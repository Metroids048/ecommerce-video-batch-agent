"""Scheduled VLM anchors and checked native-frame proposals; quality stays pending."""
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import carrier_tracking
import project
import semantic_scan as semantic
import source_observation as observation
import workflow

INTERVAL=24


def recover_packets(item,saved):
    directory=project.project_path(saved['directory']);cache={};refs={}
    for ref in saved.get('frame_artifacts',{}).values():observation.read_artifact(ref,'Hybrid frame')
    for path in sorted(directory.glob('frame-*.json')):
        p=project.read_json(path);n=p['frame'];ref=observation.artifact(path)
        if (type(n) is not int or not 0<=n<item['source_frame_count_inventory'] or n in cache
                or path.name!=f'frame-{n:08}.json' or p.get('binding_sha256')!=workflow.hash_json(saved['binding'])
                or set(p['operations'])!={o['id'] for o in item['operations']}
                or not isinstance(p.get('decoded_bgr_sha256'),str)):
            raise ValueError('Hybrid journal binding changed; preserve original result')
        if str(n) in saved.get('frame_artifacts',{}) and saved['frame_artifacts'][str(n)]!=ref:
            raise ValueError('Hybrid frame hash changed')
        if p['kind']=='model':
            original=observation.read_artifact(p['origin'],'Semantic anchor')
            inherited=saved['inherited_model_frames'].get(str(n))
            if (inherited and p['origin']!=inherited or not inherited and original.get('binding_sha256')!=workflow.hash_json(saved['binding'])):
                raise ValueError('Semantic anchor is not bound to this scan/source')
            if (original['frame']!=n or original['operations']!=p['operations']
                    or p['operations']!=semantic.parse_prediction(original['text'],item,n,'yxyx')):
                raise ValueError('Hybrid anchor differs from original semantic prediction')
        elif p['kind']=='tracking':
            if any(r['state']!='unresolved' or r['all_instances_observed'] or r['instances']
                   or r['frame']!=n for r in p['operations'].values()):
                raise ValueError('Tracking cannot sign semantic visibility/absence or full coverage')
        else:raise ValueError('Unknown hybrid frame provenance')
        cache[n]=p;refs[str(n)]=ref
    if set(saved.get('frame_artifacts',{}))-set(refs):raise ValueError('Missing hybrid indexed frame')
    recovered=sorted(int(n) for n in refs if n not in saved.get('frame_artifacts',{}))
    saved['frame_artifacts']=refs
    return cache,recovered


def tracked_rows(item,n,previous,gray,last_gray,root_gray,root_boxes):
    from semantic_tracking import quad,bbox
    rows={}
    for op in item['operations']:
        oid=op['id'];old=previous[oid]
        instances=old['instances'] if old['state']=='visible' else old.get('predicted_instances',[])
        proposals=[];checks=[]
        for index,inst in enumerate(instances):
            q=quad(inst['bbox']);found=carrier_tracking.register(last_gray,gray,q)
            if found is None:checks.append('track_loss');continue
            root=root_boxes.get(oid,[])
            score=carrier_tracking.appearance_score(root_gray,gray,quad(root[index]['bbox']),found['word_quad']) if index<len(root) else -1.
            if score<.6:checks.append('root_appearance_changed_or_occluded');continue
            box=bbox(found['word_quad']);x,y,w,h=box
            if x<0 or y<0 or x+w>item['dimensions'][0] or y+h>item['dimensions'][1]:checks.append('frame_edge')
            proposals.append(dict(id=f'{oid}:f{n}:tracked{index}',bbox=box,registration_support=found['support'],root_score=score))
        if not proposals:checks.append('semantic_refresh_pending_next_anchor')
        rows[oid]=dict(frame=n,state='unresolved',instances=[],predicted_instances=proposals,
            all_instances_observed=False,reason='Checked geometric proposal; coverage, occlusion and quality await native review',
            risks=sorted(set(checks+['semantic_coverage_and_quality_pending'])))
    return rows


def scan(case,spec):
    from production_pipeline import require_production
    require_production(case)
    if case.get('execution_policy',{}).get('mode')!='method_trial':
        raise ValueError('Hybrid localization is not a qualified production method')
    fid=spec['file_id'];item=workflow.item_for(case,fid)
    path=workflow.state_path(case)
    prior=project.read_json(path)['files'][fid].get('source_semantic_execution',{}) if path.exists() else {}
    if not prior.get('result'):
        from production_pipeline import local_edit_blockers
        blockers=local_edit_blockers(case,item)
        if blockers:raise ValueError('; '.join(blockers))
    if prior.get('error') and not (isinstance(spec.get('resume_reason'),str) and spec['resume_reason'].strip()):
        raise ValueError('Diagnose preserved local failure and supply resume_reason before another model probe')
    if not prior.get('result'):semantic.require_scan_capacity()
    observation.prepare(case,dict(file_id=fid))
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case,state);entry=state['files'][fid];old=entry.get('source_semantic_execution')
        if old and old.get('result'):return semantic.progress(case,fid,entry)
        if entry.get('route_handoff') or entry.get('material_handoff') or entry.get('current_candidate_id'):
            raise ValueError('Source stage closed; consume existing materials')
        if old and old.get('error') and not (isinstance(spec.get('resume_reason'),str) and spec['resume_reason'].strip()):
            raise ValueError('Diagnose preserved local failure and supply resume_reason before another model probe')
        auth=observation.read_artifact(spec['authorization'],'Localization strategy authorization')
        if not all(auth.get(k) is True for k in ('allow_keyframe_tracking','keep_every_frame_review','latency_first_quality_repair_later','no_download_install','zero_api')):
            raise ValueError('Explicit strategy and timing-before-quality authorization required')
        engine=semantic.engine_binding(spec)
        if old and any(old['engine'].get(k)!=engine.get(k) for k in ('options','runtime','model_files')):
            raise ValueError('Hybrid strategy cannot change existing model or runtime')
        implementation={p.name:project.digest(p) for p in (Path(__file__),Path(carrier_tracking.__file__),Path(__file__).with_name('semantic_tracking.py'),Path(__file__).with_name('text_localization.py'))}
        runtime=dict(cv2=cv2.__version__,numpy=np.__version__,build=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest())
        binding=dict(dependency=workflow.requirement_binding(case,item),source_sha256=item['source_sha256'],
            queue=entry['source_observation']['queue'],engine=engine,strategy='keyframe_tracking',coordinate_order='yxyx',
            interval=INTERVAL,implementation=implementation,runtime=runtime,authorization=spec['authorization'],
            prompt_sha256=workflow.hash_json(semantic.prompt_for(case,item,'yxyx')))
        if old and old['binding'].get('strategy')=='keyframe_tracking':
            if old['binding']!=binding:raise ValueError('Frozen hybrid strategy changed; preserve scan and diagnose')
            saved=old;anchors={int(n):r for n,r in saved['inherited_model_frames'].items()}
        else:
            if old and old['binding'].get('coordinate_order')!='yxyx':raise ValueError('Diagnose coordinate order before strategy change')
            if old:
                _,_=semantic.recover_journal(item,old)
                entry.setdefault('source_semantic_execution_history',[]).append(copy.deepcopy(old))
            anchors={int(n):r for n,r in old.get('frame_artifacts',{}).items()} if old else {}
            directory=workflow.state_path(case).parent/'semantic-scan'/(fid+'-keyframe-tracking')
            if directory.exists():raise ValueError('Existing hybrid directory needs recovery; do not overwrite')
            directory.mkdir(parents=True)
            saved=dict(binding=binding,engine=engine,directory=str(directory.relative_to(project.ROOT)),
                inherited_model_frames=copy.deepcopy(old.get('frame_artifacts',{})) if old else {},
                parent_binding_sha256=workflow.hash_json(old['binding']) if old else None,
                status='prepared',wall_seconds=0)
            entry['source_semantic_execution']=saved
        cache,recovered=recover_packets(item,saved)
        if old and old.get('error'):
            saved.setdefault('resumptions',[]).append(dict(at=workflow.now(),reason=spec['resume_reason'],cached_frames=len(cache)))
        if recovered:saved.setdefault('journal_recoveries',[]).append(dict(at=workflow.now(),frames=recovered))
        from source_timeline import load_for_item
        timeline=load_for_item(item);queue=observation.queue_for(case,fid,entry['source_observation'])
        started=time.time();cap=None;new_calls=0
        try:
            cap=cv2.VideoCapture(str(project.project_path(item['source'])))
            last_gray=root_gray=None;previous=None;root_boxes={}
            saved.update(status='running');saved.pop('error',None)
            with contextlib.ExitStack() as stack:
                model=None
                for n in range(item['source_frame_count_inventory']):
                    ok,image=cap.read()
                    if not ok or [image.shape[1],image.shape[0]]!=item['dimensions']:raise ValueError('Incomplete/native source decode failed')
                    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    if n in cache:
                        p=cache[n]
                        if p['decoded_bgr_sha256']!=hashlib.sha256(image.tobytes()).hexdigest():raise ValueError('Decoded hybrid source frame changed')
                    else:
                        p=dict(frame=n,binding_sha256=workflow.hash_json(binding),decoded_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest())
                        if n in anchors or n%INTERVAL==0 or n==item['source_frame_count_inventory']-1:
                            if n in anchors:origin=anchors[n];raw=observation.read_artifact(origin,'Cached semantic frame')
                            else:
                                directory=project.project_path(saved['directory']);rawpath=directory/f'anchor-{n:08}.json'
                                if rawpath.exists():raw=project.read_json(rawpath);origin=observation.artifact(rawpath)
                                else:
                                    semantic.require_scan_capacity()
                                    if model is None:model=stack.enter_context(semantic.LocalModel(engine,semantic.prompt_for(case,item,'yxyx'),directory,[o['id'] for o in item['operations']]))
                                    if not cv2.imwrite(str(directory/'input.png'),image):raise ValueError('Cannot save native anchor input')
                                    text=model.predict(n);new_calls+=1
                                    raw=dict(frame=n,text=text,operations=semantic.parse_prediction(text,item,n,'yxyx'),
                                        binding_sha256=workflow.hash_json(binding),image_sha256=project.digest(directory/'input.png'))
                                    with rawpath.open('x') as stream:json.dump(raw,stream,ensure_ascii=False)
                                    origin=observation.artifact(rawpath)
                            if (raw['frame']!=n or raw['operations']!=semantic.parse_prediction(raw['text'],item,n,'yxyx')
                                    or n not in anchors and raw.get('binding_sha256')!=workflow.hash_json(binding)):
                                raise ValueError('Anchor frame/parser mismatch')
                            p.update(kind='model',origin=origin,operations=raw['operations'])
                        else:p.update(kind='tracking',operations=tracked_rows(item,n,previous,gray,last_gray,root_gray,root_boxes))
                        path=project.project_path(saved['directory'])/f'frame-{n:08}.json'
                        with path.open('x') as stream:json.dump(p,stream,ensure_ascii=False)
                        saved.setdefault('frame_artifacts',{})[str(n)]=observation.artifact(path);cache[n]=p
                    if p['kind']=='model':
                        root_gray=gray
                        root_boxes={oid:r['instances'] if r['state']=='visible' else [] for oid,r in p['operations'].items()}
                    previous=p['operations'];last_gray=gray
                if cap.read()[0]:raise ValueError('Source contains extra frames')
            operations={}
            for op in item['operations']:
                oid=op['id'];records=[]
                for n in range(item['source_frame_count_inventory']):
                    row=copy.deepcopy(cache[n]['operations'][oid]);row.update(source_pts=timeline['pts'][n],time_base=timeline['time_base'])
                    row['evidence']=[{k:e[k] for k in ('path','sha256')} for e in queue['evidence'] if n in e['frames']]
                    row['localization_provenance']=cache[n]['kind'];records.append(row)
                operations[oid]=dict(records=records,quality_requirements=[json.dumps(op,ensure_ascii=False)])
            summaries,_=observation.summarize(item,operations)
            for oid,summary in summaries.items():operations[oid].update(summary)
            result=dict(schema='source_semantic_scan_v1',file_id=fid,source_sha256=item['source_sha256'],
                source_frame_count=item['source_frame_count_inventory'],dependency=binding['dependency'],queue=binding['queue'],
                engine=engine,strategy=binding['strategy'],operations=operations,accepted=False,
                source_timeline=dict(path=item['source_timeline'],sha256=item['source_timeline_sha256']),
                frame_artifacts=saved['frame_artifacts'],scope='All native frames checked geometrically; semantic coverage/occlusion/quality remain pending. No masks or quality signoff.')
            path=project.project_path(saved['directory'])/'result.json'
            if path.exists() and project.read_json(path)!=result:raise ValueError('Preserve existing hybrid result')
            if not path.exists():
                with path.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
            saved.update(status='completed',result=observation.artifact(path))
        except Exception as exc:
            saved.update(status='failed',error=str(exc));saved.setdefault('failures',[]).append(dict(at=workflow.now(),error=str(exc),frames=len(cache)))
        finally:
            if cap is not None:cap.release()
            (project.project_path(saved['directory'])/'input.png').unlink(missing_ok=True)
            saved['wall_seconds']+=time.time()-started;saved['new_model_calls']=saved.get('new_model_calls',0)+new_calls
        result=semantic.progress(case,fid,entry);error=saved.get('error')
    if error:raise ValueError(error)
    return result
