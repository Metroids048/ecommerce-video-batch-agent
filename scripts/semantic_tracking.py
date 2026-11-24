"""Bounded, cached-anchor trial; never signs semantic coverage or creates masks."""
import copy
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import carrier_tracking
import project
import semantic_scan
import source_observation as observation
import workflow


def quad(box):
    x,y,w,h=box
    return np.float32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])


def bbox(q):
    lo=q.min(axis=0);hi=q.max(axis=0)
    return [float(v) for v in (*lo,*(hi-lo))]


def iou(a,b):
    x,y,w,h=a;xx,yy,ww,hh=b
    common=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
    return common/(w*h+ww*hh-common) if common else 0.


def propagate(grays,root,seed):
    """Track one coarse instance; loss never means absence or interpolation."""
    results={root:dict(bbox=seed,root=root,root_score=1.)}
    for direction in (-1,1):
        previous=root;q=quad(seed)
        for n in range(root+direction,len(grays) if direction>0 else -1,direction):
            found=carrier_tracking.register(grays[previous],grays[n],q)
            if found is None:break
            q=found['word_quad']
            score=carrier_tracking.appearance_score(grays[root],grays[n],quad(seed),q)
            if score<.6:break
            results[n]=dict(bbox=bbox(q),root=root,root_score=score,support=found['support'])
            previous=n
    return results


def trial(case,spec):
    from production_pipeline import require_production
    require_production(case)
    if case.get('execution_policy',{}).get('mode')!='method_trial':
        raise ValueError('Tracking trial is not a production semantic scanner')
    fid=spec['file_id'];item=workflow.item_for(case,fid)
    lo,hi=spec['frames_half_open'];anchors=spec['anchor_frames']
    if (type(lo) is not int or type(hi) is not int or not 0<=lo<hi<=item['source_frame_count_inventory']
            or anchors!=[lo,hi-1] or hi-lo>32):
        raise ValueError('Freeze a bounded native interval with two endpoint anchors (at most 32 frames)')
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case,state);entry=state['files'][fid]
        if entry.get('route_handoff') or entry.get('material_handoff') or entry.get('current_candidate_id'):
            raise ValueError('Source stage already consumed; preserve downstream work')
        saved=copy.deepcopy(entry['source_semantic_execution'])
        cache,_=semantic_scan.recover_journal(item,saved)
        if any(n not in cache for n in range(lo,hi)):
            raise ValueError('First trial requires existing held-out predictions; do not repeat model inference')
        binding=dict(dependency=workflow.requirement_binding(case,item),frames_half_open=[lo,hi],
            anchor_frames=anchors,authorization=spec['authorization'],
            predictions={str(n):saved['frame_artifacts'][str(n)] for n in range(lo,hi)},
            implementation={p.name:project.digest(p) for p in (Path(__file__),Path(carrier_tracking.__file__))},
            runtime=dict(opencv=cv2.__version__,numpy=np.__version__,
                opencv_build=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()))
        authorization=observation.read_artifact(spec['authorization'],'Tracking authorization')
        if authorization.get('allow_keyframe_tracking') is not True or authorization.get('keep_every_frame_review') is not True:
            raise ValueError('Explicit localization change and preserved per-frame review required')
        old=entry.get('source_tracking_trial')
        if old:
            if old['binding']!=binding:raise ValueError('Preserve frozen tracking trial; do not retune or replace it')
            observation.read_artifact(old['result'],'Tracking trial')
            return old
        started=time.time();cap=cv2.VideoCapture(str(project.project_path(item['source'])))
        grays=[];frames=[]
        try:
            for n in range(hi):
                ok,image=cap.read()
                if not ok:raise ValueError('Cannot decode original tracking interval')
                if n<lo:continue
                if [image.shape[1],image.shape[0]]!=item['dimensions']:raise ValueError('Native dimensions changed')
                frames.append(dict(frame=n,decoded_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
                grays.append(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
        finally:cap.release()
        decode_seconds=time.time()-started;operations={};tracking_started=time.time()
        for op in item['operations']:
            oid=op['id'];roots=[]
            for n in anchors:
                prediction=cache[n]['operations'][oid]
                # Coarse multi-instance identity under occlusion is not solved by a rigid carrier tracker.
                if prediction['state']!='visible' or len(prediction['instances'])!=1:
                    raise ValueError('This fixed coarse-box trial requires one visible instance at each anchor')
                roots.append(propagate(grays,n-lo,prediction['instances'][0]['bbox']))
            rows=[]
            for offset,frame in enumerate(frames):
                candidates=[r[offset] for r in roots if offset in r];risks=[]
                overlap=iou(candidates[0]['bbox'],candidates[1]['bbox']) if len(candidates)==2 else None
                if len(candidates)!=2:risks.append('track_loss_requires_semantic_refresh')
                elif overlap<.8:risks.append('bidirectional_disagreement_requires_semantic_refresh')
                if any(c['bbox'][0]<0 or c['bbox'][1]<0 or c['bbox'][0]+c['bbox'][2]>item['dimensions'][0]
                       or c['bbox'][1]+c['bbox'][3]>item['dimensions'][1] for c in candidates):
                    risks.append('frame_edge_requires_native_review')
                heldout=cache[frame['frame']]['operations'][oid]
                comparisons=[iou(c['bbox'],i['bbox']) for c in candidates for i in heldout['instances']]
                # Cached VLM outputs are comparison data, not ground truth or a coverage verifier.
                rows.append(dict(frame=frame['frame'],state='unresolved',all_instances_observed=False,
                    predicted_instances=candidates,bidirectional_iou=overlap,risks=risks,
                    heldout_model_max_iou=max(comparisons) if comparisons else None,
                    semantic_coverage='requires_native_review',occlusion='requires_native_review',
                    quality='requires_native_review'))
            operations[oid]=dict(records=rows,refresh_frames=[r['frame'] for r in rows if r['risks']])
        result=dict(schema='source_tracking_trial_v1',binding=binding,frames=frames,operations=operations,
            accepted=False,status='native_hard_case_review_required',decode_seconds=decode_seconds,
            tracking_seconds=time.time()-tracking_started,wall_seconds=time.time()-started,
            cached_anchor_count=2,new_model_calls=0,api_calls=0,
            scope='Coarse localization only. Native coverage/occlusion/quality review required for EVERY frame; boxes are never masks.')
        directory=workflow.state_path(case).parent/'source-tracking-trial';directory.mkdir(exist_ok=True)
        path=directory/'result.json'
        if path.exists():raise ValueError('Preserve existing tracking result; recover state before rerunning')
        with path.open('x') as stream:
            import json
            json.dump(result,stream,ensure_ascii=False,indent=2)
        trial=dict(binding=binding,result=observation.artifact(path),status=result['status'],wall_seconds=result['wall_seconds'])
        entry['source_tracking_trial']=trial
        return trial
