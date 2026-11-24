"""Bounded failure-frame diagnosis; never replaces a scan or grants acceptance."""
import copy
import hashlib
import json
from pathlib import Path
import time
import cv2
import project
import workflow
import semantic_scan as semantic
import source_observation as observation


def diagnose(case, spec):
    if case.get('execution_policy',{}).get('mode')!='method_trial':
        raise ValueError('Failure-frame diagnosis requires method_trial; it cannot qualify production')
    fid=spec['file_id'];item=workflow.item_for(case,fid);frames=spec.get('frames')
    if (not isinstance(frames,list) or not 1<=len(frames)<=8 or len(set(frames))!=len(frames)
            or any(type(n) is not int or not 0<=n<item['source_frame_count_inventory'] for n in frames)):
        raise ValueError('Declare 1–8 distinct original failure frames, not a whole-video rerun')
    diagnosis=observation.read_artifact(spec['diagnosis'],'Failure diagnosis')
    if diagnosis.get('source_sha256')!=item['source_sha256'] or diagnosis.get('frames')!=frames or not diagnosis.get('reason'):
        raise ValueError('Diagnosis must bind the original source and exact failure frames')
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case,state);entry=state['files'][fid]
        old=entry.get('source_semantic_execution',{})
        if not old.get('result') or old.get('binding',{}).get('strategy')!='keyframe_tracking':
            raise ValueError('Preserved completed hybrid predictions are required')
        if entry.get('route_handoff') or entry.get('material_handoff') or entry.get('current_candidate_id'):
            raise ValueError('Reopen affected dependencies explicitly before diagnosis')
        parents={}
        for n in frames:
            ref=old['frame_artifacts'].get(str(n))
            packet=observation.read_artifact(ref,'Original failure frame') if ref else {}
            if packet.get('kind')!='model' or packet.get('frame')!=n:
                raise ValueError('Each diagnosis frame must reuse an original model anchor')
            original=observation.read_artifact(packet['origin'],'Original anchor')
            parents[str(n)]=dict(packet=ref,anchor=packet['origin'],image_sha256=original['image_sha256'],
                                decoded_bgr_sha256=packet['decoded_bgr_sha256'])
        prompt=semantic.prompt_for(case,item,'yxyx')
        binding=dict(source_sha256=item['source_sha256'],dependency=workflow.requirement_binding(case,item),
            frames=frames,diagnosis=spec['diagnosis'],parents=parents,prompt_sha256=workflow.hash_json(prompt),
            code_sha256=project.digest(Path(__file__)))
        directory=workflow.state_path(case).parent/'semantic-diagnostics'/fid
        binding_path=directory/'binding.json';result_path=directory/'result.json'
        if binding_path.exists():
            if project.read_json(binding_path)!=binding:raise ValueError('Preserve fixed diagnosis; inputs or implementation changed')
            if result_path.exists():
                result=project.read_json(result_path)
                for ref in result['artifacts']:observation.read_artifact(ref,'Diagnosis result')
                return result
            raise ValueError('Preserved interrupted diagnosis needs explicit failure investigation; do not blindly retry')
        semantic.require_scan_capacity()
        engine=semantic.engine_binding(spec)
        before=copy.deepcopy(old['engine']);after=copy.deepcopy(engine)
        before.pop('code',None);after.pop('code',None)
        if before!=after:raise ValueError('Failure-frame comparison must keep the original model and runtime')
        if old['engine'].get('code',{}).get('semantic_worker.py')!=engine.get('code',{}).get('semantic_worker.py'):
            raise ValueError('Failure-frame comparison must keep the original worker')
        if project.digest(project.project_path(item['source']))!=item['source_sha256']:
            raise ValueError('Original source changed')
        directory.mkdir(parents=True,exist_ok=True)
        binding_path.write_text(json.dumps(binding,ensure_ascii=False,indent=2))
        (directory/'prompt.txt').write_text(prompt)
        (directory/'engine.json').write_text(json.dumps(engine,indent=2))
        started=time.time();refs=[];cap=cv2.VideoCapture(str(project.project_path(item['source'])))
        try:
            with semantic.LocalModel(engine,prompt,directory,[op['id'] for op in item['operations']]) as model:
                for n in frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,image=cap.read()
                    if not ok or hashlib.sha256(image.tobytes()).hexdigest()!=parents[str(n)]['decoded_bgr_sha256']:
                        raise ValueError('Seek did not return the exact original decoded frame')
                    png=directory/'input.png'
                    if not cv2.imwrite(str(png),image) or project.digest(png)!=parents[str(n)]['image_sha256']:
                        raise ValueError('Diagnostic image differs from original model input')
                    text=model.predict(n)
                    row=dict(frame=n,text=text,operations=semantic.parse_prediction(text,item,n,'yxyx'),
                        parent=parents[str(n)],binding_sha256=workflow.hash_json(binding),accepted=False)
                    path=directory/f'frame-{n:08}.json';path.write_text(json.dumps(row,ensure_ascii=False,indent=2))
                    refs.append(observation.artifact(path))
        except Exception as exc:
            (directory/'failure.json').write_text(json.dumps(dict(error=str(exc),at=workflow.now(),artifacts=refs)))
            raise
        finally:
            cap.release();(directory/'input.png').unlink(missing_ok=True)
        result=dict(status='diagnostic_predictions_need_review',file_id=fid,frames=frames,
            artifacts=refs,seconds=time.time()-started,new_model_calls=len(frames),accepted=False,
            scope='Original scan and quality gates unchanged; no full scan, masks, edited video or acceptance')
        result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        entry.setdefault('source_semantic_diagnostics',[]).append(observation.artifact(result_path))
        return result
