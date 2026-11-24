"""One frozen component-tracking experiment inside the canonical maintenance pipeline."""
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import project
import workflow

_ACTIVE = None
REVIEW_CHECKS = ('coverage','target_asset','protection','texture_geometry','occlusion','temporal_seams')


def protection_layers(existing, replacement, retained_imports=None):
    if retained_imports is None:
        return existing + [replacement]
    if (len(set(retained_imports)) != len(retained_imports) or
            not set(retained_imports) <= {p['segmentation_import'] for p in existing}):
        raise ValueError('Assembly may retain only explicitly bound existing protection')
    return [p for p in existing if p['segmentation_import'] in retained_imports] + [replacement]


def validate_review(ticket, audit):
    output=ticket['output']
    if (audit['output_sha256'] != output['sha256'] or
            project.digest(project.project_path(output['path'])) != output['sha256'] or
            audit['source_sha256'] != ticket['recipe']['dependency']['source_sha256'] or
            audit['verdict'] not in ('pass','fail') or not audit['diagnosis'] or not audit['observed_frames']):
        raise ValueError('Review must identify this exact probe and actual observations')
    if project.digest(project.project_path(output['plan'])) != output['plan_sha256']:
        raise ValueError('Probe plan changed')
    for e in audit['evidence']:
        if project.digest(project.project_path(e['path'])) != e['sha256']:
            raise ValueError('Probe observation changed')
    if audit['verdict']=='pass':
        lo,hi=output['source_frames_half_open']
        if audit['observed_frames'] != list(range(lo,hi)):
            raise ValueError('Pass requires actual observation of all paired frames')
        if audit.get('checks') != {k:'pass' for k in REVIEW_CHECKS}:
            raise ValueError('Pass requires all six quality checks')
        for key in ('inventory','playback'):
            binding=audit[key]
            if binding not in audit['evidence']:
                raise ValueError('Pass observation must be included in bound evidence')
        inventory=project.read_json(project.project_path(audit['inventory']['path']))
        playback=project.read_json(project.project_path(audit['playback']['path']))
        details={int(Path(e['path']).stem.split('-')[-1]) for e in inventory['evidence']
                 if Path(e['path']).name.startswith('native-')}
        if (inventory['output_sha256'] != output['sha256'] or inventory['source_sha256'] != audit['source_sha256'] or
                inventory['paired_frames'] != audit['observed_frames'] or not inventory['overview_sheets'] or
                not set(ticket['recipe']['detail_frames']) <= details or
                not playback.get('ended') or playback['begin']['rate'] != 1):
            raise ValueError('Pass requires same-output full pairing, native details and 1x playback')


def reassemble(case, spec, ticket):
    """Revalidate cached materials after diagnosed assembly repair; never prepare a new unit."""
    from production_pipeline import implementation_binding, require_research, controlled
    require_research(case)
    if controlled(case):
        raise ValueError('Fixed method trial consumed; diagnose a new method in explicit development scope, retain the failed assembly')
    review=ticket['review']
    if (ticket['status'] != 'fail' or spec['failure_review_sha256'] != review['sha256'] or
            project.digest(project.project_path(review['path'])) != review['sha256']):
        raise ValueError('Cached assembly requires the exact preserved failure')
    validate_review(ticket,project.read_json(project.project_path(review['path'])))
    if (ticket['recipe_sha256'] != project.digest(project.project_path(ticket['recipe_path'])) or
            ticket['recipe']['dependency'] != workflow.requirement_binding(case,workflow.item_for(case,spec['file_id']))):
        raise ValueError('Cached source or provider recipe changed')
    plan=project.read_json(project.project_path(ticket['recipe']['base_plan']))
    protection_layers(plan.get('segmentation_protection',[]),{},spec['retained_protection_imports'])
    out=project.project_path(spec['out_dir'])
    if out.exists():raise ValueError('Preserve previous assembly output')
    binding=implementation_binding()
    modules=['test_component_probe','test_source_recolor','test_video_segmentation','test_interval_edit',
             'test_production_pipeline','test_runninghub_adapter','test_source_timeline','test_production_effort','test_execution_plan']
    r=subprocess.run([sys.executable,'-m','unittest',*modules],cwd=project.ROOT/'scripts',capture_output=True,text=True)
    receipt=project.project_path(spec['out']);receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(dict(binding=binding,tests=modules,returncode=r.returncode,
        output=r.stdout+r.stderr,at=workflow.now(),new_model_calls=0),ensure_ascii=False,indent=2)+'\n')
    if r.returncode or binding != implementation_binding():raise ValueError('Offline assembly verification failed')
    current=dict(ticket,assembly=dict(retained_protection_imports=spec['retained_protection_imports'],
        out_dir=spec['out_dir'],failure_review=review),implementation=binding,
        verification=dict(path=spec['out'],sha256=project.digest(receipt)),status='cached_assembly_pending_render')
    current.pop('output');current.pop('review')
    with workflow.locked_state(case) as state:
        entry=state['files'][spec['file_id']]
        if entry['component_probe'] != ticket:raise ValueError('Probe changed during assembly verification')
        entry.setdefault('component_probe_setup_history',[]).append(ticket)
        entry['component_probe']=current
    return dict(status=current['status'],verification=current['verification'],new_model_calls=0,accepted=False)


def active():
    if _ACTIVE is not None:
        from production_pipeline import implementation_binding
        if _ACTIVE['implementation'] != implementation_binding():
            raise ValueError('Maintenance probe implementation changed; offline verification required')
    return _ACTIVE


def matches(case, fid, op, model, span):
    p = active()
    if not p:
        return False
    s = p['recipe']['provider_spec']
    if (case['batch_id'], fid, op, model, span) != (p['batch_id'], p['file_id'],
            s['operation_id'], s['model_id'], s['source_frames_half_open']):
        raise ValueError('Only the frozen hardest component probe is allowed during maintenance')
    return True


def verify_spec(case, spec):
    if not active() or spec != active()['recipe']['provider_spec']:
        raise ValueError('Maintenance probe recipe changed; no parameter retries')


def freeze(case, spec):
    from production_pipeline import require_research, controlled
    require_research(case)
    from production_pipeline import implementation_binding
    import runninghub_adapter as provider
    if project.read_json(project.ROOT/'config/models.json').get('execution_mode') != 'maintenance':
        raise ValueError('Freeze method evaluation only during maintenance')
    recipe = project.read_json(project.project_path(spec['recipe']))
    s = recipe['provider_spec']; item = workflow.item_for(case,s['file_id'])
    state = project.read_json(workflow.state_path(case)); workflow.verify_state_case(case,state)
    e = state['files'][item['id']]
    previous=e.get('component_probe')
    if controlled(case) and previous:
        raise ValueError('Frozen method research already registered; consume its original trial without replacing the candidate')
    if previous and any(u['unit_id']==previous['recipe']['provider_spec']['unit_id'] for u in e['execution_units']):
        review=previous.get('review',{})
        if (previous['status'] != 'fail' or not review or
                project.digest(project.project_path(review['path'])) != review['sha256'] or
                recipe.get('failed_component_review_sha256') != review['sha256'] or
                s['model_id'] == previous['recipe']['provider_spec']['model_id']):
            raise ValueError('Component method already prepared; consume the original probe or diagnose a distinct method')
    if not isinstance(s.get('prompt'),str) or not s['prompt'].strip():
        raise ValueError('Frozen provider recipe requires its explicit prompt metadata')
    if (recipe['dependency'] != workflow.requirement_binding(case,item) or
            s['model_id'] not in ('sam2-video-large-multiobject','sam3-video-dense-components','kling-o3-std-edit') or
            s['source_frames_half_open'] != e['interval_edit_route']['probe_frames_half_open'] or
            e.get('material_handoff') or e.get('current_candidate_id') or
            any(u['status'].upper() not in ('PREPARED','FAILED','RAW_READY') for u in e['execution_units'])):
        raise ValueError('Probe must preserve source/requirements and the original hardest interval, with production idle')
    failed = provider.unit_for(state,item['id'],recipe['failed_native_unit'])
    review = failed['raw_review']
    if (review['status'] not in ('fail','fail_or_limited') or project.digest(project.project_path(review['path'])) != review['sha256'] or
            recipe['failed_review_sha256'] != review['sha256']):
        raise ValueError('Method change requires the exact preserved native protection failure')
    if project.digest(project.project_path(recipe['base_plan'])) != recipe['base_plan_sha256']:
        raise ValueError('Existing source material changed')
    from video_segmentation import parameters, dense_parameters, DENSE_MODEL_ID
    fps=float(__import__('fractions').Fraction(project.read_json(project.project_path(item['source_timeline']))['fps']))
    if s['model_id'] == 'kling-o3-std-edit':
        from native_colour_guide import references
        references(case,item,recipe)
        if s['parameters'] != {'keepOriginalSound':False}:
            raise ValueError('Native guide preserves source audio locally')
    elif s['model_id'] == DENSE_MODEL_ID:
        dense_parameters(s['parameters'],item['dimensions'],fps,s['source_frames_half_open'][1]-s['source_frames_half_open'][0])
        if project.digest(project.project_path(s['seed_atlas'])) != recipe['seed_atlas_sha256']:
            raise ValueError('Frozen dense source seeds changed')
    else:
        parameters(s['parameters'],item['dimensions'],fps,multiobject=True)
    modules=['test_component_probe','test_video_segmentation','test_interval_edit','test_production_pipeline',
             'test_runninghub_adapter','test_source_timeline','test_production_effort','test_execution_plan','test_source_recolor']
    binding = implementation_binding()
    r = subprocess.run([sys.executable,'-m','unittest',*modules],cwd=project.ROOT/'scripts',capture_output=True,text=True)
    out=project.project_path(spec['out']);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(binding=binding,tests=modules,returncode=r.returncode,
        output=r.stdout+r.stderr,new_model_calls=0,at=workflow.now()),ensure_ascii=False,indent=2)+'\n')
    if r.returncode or binding != implementation_binding():
        raise ValueError('Offline probe checks failed; maintenance remains active')
    ticket=dict(batch_id=case['batch_id'],file_id=item['id'],recipe=recipe,
        recipe_path=spec['recipe'],recipe_sha256=project.digest(project.project_path(spec['recipe'])),
        implementation=binding,verification=dict(path=spec['out'],sha256=project.digest(out)),status='frozen_pending_hard_probe')
    with workflow.locked_state(case) as state:
        if previous:
            state['files'][item['id']].setdefault('component_probe_setup_history',[]).append(previous)
        state['files'][item['id']]['component_probe']=ticket
    return dict(status=ticket['status'],verification=ticket['verification'],accepted=False)


def render(case, ticket):
    from source_recolor import Layer, validate
    from video_segmentation import validate_import
    recipe=ticket['recipe'];s=recipe['provider_spec'];item=workflow.item_for(case,ticket['file_id'])
    plan=project.read_json(project.project_path(recipe['base_plan']))
    if project.digest(project.project_path(recipe['base_plan'])) != recipe['base_plan_sha256']:
        raise ValueError('Base plan changed')
    assembly=ticket.get('assembly',{})
    out=project.project_path(assembly.get('out_dir',recipe['out_dir']));out.mkdir(parents=True,exist_ok=False)
    if s['model_id'] == 'kling-o3-std-edit':
        from native_colour_guide import prepare_plan
        plan=prepare_plan(case,item,ticket,plan,out)
    else:
        plan=segmentation_plan(case,item,ticket,plan)
    validate(case,item,plan)
    pixels=out/'pixels.json';pixels.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    return encode_probe(case,item,ticket,plan,out,pixels)


def segmentation_plan(case,item,ticket,plan):
    from video_segmentation import validate_import
    recipe=ticket['recipe'];s=recipe['provider_spec']
    sidecar=recipe['mask_out']+'.json';guard=project.read_json(project.project_path(sidecar))
    protection=dict(masks=guard['masks'],edit_frames_half_open=guard['frames_half_open'],
        segmentation_backend=guard['backend'],segmentation_import=sidecar,
        segmentation_import_sha256=project.digest(project.project_path(sidecar)),
        artifact_hashes={p:project.digest(project.project_path(p)) for p in (sidecar,guard['masks'])})
    # Superseded wrong-object masks must not remain in the protection union.
    assembly=ticket.get('assembly',{})
    plan['segmentation_protection']=protection_layers(plan.get('segmentation_protection',[]),protection,
        assembly.get('retained_protection_imports'))
    validate_import(case,item,dict(protection,operation_id=s['operation_id']))
    return plan


def encode_probe(case,item,ticket,plan,out,pixels):
    from source_recolor import Layer
    recipe=ticket['recipe'];s=recipe['provider_spec']
    lo,hi=s['source_frames_half_open'];fps=project.read_json(project.project_path(item['source_timeline']))['fps']
    cap=cv2.VideoCapture(str(project.project_path(item['source'])));cap.set(1,lo)
    video=out/'hard-probe.mp4';w,h=item['dimensions']
    encoder=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}',
        '-r',fps,'-i','-','-an','-c:v','libx264','-crf','12','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    layer=Layer(plan,dict(modified_source_frames=[],modified_instance_count=0))
    protected_delta=0
    try:
        for n in range(lo,hi):
            ok,frame=cap.read()
            if not ok:raise ValueError('Incomplete source probe')
            edited,_=layer.compose(frame,n)
            fixed=layer.fixed_protection(frame,n)!=0
            if fixed.any():protected_delta=max(protected_delta,int(np.abs(edited.astype(np.int16)-frame)[fixed].max()))
            encoder.stdin.write(edited.tobytes())
    finally:
        cap.release();encoder.stdin.close()
    if encoder.wait():raise ValueError('Probe encode failed; retain partial artifact')
    output=dict(path=str(video.relative_to(project.ROOT)),sha256=project.digest(video),
        plan=str(pixels.relative_to(project.ROOT)),plan_sha256=project.digest(pixels),
        source_frames_half_open=[lo,hi],protected_max_delta_before_encode=protected_delta,status='pending_actual_review')
    with workflow.locked_state(case) as state:state['files'][item['id']]['component_probe']['output']=output
    import review_queue
    return review_queue.build_pairs(case,item['id'],str((out/'review').relative_to(project.ROOT)),
        dict(maintenance_probe=True,stills=recipe['detail_frames']))


def dispatch(case, spec):
    global _ACTIVE
    if spec['action']=='freeze-component-probe':return freeze(case,spec)
    if project.read_json(project.ROOT/'config/models.json').get('execution_mode') != 'maintenance':
        raise ValueError('Method probe is isolated from production')
    ticket=project.read_json(workflow.state_path(case))['files'][spec['file_id']]['component_probe']
    if spec['probe_action']=='reassemble':return reassemble(case,spec,ticket)
    if spec['probe_action']=='review':
        audit=project.read_json(project.project_path(spec['review']))
        if ticket['recipe']['dependency'] != workflow.requirement_binding(case,workflow.item_for(case,spec['file_id'])):
            raise ValueError('Probe review source/requirements changed')
        if audit['verdict']=='pass':
            from production_pipeline import implementation_binding
            if ticket['implementation'] != implementation_binding():
                raise ValueError('Probe implementation changed before pass review')
        validate_review(ticket,audit)
        with workflow.locked_state(case) as state:
            current=state['files'][spec['file_id']]['component_probe']
            if current != ticket:raise ValueError('Probe changed during review')
            current.update(status=audit['verdict'],review=dict(path=spec['review'],sha256=project.digest(project.project_path(spec['review']))))
        return dict(status=audit['verdict'],accepted=False)
    if (ticket['batch_id'] != case['batch_id'] or ticket['recipe_sha256'] != project.digest(project.project_path(ticket['recipe_path'])) or
            ticket['verification']['sha256'] != project.digest(project.project_path(ticket['verification']['path'])) or
            ticket['recipe']['dependency'] != workflow.requirement_binding(case,workflow.item_for(case,spec['file_id']))):
        raise ValueError('Frozen probe source/recipe/offline verification changed')
    _ACTIVE=ticket
    try:
        active()
        action=spec['probe_action'];s=ticket['recipe']['provider_spec']
        if ticket.get('assembly') and action in ('prepare','submit','query','download','import'):
            raise ValueError('Cached assembly repair consumes existing imported masks only')
        import runninghub_adapter as provider
        if action=='prepare':return provider.prepare_unit(case,s)
        unit=provider.unit_for(project.read_json(workflow.state_path(case)),s['file_id'],s['unit_id'])
        request=provider.verify_unit(case,unit)
        if (request.get('component_probe_sha256') != ticket['recipe_sha256'] or
                any(request[k] != s[k] for k in ('file_id','operation_id','model_id','source_frames_half_open'))):
            raise ValueError('Probe action must consume its original frozen unit')
        if action in ('submit','query','download'):
            return provider.remote_action(case,dict(action=action,file_id=s['file_id'],unit_id=s['unit_id'],
                authorization=ticket['recipe']['authorization'],maximum_charge_cny=ticket['recipe']['maximum_charge_cny']))
        if action=='import':
            from video_segmentation import import_masks
            return import_masks(case,s,ticket['recipe']['mask_out'])
        if action=='render':return render(case,ticket)
        if action=='playback':
            import review_queue
            return review_queue.native_playback(case,s['file_id'],ticket.get('assembly',{}).get('out_dir',ticket['recipe']['out_dir'])+'/playback',spec=dict(maintenance_probe=True))
        raise ValueError('Unsupported maintenance probe action')
    finally:
        _ACTIVE=None
