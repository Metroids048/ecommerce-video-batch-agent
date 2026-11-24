"""Source-bound provider units in the canonical workflow; offline by default.

No credential is read here. The transport is reached only after policy, batch
consent, contract and budget checks. Historical replay never counts as a submit.
"""
import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import project
import source_timeline
import workflow


def policy(batch_id=None):
    p = project.read_json(project.ROOT/'config/models.json')
    return dict(p, **p.get('scoped_remote_batches', {}).get(batch_id, {}))


def require_remote(action, batch_id=None):
    if batch_id:
        saved = project.ROOT/'runs'/batch_id/'workflow.json'
        if saved.is_file() and project.read_json(saved).get('case_binding', {}).get('execution_policy', {}).get('local_only') is True:
            raise ValueError('Local-only order: every remote action is disabled, including query and download')
    from production_pipeline import require_production
    require_production(batch_id=batch_id)
    p = policy(batch_id)
    if p.get('remote_calls_allowed') is not True:
        raise ValueError('All remote calls blocked: including query, quote, upload and download')
    if action in ('upload', 'submit') and p.get('media_upload_allowed') is not True:
        raise ValueError('Media upload is not authorized')
    if action == 'submit' and not all(p.get(k) is True for k in ('paid_submission_allowed','remote_submission_allowed')):
        raise ValueError('Model submission is not authorized')
    if not p.get('adapter', {}).get('contract_current_verified'):
        raise ValueError('Current official contract has not been verified')


def model_for(model_id, batch_id=None):
    p = policy(batch_id)
    return next((m for m in [p['primary']]+p.get('comparators', []) if m['id'] == model_id), None)


def unit_for(state, file_id, unit_id):
    unit = next((u for u in state['files'][file_id].get('execution_units', []) if u['unit_id'] == unit_id), None)
    if unit is None:
        raise ValueError('Unknown execution unit')
    return unit


def reference_dimensions(item, model):
    """Fit the provider input grid without cropping any original image content."""
    w,h = item['dimensions']; limits = model['limits']
    factor = max(1, limits.get('min_reference_width',w)/w,
                 limits.get('min_reference_height',h)/h)
    dimensions = [math.ceil(w*factor/2)*2, math.ceil(h*factor/2)*2] if factor > 1 else [w,h]
    if any(n > limits.get('max_reference_dimension',float('inf')) for n in dimensions):
        raise ValueError('Reference dimensions exceed provider limits; cropping is forbidden')
    return dimensions


def verify_submission_dimensions(case, request, model):
    if request.get('source_crop_xywh') is not None and model.get('mode') != 'mask-video-app':
        raise ValueError('Source crop is not supported by this provider route; do not submit an ignored crop')
    expected = reference_dimensions(workflow.item_for(case,request['file_id']),model)
    if any(k in model['limits'] for k in ('min_reference_width','min_reference_height')):
        metadata = project.probe(project.project_path(request['reference_clip']))
        actual = [metadata['width'],metadata['height']]
        if actual != expected or request.get('reference_dimensions') != expected:
            raise ValueError('Prepared reference predates provider dimension validation; no submission')


def raw_origin_binding(unit):
    """Bind the consumed result, not unrelated changes to its parent's journal."""
    return workflow.hash_json({k:unit.get(k) for k in (
        'operation_id', 'dependency', 'request_fingerprint', 'provider_task_id', 'raw', 'raw_review')})


def verify_unit(case, unit):
    item = workflow.item_for(case, unit['file_id'])
    if unit['dependency'] != workflow.requirement_binding(case, item):
        raise ValueError('Unit requirements/source/assets changed')
    if project.digest(project.project_path(item['source'])) != item['source_sha256']:
        raise ValueError('Unit actual source changed')
    request = project.read_json(project.project_path(unit['request_path']))
    if workflow.hash_json(request) != unit['request_fingerprint']:
        raise ValueError('Unit request fingerprint changed')
    from production_effort import provider_first
    if provider_first(case) and not unit.get('local_origin'):
        adapted = request.get('editing_input')
        if adapted is None:
            raise ValueError('Prepared request lacks operation-scoped editing input')
        op = next(o for o in item['operations'] if o['id'] == request['operation_id'])
        prompt, images = editing_input(item, op, dict(prompt=adapted['context'],
            prompt_format=adapted.get('prompt_format', 1),
            image_references=adapted['image_references'], source_frames_half_open=request['source_frames_half_open']))
        if request['prompt'] != prompt or request['image_references'] != images:
            raise ValueError('Prepared request lost its target, protection or exact assets')
    if request.get('parent_candidate_id'):
        state=project.read_json(workflow.state_path(case))
        parent=workflow.verify_historical_parent(case,item['id'],workflow.candidate_for(state['files'][item['id']],request['parent_candidate_id']))
        if parent['output_sha256']!=request['parent_output_sha256']:
            raise ValueError('Unit parent changed')
    for path, digest in request['artifact_hashes'].items():
        if project.digest(project.project_path(path)) != digest:
            raise ValueError('Unit input artifact changed: '+path)
    if unit.get('local_origin'):
        origin = unit['local_origin']
        for key in ('case',) if origin.get('unit_binding') else ('case', 'workflow'):
            binding = origin[key]
            if project.digest(project.project_path(binding['path'])) != binding['sha256']:
                raise ValueError('Original raw provenance changed')
        if origin.get('unit_binding'):
            parent = project.read_json(project.project_path(origin['case']['path']))
            parent_state = project.read_json(project.project_path(origin['workflow']['path']))
            workflow.verify_state_case(parent, parent_state)
            original = unit_for(parent_state, origin['file_id'], origin['unit_id'])
            if raw_origin_binding(original) != origin['unit_binding']:
                raise ValueError('Original raw provenance changed')
            if workflow.hash_json(project.read_json(project.project_path(original['request_path']))) != original['request_fingerprint']:
                raise ValueError('Original raw request changed')
            if original.get('raw_review'):
                review = original['raw_review']
                if project.digest(project.project_path(review['path'])) != review['sha256']:
                    raise ValueError('Original raw review changed')
        if project.digest(project.project_path(unit['raw']['path'])) != unit['raw']['sha256']:
            raise ValueError('Original raw media changed')
    return request


def semantic_repair_parameters(case, spec, operation):
    """Consume one diagnosed semantic correction, retaining the fixed SAM3 graph/grid."""
    from video_segmentation import fixed_recolor_parameters
    item = workflow.item_for(case, spec['file_id'])
    fixed = fixed_recolor_parameters(item, operation)
    repair = spec.get('semantic_repair')
    if not repair:
        return fixed, {}
    state = project.read_json(workflow.state_path(case))
    workflow.verify_state_case(case, state)
    prior = unit_for(state, item['id'], repair['failed_unit_id'])
    request = verify_unit(case, prior)
    review = prior.get('raw_review', {})
    anchor_repair = 'anchor_frame' in repair
    point_repair = repair.get('conditioning') == 'instance_points'
    part_decomposition = (request.get('semantic_repair',{}).get('role') == 'target'
                          and repair.get('role') not in (None,'target'))
    if ((request.get('semantic_repair') and not anchor_repair and not part_decomposition and not point_repair)
            or (request.get('parameters',{}).get('anchor_frame') is not None and not point_repair)
            or request.get('parameters',{}).get('point_prompt') is not None
            or request['model_id'] != 'sam3-video-semantic'
            or prior['operation_id'] != operation['id'] or review.get('status') != 'fail_or_limited'
            or review.get('raw_sha256') != prior.get('raw', {}).get('sha256')):
        raise ValueError('Semantic correction requires the original failed SAM3 review; no prompt trial chains')
    path = project.project_path(review['path'])
    if project.digest(path) != review['sha256']:
        raise ValueError('Semantic failure evidence changed')
    observed = project.read_json(path)
    if not any(c['kind'] in ('coverage', 'protection') and c['status'] == 'fail' for c in observed['checks']):
        raise ValueError('Semantic correction requires an observed coverage/protection failure')
    if point_repair:
        anchor = observed.get('instance_point_anchors', {}).get(repair['role'], {})
        frame = anchor.get('frame')
        if (not observed.get('diagnosis') or type(frame) is not int
                or not 0 <= frame < item['source_frame_count_inventory']
                or frame not in observed.get('observed_frames', []) or not anchor.get('point_prompt')):
            raise ValueError('SAM3 instance conditioning requires frozen observed part points')
        fixed.update(text_prompt=request['parameters']['text_prompt'],anchor_frame=frame,
                     point_prompt=anchor['point_prompt'])
        return fixed, {review['path']: review['sha256']}
    if anchor_repair:
        anchor = observed.get('semantic_anchor', {})
        frame = repair['anchor_frame']
        if (type(frame) is not int or not 0 <= frame < item['source_frame_count_inventory']
                or frame not in observed.get('observed_frames', []) or anchor.get('frame') != frame
                or anchor.get('text_prompt') != request['parameters']['text_prompt'] or not observed.get('diagnosis')):
            raise ValueError('SAM3 anchor correction requires a frozen observed visible frame and unchanged label')
        fixed.update(text_prompt=request['parameters']['text_prompt'],anchor_frame=frame)
        return fixed, {review['path']: review['sha256']}
    labels = observed.get('semantic_repair', {})
    label = labels.get(repair['role'])
    if not observed.get('diagnosis') or not isinstance(label, str) or not 1 <= len(label) <= 200:
        raise ValueError('Semantic correction must consume the diagnosed, frozen target/protection label')
    fixed['text_prompt'] = label
    return fixed, {review['path']: review['sha256']}


def editing_input(item, operation, spec):
    intent = {k: operation[k] for k in ('id', 'kind', 'target', 'semantic_label',
        'desired_change', 'recolor_mode', 'target_color_rgb', 'protect', 'replacement_text',
        'replacement_asset', 'reference_asset') if k in operation}
    intent['source_frames_half_open'] = spec['source_frames_half_open']
    images = list(spec.get('image_references', []))
    for key in ('replacement_asset', 'reference_asset'):
        if operation.get(key) and operation[key] not in images:
            images.append(operation[key])
    if spec.get('prompt_format', 1) == 2:
        # Put the actual edit first. Keep the structured intent in the immutable
        # request binding, rather than asking a video model to interpret JSON.
        prompt = ((str(spec['prompt']).strip() + '\n' if spec.get('prompt') else '')
            + str(operation['desired_change']).strip() + '\nTarget: ' + str(operation['target'])
            + ('\nTarget color RGB: ' + ', '.join(map(str, operation['target_color_rgb']))
               if operation.get('target_color_rgb') else '')
            + '\nPreserve: ' + ', '.join(operation.get('protect', []))
            + '\nChange every visible part of this target, including after occlusion. Keep the original texture, '
              'folds, lighting, objects, text, hands, actions, camera and cuts exactly unchanged. '
              'Do not add text or objects.')
        return prompt, images
    prompt = ('Identify targets by original appearance; requested color/text is the result. '
        'Edit every visible/partial instance only. Preserve protected items, other objects, actions, camera, '
        'geometry, texture, text, frame order/duration. Use exact supplied assets; invent nothing, hide no failures. '
        'Source audio restored.\n'
        + json.dumps(intent, ensure_ascii=False, sort_keys=True)
        + ('\nClip context: ' + str(spec['prompt']).strip() if spec.get('prompt') else ''))
    return prompt, images


def prepare_unit(case, spec):
    from production_pipeline import guard_upstream, require_production
    require_production(case)
    from production_effort import check
    from component_probe import active, verify_spec
    if active():
        verify_spec(case, spec)
    from production_pipeline import guard_method
    guard_method(case, spec['file_id'], spec.get('operation_id'), provider_spec=spec)
    from production_pipeline import controlled
    if controlled(case) and case['execution_policy']['mode'] == 'method_trial':
        state = project.read_json(workflow.state_path(case))
        prior = [u for u in state['files'][spec['file_id']]['execution_units']
                 if u.get('operation_id') == spec.get('operation_id')]
        if any(u['unit_id'] != spec.get('unit_id') for u in prior):
            raise ValueError('Consume the original method trial; renaming units cannot start another experiment')
    configured = model_for(spec['model_id'], case['batch_id'])
    family = 'local_recolor' if configured and configured.get('mode') in ('video-segmentation', 'video-matting') else None
    check(case, spec['file_id'], start_method=True, method_family=family, operation_id=spec.get('operation_id'))
    guard_upstream(case, spec['file_id'], operation_id=spec.get('operation_id'))
    project.validate_case(case)
    item = workflow.item_for(case, spec['file_id'])
    if spec['operation_id'] not in {op['id'] for op in item['operations']}:
        raise ValueError('Unit operation does not belong to file')
    from production_effort import source_preserving_recolor, provider_first
    from production_pipeline import guard_interval_unit
    operation = next(op for op in item['operations'] if op['id'] == spec['operation_id'])
    interval = guard_interval_unit(case,item['id'],operation['id'],spec['model_id'],spec.get('source_frames_half_open'))
    if source_preserving_recolor(operation) and not provider_first(case) and not interval and not active() and spec['model_id'] != 'sam3-video-semantic':
        raise ValueError('Preserve-source recolor uses fixed SAM3 tracking; reconstruction and parameter branches are disabled')
    uid = spec['unit_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', uid):
        raise ValueError('Invalid unit id')
    timeline = source_timeline.require_cfr(source_timeline.load_for_item(item))
    lo, hi = spec['source_frames_half_open']
    if not all(type(n) is int for n in (lo, hi)) or not 0 <= lo < hi <= len(timeline['pts']):
        raise ValueError('Invalid unit source range')
    semantic_dependencies = {}
    if source_preserving_recolor(operation) and not provider_first(case) and not interval and not active():
        fixed, semantic_dependencies = semantic_repair_parameters(case, spec, operation)
        if [lo, hi] != [0, len(timeline['pts'])] or spec.get('parameters') != fixed:
            raise ValueError('Preserve-source recolor requires the fixed full-source SAM3 grid and object; no parameter trials')
    model = model_for(spec['model_id'], case['batch_id'])
    adapted_prompt, adapted_images = editing_input(item, operation, spec) if provider_first(case) else (spec.get('prompt', '').strip(), spec.get('image_references', []))
    if model is None:
        raise ValueError('Unknown configured model')
    if spec.get('source_crop_xywh') is not None and model.get('mode') != 'mask-video-app':
        raise ValueError('Source crop is not supported by this provider route; do not silently ignore source coordinates')
    reference_fps = interval['model_input_fps'] if interval else timeline['fps']
    duration = Fraction(hi-lo)/Fraction(reference_fps)
    limits = model['limits']
    reference_size = reference_dimensions(item,model)
    if duration > limits['max_reference_video_seconds_conservative'] or 2*duration > limits['max_input_plus_output_seconds'] or duration > limits['max_output_seconds']:
        raise ValueError('Unit exceeds locally recorded model limits')
    matte_dependencies = {}
    if model.get('mode') == 'video-matting':
        from video_matting import parameters as matting_parameters
        parameters, matte_dependencies = matting_parameters(case,item,spec,timeline['fps'],[lo,hi])
    elif model.get('mode') == 'video-segmentation':
        from video_segmentation import parameters as segmentation_parameters, graph, sam3_parameters, sam3_graph, dense_parameters, dense_graph, DENSE_MODEL_ID
        if spec['model_id'] == DENSE_MODEL_ID:
            import cv2
            import numpy as np
            parameters=dense_parameters(spec['parameters'],item['dimensions'],Fraction(timeline['fps']),hi-lo)
            atlas=project.project_path(spec['seed_atlas'])
            image=cv2.imread(str(atlas),cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (parameters['height'],parameters['width']*len(parameters['parts'])) or not set(np.unique(image)).issubset({0,255}):
                raise ValueError('Dense seeds must be source-sized binary component masks')
            parameters['workflow']=dense_graph(parameters)
        elif spec['model_id'] == 'sam3-video-semantic':
            parameters = sam3_parameters(spec['parameters'], item['dimensions'], Fraction(timeline['fps']))
            if parameters.get('anchor_frame',0) >= hi-lo:
                raise ValueError('SAM3 anchor outside full-source range')
            parameters['workflow'] = sam3_graph(parameters)
        else:
            parameters = segmentation_parameters(spec['parameters'], item['dimensions'], Fraction(timeline['fps']),
                multiobject=spec['model_id']=='sam2-video-large-multiobject')
            if any(a['frame'] >= hi-lo for a in parameters['anchors']):
                raise ValueError('SAM2 anchor outside unit source interval')
            parameters['workflow'] = graph(parameters)
    elif model.get('mode') == 'mask-video-app':
        parameters = dict(spec.get('parameters', {}))
        if (set(parameters) != {'webappId','width','height','instanceType'}
                or parameters['webappId'] != '1963199782695178242' or parameters['instanceType'] != 'default'
                or any(type(parameters[k]) is not int or not 128 <= parameters[k] <= 1280 or parameters[k] % 32 for k in ('width','height'))):
            raise ValueError('Unsupported mask application parameters')
        crop = spec.get('source_crop_xywh')
        if crop is not None:
            w,h=item['dimensions']
            if (len(crop)!=4 or any(type(n)is not int or n%2 for n in crop)
                    or min(crop[:2])<0 or min(crop[2:])<=0
                    or crop[0]+crop[2]>w or crop[1]+crop[3]>h
                    or crop[2:]!=[parameters['width'],parameters['height']]):
                raise ValueError('Invalid source crop or native crop output dimensions')
        mask_path = spec['mask_video']
        mt = source_timeline.require_cfr(source_timeline.build(project.project_path(mask_path)))
        if len(mt['pts']) != hi-lo or Fraction(mt['fps']) != Fraction(timeline['fps']):
            raise ValueError('Mask video must match every source frame')
        if crop is not None:
            meta=project.probe(project.project_path(mask_path))
            if [meta['width'],meta['height']] != crop[2:]:
                raise ValueError('Mask must use the same source crop')
    elif model.get('mode') == 'volc-text-erase':
        parameters = dict(spec.get('parameters', {}))
        if (set(parameters) != {'eraseType','encodeMode','clientToken'}
                or parameters['eraseType'] != 'text' or parameters['encodeMode'] != 'quality'
                or parameters['clientToken'] != case['batch_id']+'-'+uid):
            raise ValueError('Unsupported text erasure parameters or idempotency token')
    elif model.get('mode') == 'rhart-video-edit':
        parameters = dict(spec.get('parameters', {}))
        if parameters != {'resolution':'720p'} or len(adapted_prompt) > 800:
            raise ValueError('Unsupported video edit parameters or prompt length')
    elif model.get('mode') == 'kling-video-edit':
        parameters = dict(spec.get('parameters', {}))
        if parameters != {'keepOriginalSound': False}:
            raise ValueError('Unsupported Kling edit parameters; source audio is copied locally')
    else:
        parameters = dict(policy()['adapter']['parameters'], **spec.get('parameters', {}))
        parameters['duration'] = ('auto' if provider_first(case) and (duration.denominator != 1 or duration < 2)
                                  else str(float(duration)).rstrip('0').rstrip('.') if duration.denominator != 1 else str(int(duration)))
        if set(parameters)-{'resolution','aspectRatio','audio','promptExtend','watermark','duration','seed'} or parameters['audio'] is not False:
            raise ValueError('Unsupported parameters or generated audio')
    prompt = adapted_prompt
    if not prompt or len(prompt) > 12000:
        raise ValueError('Missing/oversized model editing prompt')
    directory = project.ROOT/'runs'/case['batch_id']/'units'/item['id']/uid
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case, state)
        guard_upstream(case, item['id'], operation_id=spec['operation_id'], entry=state['files'][item['id']])
        guard_interval_unit(case,item['id'],operation['id'],spec['model_id'],[lo,hi],state['files'][item['id']])
        parent=None
        if spec.get('parent_candidate_id'):
            parent=workflow.verify_candidate(case,item['id'],workflow.candidate_for(state['files'][item['id']],spec['parent_candidate_id']))
        units = state['files'][item['id']].setdefault('execution_units', [])
        if controlled(case) and case['execution_policy']['mode'] == 'method_trial' and any(
                u.get('operation_id') == spec['operation_id'] and u['unit_id'] != uid for u in units):
            raise ValueError('Consume the original method trial; no second unit for this operation')
        guard_method(case, item['id'], spec['operation_id'], provider_spec=spec, entry=state['files'][item['id']])
        if any(u['unit_id'] == uid for u in units) or directory.exists():
            raise ValueError('Unit already exists; inspect original, never overwrite/retry')
        # Reuse immutable source preparation when only model prompts change.
        # Re-encoding the same full source on every repair was wasted phase work.
        cached = None
        for prior in units:
            previous = verify_unit(case, prior)
            same_source = (previous['source_frames_half_open'] == [lo,hi] and
                previous.get('reference_fps', timeline['fps']) == reference_fps and
                previous.get('reference_dimensions', item['dimensions']) == reference_size and
                previous.get('source_crop_xywh') == spec.get('source_crop_xywh') and
                previous.get('dependency') == workflow.requirement_binding(case,item))
            if not same_source:
                continue
            if (previous['model_id'] == model['id'] and previous['parameters'] == parameters and
                    previous['prompt'] == prompt and previous['image_references'] == adapted_images and
                    previous.get('mask_video') == spec.get('mask_video') and
                    previous.get('parent_candidate_id') == spec.get('parent_candidate_id')):
                raise ValueError('Identical request already prepared; reuse original task and artifacts')
            if not previous.get('historical_replay'):
                cached = previous['reference_clip']
        directory.mkdir(parents=True)
        reference = spec.get('reference_clip')
        if cached and not reference:
            reference = cached
        elif reference:
            # Historical replay validates this exact reference hash against the original receipt.
            if not spec.get('historical_replay'):
                raise ValueError('External prepared clip only allowed with verified historical replay')
            project.project_path(reference)
        else:
            target = directory/'reference.mp4'
            crop_filter=(',crop='+':'.join(map(str,[crop[2],crop[3],crop[0],crop[1]]))) if model.get('mode')=='mask-video-app' and crop is not None else ''
            clock_filter = f'setpts=N/({reference_fps}*TB)' if interval else 'setpts=PTS-STARTPTS'
            scale_filter = (',scale='+':'.join(map(str,reference_size))+':flags=lanczos,setsar=1') if reference_size != item['dimensions'] else ''
            subprocess.run(['ffmpeg','-v','error','-i',str(project.project_path(item['source'])),
                '-vf',f'trim=start_frame={lo}:end_frame={hi},'+clock_filter+crop_filter+scale_filter,
                '-an','-r',reference_fps,'-c:v','libx264','-preset','fast','-crf','16',
                '-video_track_timescale',str(Fraction(timeline['time_base']).denominator),str(target)],
                check=True,capture_output=True)
            reference = str(target.relative_to(project.ROOT))
        if any(k in limits for k in ('min_reference_width','min_reference_height')):
            metadata = project.probe(project.project_path(reference))
            if [metadata['width'],metadata['height']] != reference_size:
                raise ValueError('Reference does not match validated provider dimensions')
        if not cached or spec.get('reference_clip'):
            rt = source_timeline.require_cfr(source_timeline.build(project.project_path(reference)))
            if len(rt['pts']) != hi-lo or Fraction(rt['fps']) != Fraction(reference_fps):
                raise ValueError('Reference timing differs; do not pad/retime')
        images = adapted_images
        derived = {}
        from review_completion import reference_hashes
        derived.update(reference_hashes())
        if active() and spec['model_id'] == 'kling-o3-std-edit':
            from native_colour_guide import references
            derived = references(case,item,active()['recipe'])
        if len(images)>1 and not derived:
            raise ValueError('Only the locally verified single-image reference contract is supported')
        allowed_assets = workflow.requirement_binding(case,item)['assets']
        if any(p not in allowed_assets and p not in derived for p in images):
            raise ValueError('Reference image not bound to this request')
        request = dict(unit_id=uid,file_id=item['id'],operation_id=spec['operation_id'],
            dependency=workflow.requirement_binding(case,item),source_frames_half_open=[lo,hi],
            source_time_seconds=[float(Fraction(lo)/Fraction(timeline['fps'])),float(Fraction(hi)/Fraction(timeline['fps']))],
            model_id=model['id'],route=model['route'],parameters=parameters,prompt=prompt,
            reference_clip=reference,image_references=images,
            mask_video=spec.get('mask_video'),
            source_crop_xywh=spec.get('source_crop_xywh'),
            artifact_hashes=dict({p:project.digest(project.project_path(p)) for p in [reference]+images+([spec['mask_video']] if model.get('mode') in ('mask-video-app','video-matting') else [])}, **matte_dependencies, **semantic_dependencies),
            source_timeline_sha256=item['source_timeline_sha256'],
            parent_candidate_id=spec.get('parent_candidate_id'),parent_output_sha256=parent['output_sha256'] if parent else None,
            historical_replay=bool(spec.get('historical_replay')))
        if provider_first(case):
            request['editing_input'] = dict(context=spec.get('prompt', ''), image_references=spec.get('image_references', []),
                                           prompt_format=spec.get('prompt_format', 1))
        if controlled(case):
            request['method_recipe'] = {k:spec.get(k,dict(parameters={},prompt=None,image_references=[]).get(k))
                                        for k in ('model_id','parameters','prompt','image_references')}
        if any(k in limits for k in ('min_reference_width','min_reference_height')):
            request['reference_dimensions'] = reference_size
        if spec.get('seed_atlas'):
            request['seed_atlas']=spec['seed_atlas']
            request['artifact_hashes'][spec['seed_atlas']]=project.digest(project.project_path(spec['seed_atlas']))
        if interval:
            request.update(reference_fps=reference_fps, reference_duration_seconds=float(duration),
                           interval_route_sha256=workflow.hash_json(interval),
                           frame_clock_adapter='one_to_one_reference_frames')
        if spec.get('semantic_repair'):
            request['semantic_repair'] = spec['semantic_repair']
        if active():
            request['component_probe_sha256'] = active()['recipe_sha256']
        logical_fingerprint=workflow.hash_json({k:v for k,v in request.items() if k not in ('unit_id','reference_clip','artifact_hashes','historical_replay')})
        fingerprint = workflow.hash_json(request)
        if any(u.get('logical_fingerprint') == logical_fingerprint for u in units):
            raise ValueError('Identical request already prepared')
        path = directory/'request.json'
        path.write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n')
        unit = dict(unit_id=uid,file_id=item['id'],operation_id=spec['operation_id'],
            dependency=request['dependency'],request_path=str(path.relative_to(project.ROOT)),
            request_fingerprint=fingerprint,logical_fingerprint=logical_fingerprint,status='prepared',provider_task_id=None,
            created_at=workflow.now(),new_model_calls=0,new_cost_cny=0,accepted=False)
        units.append(unit)
    return unit


def replay(case, spec):
    """Import an exact historical local receipt; no network or success promotion."""
    if spec.get('origin'):
        return replay_original_unit(case, spec)
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case,state)
        unit = unit_for(state,spec['file_id'],spec['unit_id']); request = verify_unit(case,unit)
        if unit['status'] != 'prepared' or not request['historical_replay']:
            raise ValueError('Replay requires a fresh explicitly isolated historical unit')
        receipt = project.read_json(project.project_path(spec['receipt']))
        if 'attempts' in receipt:
            attempt = next(a for a in receipt['attempts'] if a['task_id'] == spec['provider_task_id'])
            quote = receipt['quotes'][attempt['pilot']]; binding = attempt['binding']
            source_hash=binding['source_sha256']; clip_hash=binding['clip_sha256']
            interval=binding['source_frame_range_inclusive']; interval=[interval[0],interval[1]+1]
            body=quote['body']; task=attempt['task_id']; status=attempt['state']; old_fingerprint=attempt['request_sha256']
            download=project.read_json(project.project_path(spec['raw']+'.receipt.json'))
            usage=attempt.get('last_response',{}).get('usage',{})
        else:
            source_hash=receipt['source_sha256'];clip_hash=receipt['reference_clip_sha256']
            interval=receipt['source_frames_half_open'];body=receipt['body'];task=receipt['task_id']
            status=receipt['status'];old_fingerprint=receipt['request_sha256'];download=receipt['download']
            usage=receipt.get('last_response',{}).get('usage',{})
        if (receipt.get('route',receipt.get('model'))!=request['route'] or source_hash!=unit['dependency']['source_sha256'] or interval!=request['source_frames_half_open']
                or clip_hash!=request['artifact_hashes'][request['reference_clip']]
                or str(task)!=str(spec['provider_task_id']) or status!='SUCCESS'
                or old_fingerprint!=spec['historical_request_sha256']
                or body.get('prompt')!=request['prompt']
                or any(body.get(k)!=v for k,v in request['parameters'].items())):
            raise ValueError('Historical request/source/reference/parameters binding mismatch')
        if request['image_references']:
            raise ValueError('Historical image reference needs original upload hash attribution; not inferred from URL')
        if project.digest(project.project_path(spec['raw'])) != download['sha256']:
            raise ValueError('Historical raw download hash mismatch')
        raw = dict(unit_id=unit['unit_id'],path=spec['raw'],sha256=download['sha256'],
            provider_task_id=str(task),source_frames_half_open=interval,request_fingerprint=unit['request_fingerprint'],
            receipt=spec['receipt'],receipt_sha256=project.digest(project.project_path(spec['receipt'])),
            historical_request_sha256=old_fingerprint,mode='historical_offline_replay',
            historical_usage=usage,new_cost_cny=0,new_model_calls=0,status='pending_raw_review')
        unit.update(status='raw_ready',provider_task_id=str(task),raw=raw,mode='historical_offline_replay')
        state['files'][unit['file_id']]['raw_returns'].append(raw)
    return raw


def replay_original_unit(case, spec):
    """Consume an original provider unit, never an arbitrary/final video path."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', spec['unit_id']):
        raise ValueError('Invalid unit id')
    origin = spec['origin']
    binding = origin['case']
    if project.digest(project.project_path(binding['path'])) != binding['sha256']:
        raise ValueError('Original raw case changed')
    parent = project.read_json(project.project_path(binding['path']))
    parent_path = workflow.state_path(parent)
    old_state = project.read_json(parent_path); workflow.verify_state_case(parent, old_state)
    old = unit_for(old_state, origin['file_id'], origin['unit_id'])
    request = copy.deepcopy(verify_unit(parent, old))
    item = workflow.item_for(case, spec['file_id']); old_item = workflow.item_for(parent, origin['file_id'])
    op = next(o for o in item['operations'] if o['id'] == spec['operation_id'])
    old_op = next(o for o in old_item['operations'] if o['id'] == old['operation_id'])
    def meaning(operation):
        return {k:v for k,v in operation.items() if k not in ('id','occurrences','full_coverage_verified')}
    if (item['source_sha256'] != old_item['source_sha256'] or meaning(op) != meaning(old_op)
            or item.get('audio_required') != old_item.get('audio_required')):
        raise ValueError('Original raw source/operation/protection/audio mismatch')
    raw = copy.deepcopy(old.get('raw', {})); task = str(old.get('provider_task_id', ''))
    receipts = [r.get('response', {}) for r in old.get('receipts', [])]
    if (old.get('status') != 'raw_ready' or not task or raw.get('provider_task_id') != task
            or raw.get('request_fingerprint') != old['request_fingerprint']
            or not any(str(r.get('task_id')) == task and r.get('status') == 'SUCCESS' for r in receipts)
            or project.digest(project.project_path(raw['path'])) != raw['sha256']):
        raise ValueError('Original provider task/download binding missing or changed')
    provenance = dict(case=binding, workflow=dict(path=str(parent_path.relative_to(project.ROOT)),
        sha256=project.digest(parent_path)), file_id=origin['file_id'], unit_id=origin['unit_id'],
        request_fingerprint=old['request_fingerprint'], unit_binding=raw_origin_binding(old))
    review = None
    if old.get('raw_review'):
        reviewed = old['raw_review']; review_path = project.project_path(reviewed['path'])
        if project.digest(review_path) != reviewed['sha256']:
            raise ValueError('Original raw review changed')
        review = project.read_json(review_path)
        if review['raw_sha256'] != raw['sha256'] or review['request_fingerprint'] != old['request_fingerprint']:
            raise ValueError('Original raw review binding mismatch')
        for evidence in review['evidence']:
            if project.digest(project.project_path(evidence['path'])) != evidence['sha256']:
                raise ValueError('Original raw observation evidence changed')
    # Preserve the original request/artifacts, changing only the new order IDs.
    request.update(file_id=item['id'], operation_id=op['id'], unit_id=spec['unit_id'],
                   dependency=workflow.requirement_binding(case,item))
    fingerprint = workflow.hash_json(request)
    destination = project.ROOT/'runs'/case['batch_id']/'units'/item['id']/spec['unit_id']
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case, state); entry = state['files'][item['id']]
        prior = next((u for u in entry.get('execution_units', []) if u['unit_id'] == spec['unit_id']), None)
        if prior:
            verify_unit(case, prior)
            previous = copy.deepcopy(prior.get('local_origin', {}))
            expected = copy.deepcopy(provenance)
            if previous.get('unit_binding'):
                previous['workflow'].pop('sha256', None); expected['workflow'].pop('sha256', None)
            else:
                expected.pop('unit_binding', None)
            if previous != expected or prior['request_fingerprint'] != fingerprint:
                raise ValueError('Imported unit already bound to different material')
            return dict(status='raw_ready', reused=True, new_model_calls=0, new_cost_cny=0)
        if entry.get('material_handoff') or entry.get('current_candidate_id'):
            raise ValueError('Materials already frozen; reopen only the diagnosed dependency')
        destination.mkdir(parents=True, exist_ok=False)
        request_path = destination/'request.json'
        request_path.write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n')
        raw.update(unit_id=spec['unit_id'], request_fingerprint=fingerprint,
                   mode='original_raw_local_reuse', new_model_calls=0, new_cost_cny=0)
        unit = dict(unit_id=spec['unit_id'],file_id=item['id'],operation_id=op['id'],
            dependency=workflow.requirement_binding(case,item),request_path=str(request_path.relative_to(project.ROOT)),
            request_fingerprint=fingerprint,logical_fingerprint=fingerprint,status='raw_ready',
            provider_task_id=task,raw=raw,local_origin=provenance,created_at=workflow.now(),
            new_model_calls=0,new_cost_cny=0,accepted=False)
        if review is not None:
            review.update(request_fingerprint=fingerprint, reused_from=dict(path=reviewed['path'],sha256=reviewed['sha256']))
            new_review = destination/'raw-review.json'
            new_review.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
            unit['raw_review'] = dict(path=str(new_review.relative_to(project.ROOT)),sha256=project.digest(new_review),
                status=reviewed['status'],raw_sha256=raw['sha256'])
        entry.setdefault('execution_units', []).append(unit); entry['raw_returns'].append(raw)
    return dict(status='raw_ready',unit_id=spec['unit_id'],origin=provenance,
                raw_sha256=raw['sha256'],new_model_calls=0,new_cost_cny=0,accepted=False)


RAW_CHECKS = {'coverage','target_asset','protection','texture_geometry','occlusion','temporal_seams'}


def review_raw(case, spec):
    evidence_path=spec['evidence']; review=project.read_json(project.project_path(evidence_path))
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case,state)
        unit=unit_for(state,spec['file_id'],spec['unit_id']);request=verify_unit(case,unit)
        raw=unit['raw']
        if review['raw_sha256']!=raw['sha256'] or review['request_fingerprint']!=unit['request_fingerprint']:
            raise ValueError('Raw review binding mismatch')
        if project.digest(project.project_path(raw['path']))!=raw['sha256']:
            raise ValueError('Raw media changed')
        checks=review['checks']
        if len(checks)!=len(RAW_CHECKS) or {c['kind'] for c in checks}!=RAW_CHECKS or any(c['status'] not in ('pass','fail','reviewed_with_limits') or not c.get('scope') for c in checks):
            raise ValueError('Raw review needs six explicit scoped observations')
        if not review.get('observed_frames') or not review.get('evidence'):
            raise ValueError('Raw review requires actual observation evidence')
        for e in review['evidence']:
            if project.digest(project.project_path(e['path']))!=e['sha256']:
                raise ValueError('Raw observation evidence changed')
        unit['raw_review']=dict(path=evidence_path,sha256=project.digest(project.project_path(evidence_path)),
            status='pass' if all(c['status']=='pass' for c in checks) else 'fail_or_limited',
            raw_sha256=raw['sha256'])
        # Queue against the frozen parent/current candidate, while preserving the
        # raw observation basis. This is not a review of that candidate's pixels.
        entry=state['files'][unit['file_id']]
        parent_id=request.get('parent_candidate_id') or entry['current_candidate_id']
        parent=workflow.verify_historical_parent(case,unit['file_id'],workflow.candidate_for(entry,parent_id)) if parent_id else None
        for defect in review.get('defects',[]):
            body=dict(file_id=unit['file_id'],operation_id=unit['operation_id'],candidate_id=parent_id,
                candidate_sha256=parent['output_sha256'] if parent else unit['dependency']['source_sha256'],source_sha256=unit['dependency']['source_sha256'],
                expected_revision=state['revision'],source_time=defect['source_time'],text=defect['text'],
                category=defect['category'],actor='host_agent')
            issue=workflow.bind_feedback(case,state,body)
            issue.update(unit_id=unit['unit_id'],raw_sha256=raw['sha256'],observation_basis='provider_raw')
            state['files'][unit['file_id']]['issues'].append(issue)
    return unit['raw_review']


def remote_action(case,spec,transport=None):
    """Single attempt only; persist unknown before transport, never auto-resubmit."""
    action=spec['action']
    require_remote(action, case['batch_id'])  # Before credential access or network.
    if action == 'submit':
        from production_pipeline import guard_upstream
        from production_effort import check
        state=project.read_json(workflow.state_path(case))
        unit=unit_for(state,spec['file_id'],spec['unit_id'])
        request=verify_unit(case,unit)
        from production_pipeline import guard_method
        guard_method(case, spec['file_id'], request['operation_id'], provider_spec=request)
        guard_upstream(case, spec['file_id'], operation_id=request['operation_id'])
        from production_pipeline import guard_interval_unit
        guard_interval_unit(case,spec['file_id'],request['operation_id'],request['model_id'],request.get('source_frames_half_open'))
        configured=model_for(request['model_id'],case['batch_id'])
        verify_submission_dimensions(case,request,configured)
        family='local_recolor' if configured and configured.get('mode') in ('video-segmentation','video-matting') else None
        check(case, spec['file_id'], start_method=True, method_family=family, operation_id=request['operation_id'])
    grant=project.read_json(project.project_path(spec['authorization']))
    if grant.get('batch_id')!=case['batch_id'] or grant.get('remote_authorized') is not True or not grant.get('user_authorization_evidence') or grant.get('full_cost_confirmed') is not True:
        raise ValueError('Missing explicit batch authorization')
    if transport is None:
        def transport(payload):
            node=shutil.which('node') or policy()['adapter'].get('node_executable')
            if not node or not Path(node).is_file():raise ValueError('Configured Node runtime unavailable')
            result=subprocess.run([node,str(project.ROOT/'scripts/runninghub_transport.mjs')],
                cwd=project.ROOT,input=json.dumps(payload),text=True,capture_output=True,check=False)
            if result.returncode and not result.stdout.strip():raise ValueError('Provider transport failed without receipt; inspect original unit, no retry')
            return json.loads(result.stdout)
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case,state)
        unit=unit_for(state,spec['file_id'],spec['unit_id']);request=verify_unit(case,unit)
        if request['historical_replay']:raise ValueError('Replay units can never submit/query remote tasks')
        if action=='submit':
            guard_method(case, spec['file_id'], request['operation_id'], provider_spec=request, entry=state['files'][spec['file_id']])
            verify_submission_dimensions(case,request,model_for(request['model_id'],case['batch_id']))
            guard_upstream(case, spec['file_id'], operation_id=request['operation_id'], entry=state['files'][spec['file_id']])
            from production_pipeline import guard_interval_unit
            interval = guard_interval_unit(case,spec['file_id'],request['operation_id'],request['model_id'],
                                           request['source_frames_half_open'],state['files'][spec['file_id']])
            if interval and (request.get('interval_route_sha256') != workflow.hash_json(interval)
                    or request.get('reference_fps') != interval['model_input_fps']):
                raise ValueError('Prepared request predates the fixed interval route and clock adapter')
            # Current official schema accepts auto or integer seconds 2..30 only.
            # Keep historical/offline requests intact, reject incompatible live calls.
            if (request['route'] not in {'/task/openapi/create','/task/openapi/ai-app/run','/openapi/v2/volc-subtitle-erase-pro/video', '/openapi/v2/rhart-video-g-official/edit-video', '/openapi/v2/kling-video-o3-std/video-edit'}
                    and request['parameters']['duration'] not in {'auto', *map(str, range(2,31))}):
                raise ValueError('Current provider duration requires auto or integer seconds 2..30')
            origin=policy(case['batch_id']).get('adapter',{}).get('api_origin','https://www.runninghub.cn')
            rejected_migration=(unit['status']=='submission_rejected' and not unit.get('provider_task_id')
                and unit.get('last_receipt',{}).get('provider_rejected') is True
                and unit.get('provider_api_origin','https://www.runninghub.cn')!=origin)
            if unit['status']!='prepared' and not rejected_migration:raise ValueError('Already attempted; recover original task without resubmission')
            ceiling=spec['maximum_charge_cny'];total=grant.get('total_budget_cny',0)
            if not isinstance(ceiling,(float,int)) or not math.isfinite(ceiling) or ceiling<=0:
                raise ValueError('Finite positive full-charge reservation required')
            reserved=sum(float(u.get('new_cost_cny',0) if u.get('cost_settled') else u.get('reserved_cny',0)) for e in state['files'].values() for u in e.get('execution_units',[]))
            caps=grant.get('unit_caps_cny')
            if reserved+ceiling>total or (caps is not None and ceiling>caps.get(unit['unit_id'],0)):
                raise ValueError('Batch/unit budget exceeded')
            unit.update(status='submission_unknown',reserved_cny=ceiling,submitted_at=workflow.now(),provider_api_origin=origin)
        elif action in ('query','download'):
            if not unit.get('provider_task_id'):raise ValueError('No original task id; do not resubmit unknown create')
        else:raise ValueError('Unsupported remote action')
        task_id=unit.get('provider_task_id')
    # Network outside the state lock; failed submission intentionally remains unknown.
    reply=transport(dict(action=action,batch_id=case['batch_id'],request=request,task_id=task_id,
                         api_origin=unit.get('provider_api_origin','https://www.runninghub.cn'),
                         usd_to_cny=policy(case['batch_id']).get('adapter',{}).get('usd_to_cny'),
                         maximum_charge_cny=spec.get('maximum_charge_cny'),
                         output=str((project.ROOT/'runs'/case['batch_id']/'units'/unit['file_id']/unit['unit_id']/'raw.mp4')),
                         journal_path=str(project.project_path(unit['request_path']).with_name('transport-events.jsonl'))))
    with workflow.locked_state(case) as state:
        unit=unit_for(state,spec['file_id'],spec['unit_id'])
        verify_unit(case,unit)
        if task_id and reply.get('task_id') and str(reply['task_id'])!=str(task_id):
            raise ValueError('Provider response belongs to another task')
        if reply.get('task_id'):unit['provider_task_id']=str(reply['task_id'])
        unit.setdefault('receipts',[]).append(dict(action=action,at=workflow.now(),response=reply))
        unit['last_receipt']=reply
        rejected=reply.get('provider_rejected') is True and str(reply.get('provider_error_code'))=='40310' and not unit.get('provider_task_id')
        status=(reply.get('status') or 'submission_unknown') if unit.get('provider_task_id') else ('submission_rejected' if rejected else 'submission_not_created' if reply.get('create_attempted') is False else 'submission_unknown')
        # A read-only settlement query must not undo a completed local download.
        if not (action in ('query','download') and reply.get('failure_stage')):
            unit['status']='raw_ready' if status=='SUCCESS' and unit.get('raw') else status
        if action=='submit':
            unit['new_model_calls']=0 if reply.get('create_attempted') is False or rejected else 1
            if reply.get('create_attempted') is False or rejected:unit['reserved_cny']=0
        if reply.get('actual_charge_cny') is not None:
            unit['new_cost_cny']=reply['actual_charge_cny'];unit['cost_settled']=True
        if reply.get('raw_path'):
            raw_path=str(Path(reply['raw_path']).relative_to(project.ROOT))
            raw=dict(unit_id=unit['unit_id'],path=raw_path,sha256=project.digest(project.project_path(raw_path)),
                provider_task_id=unit['provider_task_id'],request_fingerprint=unit['request_fingerprint'],
                source_frames_half_open=request['source_frames_half_open'],mode='remote',status='pending_raw_review')
            if unit.get('raw') and unit['raw']['sha256']!=raw['sha256']:
                raise ValueError('Original provider raw changed')
            if not unit.get('raw'):state['files'][unit['file_id']]['raw_returns'].append(raw)
            unit['raw']=raw;unit['status']='raw_ready'
    return copy.deepcopy(unit)


def native_contexts(count, fps):
    """Overlapping source windows; never extend, repeat or retime source frames.

    16n+1 is the observed Kling return format, not an output-length guarantee.
    Every downloaded return still requires its independent frame/timing audit.
    """
    fps = Fraction(fps)
    if abs(fps / 24 - 1) > Fraction(1, 1000):
        raise ValueError('Verified native route requires 24fps or nominal 23.976fps source; no frame conversion')
    maximum = 16 * ((int(15 * fps) - 1) // 16) + 1
    minimum = 16 * math.ceil((3 * fps - 1) / 16) + 1
    if count < minimum:
        raise ValueError('Source is shorter than the verified native edit context')
    if count <= maximum:
        length = 16 * ((count - 1) // 16) + 1
        windows = [[0, length]]
        if length != count:
            windows.append([count - length, count])
        return windows
    overlap = 33
    windows = [[0, maximum]]
    while windows[-1][1] < count:
        if count - windows[-1][1] < maximum:
            length = min(maximum, max(minimum,
                16 * math.ceil((count - windows[-1][1] + overlap - 1) / 16) + 1))
            windows.append([count - length, count])
        else:
            lo = windows[-1][1] - overlap
            windows.append([lo, lo + maximum])
    return windows


def prepare_recolor(case, spec):
    """Reuse the proven route without rebuilding graphs or selecting tracking thresholds."""
    from production_effort import source_preserving_recolor
    item = workflow.item_for(case, spec['file_id'])
    operation = next(op for op in item['operations'] if op['id'] == spec['operation_id'])
    from production_pipeline import interval_route
    from production_effort import provider_first
    if provider_first(case) or interval_route(case,item['id'],operation['id']):
        return prepare_native_recolor(case,spec)
    if not source_preserving_recolor(operation):
        return prepare_native_recolor(case, spec)
    from video_segmentation import fixed_recolor_parameters
    timeline = source_timeline.require_cfr(source_timeline.load_for_item(item))
    unit = prepare_unit(case, dict(spec, unit_id=spec['unit_prefix']+'-sam3',
        model_id='sam3-video-semantic', source_frames_half_open=[0, len(timeline['pts'])],
        parameters=fixed_recolor_parameters(item, operation),
        prompt='Track the visible '+operation['semantic_label']+' including its straps; return masks only, preserve original imagery.'))
    return dict(route='source_preserving_recolor', units=[unit],
                target_color_rgb=operation['target_color_rgb'], new_model_calls=0, accepted=False)


def prepare_native_recolor(case, spec):
    """Actual default preparation for deformable recolor, using the existing provider path."""
    item = workflow.item_for(case, spec['file_id'])
    operation = next(op for op in item['operations'] if op['id'] == spec['operation_id'])
    from production_effort import source_preserving_recolor
    from production_pipeline import interval_route
    route = interval_route(case,item['id'],operation['id'])
    from production_effort import provider_first
    if source_preserving_recolor(operation) and not provider_first(case) and not route:
        raise ValueError('Strict detail preservation requires prepare-recolor, not native reconstruction')
    if not provider_first(case) and (operation.get('kind') != 'modify' or operation.get('surface_motion') != 'deformable'):
        raise ValueError('Native recolor preparation requires a deformable modification')
    timeline = source_timeline.require_cfr(source_timeline.load_for_item(item))
    if route:
        # Preparing all intervals is an explicit second step, gated by the
        # original hard return review at prepare_unit and again at submit.
        windows = ([route['probe_frames_half_open']] if not spec.get('expand_targets') else
                   [[lo+a,lo+b] for lo,hi in route['target_ranges_half_open']
                    for a,b in native_contexts(hi-lo,route['model_input_fps'])])
    elif provider_first(case):
        from production_pipeline import verify_route
        state = project.read_json(workflow.state_path(case))
        handoff = verify_route(case, item['id'], state['files'][item['id']], required=True)
        observed = project.read_json(project.project_path(handoff['observation']['path']))
        ranges = observed['operations'][operation['id']]['appearance_ranges_half_open']
        model = model_for(spec.get('model_id', policy(case['batch_id'])['primary']['id']), case['batch_id'])
        if not model:
            raise ValueError('Requested provider model is not configured for this batch')
        limits = model['limits']
        maximum = int(min(limits['max_reference_video_seconds_conservative'],
            limits['max_input_plus_output_seconds']/2, limits['max_output_seconds']) * Fraction(timeline['fps']))
        if maximum < 1:
            raise ValueError('Invalid recorded provider duration limit')
        windows = ([[lo+a,lo+b] for lo,hi in ranges for a,b in native_contexts(hi-lo,timeline['fps'])]
                   if model['mode'] == 'kling-video-edit' else
                   [[a,min(hi,a+maximum)] for lo,hi in ranges for a in range(lo,hi,maximum)])
    else:
        windows = native_contexts(len(timeline['pts']), timeline['fps'])
    model_id = spec.get('model_id', policy(case['batch_id'])['primary']['id']) if provider_first(case) else 'kling-o3-std-edit'
    model = model_for(model_id, case['batch_id']) if provider_first(case) else {'mode': 'kling-video-edit'}
    if model is None:
        raise ValueError('Requested provider model is not configured for this batch')
    parameters = spec.get('parameters', {'keepOriginalSound': False} if model['mode'] == 'kling-video-edit' else
        {'resolution': '720p'} if model['mode'] == 'rhart-video-edit' else {})
    units = []
    for i, interval in enumerate(windows, 1):
        unit_spec = dict(spec, unit_id=f"{spec['unit_prefix']}-{i:02d}",
                         source_frames_half_open=interval, model_id=model_id, parameters=parameters)
        unit = prepare_unit(case, unit_spec)
        units.append(unit)
    return dict(route='native_video_edit', source_windows=windows, units=units,
                new_model_calls=0, accepted=False)


def plan_native_intervals(case, spec):
    """Normal timeline refill: no segmentation, per-frame masks or local reconstruction."""
    from aigc_composite import prepare_plan, validate
    item = workflow.item_for(case, spec['file_id'])
    state = project.read_json(workflow.state_path(case))
    units = [unit_for(state, item['id'], uid) for uid in spec['unit_ids']]
    requests = [verify_unit(case, u) for u in units]
    pairs = sorted(zip(units, requests), key=lambda pair: pair[1]['source_frames_half_open'])
    if not pairs or len({u['unit_id'] for u in units}) != len(units):
        raise ValueError('Provide distinct reviewed edit units')
    if spec.get('target_matte') or spec.get('diagnostic'):
        raise ValueError('Normal native refill does not use masks or failed diagnostic returns')
    parts = []
    for i,(unit, request) in enumerate(pairs):
        lo,hi = request['source_frames_half_open']
        wl,wh = lo,hi
        if i and pairs[i-1][1]['source_frames_half_open'][1] > lo:
            wl = (lo + pairs[i-1][1]['source_frames_half_open'][1]) // 2
        if i+1 < len(pairs) and pairs[i+1][1]['source_frames_half_open'][0] < hi:
            wh = (hi + pairs[i+1][1]['source_frames_half_open'][0]) // 2
        if not lo <= wl < wh <= hi:
            raise ValueError('Overlapping contexts cannot be resolved to sequential timeline intervals')
        parts.append(dict(unit_id=unit['unit_id'], blend_mode='native_replace', write_frames_half_open=[wl,wh]))
        if spec.get('timing_mappings', {}).get(unit['unit_id']):
            parts[-1]['timing_mapping'] = spec['timing_mappings'][unit['unit_id']]
    out = project.project_path(spec['out'])
    pixel_plan = out.with_name(out.stem+'-pixels.json')
    fingerprint = workflow.hash_json({k:v for k,v in spec.items() if k != 'action'})
    if out.exists() or pixel_plan.exists():
        if not out.is_file() or not pixel_plan.is_file():
            raise ValueError('Partial native assembly; reconcile original files')
        manifest = project.read_json(out)
        plan = project.read_json(pixel_plan)
        if (manifest.get('assembly_request_fingerprint') != fingerprint
                or manifest['dependency'] != workflow.requirement_binding(case,item)
                or manifest['operations'][0]['plan_sha256'] != project.digest(pixel_plan)):
            raise ValueError('Native refill input changed; preserve original plan')
        validate(case,item,plan)
        return dict(manifest=spec['out'],assembly_reused=True,new_model_calls=0,new_renders=0,accepted=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    prepare_plan(case,dict(file_id=item['id'],operation_id=spec['operation_id'],parts=parts,
        frame_states=spec['frame_states'], observation_evidence=spec['observation_evidence'],
        frame_records=spec.get('frame_records'),
        out=str(pixel_plan.relative_to(project.ROOT))))
    manifest=dict(schema='execution_manifest_v1',file_id=item['id'],dependency=workflow.requirement_binding(case,item),
        risk_preview_ranges_half_open=spec['risk_preview_ranges_half_open'],assembly_request_fingerprint=fingerprint,
        operations=[dict(operation_id=spec['operation_id'],disposition='ready',
            plan=str(pixel_plan.relative_to(project.ROOT)),plan_sha256=project.digest(pixel_plan))])
    out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return dict(manifest=spec['out'],status='native_plan_pending_shared_preview',new_model_calls=0,new_renders=0,accepted=False)


def plan_native_recolor(case, spec):
    """Reuse reviewed native returns; legacy masked assemblies remain recoverable."""
    from production_effort import provider_first
    if provider_first(case):
        return plan_native_intervals(case, spec)
    import cv2
    import numpy as np
    from aigc_composite import prepare_plan, mapping_binding, validate_frame_index_mapping, validate
    item = workflow.item_for(case, spec['file_id'])
    timeline = source_timeline.load_for_item(item)
    state = project.read_json(workflow.state_path(case))
    units = [unit_for(state, item['id'], uid) for uid in spec['unit_ids']]
    requests = [verify_unit(case, u) for u in units]
    spans = [r['source_frames_half_open'] for r in requests]
    if any(r['model_id'] != 'kling-o3-std-edit' for r in requests):
        raise ValueError('Native assembly requires the reviewed original-video edit route')
    out = project.project_path(spec['out'])
    out.parent.mkdir(parents=True, exist_ok=True)
    pixel_plan = out.with_name(out.stem+'-pixels.json')
    paths = [out, pixel_plan] + [out.parent/(u['unit_id']+suffix) for u in units
                               for suffix in ('-alpha.npz', '-timing.json')]
    request_fingerprint = workflow.hash_json({k:v for k,v in spec.items() if k != 'action'})
    if any(p.exists() for p in paths):
        if not all(p.is_file() for p in paths):
            raise ValueError('Partial native assembly; reconcile original files, never overwrite or rename')
        manifest = project.read_json(out)
        expected = dict(schema='execution_manifest_v1',file_id=item['id'],
            dependency=workflow.requirement_binding(case,item),
            risk_preview_ranges_half_open=spec['risk_preview_ranges_half_open'],
            assembly_request_fingerprint=request_fingerprint,
            operations=[dict(operation_id=spec['operation_id'],disposition='ready',
                plan=str(pixel_plan.relative_to(project.ROOT)),plan_sha256=project.digest(pixel_plan))])
        if manifest != expected:
            raise ValueError('Native assembly request/binding changed or legacy binding absent; consume the saved manifest via pipeline or reconcile it')
        prepared = project.read_json(pixel_plan)
        if (prepared.get('source_target_matte') != dict(path=spec.get('target_matte'),
                    sha256=project.digest(project.project_path(spec['target_matte'])))
                or prepared['review']['frame_states'] != spec['frame_states']
                or [p['unit_id'] for p in prepared['parts']] != spec['unit_ids']
                or prepared['operation_id'] != spec['operation_id']
                or prepared['diagnostic'] != bool(spec.get('diagnostic'))):
            raise ValueError('Native assembly input binding changed')
        validate(case,item,prepared)  # Original raw, review, mapping, masks and evidence.
        return dict(manifest=spec['out'],manifest_sha256=project.digest(out),
                    status='native_plan_pending_shared_preview',assembly_reused=True,
                    new_model_calls=0,new_renders=0,accepted=False)
    parts = []
    w, h = item['dimensions']
    target_matte = spec.get('target_matte')
    if not target_matte:
        raise ValueError('Regional native edit requires a source-bound target matte; full-frame alpha would overwrite protected source details')
    target_sha = project.digest(project.project_path(target_matte))
    with np.load(project.project_path(target_matte), allow_pickle=False) as z:
        from source_matte import validate_metadata
        if not {'source_sha256','source_dimensions','mask_dimensions','frames_half_open','alpha','protected'} <= set(z.files):
            raise ValueError('Native target matte requires source/grid/range/protection metadata')
        validate_metadata(z, item['source_sha256'], [w,h], [0,item['source_frame_count_inventory']])
        target_alpha = z['alpha'].copy(); target_protected = z['protected']
        if target_alpha.dtype != np.uint8 or target_alpha.shape != (item['source_frame_count_inventory'],h,w):
            raise ValueError('Native target matte must match the complete source grid')
        if target_protected.dtype != np.uint8 or target_protected.shape != target_alpha.shape:
            raise ValueError('Native source protection must match the target grid')
        target_alpha[target_protected > 0] = 0
        if np.any(np.all(target_alpha > 0, axis=(1,2))):
            raise ValueError('Regional target matte cannot cover an entire source frame; preserve protected original pixels')
    for unit, request in zip(units, requests):
        lo, hi = request['source_frames_half_open']
        review = project.read_json(project.project_path(unit['raw_review']['path']))
        if ((unit['raw_review']['status'] != 'pass' and not spec.get('diagnostic'))
                or review['observed_frames'] != list(range(hi-lo))):
            raise ValueError('Native assembly needs actual complete source/raw observations first')
        raw = project.project_path(unit['raw']['path'])
        rt = source_timeline.require_cfr(source_timeline.build(raw))
        mapping = dict(method='one_to_one_source_pts', binding=mapping_binding(item,unit,[lo,hi]),
                       observed_source_frames=list(range(lo,hi)), evidence=review['evidence'])
        reference_timeline = None
        if request.get('frame_clock_adapter'):
            if review.get('frame_correspondence') != 'one_to_one_reference_frames':
                raise ValueError('Native clock adaptation requires observed source/reference/raw correspondence')
            reference_timeline = source_timeline.build(project.project_path(request['reference_clip']))
            mapping.update(reference_sha256=reference_timeline['source_sha256'],
                           frame_correspondence=review['frame_correspondence'])
        validate_frame_index_mapping(timeline,rt,[lo,hi],mapping,mapping['binding'],reference_timeline)
        timing = out.parent/(unit['unit_id']+'-timing.json')
        timing.write_text(json.dumps(mapping,indent=2)+'\n')
        cap = cv2.VideoCapture(str(raw));rw,rh=int(cap.get(3)),int(cap.get(4));cap.release()
        weights = np.full(hi-lo,255,np.uint8)
        for other_lo, other_hi in spans:
            if [lo,hi] == [other_lo,other_hi]:continue
            a,b=max(lo,other_lo),min(hi,other_hi)
            if a>=b:continue
            ramp=np.rint(np.linspace(0,255,b-a)).astype(np.uint8)
            weights[a-lo:b-lo] = 255-ramp if lo<other_lo else ramp
        mask = out.parent/(unit['unit_id']+'-alpha.npz')
        matrix=np.diag([(w-1)/(rw-1),(h-1)/(rh-1),1.0])
        alpha=np.rint(target_alpha[lo:hi].astype(np.float32)*weights[:,None,None]/255).astype(np.uint8)
        np.savez_compressed(mask, alpha=alpha,
            protected=target_protected[lo:hi],
            raw_to_source=np.broadcast_to(matrix,(hi-lo,3,3)), source_sha256=item['source_sha256'],
            source_dimensions=item['dimensions'],mask_dimensions=item['dimensions'],frames_half_open=[lo,hi])
        parts.append(dict(unit_id=unit['unit_id'],masks=str(mask.relative_to(project.ROOT)),
            masks_sha256=project.digest(mask),blend_mode='temporal_crossfade',
            timing_mapping=str(timing.relative_to(project.ROOT))))
    prepare_plan(case,dict(file_id=item['id'],operation_id=spec['operation_id'],parts=parts,
        frame_states=spec['frame_states'],observation_evidence=spec['observation_evidence'],
        color_match='none',diagnostic=bool(spec.get('diagnostic')),out=str(pixel_plan.relative_to(project.ROOT))))
    if project.digest(project.project_path(target_matte)) != target_sha:
        raise ValueError('Native target matte changed during preparation')
    prepared = project.read_json(pixel_plan)
    prepared['artifact_hashes'][target_matte] = target_sha
    prepared['source_target_matte'] = dict(path=target_matte,sha256=target_sha)
    pixel_plan.write_text(json.dumps(prepared,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(schema='execution_manifest_v1',file_id=item['id'],dependency=workflow.requirement_binding(case,item),
        risk_preview_ranges_half_open=spec['risk_preview_ranges_half_open'],
        assembly_request_fingerprint=request_fingerprint,
        operations=[dict(operation_id=spec['operation_id'],disposition='ready',
            plan=str(pixel_plan.relative_to(project.ROOT)),plan_sha256=project.digest(pixel_plan))])
    out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return dict(manifest=spec['out'],status='native_plan_pending_shared_preview',new_model_calls=0)


def observe_collected_unit(case, spec, uid):
    """Expose each original return immediately; never infer a quality verdict."""
    if not spec.get('observation_out'):return None
    import review_queue
    observation_spec=dict(unit_id=uid,stills=spec.get('stills',[]))
    _,_,_,start,raw_sha=review_queue.pair_inputs(case,spec['file_id'],observation_spec)
    destination=project.project_path(spec['observation_out'])/uid
    inventory_path=destination/'inventory.json'
    if destination.exists():
        inventory=project.read_json(inventory_path)
        if (inventory['file_id']!=spec['file_id'] or inventory['output_sha256']!=raw_sha
                or inventory['source_frames_half_open'][0]!=start
                or inventory['source_sha256']!=workflow.item_for(case,spec['file_id'])['source_sha256']):
            raise ValueError('Collected observation binding changed; preserve previous evidence')
        for evidence in inventory['evidence']:
            if project.digest(project.project_path(evidence['path']))!=evidence['sha256']:
                raise ValueError('Collected observation evidence changed')
        result=dict(status=inventory['status'],inventory=str(inventory_path.relative_to(project.ROOT)),reused=True)
    else:
        result=review_queue.build_pairs(case,spec['file_id'],str(destination.relative_to(project.ROOT)),observation_spec)
    print(json.dumps(dict(event='raw_observation_ready',unit_id=uid,observation=result),ensure_ascii=False),flush=True)
    return result


def settle_unit(case,spec):
    from decimal import Decimal
    require_remote('query',case['batch_id'])
    state=project.read_json(workflow.state_path(case));unit=unit_for(state,spec['file_id'],spec['unit_id'])
    verify_unit(case,unit)
    node=shutil.which('node') or policy()['adapter']['node_executable']
    result=subprocess.run([node,str(project.ROOT/'scripts/runninghub_transport.mjs')],cwd=project.ROOT,
        input=json.dumps(dict(action='settlement-query',batch_id=case['batch_id'],task_id=unit['provider_task_id'])),
        text=True,capture_output=True,check=False)
    if result.returncode:raise ValueError('Original task settlement failed; no resubmission')
    receipt=json.loads(result.stdout);usage=receipt.get('usage') or {}
    charge=usage.get('thirdPartyConsumeMoney')
    if charge is None:charge=usage.get('consumeMoney')
    if receipt['status'] not in ('SUCCESS','FAILED') or charge is None:
        raise ValueError('Original task has no final CNY settlement')
    receipt['actual_charge_cny']=float(Decimal(str(charge)))
    with workflow.locked_state(case) as state:
        current=unit_for(state,spec['file_id'],spec['unit_id'])
        if current['provider_task_id']!=receipt['task_id']:raise ValueError('Settlement task changed')
        current.update(new_cost_cny=float(Decimal(str(charge))),cost_settled=True)
        current.setdefault('receipts',[]).append(dict(action='settle',at=workflow.now(),response=receipt))
    return dict(status='settled',task_id=receipt['task_id'],actual_charge_cny=float(Decimal(str(charge))),accepted=False)


def link_preservation_repair(case,spec):
    """Bind rejected reconstruction issues to a source-pixel replacement for recheck."""
    from production_effort import preservation_fallback
    from source_recolor import validate
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case,state)
        file_id=spec['file_id'];entry=state['files'][file_id]
        if not preservation_fallback(case,state,file_id):
            raise ValueError('Missing observed preservation fallback')
        candidate=workflow.verify_candidate(case,file_id,workflow.candidate_for(entry,spec['candidate_id']))
        manifest=project.read_json(project.project_path(candidate['plan']))
        plans={}
        for operation in manifest['operations']:
            plan=project.read_json(project.project_path(operation['plan']))
            if (plan.get('composition_mode')!='source_recolor' or
                    plan.get('segmentation_backend')!='sam3-video-semantic' or
                    not plan.get('full_appearance_required')):
                raise ValueError('Repair must use bound whole-source semantic recoloring')
            validate(case,workflow.item_for(case,file_id),plan)
            plans[operation['operation_id']]=plan
        issues=[]
        for issue_id in spec['issue_ids']:
            issue=next((i for i in entry['issues'] if i['id']==issue_id),None)
            if (not issue or issue.get('observation_basis')!='provider_raw' or
                    issue.get('category')!='model_internal' or issue.get('operation_id') not in plans or
                    issue.get('status') not in ('open','pending_recheck')):
                raise ValueError('Unrelated preservation repair issue')
            unit=unit_for(state,file_id,issue['unit_id']);request=verify_unit(case,unit)
            raw=unit['raw'];review=unit['raw_review']
            if (request['operation_id']!=issue['operation_id'] or request['model_id']!='kling-o3-std-edit' or
                    issue['raw_sha256']!=raw['sha256'] or
                    project.digest(project.project_path(raw['path']))!=raw['sha256'] or
                    project.digest(project.project_path(review['path']))!=review['sha256'] or
                    not any(c['kind']=='texture_geometry' and c['status']=='fail' for c in
                            project.read_json(project.project_path(review['path']))['checks'])):
                raise ValueError('Preservation repair failure evidence changed')
            if issue.get('result_candidate_id') not in (None,candidate['id']):
                raise ValueError('Issue already linked to another repair')
            issues.append(issue)
        for issue in issues:
            issue.update(status='pending_recheck',result_candidate_id=candidate['id'],
                         replacement_basis='source_pixels_after_native_detail_failure')
    return dict(status='pending_actual_candidate_recheck',candidate_id=candidate['id'],accepted=False)


def collect_originals(case, spec):
    """Parallel work within B, then return control while original models run."""
    from concurrent.futures import ThreadPoolExecutor
    import time
    pending = list(dict.fromkeys(spec['unit_ids'])); collected = []
    if not pending:
        raise ValueError('Collection requires original unit IDs')
    state = project.read_json(workflow.state_path(case))
    workflow.verify_state_case(case, state)
    # Validate the whole batch before any side effect, including a partial query.
    for uid in pending:
        if not unit_for(state, spec['file_id'], uid).get('provider_task_id'):
            raise ValueError('Missing original task; collection cannot submit')

    def collect(uid):
        state = project.read_json(workflow.state_path(case))
        unit = unit_for(state, spec['file_id'], uid)
        if unit.get('raw'):
            verify_unit(case, unit)
            if project.digest(project.project_path(unit['raw']['path'])) != unit['raw']['sha256']:
                raise ValueError('Collected raw changed; do not download again')
        else:
            body = dict(spec, unit_id=uid, action='query')
            unit = remote_action(case, body)
            if unit['status'] in ('FAILED', 'FAILURE', 'CANCELLED'):
                raise ValueError('Original provider task failed: '+uid+'; no resubmission')
            if unit['status'] != 'SUCCESS':
                return False
            remote_action(case, dict(body, action='download'))
        observe_collected_unit(case, spec, uid)
        return True

    while pending:
        with ThreadPoolExecutor(max_workers=min(3, len(pending))) as pool:
            results = list(pool.map(collect, pending))
        collected.extend(uid for uid, ready in zip(pending, results) if ready)
        pending = [uid for uid, ready in zip(pending, results) if not ready]
        if not pending or not spec.get('wait', False):
            break
        time.sleep(30)
    return dict(status='waiting_original_tasks' if pending else 'raw_ready', collected=collected, pending=pending,
                next_poll_seconds=30 if pending else None, new_model_calls=0, accepted=False,
                next_action='review_collected_and_prepare_independent_local_work' if pending else 'review_original_raws')


def dispatch(case,spec):
    action=spec['action']
    if action=='preservation-fallback':
        from production_effort import preservation_fallback
        with workflow.locked_state(case) as state:
            workflow.verify_state_case(case,state)
            failed=unit_for(state,spec['file_id'],spec['failed_unit_id'])
            state.setdefault('preservation_fallbacks',{})[spec['file_id']]=dict(
                failed_unit_id=spec['failed_unit_id'],operation_id=spec['operation_id'],
                review_sha256=failed['raw_review']['sha256'],user_instruction=spec['user_instruction'],at=workflow.now())
            if not preservation_fallback(case,state,spec['file_id']):
                raise ValueError('Missing preservation fallback authorization/evidence')
        return dict(status='source_preservation_fallback_enabled',accepted=False)
    if action=='settle':
        return settle_unit(case,spec)
    if action=='link-preservation-repair':
        return link_preservation_repair(case,spec)
    if action=='plan-source-recolor':
        from source_recolor import prepare_plan
        return prepare_plan(case,spec)
    if action=='import-segmentation':
        from video_segmentation import import_masks
        return import_masks(case,spec,spec['out'])
    if action=='prepare':return prepare_unit(case,spec)
    if action=='prepare-recolor':return prepare_recolor(case,spec)
    if action in ('prepare-native-recolor', 'prepare-native-edit'):return prepare_native_recolor(case,spec)
    if action in ('plan-native-recolor', 'plan-native-edit'):return plan_native_recolor(case,spec)
    if action=='collect-native-recolor':
        return collect_originals(case, spec)
    if action=='replay':return replay(case,spec)
    if action=='review-raw':return review_raw(case,spec)
    if action=='plan':
        from aigc_composite import prepare_plan
        return prepare_plan(case,spec)
    if action in ('submit','query','download'):return remote_action(case,spec)
    raise ValueError('Unknown provider action')
