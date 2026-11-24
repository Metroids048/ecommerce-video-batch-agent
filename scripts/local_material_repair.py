"""One diagnosed local material repair in the existing order, using cached inputs.

This is a handoff consumer for the normal preview/export/review chain. It does
not submit jobs, certify visual quality, erase failures or create another case.
"""
import json
import cv2
import numpy as np
import project
import workflow


def repair_scope(case, kind):
    from production_effort import provider_first
    return (case.get('execution_policy', {}).get('local_only') is True or
            (provider_first(case) and kind == 'observed_colour_regions'))


def observed_region(frame, record, coordinates):
    """Colour isolation inside observed shot regions, with explicit foreground protection."""
    from source_matte import polygon_union
    size = list(frame.shape[1::-1])
    region = polygon_union(record['regions'], coordinates, size)
    protected = polygon_union(record.get('protected', []), coordinates, size)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gate = record['material_gate']
    if gate.get('soft_red'):
        hue = hsv[:, :, 0].astype(np.float32)
        centre = gate['soft_red'].get('centre_hue', 178)
        distance = np.minimum(abs(hue-centre), 180-abs(hue-centre))
        chroma = np.clip((gate['soft_red']['outer_hue']-distance)/
                         (gate['soft_red']['outer_hue']-gate['soft_red']['inner_hue']), 0, 1)
        low, high = gate['soft_red']['saturation_ramp']
        saturation = hsv[:, :, 1].astype(np.float32)
        if gate['soft_red'].get('preserve_neutral_reflections'):
            dark = np.clip((150-hsv[:, :, 2].astype(np.float32))/65, 0, 1)
            low = low+(100-low)*dark
            high = high+(150-high)*dark
        chroma *= np.clip((saturation-low)/(high-low), 0, 1)
        # Skin protection follows its observed colour inside the foreground ROI;
        # coarse hand polygons must not cut holes in adjacent fabric.
        skin = (protected != 0) & (hue >= 3) & (hue <= 35) & (saturation < 115)
        chroma[skin] = 0
        result = np.rint(region*chroma).astype(np.uint8)
        if gate['soft_red'].get('close_weave_holes'):
            scale = size[0]/coordinates[0]
            kernel = max(3, int(5*scale)//2*2+1)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((kernel,kernel), np.uint8))
            result = cv2.GaussianBlur(result, (kernel,kernel), max(.55,scale))
        else:
            result = cv2.GaussianBlur(result, (3,3), .55)
        result[(region == 0) | skin] = 0
        result[polygon_union(record.get('hard_protected', []), coordinates, size) != 0] = 0
        return result
    selected = np.zeros(frame.shape[:2], bool)
    for lo, hi in gate['hue_ranges']:
        selected |= (hsv[:, :, 0] >= lo) & (hsv[:, :, 0] <= hi)
    selected &= hsv[:, :, 1] >= gate['minimum_saturation']
    region[~selected | (protected != 0)] = 0
    return region


def verify(case, file_id, entry):
    repair = entry.get('local_material_repair')
    if not repair:
        return None
    item = workflow.item_for(case, file_id)
    if (not repair_scope(case, repair.get('kind')) or
            repair['dependency'] != workflow.requirement_binding(case, item)):
        raise ValueError('Local repair must retain the exact local-only source and requirements')
    for path, sha in repair['artifacts'].items():
        if project.digest(project.project_path(path)) != sha:
            raise ValueError('Fixed local repair input changed: ' + path)
    parents = [u for u in entry.get('execution_units', []) if u['unit_id'] in repair['parent_unit_ids']]
    if (len(set(repair['parent_unit_ids'])) != len(repair['parent_unit_ids']) or
            len(parents) != len(repair['parent_unit_ids']) or
            {u['unit_id'] for u in parents} != set(repair['parent_unit_ids'])):
        raise ValueError('Fixed local repair requires its original parent units exactly once')
    for unit in parents:
        if unit.get('raw_review') != repair['parent_reviews'][unit['unit_id']]:
            raise ValueError('Diagnosed raw review changed')
        if repair.get('parent_unit_bindings'):
            from runninghub_adapter import raw_origin_binding
            if raw_origin_binding(unit) != repair['parent_unit_bindings'].get(unit['unit_id']):
                raise ValueError('Diagnosed original parent unit binding changed')
    if set(repair['parent_unit_ids']) != set(repair['parent_reviews']):
        raise ValueError('Incomplete local repair parent binding')
    return repair


def verify_plan(case, file_id, entry, plan):
    repair = verify(case, file_id, entry)
    if repair and workflow.hash_json(plan) != repair['pixel_plan_fingerprint']:
        raise ValueError('Local repair consumes exactly its fixed plan; no parameter or mask retry')


def repairs_issue(case, file_id, entry, candidate, issue):
    """Bind a reviewed local result to its preserved raw failure, never pass quality.

    The caller still requires all operation observations to pass. This only
    supplies the lineage a source-pixel repair lacks in provider write regions.
    """
    repair = verify(case, file_id, entry)
    if not repair or not issue or issue.get('id') not in repair['parent_issue_ids']:
        return False
    if (issue.get('observation_basis') != 'provider_raw' or
            issue.get('source_sha256') != candidate['source_sha256'] or
            issue.get('dependency') != candidate['dependency'] or
            candidate['dependency'] != repair['dependency'] or
            candidate.get('file_id') != file_id or not candidate.get('execution_manifest') or
            candidate.get('plan') != repair['manifest']):
        return False
    unit_id = issue.get('unit_id')
    parent = repair['parent_reviews'].get(unit_id)
    if (unit_id not in repair['parent_unit_ids'] or not parent or
            parent.get('raw_sha256') != issue.get('raw_sha256')):
        return False
    units = [u for u in entry.get('execution_units', []) if u['unit_id'] == unit_id]
    if len(units) != 1 or units[0].get('raw_review') != parent:
        return False
    if any(candidate.get('artifact_hashes', {}).get(p) != sha
           for p, sha in repair['artifacts'].items()):
        return False
    manifest = project.read_json(project.project_path(repair['manifest']))
    operation = next((op for op in manifest['operations']
                      if op['operation_id'] == issue.get('operation_id')), None)
    if not operation:
        return False
    if (manifest['file_id'] != file_id or manifest['dependency'] != repair['dependency'] or
            project.digest(project.project_path(operation['plan'])) != operation['plan_sha256']):
        return False
    pixel_plan = project.read_json(project.project_path(operation['plan']))
    verify_plan(case, file_id, entry, pixel_plan)
    result = next((r for r in candidate.get('operation_results', [])
                   if r.get('operation_id') == issue.get('operation_id')), {})
    return (pixel_plan.get('operation_id') == issue.get('operation_id') and
            result.get('method') == 'source_recolor' and
            result.get('regenerated_source_geometry') is False)


def native_region(source, returned, rgb):
    """Changed target chroma locates the region; source pixels supply texture.

    Already-blue source objects are excluded, so the API's untouched blue
    background cannot become an editable region. This is a colour-edit guide,
    not a general semantic object segmenter.
    """
    hsv = cv2.cvtColor(returned, cv2.COLOR_BGR2HSV)
    original = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hue = int(cv2.cvtColor(np.array([[rgb[::-1]]], np.uint8), cv2.COLOR_BGR2HSV)[0, 0, 0])
    distance = lambda x: np.minimum(abs(x.astype(np.int16)-hue), 180-abs(x.astype(np.int16)-hue))
    selected = ((distance(hsv[:, :, 0]) <= 15) & (hsv[:, :, 1] >= 80) &
                ((distance(original[:, :, 0]) > 15) | (original[:, :, 1] < 55)))
    return selected.astype(np.uint8)*255


def material_components(source, mask, gate, guide=None):
    """Reject neutral spill components, retaining small matching fabric parts.

    Material colour only classifies components inside an existing semantic
    matte; it never finds an object globally or deletes parts by their size.
    """
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    lo, hi = gate['hue_range']
    material = ((hsv[:, :, 0] >= lo) & (hsv[:, :, 0] <= hi) &
                (hsv[:, :, 1] >= gate['minimum_saturation']))
    count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    result = mask.copy()
    for label in range(1, count):
        selected = labels == label
        if np.mean(material[selected]) < gate['minimum_component_fraction']:
            # A connected spill can include a real, narrow fabric edge. Do not
            # delete that edge with the entire neutral component. The existing
            # same-source colour return disambiguates it from skin/board pixels.
            keep = material if guide is None else material & (guide > 0)
            result[selected & ~keep] = 0
    return result


def prepare(case, spec):
    from production_pipeline import (implementation_binding, require_production, route_for)
    from runninghub_adapter import verify_unit
    from production_effort import require_lightweight_edit
    require_lightweight_edit(case)
    require_production(case)
    if (not repair_scope(case, spec.get('kind')) or
            not spec.get('user_instruction') or not spec.get('diagnosis')):
        raise ValueError('Diagnosed local-only repair and original instruction required')
    fid = spec['file_id']; item = workflow.item_for(case, fid)
    state = project.read_json(workflow.state_path(case)); workflow.verify_state_case(case, state)
    entry = state['files'][fid]
    old = entry.get('local_material_repair')
    parent_trial = None
    if old:
        verify(case, fid, entry)
        if old['request_fingerprint'] == workflow.hash_json(spec):
            return dict(status='local_materials_fixed', manifest=old['manifest'], reused=True, new_model_calls=0)
        parent_trial = next((t for t in entry.get('preview_trials', [])
                             if t['id'] == spec.get('repair_of_trial')), None)
        if (not parent_trial or parent_trial.get('verdict') != 'fail' or
                parent_trial.get('failure_scope') not in ('localization','protection','composition') or
                not parent_trial.get('diagnosis') or not parent_trial.get('next_change') or
                entry.get('current_candidate_id')):
            raise ValueError('Local repair is fixed; preserve the attempt instead of starting another loop')
        if project.digest(project.project_path(parent_trial['report'])) != parent_trial['report_sha256']:
            raise ValueError('Diagnosed preview report changed')
        if parent_trial.get('handoff_fingerprint') != entry.get('material_handoff', {}).get('fingerprint'):
            raise ValueError('Repair must consume this fixed material handoff failure')
        for e in parent_trial['evidence']:
            if project.digest(project.project_path(e['path'])) != e['sha256']:
                raise ValueError('Diagnosed preview evidence changed')
    if not parent_trial and (entry.get('material_handoff') or entry.get('current_candidate_id') or entry.get('preview_trials')):
        raise ValueError('Preserve existing material/candidate; reopen its bound defect before local repair')
    if {spec['operation_id']} != {op['id'] for op in item['operations']}:
        raise ValueError('Local repair currently requires one explicitly scoped operation')
    units = [u for u in entry.get('execution_units', []) if u['unit_id'] in spec['unit_ids']]
    if len(units) != len(spec['unit_ids']) or not units:
        raise ValueError('Reuse the existing original units, never create replacement jobs')
    if not all(u.get('raw_review', {}).get('status') in ('fail', 'fail_or_limited') for u in units):
        raise ValueError('Concrete preserved raw failures required')
    artifacts = {}
    def bind(path):
        artifacts[path] = project.digest(project.project_path(path))
        return project.read_json(project.project_path(path))
    observed = bind(spec['observation'])
    if observed['source_sha256'] != item['source_sha256']:
        raise ValueError('Local repair source observation differs')
    for evidence in spec['failure_evidence']:
        if project.digest(project.project_path(evidence['path'])) != evidence['sha256']:
            raise ValueError('Original failure evidence changed')
        artifacts[evidence['path']] = evidence['sha256']
    if not spec['failure_evidence']:
        raise ValueError('Preserve the actual failed preview and diagnosis')
    requests = []
    for u in units:
        requests.append(verify_unit(case, u))
        for p, sha in ((u['request_path'], project.digest(project.project_path(u['request_path']))),
                       (u['raw']['path'], u['raw']['sha256']),
                       (u['raw_review']['path'], u['raw_review']['sha256'])):
            if project.digest(project.project_path(p)) != sha:
                raise ValueError('Original unit artifact changed')
            artifacts[p] = sha
    count = item['source_frame_count_inventory']; w, h = item['dimensions']
    grid = spec['mask_dimensions']; mw, mh = grid
    alpha = np.zeros((count, mh, mw), np.uint8)
    source = cv2.VideoCapture(str(project.project_path(item['source'])))
    caps = []
    guides = []
    try:
        if spec['kind'] == 'native_colour_region':
            if any(r['model_id'] != 'kling-o3-std-edit' for r in requests):
                raise ValueError('Native colour repair requires original colour-edit returns')
            caps = [cv2.VideoCapture(str(project.project_path(u['raw']['path']))) for u in units]
        elif spec['kind'] == 'semantic_material_components':
            if len(units) != 1 or requests[0]['model_id'] != 'sam3-video-local':
                # Also accept the deployed model identifier, never an arbitrary donor.
                if len(units) != 1 or 'sam3' not in requests[0]['model_id']:
                    raise ValueError('Semantic repair requires the existing SAM3 result')
            masks = spec['input_masks']; artifacts[masks] = project.digest(project.project_path(masks))
            with np.load(project.project_path(masks), allow_pickle=False) as z:
                from source_matte import validate_metadata
                validate_metadata(z, item['source_sha256'], [w, h], [0, count])
                prior = z['alpha']
            if prior.shape != alpha.shape:
                raise ValueError('Reuse the original mask grid without another segmentation pass')
            for ref in spec.get('colour_guides', []):
                original = bind(ref['case'])
                donor = workflow.item_for(original, ref['file_id'])
                if donor['source_sha256'] != item['source_sha256']:
                    raise ValueError('Repair guide must refer to the exact original source')
                saved = project.read_json(workflow.state_path(original))
                workflow.verify_state_case(original, saved)
                unit = next(u for u in saved['files'][ref['file_id']]['execution_units']
                            if u['unit_id'] == ref['unit_id'])
                request = verify_unit(original, unit)
                if request['model_id'] != 'kling-o3-std-edit' or not unit.get('provider_task_id') or not unit.get('raw'):
                    raise ValueError('Only existing original colour-edit returns may guide this repair')
                bind(unit['request_path'])
                raw = unit['raw']; artifacts[raw['path']] = raw['sha256']
                if project.digest(project.project_path(raw['path'])) != raw['sha256']:
                    raise ValueError('Original guide changed')
                guides.append((cv2.VideoCapture(str(project.project_path(raw['path']))), request))
        elif spec['kind'] == 'observed_colour_regions':
            records = bind(spec['regions'])
            if (records['source_sha256'] != item['source_sha256'] or
                    [r['frame'] for r in records['records']] != list(range(count))):
                raise ValueError('Observed colour repair must bind every original frame')
        else:
            raise ValueError('Unknown diagnosed local material repair')
        for n in range(count):
            ok, frame = source.read()
            if not ok: raise ValueError('Source ended before the complete timeline')
            frame = cv2.resize(frame, (mw, mh), interpolation=cv2.INTER_AREA)
            if spec['kind'] == 'observed_colour_regions':
                record = records['records'][n]
                alpha[n] = observed_region(frame, record, records['coordinate_size'])
            elif caps:
                for cap, request in zip(caps, requests):
                    lo, hi = request['source_frames_half_open']
                    if lo <= n < hi:
                        ok, raw = cap.read()
                        if not ok: raise ValueError('Original return ended before its source range')
                        raw = cv2.resize(raw, (mw, mh), interpolation=cv2.INTER_LINEAR)
                        alpha[n] = np.maximum(alpha[n], native_region(frame, raw, spec['color_rgb']))
            else:
                guide = np.zeros((mh, mw), np.uint8) if guides else None
                for cap, request in guides:
                    lo, hi = request['source_frames_half_open']
                    if lo <= n < hi:
                        ok, raw = cap.read()
                        if not ok: raise ValueError('Existing guide ended before its source range')
                        raw = cv2.resize(raw, (mw, mh), interpolation=cv2.INTER_LINEAR)
                        guide = np.maximum(guide, native_region(frame, raw, spec['color_rgb']))
                alpha[n] = material_components(frame, prior[n], spec['material_gate'], guide)
    finally:
        source.release()
        for cap in caps: cap.release()
        for cap, _ in guides: cap.release()
    states = ['absent']*count
    op_observed = observed['operations'][spec['operation_id']]
    for lo, hi in op_observed['appearance_ranges_half_open']:
        states[lo:hi] = ['visible']*(hi-lo)
    for lo, hi in spec.get('observed_occluded_ranges_half_open', []):
        states[lo:hi] = ['occluded']*(hi-lo)
        alpha[lo:hi] = 0
    if spec['kind'] == 'observed_colour_regions':
        states = [r['state'] for r in records['records']]
    empty = [n for n, (s, a) in enumerate(zip(states, alpha)) if s == 'visible' and not np.any(a)]
    if empty:
        raise ValueError('Repair leaves observed visible target empty at source frames ' +
                         str(empty[:20]) + '; inspect fixed source evidence, never relabel to bypass coverage')
    out = project.project_path(spec['out']); out.parent.mkdir(parents=True, exist_ok=True)
    pixels = out.with_name(out.stem+'-pixels.json'); masks = out.with_name(out.stem+'-alpha.npz')
    if any(p.exists() for p in (out, pixels, masks)):
        raise ValueError('Preserve existing local repair outputs; do not overwrite or retry')
    np.savez_compressed(masks, alpha=alpha, protected=np.zeros_like(alpha), source_sha256=item['source_sha256'],
                        source_dimensions=[w,h], mask_dimensions=grid, frames_half_open=[0,count])
    masks_path = str(masks.relative_to(project.ROOT)); artifacts[masks_path] = project.digest(masks)
    plan = dict(file_id=fid, operation_id=spec['operation_id'], case_version=case['case_version'],
        source_sha256=item['source_sha256'], dependency=workflow.requirement_binding(case,item),
        composition_mode='source_recolor', segmentation_backend='cached_local_material_repair',
        source_frame_count=count, source_frames_half_open=[0,count], edit_frames_half_open=[0,count],
        masks=masks_path, color_asset=None, color_rgb=spec['color_rgb'],
        recolor_mode=spec.get('recolor_mode', 'hsv_chroma' if caps or spec['kind'] == 'observed_colour_regions' else 'hsv_hue'), full_appearance_required=True,
        artifact_hashes=artifacts.copy(), review=dict(frame_states=states,
            records=[dict(frame=n,state=s, **({'semantic_instance_ids': records['records'][n]['semantic_instance_ids']}
                if spec['kind'] == 'observed_colour_regions' and records['records'][n].get('semantic_instance_ids') else {}))
                for n,s in enumerate(states)], evidence=spec['failure_evidence']))
    pixels.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    manifest = dict(schema='execution_manifest_v1', file_id=fid, dependency=plan['dependency'],
        risk_preview_ranges_half_open=op_observed['risk_ranges_half_open'], operations=[dict(
            operation_id=spec['operation_id'], disposition='ready', plan=str(pixels.relative_to(project.ROOT)),
            plan_sha256=project.digest(pixels))])
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    artifacts[str(pixels.relative_to(project.ROOT))] = project.digest(pixels)
    artifacts[str(out.relative_to(project.ROOT))] = project.digest(out)
    repair = dict(kind=spec['kind'], dependency=plan['dependency'], request_fingerprint=workflow.hash_json(spec),
        instruction=spec['user_instruction'], diagnosis=spec['diagnosis'], artifacts=artifacts,
        pixel_plan_fingerprint=workflow.hash_json(plan), manifest=str(out.relative_to(project.ROOT)),
        parent_unit_ids=[u['unit_id'] for u in units], parent_reviews={u['unit_id']:u['raw_review'] for u in units},
        parent_issue_ids=[i['id'] for i in entry['issues']],
        parent_trial_ids=(old.get('parent_trial_ids', [])+[parent_trial['id']]) if parent_trial else [], accepted=False)
    from runninghub_adapter import raw_origin_binding
    repair['parent_unit_bindings'] = {u['unit_id']:raw_origin_binding(u) for u in units}
    with workflow.locked_state(case) as state:
        entry = state['files'][fid]
        if entry.get('local_material_repair') != old: raise ValueError('Another repair owns this file')
        if old:
            entry.setdefault('local_material_repair_history', []).append(old)
            entry.setdefault('material_handoff_history', []).append(entry.pop('material_handoff'))
        entry['local_material_repair'] = repair
        if entry.get('route_handoff'): entry.setdefault('route_handoff_history', []).append(entry['route_handoff'])
        entry['route_handoff'] = dict(dependency=plan['dependency'],
            observation=dict(path=spec['observation'],sha256=artifacts[spec['observation']]),
            routes={op['id']:route_for(op, case) for op in item['operations']}, implementation=implementation_binding(),
            methods={spec['operation_id']:dict(id=spec['kind'],local={k:plan[k] for k in
                ('composition_mode','segmentation_backend')})})
    return dict(status='local_materials_fixed_pending_normal_preview', manifest=str(out.relative_to(project.ROOT)),
                new_model_calls=0, accepted=False)
