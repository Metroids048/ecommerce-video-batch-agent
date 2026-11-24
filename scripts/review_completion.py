"""Explicit full-coverage review delivery; ordinary qualified production stays closed."""
import copy
import project
import workflow

_ACTIVE = None


def active():
    return _ACTIVE is not None


def reference_hashes():
    return _ACTIVE.get('image_hashes', {}) if _ACTIVE else {}


def matches(case, file_id, operation_id, model_id, span):
    return bool(_ACTIVE and _ACTIVE['batch_id'] == case['batch_id']
                and _ACTIVE['file_id'] == file_id and _ACTIVE['operation_id'] == operation_id
                and any(u['model_id'] == model_id and u['source_frames_half_open'] == span
                        for u in _ACTIVE['unit_specs']))


def freeze(case, spec):
    from production_pipeline import implementation_binding, interval_route, current_issue, require_research
    require_research(case)
    import runninghub_adapter as provider
    if project.read_json(project.ROOT/'config/models.json').get('execution_mode') != 'maintenance':
        raise ValueError('Review completion is a scoped maintenance exception')
    fid = spec['file_id']; item = workflow.item_for(case, fid)
    contract = project.read_json(project.project_path(spec['contract']))
    state = project.read_json(workflow.state_path(case)); workflow.verify_state_case(case, state)
    entry = state['files'][fid]
    if (contract['batch_id'] != case['batch_id'] or contract['file_id'] != fid
            or not contract.get('user_instruction') or contract.get('quality_policy') != 'review_with_limits'):
        raise ValueError('Explicit full-coverage review instruction required')
    issue = next((i for i in entry['issues'] if i['id'] == contract['issue_id']), None)
    if not issue or issue['category'] != 'localization' or not current_issue(case, fid, entry, issue):
        raise ValueError('Coverage repair must bind the current candidate and original requirements')
    if any(j['status'] in ('running', 'queued') for j in entry['jobs']):
        raise ValueError('Recover original local job first')
    if any(u['status'].upper() not in ('PREPARED', 'FAILED', 'RAW_READY') for u in entry['execution_units']):
        raise ValueError('Reconcile original provider tasks first')
    original = provider.unit_for(state, fid, contract['reuse_unit_id'])
    request = provider.verify_unit(case, original)
    if project.digest(project.project_path(original['raw']['path'])) != original['raw']['sha256']:
        raise ValueError('Reusable original return changed')
    route = interval_route(case, fid, contract['operation_id'], entry)
    covered = set(range(*request['source_frames_half_open']))
    ids = set()
    for unit in contract['unit_specs']:
        lo, hi = unit['source_frames_half_open']
        if (unit['unit_id'] in ids or unit['file_id'] != fid or unit['operation_id'] != contract['operation_id']
                or unit['model_id'] != request['model_id'] or unit['model_id'] != route['model_id']
                or any(unit.get(k) != request.get(k) for k in ('prompt', 'parameters', 'image_references'))
                or not any(a <= lo < hi <= b for a, b in route['target_ranges_half_open'])
                or (hi-lo-1) % 16 or not 3 <= (hi-lo)/24 <= 15):
            raise ValueError('Only missing intervals with the unchanged API method may be prepared')
        ids.add(unit['unit_id']); covered.update(range(lo, hi))
    targets = {f for a,b in route['target_ranges_half_open'] for f in range(a,b)}
    if covered != targets:
        raise ValueError('Review completion must cover every observed target frame')
    grant_path = project.project_path(contract['authorization'])
    grant = project.read_json(grant_path)
    occupied = sum(float(u.get('new_cost_cny', 0) if u.get('cost_settled') else u.get('reserved_cny', 0))
                   for e in state['files'].values() for u in e['execution_units'])
    caps = contract['unit_caps_cny']
    if (set(caps) != ids or grant['batch_id'] != case['batch_id'] or not grant['remote_authorized']
            or not grant['full_cost_confirmed'] or any(caps[u] <= 0 or caps[u] != grant['unit_caps_cny'].get(u) for u in ids)
            or occupied + sum(caps.values()) > grant['total_budget_cny']):
        raise ValueError('Original total budget including failures and unknown reservations exceeded')
    binding = dict(contract=spec['contract'], sha256=project.digest(project.project_path(spec['contract'])),
                   dependency=workflow.requirement_binding(case,item), implementation=implementation_binding(),
                   authorization_sha256=project.digest(grant_path), reuse_raw_sha256=original['raw']['sha256'],
                   reuse_request_fingerprint=original['request_fingerprint'], frozen_at=workflow.now())
    with workflow.locked_state(case) as state:
        entry = state['files'][fid]
        if entry.get('review_completion'):
            raise ValueError('Full review method already fixed; consume it without parameter retries')
        entry['review_completion'] = binding
    return dict(status='full_coverage_review_fixed', target_frames=len(targets), new_units=sorted(ids),
                qualified=False, accepted=False)


def run(case, spec):
    global _ACTIVE
    from production_pipeline import implementation_binding, reopen, require_research
    require_research(case)
    import runninghub_adapter as provider
    fid = spec['file_id']; state = project.read_json(workflow.state_path(case))
    workflow.verify_state_case(case,state)
    entry = state['files'][fid]; binding = entry['review_completion']
    path = project.project_path(binding['contract'])
    contract = project.read_json(path)
    original = provider.unit_for(state, fid, contract['reuse_unit_id'])
    original_request = provider.verify_unit(case, original)
    if (project.digest(path) != binding['sha256'] or binding['implementation'] != implementation_binding()
            or binding['dependency'] != workflow.requirement_binding(case,workflow.item_for(case,fid))
            or project.digest(project.project_path(contract['authorization'])) != binding['authorization_sha256']
            or original['request_fingerprint'] != binding['reuse_request_fingerprint']
            or project.digest(project.project_path(original['raw']['path'])) != binding['reuse_raw_sha256']):
        raise ValueError('Frozen full review inputs or implementation changed')
    action = spec['review_action']
    if _ACTIVE:
        raise ValueError('Nested review completion is not allowed')
    _ACTIVE = dict(contract, image_hashes={p:original_request['artifact_hashes'][p]
                                         for p in original_request['image_references']})
    try:
        if action == 'reopen-materials':
            return reopen(case, dict(file_id=fid, issue_id=contract['issue_id']))
        unit = next((u for u in contract['unit_specs'] if u['unit_id'] == spec['unit_id']), None)
        if not unit:
            raise ValueError('Only fixed missing-range units may run')
        if action == 'prepare':
            return provider.prepare_unit(case, copy.deepcopy(unit))
        if action not in ('submit', 'query', 'download', 'settle'):
            raise ValueError('Only original fixed-unit actions may run; no trial or replacement methods')
        prepared = provider.unit_for(state,fid,unit['unit_id'])
        request = provider.verify_unit(case,prepared)
        if any(request.get(k) != unit.get(k) for k in
               ('file_id','operation_id','unit_id','model_id','source_frames_half_open','prompt','parameters','image_references')):
            raise ValueError('Prepared unit differs from the fixed full review method')
        body = dict(action=action,file_id=fid,unit_id=unit['unit_id'],authorization=contract['authorization'],
                    maximum_charge_cny=contract['unit_caps_cny'][unit['unit_id']])
        return provider.dispatch(case,body)
    finally:
        _ACTIVE = None
