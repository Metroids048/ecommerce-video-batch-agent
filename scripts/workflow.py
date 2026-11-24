"""Case/run adapter and local, version-bound review/repair. Never submits models."""
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import project


def now():
    return datetime.now(timezone.utc).isoformat()


def hash_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def requirement_binding(case, item):
    assets = {op[key]: project.digest(project.project_path(op[key]))
              for op in item['operations'] for key in ('replacement_asset','reference_asset') if op.get(key)}
    return {'case_version': case['case_version'], 'source_sha256': item['source_sha256'],
            'requirements_sha256': hash_json({'operations': item['operations'],
                'user_input_sha256': project.digest(project.project_path(case['user_input']))}), 'assets': assets}


def item_for(case, file_id):
    item = next((f for f in case['files'] if f['id'] == file_id), None)
    if item is None:
        raise ValueError('Unknown file id')
    return item


def case_binding(case):
    binding = {'case_version': case['case_version'], 'files': {
        item['id']: dict(requirement_binding(case, item), source=item['source']) for item in case['files']}}
    if 'execution_policy' in case:
        binding['execution_policy'] = case['execution_policy']
    return binding


def verify_state_case(case, state):
    if state['case_version'] != case['case_version'] or set(state['files']) != {f['id'] for f in case['files']}:
        raise ValueError('State/case binding changed; preserve the old run and use a new batch')
    if 'case_binding' in state:
        if state['case_binding'] != case_binding(case):
            raise ValueError('State source/requirements/asset binding changed; use a new batch')
    else:
        # shortcut: legacy batch has per-candidate binding, retain it without inventing a historical case hash.
        if case['batch_id'] != 'ecommerce-batch-002':
            raise ValueError('Legacy state has no case binding; explicit migration is required')


def verify_candidate(case, file_id, candidate):
    item = item_for(case, file_id)
    if candidate['file_id'] != file_id or candidate['source_sha256'] != item['source_sha256']:
        raise ValueError('Candidate source/file mismatch')
    if project.digest(project.project_path(item['source'])) != item['source_sha256']:
        raise ValueError('Actual source hash changed')
    if candidate['dependency'] != requirement_binding(case, item):
        raise ValueError('Candidate requirements/asset revision is stale')
    if candidate.get('cache_origin'):
        verify_cache_origin(case, file_id, candidate['cache_origin'], candidate['output_sha256'])
    if project.digest(project.project_path(candidate['output'])) != candidate['output_sha256']:
        raise ValueError('Candidate output hash changed')
    for key, expected in candidate.get('artifact_hashes', {}).items():
        if project.digest(project.project_path(key)) != expected:
            archive = code_archive(key, expected)
            if archive is None or not archive.is_file() or project.digest(archive) != expected:
                raise ValueError('Candidate dependency hash changed: ' + key)
    return candidate


def verify_cache_origin(case, file_id, origin, output_sha256=None):
    """Validate an exact finished-output hit, never certify a new editing method."""
    path = project.project_path(origin['case'])
    if project.digest(path) != origin['case_sha256']:
        raise ValueError('Cache origin case changed')
    parent = project.read_json(path)
    if parent['batch_id'] == case['batch_id']:
        raise ValueError('Cache origin must be a different preserved batch')
    item, previous = item_for(case, file_id), item_for(parent, origin['file_id'])
    if (item['source_sha256'] != previous['source_sha256']
            or item['operations'] != previous['operations']
            or item.get('audio_required') != previous.get('audio_required')
            or requirement_binding(case, item)['assets'] != requirement_binding(parent, previous)['assets']):
        raise ValueError('Cache miss: source, operations, protection, audio or assets differ')
    state = project.read_json(state_path(parent)); verify_state_case(parent, state)
    entry = state['files'][origin['file_id']]
    candidate = candidate_for(entry, origin['candidate_id'])
    if candidate.get('cache_origin'):
        raise ValueError('Cache chains are unsupported; refer to the original production batch')
    verify_candidate(parent, origin['file_id'], candidate)
    if (entry['current_candidate_id'] != candidate['id'] or candidate['status'] != 'qualified'
            or not quality_reviews_pass(candidate) or any(i['status'] != 'closed' for i in entry['issues'])
            or any(j['status'] in ('running', 'queued') for j in entry['jobs'])):
        raise ValueError('Cache origin is not a qualified, idle current result')
    if output_sha256 and candidate['output_sha256'] != output_sha256:
        raise ValueError('Cache origin output differs')
    if origin.get('candidate_sha256') and hash_json(candidate) != origin['candidate_sha256']:
        raise ValueError('Cache origin candidate changed')
    return candidate


def reuse_candidate(case, spec):
    """Attach verified media to a fresh order; semantic reviews are not inherited."""
    if case.get('execution_policy', {}).get('reuse_only') is not True:
        raise ValueError('Exact cache import requires reuse-only intake')
    file_id = spec['file_id']; item = item_for(case, file_id)
    origin = copy.deepcopy(spec['origin'])
    parent = verify_cache_origin(case, file_id, origin)
    origin['candidate_sha256'] = hash_json(parent)
    candidate = copy.deepcopy(parent)
    candidate.update(id='cached-full-v1', file_id=file_id, parent_id=None, cache_origin=origin,
        dependency=requirement_binding(case, item), status='pending_review', accepted=False,
        note='Exact historical full-output reuse; original rendering receipts remain at cache_origin. New order requires new review.',
        reviews=[dict(r, dependency=requirement_binding(case, item)) for r in parent['reviews']
                 if r['kind'] in ('technical', 'audio_preservation', 'audio_listening')], operation_reviews=[])
    verify_candidate(case, file_id, candidate)
    with locked_state(case) as state:
        verify_state_case(case, state); entry = state['files'][file_id]
        if entry['candidates']:
            existing = candidate_for(entry, entry['current_candidate_id'])
            verify_candidate(case, file_id, existing)
            if existing.get('cache_origin') != origin:
                raise ValueError('Cache import already belongs to another result; preserve it')
            reused = True
        else:
            if entry['jobs'] or entry.get('execution_units'):
                raise ValueError('Reconcile original jobs before cache import')
            entry['candidates'].append(candidate)
            entry.update(current_candidate_id=candidate['id'], missing_reason=None)
            reused = False
    return dict(status='cached_candidate_pending_review' if not reused else 'existing_cache_reused',
                file_id=file_id, candidate_id=candidate['id'], output_sha256=candidate['output_sha256'],
                cache_origin=origin, new_model_calls=0, new_renders=0, new_cost_cny=0, accepted=False)


def code_archive(key, digest):
    # Only executable provenance can be frozen. Masks, plans, sources, assets,
    # output and review evidence must still match their actual bound paths.
    path = Path(key)
    if path.parts[:1] != ('scripts',) or '..' in path.parts or path.suffix != '.py':
        return None
    return project.ROOT / 'evidence/artifact-history' / (digest + '.py')


def archive_candidate_code(candidate):
    for key, expected in candidate.get('artifact_hashes', {}).items():
        archive = code_archive(key, expected)
        if archive is None:
            continue
        source = project.project_path(key)
        if project.digest(source) != expected:
            raise ValueError('Export code changed before registration: ' + key)
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            with archive.open('xb') as stream:
                stream.write(source.read_bytes())
        if project.digest(archive) != expected:
            raise ValueError('Exact code archive corrupted: ' + key)


def verify_historical_parent(case, file_id, candidate):
    """Frozen media keeps exact code provenance, without inheriting quality."""
    return verify_candidate(case, file_id, candidate)


def effective_reviews(candidate):
    return [dict(r, status=('stale' if r.get('output_sha256') != candidate['output_sha256']
                            or r.get('dependency', candidate['dependency']) != candidate['dependency']
                            else r['status'])) for r in candidate.get('reviews', [])]


def quality_reviews_pass(candidate):
    if any(r.get('disposition') == 'blocked' or r.get('unresolved_ranges')
           or r.get('known_unedited_instances') or r.get('replacement_dropout_risks')
           or r.get('diagnostic') or (r.get('raw_reviews_pass') is False and not r.get('write_regions_reviewed'))
           for r in candidate.get('operation_results', [])):
        return False
    reviews = effective_reviews(candidate)
    required = {'technical', 'audio_preservation', 'target_coverage', 'visual', 'playback', 'delivery'}
    if candidate.get('execution_manifest') or len(candidate.get('operation_ids', [])) > 1:
        operations = candidate.get('operation_reviews', [])
        expected = candidate.get('operation_ids', [])
        kinds = {'target_coverage', 'visual', 'protection', 'temporal'}
        if (not expected or len(operations) != len(expected)
                or {r.get('operation_id') for r in operations} != set(expected)):
            return False
        for operation in operations:
            checks = operation.get('checks', [])
            if (operation.get('output_sha256') != candidate['output_sha256']
                    or operation.get('dependency') != candidate['dependency']
                    or len(checks) != len(kinds) or {c.get('kind') for c in checks} != kinds
                    or any(c.get('status') != 'pass' or not c.get('scope') or not c.get('evidence')
                           for c in checks)):
                return False
    return (required <= {r['kind'] for r in reviews}
            and all(r['status'] == 'pass' or
                    (r['kind'] == 'audio_listening' and r['status'] == 'not_required')
                    for r in reviews))


def require_candidate_coverage(case, candidate):
    if not candidate.get('execution_manifest'):
        return
    # Exact cache imports retain the origin's frozen source observation/plan.
    origin = candidate.get('cache_origin')
    scope_case = project.read_json(project.project_path(origin['case'])) if origin else case
    from execution_plan import require_frozen_appearances
    require_frozen_appearances(scope_case, project.read_json(project.project_path(candidate['plan'])))


def candidate_for(entry, candidate_id):
    candidate = next((c for c in entry['candidates'] if c['id'] == candidate_id), None)
    if candidate is None:
        raise ValueError('Unknown candidate/parent; original source has no edited candidate')
    return candidate


def state_path(case):
    return project.ROOT / 'runs' / case['batch_id'] / 'workflow.json'


@contextlib.contextmanager
def locked_state(case):
    path = state_path(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = project.read_json(path)
        original = copy.deepcopy(state)
        yield state
        if state == original:
            return  # Exact recovery/read must not create a new production revision.
        state['revision'] += 1
        state['updated_at'] = now()
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
        os.replace(temp, path)
        fcntl.flock(lock, fcntl.LOCK_UN)


def resolve_parent(case, file_id, parent):
    path = state_path(case)
    if path.exists():
        entry = project.read_json(path)['files'].get(file_id, {})
        candidate = next((c for c in entry.get('candidates', []) if parent in (c['id'], c['output'])), None)
        if candidate:
            return verify_candidate(case, file_id, candidate)
    # A sidecar without requirement/asset binding cannot establish a repair parent.
    try:
        media = project.project_path(parent)
        sidecar = project.read_json(Path(str(media)+'.json'))
        item = item_for(case, file_id)
        candidate = {'id': parent, 'file_id': sidecar['file_id'], 'source_sha256': sidecar['source_sha256'],
                     'output': parent, 'output_sha256': sidecar['output_sha256'],
                     'dependency': sidecar['dependency']}
        if sidecar['case_version'] != case['case_version']:
            raise ValueError('parent case version mismatch')
        return verify_candidate(case, file_id, candidate)
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError('Unverified parent artifact: '+str(exc)) from exc


def dependency_plan(case, candidate, interval, category):
    recompute = {
        'localization': ['affected target masks/trajectory', 'regional composition', 'complete export', 'encoded visual/seam review'],
        'asset': ['exact asset composition using existing carrier track', 'complete export', 'encoded visual/seam review'],
        'protection': ['affected protection/masks', 'regional composition', 'complete export', 'encoded visual/seam review'],
        'composition': ['affected registration/color', 'regional composition', 'complete export', 'encoded visual/seam review'],
        'packaging': ['original audio remux', 'timeline/audio/tail review'],
        'model_internal': ['affected provider unit input/method review; no automatic resubmission']}
    if category not in recompute:
        raise ValueError('Unsupported cause; choose an explicit cause before execution')
    item = item_for(case, candidate['file_id'])
    start, end = interval
    if not all(isinstance(t, (float, int)) and math.isfinite(t) for t in interval) or not 0 <= start < end <= item['duration_seconds']:
        raise ValueError('Repair requires valid source interval')
    return {'file_id': item['id'], 'source_sha256': item['source_sha256'], 'parent_version': candidate['id'],
            'parent_output_sha256': candidate['output_sha256'], 'parent_artifact_verified': True,
            'source_writeback_seconds': interval, 'category': category, 'recompute': recompute[category],
            'reuse': ['source', 'valid raw model returns', 'unaffected carrier tracks', 'other files'],
            'remote_submission_allowed': False, 'actual_repair_executed': False,
            'status': 'needs_manual_or_valid_cache' if category == 'model_internal' else 'local_dependency_plan',
            'accepted': False}


def require_file_revision(state, body, entry, message):
    """Keep optimistic locking on this complete file, not unrelated batch work.

    Old clients retain the strict global revision check. A supplied token must
    always match, including when the global revision happens to be current.
    """
    expected = body.get('expected_revision')
    token = body.get('expected_file_state')
    if (type(expected) is not int or not 0 <= expected <= state['revision']
            or (token is not None and token != hash_json(entry))
            or (token is None and expected != state['revision'])):
        raise ValueError(message)


def bind_feedback(case, state, body):
    try:
        file_id, candidate_id = body['file_id'], body['candidate_id']
        entry = state['files'][file_id]
        item = item_for(case, file_id)
        if candidate_id is None and not entry['candidates']:
            if project.digest(project.project_path(item['source'])) != item['source_sha256']:
                raise ValueError('Feedback actual source hash changed')
            candidate = {'id': None, 'output_sha256': item['source_sha256'],
                         'source_sha256': item['source_sha256'], 'parent_id': None}
        else:
            candidate = candidate_for(entry, candidate_id)
            # Feedback may identify a defect in exact archived parent bytes.
            # Current review/delivery still require current dependencies.
            verify_historical_parent(case, file_id, candidate)
        require_file_revision(state, body, entry, 'Revision conflict; reload without discarding the feedback')
        if body['candidate_sha256'] != candidate['output_sha256'] or body['source_sha256'] != candidate['source_sha256']:
            raise ValueError('Feedback source/candidate hash mismatch')
        t = body['source_time']
        if not isinstance(t, (float, int)) or not math.isfinite(t) or not 0 <= t < item_for(case, file_id)['duration_seconds']:
            raise ValueError('Feedback source time outside complete source')
        text = body['text'].strip()
        if not text or len(text) > 2000:
            raise ValueError('Feedback requires a description of at most 2000 characters')
        region = body.get('region')
        if region is not None and (len(region) < 3 or len(region) > 32 or any(len(p)!=2 or any(not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1 for v in p) for p in region)):
            raise ValueError('Region must be 3–32 normalized source-image points')
        category = body.get('category', 'protection')
        actor = body.get('actor', 'browser_feedback')
        if actor not in ('browser_feedback', 'host_agent'):
            raise ValueError('Unknown feedback actor')
        if category not in ('localization','asset','protection','composition','packaging','model_internal'):
            raise ValueError('Unknown cause')
        operations = item['operations']
        operation_id = body.get('operation_id', operations[0]['id'] if len(operations) == 1 else None)
        if operation_id not in {op['id'] for op in operations}:
            raise ValueError('Feedback must identify the actual operation when a file has several operations')
    except (KeyError, TypeError) as exc:
        raise ValueError('Feedback must bind file, candidate, hashes, revision and source time') from exc
    timeline_binding = {}
    if item.get('source_timeline'):
        import source_timeline
        timeline = source_timeline.load_for_item(item)
        source_frame = source_timeline.frame_at_time(timeline, t)
        timeline_binding = {'source_pts': timeline['pts'][source_frame],
                            'source_time_base': timeline['time_base'],
                            'time_mapping_basis': 'decoded_source_pts',
                            'source_timeline_sha256': item['source_timeline_sha256']}
    else:
        # shortcut: historical cases have no PTS index; new intake supplies one.
        source_frame = min(math.floor(t*item['fps_container_average']+1e-6), entry.get('source_frame_count',10**9)-1)
        timeline_binding = {'time_mapping_basis': 'legacy_average_fps_not_pts_verified'}
    return {'id': 'issue-'+uuid.uuid4().hex[:12], 'created_at': now(), 'file_id': file_id,
            'candidate_id': candidate_id, 'candidate_sha256': candidate['output_sha256'],
            'source_sha256': candidate['source_sha256'], 'parent_id': candidate.get('parent_id'),
            'operation_id': operation_id, 'source_time_seconds': t,
            'source_frame': source_frame, **timeline_binding,
            'region': region, 'text': text, 'category': category, 'status': 'open',
            'dependency': requirement_binding(case,item),
            'observation_basis': 'source_without_candidate' if candidate_id is None else 'edited_candidate',
            'actor': actor, 'accepted': False}


def add_feedback(case, body):
    with locked_state(case) as state:
        issue = bind_feedback(case, state, body)
        state['files'][body['file_id']]['issues'].append(issue)
    return issue


def ranges(indices):
    out = []
    for i in indices:
        if out and out[-1][1] == i:
            out[-1][1] = i+1
        else:
            out.append([i,i+1])
    return out


def register_existing(case, file_id, candidate_id, output, plan_path, parent_id, status, note):
    item = item_for(case,file_id)
    plan = project.read_json(project.project_path(plan_path))
    sidecar = project.read_json(project.project_path(output+'.json'))
    if sidecar['source_sha256'] != item['source_sha256'] or sidecar['file_id'] != file_id or sidecar['case_version'] != case['case_version']:
        raise ValueError('Legacy candidate source/version mismatch')
    candidate = {'id': candidate_id, 'file_id': file_id, 'source_sha256': item['source_sha256'],
                 'output': output, 'output_sha256': sidecar['output_sha256'],
                 'sidecar': output+'.json', 'plan': plan_path, 'parent_id': parent_id,
                 'dependency': requirement_binding(case,item), 'artifact_hashes': {
                     output+'.json': project.digest(project.project_path(output+'.json')),
                     plan_path: project.digest(project.project_path(plan_path)),
                     **(sidecar.get('artifact_hashes', {}) if plan.get('schema') == 'execution_manifest_v1' else
                        {plan['masks']: plan['masks_sha256'], plan['cloud']: plan['cloud_sha256']})},
                 'source_mapping': {'mode': 'source_pts_identity', 'retimed': False},
                 'modified_frame_ranges': ranges(sidecar['modified_source_frames']),
                 'status': status, 'note': note, 'accepted': False,
                 'reviews': [{'kind': 'technical', 'status': 'pass' if sidecar['timeline_exact'] and sidecar['audio_packets_pts_payload_exact'] and sidecar['preencode_changes_outside_effective_mask']==0 else 'fail',
                    'output_sha256': sidecar['output_sha256'], 'evidence': output+'.json',
                    'scope': 'frame PTS/audio packets/preencode mask only; no listening or semantic judgment'},
                    {'kind': 'visual', 'status': 'pending', 'output_sha256':sidecar['output_sha256']},
                    {'kind':'audio_preservation','status':'pass' if sidecar['timeline_exact'] and sidecar['audio_packets_pts_payload_exact'] else 'fail',
                     'output_sha256':sidecar['output_sha256'], 'evidence':output+'.json',
                     'scope':'Source audio packet payload/PTS and video PTS identity; no claim of listening.'},
                    {'kind':'audio_listening','status':'not_required' if sidecar['timeline_exact'] and sidecar['audio_packets_pts_payload_exact'] else 'pending',
                     'output_sha256':sidecar['output_sha256'],
                     'scope':'Visual-only operation with exact copied source audio. Changed audio or semantic listening requests require separate Agent audio review.'}],
                 'supervised': True}
    if plan.get('schema') == 'execution_manifest_v1':
        candidate['execution_manifest'] = True
        candidate['operation_ids'] = [op['id'] for op in item['operations']]
        candidate['operation_results'] = sidecar['operations']
    archive_candidate_code(candidate)
    verify_candidate(case,file_id,candidate)
    return candidate


def execute_plan(case, plan_relative, candidate_id, parent_id=None, issue_id=None, diagnostic=False):
    """Run a source-bound plan through the production compositor and register it.

    Host Agent owns localization/visual judgment. Exporting never sets a quality
    pass. A repair names both the exact parent and its persisted issue.
    """
    if not __import__('re').fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', candidate_id):
        raise ValueError('Invalid candidate id')
    plan = project.read_json(project.project_path(plan_relative))
    file_id = plan['file_id']
    # Recovery consumes the original candidate before current-code preview checks.
    # A policy/entry edit must never force rendering an unchanged, archived film.
    existing_state = project.read_json(state_path(case))
    verify_state_case(case, existing_state)
    existing = next((c for c in existing_state['files'][file_id]['candidates'] if c['id'] == candidate_id), None)
    if existing:
        verify_candidate(case, file_id, existing)
        if project.read_json(project.project_path(existing['plan'])) != plan or existing.get('parent_id') != parent_id:
            raise ValueError('Candidate id belongs to different inputs')
        return dict(existing, execution_reused=True, new_model_calls=0, new_renders=0)
    if case.get('execution_policy', {}).get('reuse_only') is True:
        from production_pipeline import require_production
        require_production(case)  # Diagnostic flags cannot reopen a cache-only order.
    from production_effort import require_reviewed_repair
    require_reviewed_repair(case, file_id, [o['id'] for o in item_for(case, file_id)['operations']])
    from production_pipeline import require_production
    from production_effort import require_preview
    if diagnostic:
        # Explicit review returns retain failed verdicts and use the same source
        # compositor. They cannot become qualified deliveries or reopen models.
        from execution_plan import validate
        loaded = validate(case, plan)[2]
        ready = [p for _, p in loaded if p is not None]
        if not ready or any(p.get('composition_mode') != 'aigc_regional' or not p.get('diagnostic') for p in ready):
            raise ValueError('Diagnostic export requires explicitly diagnostic provider-pixel plans')
    else:
        require_production(case)
        require_preview(case, plan)
    with locked_state(case) as state:
        verify_state_case(case, state)
        entry = state['files'][file_id]
        if (any(c['id'] == candidate_id for c in entry['candidates'])
                or any(j['id'] == candidate_id for j in entry['jobs'])):
            raise ValueError('Candidate/job exists; inspect it and preserve failed attempts before using a new id')
        if any(j['status'] in ('queued', 'running') for j in entry['jobs']):
            raise ValueError('Another job owns this file')
        if parent_id:
            parent = verify_historical_parent(case, file_id, candidate_for(entry, parent_id))
            issue = next((i for i in entry['issues'] if i['id'] == issue_id and i['candidate_id'] == parent_id), None)
            if not issue or issue['status'] != 'open':
                raise ValueError('Repair requires an open issue bound to this exact parent')
            if plan.get('parent_output_sha256') != parent['output_sha256']:
                raise ValueError('Plan parent hash mismatch')
            previous_plan = project.read_json(project.project_path(parent['plan']))
            render_inputs = lambda p: {k: v for k, v in p.items()
                                      if k not in ('parent_output_sha256', 'parent_version')}
            same_code = all(project.digest(project.project_path(k)) == h
                            for k, h in parent.get('artifact_hashes', {}).items()
                            if code_archive(k, h) is not None)
            if render_inputs(plan) == render_inputs(previous_plan) and same_code:
                raise ValueError('Unchanged render inputs; reuse parent ' + parent_id + ' and review it')
        elif entry['candidates']:
            raise ValueError('Further exports require an exact parent and issue')
        from runtime_checkpoint import capture
        checkpoint = capture(case)
        job = {'runtime_checkpoint': checkpoint, 'id': candidate_id, 'file_id': file_id, 'status': 'running', 'owner_pid': os.getpid(),
               'parent_id': parent_id, 'issue_id': issue_id, 'started_at': now(), 'plan': plan_relative,
               'new_model_calls': 0, 'new_cost_cny': 0}
        entry['jobs'].append(job)
    output = project.ROOT / 'runs' / case['batch_id'] / file_id / (candidate_id + '.mp4')
    try:
        if diagnostic:
            from execution_plan import render
            report = render(case, plan, output, diagnostic=True)
        else:
            from regional_composite import render
            report = render(case, plan, output)
        Path(str(output)+'.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        if report['status'] == 'failed_integrity_check':
            raise ValueError('Media integrity failed; failed output preserved without activation')
        candidate = register_existing(case, file_id, candidate_id, str(output.relative_to(project.ROOT)),
                                      plan_relative, parent_id, 'pending_agent_quality_review',
                                      'Host Agent plan; full source-based export. Export is not visual acceptance.')
        with locked_state(case) as state:
            entry = state['files'][file_id]
            entry['candidates'].append(candidate)
            entry['current_candidate_id'] = candidate_id
            entry.pop('missing_reason', None)
            live = next(j for j in entry['jobs'] if j['id'] == candidate_id)
            live.update(status='completed', completed_at=now(), output=candidate['output'])
            if issue_id:
                issue = next(i for i in entry['issues'] if i['id'] == issue_id)
                issue.update(status='pending_recheck', result_candidate_id=candidate_id)
        return candidate
    except Exception as exc:
        with locked_state(case) as state:
            job = next(j for j in state['files'][file_id]['jobs'] if j['id'] == candidate_id)
            job.update(status='failed', error=str(exc), completed_at=now())
        raise


def initialize(case, case_relative):
    project.validate_case(case)
    path = state_path(case)
    if path.exists():
        verify_state_case(case, project.read_json(path))
        return {'status':'existing_state_reused','state':str(path.relative_to(project.ROOT))}
    if case_relative == 'cases/ecommerce-batch-002/case.json' and case['case_version'] == '2026-11-12-new-inputs-v1':
        return import_legacy_batch(case, case_relative)
    state = {'schema_version': 1, 'batch_id': case['batch_id'], 'case_version': case['case_version'],
             'case_path': case_relative, 'case_binding': case_binding(case),
             'created_at': now(), 'revision': 1, 'accepted': False, 'files': {}}
    for item in case['files']:
        inventory = project.source_inventory(project.project_path(item['source']))
        state['files'][item['id']] = {
            'source_frame_count': inventory['frame_count'], 'source_inventory': inventory,
            'current_candidate_id': None, 'candidates': [], 'raw_returns': [], 'issues': [], 'jobs': [],
            'missing_reason': '待Agent全片找点、判断可用方法并执行；尚无修改候选。建档成功不代表视频编辑已完成。'}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2)
    return {'status': 'initialized_new_case_pending_localization',
            'state': str(path.relative_to(project.ROOT)), 'accepted': False}


def import_legacy_batch(case, case_relative):
    """One-time migration of verified experiments, separate from new-case intake."""
    path = state_path(case)
    audit = project.read_json(project.ROOT/'evidence/prime-pilot-20261112/source-audit.json')
    state = {'schema_version': 1, 'batch_id':case['batch_id'], 'case_version':case['case_version'],
             'case_path':case_relative,'created_at':now(),'revision':1,'accepted':False,'files':{}}
    notes = {'fashion-aboutyou':'门结构在原始回片中损坏；没有可用完整候选。不能用调色或拉墙修饰内部失败。需有效重建素材或专业后期。',
             'makeup-08':'监督全片v8可审查；衣料纹理软化、局部色差、手边浅点仍待画质验收。未人工听审；本片无字幕。',
             'makeup-03':'历史候选仅源帧1317–1346共30帧写回，其他目标区未完成；原始回片字幕提前结束。新标精确资产及全目标覆盖仍待查。'}
    for item in case['files']:
        fid=item['id']; source_audit=next(a for a in audit['files'] if a['id']==fid)
        pilot=next(p for p in case['pilot_segments'] if p['file_id']==fid)
        raw={'id':pilot['id'],'path':pilot['raw_output'],'sha256':pilot['raw_output_sha256'],
             'source_seconds':[pilot['source_start_seconds'],pilot['source_end_seconds']],
             'source_mapping':'index aligned at source start; no retiming; model may redraw actions',
             'quality_status':pilot['quality_status']}
        if project.digest(project.project_path(raw['path'])) != raw['sha256']:
            raise ValueError('Raw hash changed')
        entry={'source_frame_count':source_audit['frame_count'],'current_candidate_id':None,'candidates':[],
               'raw_returns':[raw],'issues':[],'jobs':[], 'missing_reason':notes[fid],
               'source_audit':'evidence/prime-pilot-20261112/source-audit.json'}
        if fid=='makeup-08':
            for version,plan,parent,status in [('full-v7','v5','full-v5','failed_parent'),('full-v8','v6','full-v7','review_copy')]:
                entry['candidates'].append(register_existing(case,fid,version,
                    f'evidence/full-erase-20261112/candidates/makeup-08-{version}.mp4',
                    f'evidence/full-erase-20261112/track-plan-{plan}.json',parent,status,
                    '失败父版：源帧712粉盒误涂黑、1135银边侵入；保留用于同点返修回归。' if version=='full-v7' else notes[fid]))
            entry['current_candidate_id']='full-v8'
            qa_path='evidence/full-erase-20261112/final-qa-v8.json';qa=project.read_json(project.project_path(qa_path))
            v8=entry['candidates'][-1]
            if qa['output_sha256']==v8['output_sha256'] and qa['source_sha256']==item['source_sha256']:
                v8['reviews'][1].update(status='reviewed_with_limits',evidence=qa_path,
                    scope='historical 1234 target crops + 43 full pairs; no user acceptance or full continuous listening')
            entry['raw_returns'].append({'id':'standard-donor','path':'evidence/workflow-diagnosis-20261112/raw/P-ERASE-wan30-raw.mp4',
                'sha256':'d62d8a4aec2ac2568c9b1b2fe5a0d890f408bf69776e55e9b5f31e141cded8df',
                'source_seconds':[13,18], 'source_mapping':'source interval only; final v8 uses frame0 as supervised fabric patch',
                'quality_status':'fabric donor only; not proof of action preservation'})
        if fid=='makeup-03':
            entry['candidates']=[register_existing(case,fid,'regional-v2',
                 'evidence/workflow-diagnosis-20261112/candidates/makeup-03-prime-regional-v2.mp4',
                 'evidence/workflow-diagnosis-20261112/replace-prime-plan.json','source','partial_review_copy',notes[fid])]
            entry['current_candidate_id']='regional-v2'
        entry['issues'].append({'id':'initial-'+fid,'candidate_id':entry['current_candidate_id'],
            'candidate_sha256':entry['candidates'][-1]['output_sha256'] if entry['candidates'] else None,
            'source_sha256':item['source_sha256'],'operation_id':item['operations'][0]['id'],'file_id':fid,
            'source_time_seconds':pilot['source_start_seconds'],'text':notes[fid], 'category':'model_internal' if fid=='fashion-aboutyou' else 'visual_review',
            'status':'open','actor':'legacy_evidence_import','accepted':False})
        state['files'][fid]=entry
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents another init overwriting feedback.
    with path.open('x') as stream: json.dump(state,stream,ensure_ascii=False,indent=2)
    return {'status':'initialized_from_verified_existing_artifacts','state':str(path.relative_to(project.ROOT)), 'accepted':False}


def snapshot(case):
    state=project.read_json(state_path(case));out=copy.deepcopy(state)
    verify_state_case(case, state)
    out['user_input']=project.project_path(case['user_input']).read_text()
    out['paid_submission_allowed']=False
    out['original_goal_complete']=False
    out['qa_owner']='agent'
    out['provenance']=case.get('provenance','未提供素材来源；不推断客户委托或发布授权')
    out['legacy_regression']=case.get('legacy_regression')
    for item in case['files']:
        entry=out['files'][item['id']]
        entry['file_state_token'] = hash_json(state['files'][item['id']])
        entry['input']=item
        if item.get('source_timeline'):
            from source_timeline import load_for_item
            timeline = load_for_item(item)
            entry['presentation_timeline'] = {k:timeline[k] for k in ('pts','durations','time_base')}
        entry['source_valid']=project.project_path(item['source']).is_file() and project.digest(project.project_path(item['source']))==item['source_sha256']
        entry['requirements_binding']=requirement_binding(case,item)
        for raw in entry['raw_returns']:
            raw['valid']=project.project_path(raw['path']).is_file() and project.digest(project.project_path(raw['path']))==raw['sha256']
            if raw.get('unit_id'):
                unit=next(u for u in entry['execution_units'] if u['unit_id']==raw['unit_id'])
                request=project.read_json(project.project_path(unit['request_path']))
                raw.update(id=raw['unit_id'],source_seconds=request['source_time_seconds'],
                    source_mapping='单元源区间一帧对一帧；'+raw['mode'],
                    quality_status=unit.get('raw_review',{}).get('status','pending_raw_review'))
        for candidate in entry['candidates']:
            try: verify_candidate(case,item['id'],candidate);candidate['binding_status']='valid'
            except (ValueError,OSError,KeyError) as exc:
                candidate['binding_status']='invalid';candidate['binding_error']=str(exc);candidate['status']='blocked'
            candidate['reviews']=effective_reviews(candidate)
            if candidate['binding_status']!='valid':
                candidate['reviews']=[dict(r,status='stale',reason=candidate['binding_error']) for r in candidate['reviews']]
            elif candidate['status'] == 'qualified' and (not quality_reviews_pass(candidate)
                    or any(i['status'] != 'closed' for i in entry['issues'])):
                candidate['status'] = 'review_copy_with_limits'
            if candidate['status'] == 'qualified' and candidate.get('execution_manifest'):
                try:
                    require_candidate_coverage(case, candidate)
                except (ValueError,OSError,KeyError) as exc:
                    candidate['status'] = 'review_copy_with_limits'
                    candidate['coverage_error'] = str(exc)
        current=next((c for c in entry['candidates'] if c['id']==entry['current_candidate_id']),None)
        entry['delivery_status']='unresolved' if not current else ('blocked' if current['binding_status']!='valid' else current['status'])
    return out


def plan_feedback(case, body):
    state=project.read_json(state_path(case));entry=state['files'][body['file_id']]
    issue=next((i for i in entry['issues'] if i['id']==body['issue_id']),None)
    if not issue: raise ValueError('Unknown issue')
    candidate=verify_candidate(case,body['file_id'],candidate_for(entry,issue['candidate_id']))
    t=issue['source_time_seconds'];fps=item_for(case,body['file_id'])['fps_container_average']
    interval=body.get('source_seconds',[max(0,t-.5),min(item_for(case,body['file_id'])['duration_seconds'],t+.5)])
    result=dependency_plan(case,candidate,interval,issue['category'])
    if issue['category']=='packaging':
        result['pipeline_request'] = dict(action='remux-source-audio',file_id=body['file_id'],issue_id=issue['id'])
    if issue['category']=='asset':
        results = candidate.get('operation_results', [])
        selected = next((r for r in results if r['operation_id'] == issue.get('operation_id')), None)
        result['asset_writeback_source_frame_ranges'] = ranges(selected['modified_source_frames']) if selected else candidate['modified_frame_ranges']
        result['asset_scope']='Replacing an asset recomposes all existing carrier frames, not only the feedback context interval; it does not extend target coverage.'
    return result


def begin_repair(case, body):
    with locked_state(case) as state:
        entry=state['files'][body['file_id']]
        require_file_revision(state, body, entry, 'Revision conflict; reload before repair')
        issue=next((i for i in entry['issues'] if i['id']==body['issue_id']),None)
        if not issue: raise ValueError('Repair requires a persisted issue')
        candidate=verify_candidate(case,body['file_id'],candidate_for(entry,issue['candidate_id']))
        if body.get('candidate_sha256')!=candidate['output_sha256']: raise ValueError('Parent candidate hash mismatch')
        if any(j['status'] in ('queued','running') for j in entry['jobs']): raise ValueError('Repair conflict: another job owns this file')
        if issue['status']!='open': raise ValueError('Issue already has an execution; inspect its job instead of resubmitting')
        action=body.get('action')
        if action not in ('reviewed_protection_patch','protect_region','disable_edit','shift_registration','set_color_match','replace_exact_asset'):
            raise ValueError('Unsupported local action; no remote fallback')
        if issue['category']=='model_internal': raise ValueError('Target internal failure needs valid cache or professional handling; no model submission')
        if issue['category']=='packaging':raise ValueError('Pixel actions cannot repair audio; use project.py pipeline action remux-source-audio with the saved issue_id')
        if not candidate.get('plan'):raise ValueError('Parent has no bound composition plan')
        parent_plan=project.read_json(project.project_path(candidate['plan']))
        if parent_plan.get('schema') == 'execution_manifest_v1':
            raise ValueError('Unified candidate: use project.py pipeline action prepare-repair with the saved issue_id and operation_plan; feedback is preserved')
        if parent_plan.get('composition_mode')=='exact_asset_on_source_label' and action in ('shift_registration','set_color_match'):
            raise ValueError('Exact asset uses the source carrier; cloud registration/color actions do not apply')
        if action=='reviewed_protection_patch' and (body['file_id']!='makeup-08' or candidate['id']!='full-v7'):
            raise ValueError('Reviewed corrections are bound only to actual makeup-08 v7')
        if action=='protect_region' and not issue.get('region'):raise ValueError('Select the source-image protection region first')
        if action=='replace_exact_asset' and (body['file_id']!='makeup-03' or issue['category']!='asset'):
            raise ValueError('Exact-asset action is limited to the existing makeup-03 carrier track and an asset issue')
        job={'id':'repair-'+uuid.uuid4().hex[:12],'file_id':body['file_id'],'issue_id':issue['id'],
             'owner_pid':os.getpid(),
             'parent_id':candidate['id'],'parent_sha256':candidate['output_sha256'],
             'expected_current_candidate_id':entry['current_candidate_id'],'action':action,
             'parameters':{k:body[k] for k in ('frames','dx','dy','color_match') if k in body},
             'status':'queued','created_at':now(),'accepted':False,'new_model_calls':0,'new_cost_cny':0}
        entry['jobs'].append(job);issue['status']='repair_queued';issue['job_id']=job['id']
    threading.Thread(target=execute_repair,args=(case,job['id'],body['file_id']),daemon=True).start()
    return job


def reconcile_interrupted(case):
    """Do not silently retry a worker lost on restart, even though execution is local."""
    state=project.read_json(state_path(case))
    lost=[]
    for fid,entry in state['files'].items():
        for job in entry['jobs']:
            if job['status'] not in ('queued','running'):continue
            try:os.kill(job['owner_pid'],0)
            except (KeyError,ProcessLookupError):lost.append((fid,job['id']))
    if lost:
        with locked_state(case) as state:
            for fid,jid in lost:
                entry=state['files'][fid];job=next(j for j in entry['jobs'] if j['id']==jid)
                job.update(status='interrupted',completed_at=now(),error='Worker exited; inspect preserved run artifacts. No automatic retry or activation.')
                if job.get('issue_id'):
                    next(i for i in entry['issues'] if i['id']==job['issue_id'])['status']='repair_interrupted'
    return lost


def execute_repair(case, job_id, file_id):
    start_time=time.monotonic();directory=project.ROOT/'runs'/case['batch_id']/job_id
    try:
        with locked_state(case) as state:
            entry=state['files'][file_id];job=next(j for j in entry['jobs'] if j['id']==job_id)
            candidate=copy.deepcopy(candidate_for(entry,job['parent_id']));verify_candidate(case,file_id,candidate)
            issue=copy.deepcopy(next(i for i in entry['issues'] if i['id']==job['issue_id']))
            job['status']='running';job['started_at']=now();job=copy.deepcopy(job)
        directory.mkdir(parents=True,exist_ok=False)
        import cv2
        import numpy as np
        from regional_composite import render
        plan=project.read_json(project.project_path(candidate['plan']));original_plan=copy.deepcopy(plan)
        masks_path=project.project_path(plan['masks'])
        with np.load(masks_path,allow_pickle=False) as stored: arrays={k:stored[k] for k in stored.files}
        base=plan['source_frames_half_open'][0];fps=item_for(case,file_id)['fps_container_average']
        requested=job['parameters'].get('frames',[issue['source_frame']])
        if job['action']=='replace_exact_asset':
            requested=[base+r['frame'] for r in plan['review']['records']]
        if not isinstance(requested,list) or not requested or len(requested)>120 or any(not isinstance(f,int) or not base<=f<plan['source_frames_half_open'][1] for f in requested):
            raise ValueError('Use 1–120 valid source keyframes; no automatic full-timeline mask extension')
        if issue['source_frame'] not in requested:raise ValueError('Edit must include the feedback source frame')
        changed=[]
        if job['action']=='replace_exact_asset':
            asset=next(op['replacement_asset'] for op in item_for(case,file_id)['operations'] if op.get('replacement_asset'))
            plan['composition_mode']='exact_asset_on_source_label'
            plan['sampling_mode']='exact_asset_source_carrier'
            plan['exact_asset']=asset;plan['exact_asset_sha256']=project.digest(project.project_path(asset))
            plan['color_match']='none'
            plan['review']['local_cleanup_limit']='Bright old glyphs are locally inpainted in original label interior; cloth texture under old glyphs is estimated, not verified intact. No tracking extension.'
        elif job['action']=='reviewed_protection_patch':
            if set(requested)!={712,1135}:raise ValueError('Reviewed v7 repair is exactly bound to source frames712/1135')
            fixed_path='evidence/full-erase-20261112/track-plan-v6.json'
            fixed=project.read_json(project.project_path(fixed_path))
            if project.digest(project.project_path(fixed['masks']))!=fixed['masks_sha256']:raise ValueError('Reviewed correction cache changed')
            with np.load(project.project_path(fixed['masks']),allow_pickle=False) as corrected:
                corrected_alpha=corrected['alpha']
                for frame in requested:
                    arrays['alpha'][frame-base]=corrected_alpha[frame]
                    plan['review']['frame_states'][frame]=fixed['review']['frame_states'][frame]
                del corrected_alpha
            plan['local_correction_reference']={'plan':fixed_path,'plan_sha256':project.digest(project.project_path(fixed_path)),
                'frames':requested,'scope':'reviewed supervised alpha keyframes only; not whole v8 plan/video reuse'}
        else:
            width,height=item_for(case,file_id)['dimensions']
            for frame in requested:
                index=frame-base
                if job['action']=='disable_edit':
                    arrays['alpha'][index]=0
                    if 'frame_states' in plan['review']:plan['review']['frame_states'][frame]='occluded'
                elif job['action']=='protect_region':
                    polygon=np.rint(np.asarray(issue['region'])*[width-1,height-1]).astype(np.int32)
                    region=np.zeros((height,width),np.uint8);cv2.fillPoly(region,[polygon],255)
                    arrays['alpha'][index][region>0]=0
                elif job['action']=='shift_registration':
                    dx,dy=job['parameters'].get('dx',0),job['parameters'].get('dy',0)
                    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>20 for v in (dx,dy)):
                        raise ValueError('Registration shifts limited to 20 source pixels')
                    shift=np.array([[1,0,dx],[0,1,dy],[0,0,1]],np.float32)
                    arrays['raw_to_source'][index]=shift@arrays['raw_to_source'][index]
                elif job['action']=='set_color_match':
                    value=job['parameters'].get('color_match')
                    if value not in ('none','context_median'):raise ValueError('Unsupported bounded color mode')
                    plan.setdefault('color_match_overrides',{})[str(frame)]=value
        # Persist actual changed mask/warp frames, not the requested range alone.
        with np.load(masks_path,allow_pickle=False) as old:
            old_alpha=old['alpha']
            changed=[int(i+base) for i in range(len(old_alpha)) if not np.array_equal(old_alpha[i],arrays['alpha'][i])]
            del old_alpha
            old_warp=old['raw_to_source'];changed=sorted(set(changed)|{int(i+base) for i in range(len(old_warp)) if not np.array_equal(old_warp[i],arrays['raw_to_source'][i])})
        if job['action']=='replace_exact_asset':changed=requested
        elif job['action']=='set_color_match':
            changed=[f for f in requested if plan.get('color_match_overrides',{}).get(str(f),plan.get('color_match','none')) != original_plan.get('color_match_overrides',{}).get(str(f),original_plan.get('color_match','none'))]
        if not changed:raise ValueError('No real mask/registration/color change; no export performed')
        target_mask=directory/'masks.npz';np.savez_compressed(target_mask,**arrays);del arrays
        plan['masks']=str(target_mask.relative_to(project.ROOT));plan['masks_sha256']=project.digest(target_mask)
        plan['parent_version']=candidate['output'];plan['parent_output_sha256']=candidate['output_sha256']
        plan['feedback_issue_id']=issue['id'];plan['supervised_local_action']=job['action']
        plan['changed_source_frames']=changed
        plan_relative=str((directory/'plan.json').relative_to(project.ROOT))
        project.project_path(plan_relative).write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
        output=directory/(file_id+'-'+job_id+'.mp4')
        report=render(case,plan,output)
        Path(str(output)+'.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        if report['status']=='failed_integrity_check':raise ValueError('New candidate failed media integrity; preserve it without activation')
        new=register_existing(case,file_id,job_id,str(output.relative_to(project.ROOT)),plan_relative,candidate['id'],'regression_review_copy',
            '浏览器反馈触发的监督局部返修；隔离候选，未替代活动版本。新编码画质与接缝待复查。'+
            (' 本地清旧字并合成准确MORU，沿用1317–1346帧轨迹；原布牌旧字下纹理为估计，其余目标区未完成。' if job['action']=='replace_exact_asset' else ''))
        new['review_invalidation']={'parent_id':candidate['id'],'changed_source_frames':changed,
            'invalidated':['target coverage','visual/non-target','seams','continuous playback','user acceptance'],
            'reason':'new mask/registration/export; lossy decoded changes may extend to GOPs; no visual pass inherited',
            'recomputed':['complete frame PTS','original AAC packet payload/time','preencode outside mask'],
            'inherited':[]}
        new['reviews'][1]['status']='pending'
        with locked_state(case) as state:
            entry=state['files'][file_id];live=next(j for j in entry['jobs'] if j['id']==job_id)
            verify_candidate(case,file_id,candidate)
            if entry['current_candidate_id']!=job['expected_current_candidate_id']:
                new['activation_conflict']=True
            entry['candidates'].append(new)
            live.update(status='completed',completed_at=now(),candidate_id=new['id'],output=new['output'],
                output_sha256=new['output_sha256'],changed_source_frames=changed,machine_seconds=time.monotonic()-start_time)
            live_issue=next(i for i in entry['issues'] if i['id']==issue['id']);live_issue['status']='pending_recheck';live_issue['result_candidate_id']=new['id']
        (directory/'execution.json').write_text(json.dumps(live,ensure_ascii=False,indent=2)+'\n')
    except Exception as exc:
        with locked_state(case) as state:
            entry=state['files'][file_id];live=next(j for j in entry['jobs'] if j['id']==job_id)
            live.update(status='failed',error=str(exc),completed_at=now(),machine_seconds=time.monotonic()-start_time)
            issue=next(i for i in entry['issues'] if i['id']==live['issue_id']);issue['status']='repair_failed'
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'failure.json').write_text(json.dumps(live,ensure_ascii=False,indent=2)+'\n')


def cost_summary(state):
    units=[u for entry in state['files'].values() for u in entry.get('execution_units',[])]
    # One remote task may have several local references. Missing task IDs are
    # separate unknown requests, never deduplicated by amount or unit name.
    groups={}
    for index,unit in enumerate(units):
        key=('task',unit['provider_task_id']) if unit.get('provider_task_id') else ('unit',index)
        groups.setdefault(key,[]).append(unit)
    uncertain=[];known=Decimal(0)
    for group in groups.values():
        amounts=set();invalid=False
        for unit in group:
            charges=[r['response']['actual_charge_cny'] for r in unit.get('receipts',[])
                     if r.get('response',{}).get('actual_charge_cny') is not None]
            # Only the explicit settlement route reconciles historical CNY usage.
            if not charges:
                for receipt in unit.get('receipts',[]):
                    response=receipt.get('response',{})
                    if (receipt.get('action')=='settle' and response.get('status') in ('SUCCESS','FAILED')
                            and response.get('task_id')==unit.get('provider_task_id')):
                        usage=response.get('usage') or {}
                        amount=usage.get('thirdPartyConsumeMoney')
                        if amount is None:amount=usage.get('consumeMoney')
                        if amount is not None:charges.append(amount)
            if charges:
                try:
                    amount=Decimal(str(charges[-1]))
                    if not amount.is_finite() or amount<0:raise InvalidOperation
                    amounts.add(amount)
                except (InvalidOperation,ValueError,TypeError):invalid=True
        if len(amounts)==1 and not invalid:
            known+=next(iter(amounts))
        elif amounts or invalid or any(u.get('reserved_cny') for u in group):
            uncertain.extend(u['unit_id'] for u in group)
    # Workflow-app outputs repeat task-level usage on nodes and later queries.
    # Keep these provider amounts separate from a reconciled CNY charge.
    reported = Decimal(0); compute = Decimal(0); missing = []; ambiguous = []
    for group in groups.values():
        outputs=[]
        for unit in group:
            outputs.extend(next((r['response']['output_receipt']['outputs']
                for r in reversed(unit.get('receipts', []))
                if r.get('response', {}).get('output_receipt', {}).get('outputs')), []))
        unit_ids=[u['unit_id'] for u in group];invalid=False
        values = {}
        for field in ('consumeMoney', 'taskCostTime'):
            try:
                parsed = {Decimal(str(o[field])) for o in outputs or [] if o.get(field) is not None}
                if any(not n.is_finite() or n < 0 for n in parsed):
                    raise InvalidOperation
                values[field] = parsed
            except (InvalidOperation, ValueError, TypeError):
                values[field] = set()
                invalid=True
        if invalid or any(len(v) > 1 for v in values.values()):
            ambiguous.extend(unit_ids)
        if len(values['consumeMoney']) == 1 and not any(u in ambiguous for u in unit_ids):
            reported += next(iter(values['consumeMoney']))
        else:
            missing.extend(unit_ids)
        if len(values['taskCostTime']) == 1 and not any(u in ambiguous for u in unit_ids):
            compute += next(iter(values['taskCostTime']))
    return dict(new_model_calls=sum(max(u.get('new_model_calls',0) for u in g) for g in groups.values()),
        new_cost_cny=None if uncertain else float(known),known_actual_receipts_cny=float(known),
        reserved_total_cny=sum(max(u.get('reserved_cny',0) for u in g) for g in groups.values()),
        unsettled_provider_units=uncertain,account_bill_verified=False,
        provider_reported_consumeMoney_sum=str(reported),
        provider_reported_compute_seconds=str(compute),
        provider_reported_missing_units=missing, provider_reported_ambiguous_units=ambiguous,
        provider_reported_currency='unverified; do not infer CNY from the field name')


def delivery_report(case):
    state=snapshot(case)
    return {'batch_id':case['batch_id'],'case_version':case['case_version'],'revision':state['revision'],
            'original_goal_complete':False,'accepted':False,**cost_summary(state),
            'files':[{'id':fid,'delivery_status':e['delivery_status'],'current_candidate_id':e['current_candidate_id'],
                'candidate':next((c for c in e['candidates'] if c['id']==e['current_candidate_id']),None),
                'unresolved_issues':[i for i in e['issues'] if i['status']!='closed'],
                'missing_reason':e.get('missing_reason')} for fid,e in state['files'].items()]}


def record_agent_review(case, evidence_relative):
    """Persist the host Agent's actual observations, never infer a quality pass."""
    path = project.project_path(evidence_relative)
    review = project.read_json(path)
    with locked_state(case) as state:
        verify_state_case(case, state)
        entry = state['files'][review['file_id']]
        require_file_revision(state, review, entry, 'Review revision conflict')
        candidate = verify_candidate(case, review['file_id'], candidate_for(entry, review['candidate_id']))
        if (review['source_sha256'] != candidate['source_sha256']
                or review['output_sha256'] != candidate['output_sha256']
                or review['dependency'] != candidate['dependency']):
            raise ValueError('Review source/output/requirements binding mismatch')
        allowed = {'target_coverage', 'visual', 'playback', 'delivery'}
        checks = review['checks']
        if {c['kind'] for c in checks} != allowed or len(checks) != len(allowed):
            raise ValueError('Review needs coverage, visual, playback and delivery observations')
        if any(c['status'] not in ('pass', 'reviewed_with_limits', 'fail') or not c.get('scope') for c in checks):
            raise ValueError('Invalid Agent review observation')
        operations = review.get('operation_reviews', [])
        operation_artifacts = {}
        if candidate.get('execution_manifest') or len(candidate.get('operation_ids', [])) > 1:
            expected = candidate.get('operation_ids', [])
            requested = {op['id'] for op in item_for(case, review['file_id'])['operations']}
            kinds = {'target_coverage', 'visual', 'protection', 'temporal'}
            if (not expected or set(expected) != requested or len(expected) != len(requested)
                    or len(operations) != len(expected)
                    or {r.get('operation_id') for r in operations} != set(expected)):
                raise ValueError('Review must cover every requested operation exactly once')
            for operation in operations:
                if (operation.get('output_sha256') != candidate['output_sha256']
                        or operation.get('dependency') != candidate['dependency']):
                    raise ValueError('Operation review binding mismatch')
                observations = operation.get('checks', [])
                if (len(observations) != len(kinds)
                        or {c.get('kind') for c in observations} != kinds
                        or any(c.get('status') not in ('pass', 'reviewed_with_limits', 'fail')
                               or not c.get('scope') or not c.get('evidence') for c in observations)):
                    raise ValueError('Invalid operation observations or missing evidence')
                for observation in observations:
                    for evidence in observation['evidence']:
                        if (not isinstance(evidence, dict) or not evidence.get('path')
                                or project.digest(project.project_path(evidence['path'])) != evidence.get('sha256')):
                            raise ValueError('Operation review evidence hash mismatch')
                        operation_artifacts[evidence['path']] = evidence['sha256']
        resolved = []
        ancestry = set()
        ancestor = candidate
        while ancestor and ancestor['id'] not in ancestry:
            ancestry.add(ancestor['id'])
            ancestor = next((c for c in entry['candidates']
                             if c['id'] == ancestor.get('parent_id')), None)
        for issue_id in review.get('resolved_issue_ids', []):
            issue = next((i for i in entry['issues'] if i['id'] == issue_id), None)
            raw_result = next((r for r in candidate.get('operation_results', [])
                if issue and r.get('operation_id') == issue.get('operation_id')
                and r.get('write_regions_reviewed') and not r.get('diagnostic')
                and r.get('provider_raw_hashes', {}).get(issue.get('unit_id')) == issue.get('raw_sha256')), None)
            same_raw = issue and issue.get('observation_basis') == 'provider_raw' and raw_result is not None
            local_parent = False
            if issue and entry.get('local_material_repair'):
                from local_material_repair import repairs_issue
                local_parent = repairs_issue(case, review['file_id'], entry, candidate, issue)
            if not issue or (issue.get('result_candidate_id') not in ancestry and not same_raw and not local_parent):
                raise ValueError('Review cannot resolve an unrelated repair issue')
            resolved.append(issue)
        if any(c['status'] == 'fail' for c in checks) and resolved:
            raise ValueError('Failed review cannot close repair issues')
        for issue in resolved:
            if candidate.get('execution_manifest') or len(candidate.get('operation_ids', [])) > 1:
                operation = next((op for op in operations if op['operation_id'] == issue.get('operation_id')), None)
                if operation is None or any(c['status'] != 'pass' for c in operation['checks']):
                    raise ValueError('Unverified operation cannot close its repair issue')
        candidate['reviews'] = [r for r in candidate['reviews'] if r['kind'] not in allowed]
        for check in checks:
            candidate['reviews'].append(dict(check, evidence=evidence_relative,
                evidence_sha256=project.digest(path), output_sha256=candidate['output_sha256'],
                dependency=candidate['dependency'], actor='host_agent', reviewed_at=now()))
        if candidate.get('execution_manifest') or len(candidate.get('operation_ids', [])) > 1:
            candidate['operation_reviews'] = copy.deepcopy(operations)
        candidate.setdefault('artifact_hashes', {}).update(operation_artifacts)
        candidate.setdefault('artifact_hashes', {})[evidence_relative] = project.digest(path)
        for issue in resolved:
            issue.update(status='closed', resolution_evidence=evidence_relative,
                         resolution_candidate_id=candidate['id'])
        if quality_reviews_pass(candidate) and candidate.get('execution_manifest'):
            require_candidate_coverage(case, candidate)
        candidate['status'] = ('qualified' if quality_reviews_pass(candidate)
                               and not any(i['status'] != 'closed' for i in entry['issues'])
                               else 'review_copy_with_limits')
        candidate['accepted'] = False
        return {'candidate_id': candidate['id'], 'status': candidate['status'],
                'evidence': evidence_relative, 'accepted': False}
