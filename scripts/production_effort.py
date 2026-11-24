"""Preserve phase results and gate real method probes, without a delivery timeout."""
from datetime import datetime, timezone
import uuid

import project
import workflow


class RepairLimitReached(ValueError):
    pass


class CandidateReviewRequired(ValueError):
    pass


def provider_first(case):
    return case.get('execution_policy', {}).get('editing_strategy') == 'provider_first_v1'


def require_lightweight_edit(case):
    """Creation guard only; historical artifacts remain readable/reviewable."""
    if provider_first(case):
        raise ValueError('Use existing editor tools for lightweight corrections; '
                         'source colour separation, hand masks and local recolor development are disabled')


def new_policy(mode="production"):
    if mode not in ("production", "method_trial"):
        raise ValueError("Unknown workflow mode")
    return dict(version=3, mode=mode, editing_strategy="provider_first_v1", stage_handoff_required=True, source_semantic_review_required=True,
                source_semantic_execution='host_agent', source_localization_strategy='keyframe_tracking',
                started_at=datetime.now(timezone.utc).isoformat(),
                max_failed_previews_per_method=2,
                deformable_recolor='native_video_edit')


def _control(case, state):
    workflow.verify_state_case(case, state)
    return state.get('case_binding', {}).get('execution_policy')


def candidate_failures(entry, operation_id=None):
    """Count bound observations, even if no separate feedback issue was saved."""
    failed = set()
    for candidate in entry.get('candidates', []):
        operations = [op for op in candidate.get('operation_reviews', [])
                      if (operation_id is None or op.get('operation_id') == operation_id)
                      and op.get('output_sha256') == candidate.get('output_sha256')
                      and op.get('dependency') == candidate.get('dependency')]
        observations = [c for op in operations for c in op.get('checks', [])]
        # Older single-operation candidates only had file-level observations.
        if not candidate.get('operation_reviews') and (operation_id is None or
                operation_id in candidate.get('operation_ids', [])):
            observations += workflow.effective_reviews(candidate)
        if any(c.get('kind') in ('target_coverage', 'visual', 'protection', 'temporal')
               and c.get('status') == 'fail' for c in observations):
            failed.add(candidate['id'])
    return failed


def require_reviewed_repair(case, file_id, operation_ids):
    """New renders consume the current review; exact cached reuse happens first.

    This also applies to diagnostic/legacy exports. A diagnostic label grants
    access to failure evidence, not another unreviewed production attempt.
    """
    path = workflow.state_path(case)
    if not path.is_file():
        return
    state = project.read_json(path)
    workflow.verify_state_case(case, state)
    entry = state['files'][file_id]
    threshold = 1 if case.get('execution_policy', {}).get('mode') == 'method_trial' else 2
    if any(len(candidate_failures(entry, op)) >= threshold for op in operation_ids):
        raise RepairLimitReached('Candidate repairs failed repeatedly; preserve results and diagnose the method before another export')
    current = entry.get('current_candidate_id')
    if not current:
        return
    candidate = workflow.candidate_for(entry, current)
    reviews = workflow.effective_reviews(candidate)
    required = {'target_coverage', 'visual', 'playback', 'delivery'}
    observed = {r['kind'] for r in reviews if r.get('status') in ('pass','fail','reviewed_with_limits')}
    if not required <= observed:
        raise CandidateReviewRequired('Review current candidate before another export; persist the same-version coverage/visual/playback/delivery observations')
    if candidate.get('execution_manifest') or len(candidate.get('operation_ids', [])) > 1:
        kinds = {'target_coverage', 'visual', 'protection', 'temporal'}
        for op in operation_ids:
            review = next((r for r in candidate.get('operation_reviews', []) if r.get('operation_id') == op), {})
            if (review.get('output_sha256') != candidate['output_sha256'] or
                    review.get('dependency') != candidate['dependency'] or
                    not kinds <= {c.get('kind') for c in review.get('checks', [])
                                  if c.get('status') in ('pass','fail','reviewed_with_limits')}):
                raise CandidateReviewRequired('Review current candidate before another export; operation observations are missing or stale')


def source_preserving_recolor(operation):
    """Explicit pure-color intent, frozen with requirements; never infer from a filename."""
    if operation.get('recolor_mode') != 'preserve_source_details':
        return False
    color = operation.get('target_color_rgb')
    if (operation.get('kind') != 'modify' or operation.get('surface_motion') != 'deformable'
            or not isinstance(color, list) or len(color) != 3
            or any(type(c) is not int or not 0 <= c <= 255 for c in color)
            or not isinstance(operation.get('semantic_label'), str)
            or not 1 <= len(operation['semantic_label']) <= 200):
        raise ValueError('Source-preserving recolor requires pure color, RGB and semantic object label')
    return True


def preservation_fallback(case, state, file_id):
    """A failed native reconstruction may switch to source-preserving tracking."""
    import runninghub_adapter as provider
    switch = state.get('preservation_fallbacks', {}).get(file_id)
    if not switch or not switch.get('user_instruction'):
        return False
    unit = provider.unit_for(state, file_id, switch['failed_unit_id'])
    request = provider.verify_unit(case, unit)
    review = unit.get('raw_review', {})
    if (request['operation_id'] != switch['operation_id'] or
            request['model_id'] != 'kling-o3-std-edit' or
            review.get('sha256') != switch['review_sha256'] or
            project.digest(project.project_path(review['path'])) != review['sha256']):
        raise ValueError('Preservation fallback failure evidence changed')
    observed = project.read_json(project.project_path(review['path']))
    if not any(c['kind'] == 'texture_geometry' and c['status'] == 'fail'
               for c in observed['checks']):
        raise ValueError('Preservation fallback requires observed native detail failure')
    return True


def check(case, file_id=None, start_method=False, method_family=None, operation_id=None):
    if start_method and method_family == 'local_recolor':
        require_lightweight_edit(case)
    from component_probe import active
    if active() and active()['batch_id'] == case.get('batch_id') and active()['file_id'] == file_id:
        return  # One frozen, diagnosed multi-anchor method probe; never a production preview.
    if not case.get('batch_id') and not case.get('execution_policy'):
        return  # Geometry-only legacy/unit inputs have no production workflow.
    path = workflow.state_path(case)
    if not path.exists():
        if case.get('execution_policy'):
            raise ValueError('Initialize workflow before production work')
        return
    state = project.read_json(path)
    policy = _control(case, state)
    if not policy:
        return  # Frozen legacy runs retain their original contract.
    if policy.get('version', 0) >= 3:
        if start_method and file_id:
            from production_pipeline import check_method_failure
            check_method_failure(case, file_id, operation_id)
        return
    if policy.get('version') == 2:
        if (file_id and method_family == 'local_recolor' and policy.get('deformable_recolor') == 'native_video_edit'
                and not preservation_fallback(case, state, file_id)):
            item = workflow.item_for(case, file_id)
            if any(op.get('kind') == 'modify' and op.get('surface_motion') == 'deformable'
                   and not source_preserving_recolor(op)
                   for op in item['operations']):
                raise ValueError('Deformable modification starts with native video editing; segmentation/matting parameter trials are disabled for this route')
        # Changing trimap/feather parameters must not reopen the same failed
        # semantic strategy at prepare/submit before preview catches it again.
        if file_id and method_family:
            failures = 0
            for trial in reversed(state['files'][file_id].get('preview_trials', [])):
                families = trial.get('method_families')
                if families is None and trial.get('report'):
                    path = project.project_path(trial['report'])
                    if project.digest(path) != trial['report_sha256']:
                        raise ValueError('Preview report changed')
                    report = project.read_json(path)
                    families = ['local_recolor' for op in report['operations']
                                if op.get('method') == 'source_recolor']
                if method_family not in (families or []):
                    continue
                if trial['verdict'] == 'pass':
                    break
                failures += trial['verdict'] in ('fail', 'render_failed')
            if failures >= policy['max_failed_previews_per_method']:
                raise ValueError('Semantic strategy failed twice; native editing or another actual strategy required; reuse prior phases')
        return  # Elapsed time never makes a video qualified.
    at = datetime.now(timezone.utc)
    if at >= datetime.fromisoformat(policy['deadline']):
        raise ValueError('Production deadline reached; preserve results and report the gap, do not extend or rename the batch')
    if file_id is None:
        return
    entry = state['files'][file_id]
    trials = entry.get('preview_trials', [])
    if sum(t.get('verdict') in ('fail', 'render_failed') for t in trials) >= policy['max_failed_previews']:
        raise ValueError('Method failed initial probe and diagnosed repair; capability review required, no further parameter loop')
    first = entry.get('method_started_at')
    if first and not any(t.get('verdict') == 'pass' for t in trials):
        if (at-datetime.fromisoformat(first)).total_seconds() >= policy['feasibility_seconds']:
            raise ValueError('Ten-minute method feasibility window expired without a passing preview; report capability gap')
    if start_method and not first:
        with workflow.locked_state(case) as state:
            _control(case, state)
            state['files'][file_id].setdefault('method_started_at', at.isoformat())


def fingerprint(manifest):
    # A Skill/doc/provider-config edit cannot invalidate already rendered pixels.
    # Freeze only shared execution code and the compositor used by this manifest.
    paths = {'project.py', 'workflow.py', 'execution_plan.py', 'source_timeline.py',
             'regional_composite.py', 'production_effort.py'}
    modes = {'source_recolor': 'source_recolor.py', 'aigc_regional': 'aigc_composite.py',
             'cached_clean_plate': 'cached_clean_plate.py', 'exact_asset_on_source_label': 'surface_text.py'}
    for op in manifest['operations']:
        if op.get('plan'):
            plan = project.read_json(project.project_path(op['plan']))
            mode = plan.get('composition_mode')
            if mode in modes:
                paths.add(modes[mode])
    hashes = {p: project.digest(project.ROOT/'scripts'/p) for p in sorted(paths)
              if (project.ROOT/'scripts'/p).is_file()}
    return workflow.hash_json(dict(manifest=manifest, implementation=hashes))


def method_key(manifest):
    methods = []
    for op in manifest['operations']:
        if op.get('plan'):
            plan = project.read_json(project.project_path(op['plan']))
            methods.append((op['operation_id'], plan.get('composition_mode'),
                            plan.get('segmentation_backend') if plan.get('composition_mode') == 'source_recolor' else None,
                            plan.get('alpha_refinement', {}).get('method'),
                            bool(plan.get('alpha_envelope')),
                            tuple(sorted(r['segmentation_backend'] for r in
                                plan.get('alpha_envelope',{}).get('foreground_occluders',[])))))
    return workflow.hash_json(methods)


def recover_preview(case, manifest, output, frames):
    """Read the exact saved result, independently of permission to render again.

    The material fingerprint binds the original manifest and artifact hashes.
    Current implementation changes never turn this receipt into a new passing
    method proof: require_preview still checks its original implementation lock.
    """
    path = workflow.state_path(case)
    if not path.is_file():
        return None
    state = project.read_json(path)
    workflow.verify_state_case(case, state)
    fid = manifest['file_id']
    relative = str(project.project_path(output).relative_to(project.ROOT))
    trial = next((t for t in reversed(state['files'][fid].get('preview_trials', []))
                  if t.get('preview') == relative and t.get('requested_source_frames') == frames), None)
    if trial is None:
        return None
    if not trial.get('report') or not trial.get('report_sha256'):
        raise ValueError('Reconcile incomplete original preview; do not start another trial')
    report_path = project.project_path(trial['report'])
    if project.digest(report_path) != trial['report_sha256']:
        raise ValueError('Original preview report changed')
    report = project.read_json(report_path)
    item = workflow.item_for(case, fid)
    artifacts = report['artifact_hashes']
    if (manifest.get('dependency') != workflow.requirement_binding(case, item)
            or trial.get('handoff_fingerprint') != workflow.hash_json(dict(manifest=manifest, artifacts=artifacts))
            or report.get('file_id') != fid or report.get('source_sha256') != item['source_sha256']
            or report.get('source_frames_half_open') != frames or report.get('output') != relative):
        raise ValueError('Original preview source/range/material binding mismatch')
    if project.digest(project.project_path(item['source'])) != item['source_sha256']:
        raise ValueError('Original preview source changed')
    for name, expected in artifacts.items():
        if project.digest(project.project_path(name)) != expected:
            raise ValueError('Original preview dependency changed: '+name)
    for evidence in trial.get('evidence', []):
        if project.digest(project.project_path(evidence['path'])) != evidence['sha256']:
            raise ValueError('Original preview observation evidence changed')
    if project.digest(project.project_path(relative)) != report['output_sha256']:
        raise ValueError('Original preview media changed')
    return dict(report, trial_id=trial['id'], preview_verdict=trial['verdict'],
                preview_reused=True, report=trial['report'], report_sha256=trial['report_sha256'],
                next_action='review_existing_preview' if trial['verdict'] == 'pending' else 'consume_saved_preview_verdict',
                new_model_calls=0, new_renders=0, accepted=False)


def begin_preview(case, manifest, output, frames=None):
    require_reviewed_repair(case, manifest['file_id'], [e['operation_id'] for e in manifest['operations']])
    from execution_plan import validate
    _, _, _, artifacts = validate(case, manifest)  # No backend-renaming escape.
    from production_pipeline import pin_materials
    fid = manifest['file_id']
    families = ['local_recolor' for op in manifest['operations'] if op.get('plan')
                and project.read_json(project.project_path(op['plan'])).get('composition_mode') == 'source_recolor']
    if families:
        require_lightweight_edit(case)
    for family in families:
        check(case, fid, start_method=True, method_family=family)
    check(case, fid, start_method=True)
    if not case.get('execution_policy'):
        pin_materials(case, manifest, artifacts)
        return None
    trial = dict(id=uuid.uuid4().hex, fingerprint=fingerprint(manifest),
                 handoff_fingerprint=workflow.hash_json(dict(manifest=manifest, artifacts=artifacts)),
                 method_key=method_key(manifest), method_families=sorted(set(families)),
                 requested_source_frames=frames,
                 preview=str(project.project_path(output).relative_to(project.ROOT)), verdict='pending')
    with workflow.locked_state(case) as state:
        _control(case, state)
        check(case, fid)
        trials = state['files'][fid].setdefault('preview_trials', [])
        if any(t['verdict'] == 'pending' for t in trials):
            raise ValueError('Review the previous preview before preparing another version')
        if any(t['fingerprint'] == trial['fingerprint'] and t.get('requested_source_frames') == frames for t in trials):
            raise ValueError('Identical preview already exists; reuse its result')
        policy = state['case_binding']['execution_policy']
        consecutive_failures = 0
        for previous in reversed(trials):
            if previous.get('method_key') != trial['method_key']:
                continue
            if previous['verdict'] == 'pass':
                break
            if previous['verdict'] in ('fail', 'render_failed'):
                consecutive_failures += 1
        if policy.get('version') == 2 and consecutive_failures >= policy['max_failed_previews_per_method']:
            raise ValueError('Same method failed twice; change the actual backend, preserve prior phases, do not continue parameter edits')
        pin_materials(case, manifest, artifacts, entry=state['files'][fid])
        trials.append(trial)
    return trial


def finish_preview(case, file_id, trial_id, report=None, error=None):
    with workflow.locked_state(case) as state:
        trial = next(t for t in state['files'][file_id]['preview_trials'] if t['id'] == trial_id)
        if error:
            trial.update(verdict='render_failed', diagnosis=str(error))
        else:
            trial.update(report=report, report_sha256=project.digest(project.project_path(report)))


def review_preview(case, spec):
    """Host observation, not an automatic visual score or permission to deliver."""
    fid = spec['file_id']
    with workflow.locked_state(case) as state:
        _control(case, state)
        trial = next((t for t in state['files'][fid].get('preview_trials', []) if t['id'] == spec['trial_id']), None)
        retract = spec.get('retract_previous_pass') is True
        if (not trial or not trial.get('report') or
                trial['verdict'] != ('pass' if retract else 'pending')):
            raise ValueError('Unknown, incomplete or already reviewed preview')
        if project.digest(project.project_path(trial['report'])) != trial['report_sha256']:
            raise ValueError('Preview report changed')
        report = project.read_json(project.project_path(trial['report']))
        if project.digest(project.project_path(report['output'])) != report['output_sha256']:
            raise ValueError('Preview media changed')
        lo, hi = report['source_frames_half_open']
        observed=spec['observed_source_frames']
        if (not observed or observed != sorted(set(observed)) or
                any(type(n) is not int or not lo <= n < hi for n in observed)):
            raise ValueError('Preview observations must identify actual frames inside its range')
        if not any(op.get('modified_source_frames') for op in report['operations']):
            raise ValueError('Method preview must exercise an actual target edit')
        checks = spec['checks']
        if set(checks) != {'target', 'protection', 'temporal'} or any(
                c.get('status') not in ('pass', 'fail') or not c.get('scope') for c in checks.values()):
            raise ValueError('Preview needs scoped target/protection/temporal observations')
        evidence = spec['evidence']
        if not evidence or any(project.digest(project.project_path(e['path'])) != e['sha256'] for e in evidence):
            raise ValueError('Preview observation evidence missing or changed')
        verdict = 'pass' if all(c['status'] == 'pass' for c in checks.values()) else 'fail'
        if verdict == 'pass' and observed != list(range(lo, hi)):
            raise ValueError('Observe every frame of the short risk preview, including its boundaries')
        if verdict == 'fail' and (not spec.get('diagnosis') or not spec.get('next_change')):
            raise ValueError('Failed probe needs a cause and a discriminating next change')
        modified = {op['operation_id'] for op in report['operations'] if op.get('modified_source_frames')}
        affected = spec.get('operation_ids', list(modified) if len(modified)==1 else []) if verdict=='fail' else []
        if policy := case.get('execution_policy'):
            if policy.get('version', 0) >= 3 and verdict == 'fail':
                if spec.get('failure_scope') not in ('method', 'localization', 'protection', 'composition', 'packaging'):
                    raise ValueError('Classify the observed failure before repair: method or a concrete local defect')
                trial['failure_scope'] = spec['failure_scope']
        if verdict=='fail' and (not affected or len(affected)!=len(set(affected)) or not set(affected)<=modified):
            raise ValueError('Failed preview must identify the affected operation_ids, not reopen every operation')
        if retract:
            if verdict != 'fail':
                raise ValueError('A counterexample may only retract a previous pass')
            trial.setdefault('review_history', []).append({
                key: trial.get(key) for key in ('verdict', 'checks', 'evidence', 'diagnosis', 'next_change', 'operation_ids')})
        trial.update(verdict=verdict, checks=checks, evidence=evidence, observed_source_frames=observed,
                     diagnosis=spec.get('diagnosis'), next_change=spec.get('next_change'), operation_ids=affected,
                     reviewed_at=datetime.now(timezone.utc).isoformat())
    return dict(trial_id=trial['id'], status=verdict, accepted=False)


def require_preview(case, manifest):
    fid = manifest['file_id']
    check(case, fid)
    if not case.get('execution_policy'):
        return
    state = project.read_json(workflow.state_path(case))
    from production_pipeline import controlled, guard_method
    if controlled(case):
        for operation in manifest['operations']:
            if operation.get('plan'):
                guard_method(case, fid, operation['operation_id'],
                             plan=project.read_json(project.project_path(operation['plan'])))
    current = fingerprint(manifest)
    trials = state['files'][fid].get('preview_trials', [])
    matching = [t for t in trials if t['fingerprint'] == current]
    if not matching or any(t['verdict'] != 'pass' for t in matching):
        raise ValueError('Full export requires a passing preview of this exact manifest and implementation')
    observed = set()
    for trial in matching:
        report_path = project.project_path(trial['report'])
        if project.digest(report_path) != trial['report_sha256']:
            raise ValueError('Preview report changed after review')
        report = project.read_json(report_path)
        if project.digest(project.project_path(report['output'])) != report['output_sha256']:
            raise ValueError('Preview media changed after review')
        if any(project.digest(project.project_path(e['path'])) != e['sha256'] for e in trial['evidence']):
            raise ValueError('Preview observation evidence changed after review')
        observed.update(range(*report['source_frames_half_open']))
    item = workflow.item_for(case, fid)
    risks = manifest.get('risk_preview_ranges_half_open', [])
    if any(op.get('surface_motion') == 'deformable' for op in item['operations']) and not risks:
        raise ValueError('Deformable source needs explicit motion/occlusion/reentry risk preview ranges')
    count = item.get('source_frame_count_inventory')
    for span in risks:
        if (not isinstance(span, list) or len(span) != 2 or any(type(n) is not int for n in span)
                or not 0 <= span[0] < span[1] or (count and span[1] > count)):
            raise ValueError('Invalid risk preview source range')
        if not set(range(*span)) <= observed:
            raise ValueError('Risk preview ranges are not fully observed for this exact candidate')
