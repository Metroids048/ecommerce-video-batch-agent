"""Consume fixed stage handoffs in the existing workflow, never a second runner."""
import project
import workflow
from production_effort import provider_first


def implementation_binding():
    """Freeze the actual production code and policy, rather than a verbal phase."""
    config = project.read_json(project.ROOT / 'config/models.json')
    config = {k:v for k,v in config.items() if k not in ('execution_mode', 'implementation_lock')}
    paths = sorted(p for p in (project.ROOT/'scripts').iterdir()
                   if p.suffix in ('.py', '.mjs') and not p.name.startswith('test'))
    return dict(code={str(p.relative_to(project.ROOT)):project.digest(p) for p in paths},
                policy_sha256=workflow.hash_json(config))


def require_production(case=None, batch_id=None):
    batch = case.get('batch_id') if case else batch_id
    if batch and (project.ROOT/'runs'/batch/'retired-after-cleanup.json').is_file():
        raise ValueError('Historical batch retired after user cleanup; do not reconstruct media or resubmit old tasks')
    if case is None and batch_id:
        saved = project.ROOT/'runs'/batch_id/'workflow.json'
        if saved.is_file():
            policy = project.read_json(saved).get('case_binding', {}).get('execution_policy', {})
            if policy.get('reuse_only') is True:
                raise ValueError('Reuse-only batch: remote work is disabled for this saved order')
    if case and case.get('execution_policy', {}).get('reuse_only') is True:
        raise ValueError('Reuse-only batch: generation, upload and rendering are disabled; a cache miss must stop')
    if case and case.get('execution_policy', {}).get('local_only') is True:
        saved = workflow.state_path(case)
        if saved.is_file():
            workflow.verify_state_case(case, project.read_json(saved))
        # Method proofs, frozen materials and all review/export gates still run
        # at their normal consumers. This grants no remote capability.
        return
    path = project.ROOT / 'config/models.json'
    if path.is_file() and project.read_json(path).get('execution_mode') == 'maintenance':
        from component_probe import active
        from review_completion import active as review_active
        probe = active()
        from review_completion import _ACTIVE as completion
        scope = probe or (completion if review_active() else None)
        batch = case.get("batch_id") if case else batch_id
        if batch and scope and scope.get("batch_id") == batch:
            return
        raise ValueError('Project maintenance: production is stopped; inspect pipeline status and repair code, do not run media or remote tasks')
    if path.is_file():
        lock = project.read_json(path).get('implementation_lock')
        if lock:
            if (lock['binding'] != implementation_binding() or
                    project.digest(project.project_path(lock['verification'])) != lock['sha256']):
                raise ValueError('Production implementation changed; return to maintenance and rerun offline verification')


def interval_route(case, file_id, operation_id, entry=None):
    if entry is None:
        path = workflow.state_path(case)
        if not path.is_file():
            return None
        entry = project.read_json(path)['files'][file_id]
    route = entry.get('interval_edit_route')
    if not route or route['operation_id'] != operation_id:
        return None
    if route['dependency'] != workflow.requirement_binding(case, workflow.item_for(case, file_id)):
        raise ValueError('Interval route source/requirements changed')
    return route


def configure_interval_edit(case, spec):
    """Maintenance-only method change; original intent, source and failures survive."""
    if not provider_first(case):
        require_research(case)
    if project.read_json(project.ROOT/'config/models.json').get('execution_mode') != 'maintenance':
        raise ValueError('Route changes require maintenance; production and development must be isolated')
    fid = spec['file_id']; item = workflow.item_for(case, fid)
    if spec['operation_id'] not in {o['id'] for o in item['operations']} or not spec.get('user_instruction'):
        raise ValueError('Explicit operation and user instruction required')
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case, state); entry = state['files'][fid]
        verify_route(case, fid, entry, required=True)
        if (any(j['status'] in ('running','queued') for j in entry['jobs']) or
                any(u['status'].upper() not in ('PREPARED','FAILED','RAW_READY') for u in entry.get('execution_units',[]))):
            raise ValueError('Reconcile original jobs before maintenance route change')
        if entry.get('material_handoff') or entry.get('current_candidate_id'):
            raise ValueError('Reopen diagnosed materials before route maintenance')
        parent = None
        if provider_first(case):
            parent = next((u for u in entry.get('execution_units', [])
                           if u['unit_id'] == spec.get('parent_unit_id')), None)
            if (not parent or parent['operation_id'] != spec['operation_id']
                    or parent.get('raw_review', {}).get('status') not in ('fail','fail_or_limited')
                    or not spec.get('diagnosis') or not spec.get('prompt')):
                raise ValueError('Provider correction needs the exact failed raw, diagnosis and fixed prompt')
            import runninghub_adapter as provider
            parent_request = provider.verify_unit(case, parent)
            review = parent['raw_review']
            if project.digest(project.project_path(review['path'])) != review['sha256']:
                raise ValueError('Parent failure evidence changed')
        if not parent and not any(t.get('verdict') == 'fail' and t.get('diagnosis') for t in entry.get('preview_trials',[])):
            raise ValueError('Method change needs the preserved observed failure')
        lo,hi = spec['probe_frames_half_open']
        observed = project.read_json(project.project_path(entry['route_handoff']['observation']['path']))
        targets = observed['operations'][spec['operation_id']]['appearance_ranges_half_open']
        contexts = spec.get('context_ranges_half_open', targets) if parent else targets
        model_id = spec['model_id'] if parent else 'kling-o3-std-edit'
        if parent:
            model = provider.model_for(model_id, case['batch_id'])
            if model_id != provider.verify_unit(case, parent)['model_id'] or model['mode'] != 'rhart-video-edit':
                raise ValueError('Scoped correction retains the diagnosed provider model')
            if (any(not 0 <= a < b <= entry['source_frame_count'] for a,b in contexts)
                    or any(not any(c <= a < b <= d for c,d in contexts) for a,b in targets)):
                raise ValueError('Real source contexts must cover every observed target interval')
        if (any(type(n) is not int for n in (lo,hi)) or not 0 <= lo < hi <= entry['source_frame_count']
                or not any(a <= lo < hi <= b for a,b in contexts) or (hi-lo-1)%16
                or not 3 <= (hi-lo)/24 <= (8 if parent else 15)):
            raise ValueError('Hard probe must preserve a valid target context on the model frame clock')
        if parent and not (lo <= parent_request['source_frames_half_open'][0] < parent_request['source_frames_half_open'][1] <= hi):
            raise ValueError('Correction probe must retain the complete original failed interval')
        route = dict(operation_id=spec['operation_id'], model_id=model_id, model_input_fps='24',
                     probe_frames_half_open=[lo,hi], target_ranges_half_open=targets,
                     dependency=workflow.requirement_binding(case,item), user_instruction=spec['user_instruction'])
        if parent:
            route.update(context_ranges_half_open=contexts, prompt=spec['prompt'], prompt_format=2,
                diagnosis=spec['diagnosis'], parent_failure=dict(unit_id=parent['unit_id'],
                    request_fingerprint=parent['request_fingerprint'], raw_sha256=parent['raw']['sha256'],
                    review_sha256=parent['raw_review']['sha256']))
        if entry.get('interval_edit_route') and entry['interval_edit_route'] != route:
            raise ValueError('Interval method already fixed; do not start another parameter loop')
        entry['interval_edit_route'] = route
    return dict(status='interval_method_fixed_pending_offline_verification', route=route, accepted=False)


def guard_interval_unit(case, file_id, operation_id, model_id, span, entry=None):
    """Both prepare and direct submit consume the same hardest-first gate."""
    from component_probe import matches
    if matches(case, file_id, operation_id, model_id, span):
        route = interval_route(case, file_id, operation_id, entry)
        return route if route and model_id == route['model_id'] else None
    route = interval_route(case, file_id, operation_id, entry)
    if not route:
        return None
    if model_id != route['model_id']:
        raise ValueError('Interval model is fixed; no segmentation or alternate model trials')
    lo,hi = span
    if not any(a <= lo < hi <= b for a,b in route.get('context_ranges_half_open', route['target_ranges_half_open'])):
        raise ValueError('Only observed target intervals may enter the model')
    if (hi-lo-1)%16 or not 3 <= (hi-lo)/24 <= (8 if route.get('parent_failure') else 15):
        raise ValueError('Context must keep all source frames within the fixed model clock limits')
    from review_completion import matches as review_matches
    if review_matches(case, file_id, operation_id, model_id, span):
        return route
    if span == route['probe_frames_half_open']:
        return route
    if entry is None:
        entry = project.read_json(workflow.state_path(case))['files'][file_id]
    import runninghub_adapter as provider
    for unit in entry.get('execution_units',[]):
        request = provider.verify_unit(case, unit)
        if (request['model_id'] != route['model_id'] or request['source_frames_half_open'] != route['probe_frames_half_open']
                or request.get('interval_route_sha256') != workflow.hash_json(route)):
            continue
        review = unit.get('raw_review',{})
        if review.get('status') != 'pass' or review.get('raw_sha256') != unit.get('raw',{}).get('sha256'):
            continue
        if project.digest(project.project_path(review['path'])) != review['sha256']:
            raise ValueError('Hard probe review changed')
        audit = project.read_json(project.project_path(review['path']))
        if (audit.get('observed_frames') == list(range(route['probe_frames_half_open'][1]-route['probe_frames_half_open'][0]))
                and audit.get('frame_correspondence') == 'one_to_one_reference_frames'
                and audit.get('normal_speed_playback') == dict(rate=1, ended=True)):
            for e in audit['evidence']:
                if project.digest(project.project_path(e['path'])) != e['sha256']:
                    raise ValueError('Hard probe evidence changed')
            return route
    raise ValueError('Hardest interval has not passed full-frame protection, correspondence and normal-speed playback review')


def finish_maintenance(case, spec):
    require_research(case)
    import subprocess
    import sys
    path = project.ROOT/'config/models.json'; config = project.read_json(path)
    if config.get('execution_mode') != 'maintenance':
        raise ValueError('Offline verification must run during maintenance')
    state = project.read_json(workflow.state_path(case))
    if any(e.get('component_probe') and e['component_probe'].get('status') != 'pass'
           for e in state['files'].values()):
        raise ValueError('Component method must pass actual hard-interval review before production resumes')
    from component_probe import validate_review
    for entry in state['files'].values():
        probe=entry.get('component_probe')
        if probe:
            if (probe['implementation'] != implementation_binding() or
                    probe['recipe']['dependency'] != workflow.requirement_binding(case,workflow.item_for(case,probe['file_id']))):
                raise ValueError('Component implementation or requirements changed')
            review=probe['review']
            if project.digest(project.project_path(review['path'])) != review['sha256']:
                raise ValueError('Component review changed')
            validate_review(probe,project.read_json(project.project_path(review['path'])))
    modules = ['test_interval_edit', 'test_production_pipeline', 'test_runninghub_adapter',
               'test_source_timeline', 'test_production_effort', 'test_execution_plan']
    binding = implementation_binding()
    result = subprocess.run([sys.executable,'-m','unittest',*modules], cwd=project.ROOT/'scripts',
                            capture_output=True, text=True)
    receipt = dict(binding=binding, tests=modules, returncode=result.returncode,
                   output=result.stdout+result.stderr, verified_at=workflow.now(), new_model_calls=0)
    out = project.project_path(spec['out']); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(__import__('json').dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    if result.returncode or implementation_binding() != binding:
        raise ValueError('Offline verification failed or implementation changed; production remains stopped')
    if controlled(case):
        return dict(status='offline_verified_method_review_pending', verification=dict(path=spec['out'],sha256=project.digest(out)),
                    execution_mode='maintenance', accepted=False)
    config['implementation_lock'] = dict(binding=binding, verification=str(out.relative_to(project.ROOT)),
                                          sha256=project.digest(out))
    config['execution_mode'] = 'production'
    path.write_text(__import__('json').dumps(config,ensure_ascii=False,indent=2)+'\n')
    return dict(status='offline_verified_production_resumed', verification=config['implementation_lock'], accepted=False)


def controlled(case):
    return case.get('execution_policy', {}).get('version', 0) >= 3


def require_research(case):
    if controlled(case) and case['execution_policy'].get('mode') != 'method_trial':
        raise ValueError('Production retains its results; method research requires its own task and frozen counterexample')


def method_recipe(method):
    """Separate declared per-input data from the frozen algorithm and parameters."""
    from copy import deepcopy
    result=deepcopy({k:v for k,v in method.items() if k != 'proof'})
    provider=result.get('provider') or {}
    fields=provider.get('input_fields', [])
    allowed={'parameters.width','parameters.height','parameters.text_prompt','prompt','image_references'}
    if not isinstance(fields,list) or len(fields)!=len(set(fields)) or not set(fields)<=allowed:
        raise ValueError('Only source dimensions, semantic text and bound request assets may vary as method inputs')
    for field in fields:
        keys=field.split('.'); container=provider
        for key in keys[:-1]:container=container.setdefault(key,{})
        container[keys[-1]]='$bound_input'
    return result


def verify_method_proof(method, item=None, operation_id=None, observation=None):
    """Consume an actual reviewed trial; a label or offline green is not proof."""
    proof = method.get('proof')
    if not proof:
        raise ValueError('Production requires a consumable method proof, not a method name or offline tests')
    binding = proof['case']
    if project.digest(project.project_path(binding['path'])) != binding['sha256']:
        raise ValueError('Method proof case changed')
    parent = project.read_json(project.project_path(binding['path']))
    if not controlled(parent) or parent['execution_policy'].get('mode') != 'method_trial':
        raise ValueError('Method proof must originate in an explicit method trial')
    state = project.read_json(workflow.state_path(parent)); workflow.verify_state_case(parent, state)
    entry = state['files'][proof['file_id']]
    trial = next((t for t in entry.get('preview_trials', []) if t['id'] == proof['trial_id']), None)
    if not trial or trial.get('verdict') != 'pass' or not trial.get('evidence'):
        raise ValueError('Method proof requires a passing actual trial and bound observations')
    for e in [dict(path=trial['report'], sha256=trial['report_sha256']), *trial['evidence']]:
        if project.digest(project.project_path(e['path'])) != e['sha256']:
            raise ValueError('Method proof artifact changed')
    report = project.read_json(project.project_path(trial['report']))
    if project.digest(project.project_path(report['output'])) != report['output_sha256']:
        raise ValueError('Method proof output changed')
    handoff = entry.get('route_handoff', {})
    if handoff.get('implementation') != implementation_binding():
        raise ValueError('Method proof implementation changed')
    if project.digest(project.project_path(handoff['observation']['path'])) != handoff['observation']['sha256']:
        raise ValueError('Method proof source observation changed')
    proof_observation = project.read_json(project.project_path(handoff['observation']['path']))
    op_observation = proof_observation['operations'][proof['operation_id']]
    risk_frames = {n for lo,hi in op_observation['risk_ranges_half_open'] for n in range(lo,hi)}
    if not risk_frames <= set(range(*report['source_frames_half_open'])):
        raise ValueError('Method proof does not cover its frozen hardest risk ranges')
    if not any(o['operation_id'] == proof['operation_id'] and o.get('modified_source_frames') for o in report['operations']):
        raise ValueError('Method proof must actually edit the proven operation')
    proven = handoff.get('methods', {}).get(proof['operation_id'])
    if not proven or method_recipe(proven) != method_recipe(method):
        raise ValueError('Method proof must consume the same frozen implementation and recipe')
    if trial.get('observed_source_frames') != list(range(*report['source_frames_half_open'])):
        raise ValueError('Method proof must preserve the actual full-trial observation receipt')
    check_method_failure(parent, proof['file_id'], proof['operation_id'])
    source = workflow.item_for(parent, proof['file_id'])
    if project.digest(project.project_path(source['source'])) != source['source_sha256']:
        raise ValueError('Method proof source changed')
    if item is not None:
        proven_op = next(o for o in source['operations'] if o['id'] == proof['operation_id'])
        requested_op = next(o for o in item['operations'] if o['id'] == operation_id)
        def meaning(op):
            return {k:v for k,v in op.items() if k not in ('id', 'occurrences', 'full_coverage_verified')}
        # The current evidence is case-scoped. No implemented semantic assessor
        # establishes generalisation to a different object/source or harder
        # protection. Keep that capability gap explicit instead of inferring it
        # from matching model names or caller-written applicability labels.
        if source['source_sha256'] != item['source_sha256'] or meaning(proven_op) != meaning(requested_op):
            raise ValueError('Method applicability unproven for this source/operation/protection; preserved case evidence is not general maturity')
        requested = observation['operations'][operation_id]
        requested_risks = {n for lo,hi in requested['risk_ranges_half_open'] for n in range(lo,hi)}
        if (not requested_risks <= set(trial['observed_source_frames']) or
                not set(requested['quality_requirements']) <= set(op_observation['quality_requirements'])):
            raise ValueError('Method applicability lacks actual review of the requested risks/quality')
    if set(trial.get('checks', {})) != {'target', 'protection', 'temporal'} or any(c.get('status') != 'pass' for c in trial['checks'].values()):
        raise ValueError('Method proof needs actual scoped target/protection/temporal review')
    return proof


def check_method_failure(case, file_id, operation_id=None):
    if not controlled(case):
        return
    entry = project.read_json(workflow.state_path(case))['files'][file_id]
    from local_material_repair import verify as verify_local_repair
    repair = verify_local_repair(case, file_id, entry)
    failures = [t for t in entry.get('preview_trials', []) if t.get('verdict') in ('fail', 'render_failed')
                and (operation_id is None or not t.get('operation_ids') or operation_id in t['operation_ids'])]
    # Candidate failures also count: older paths could skip preview entirely.
    issues = [i for i in entry.get('issues', []) if i.get('category') in ('localization','protection','model_internal')
              and (operation_id is None or i.get('operation_id') == operation_id)]
    from production_effort import candidate_failures
    failed_candidates = ({i['candidate_id'] for i in issues if i.get('candidate_id')}
                         | candidate_failures(entry, operation_id))
    raw_failures = [u for u in entry.get('execution_units', [])
                    if u.get('raw_review', {}).get('status') in ('fail','fail_or_limited')
                    and (operation_id is None or u.get('operation_id') == operation_id)]
    route = (interval_route(case, file_id, operation_id, entry) if operation_id
             else entry.get('interval_edit_route'))
    if provider_first(case) and route and route.get('parent_failure'):
        parent = route['parent_failure']
        for unit in raw_failures:
            if unit['unit_id'] == parent['unit_id']:
                if (unit['request_fingerprint'] != parent['request_fingerprint']
                        or unit['raw']['sha256'] != parent['raw_sha256']
                        or unit['raw_review']['sha256'] != parent['review_sha256']
                        or project.digest(project.project_path(unit['raw_review']['path'])) != parent['review_sha256']):
                    raise ValueError('Diagnosed parent failure changed')
        raw_failures = [u for u in raw_failures if u['unit_id'] != parent['unit_id']]
        issues = [i for i in issues if not (i.get('observation_basis') == 'provider_raw'
                  and i.get('unit_id') == parent['unit_id'] and i.get('raw_sha256') == parent['raw_sha256'])]
    if repair:
        # Only the exact diagnosed parents are consumed. New failures still stop.
        issues = [i for i in issues if i['id'] not in repair['parent_issue_ids']]
        raw_failures = [u for u in raw_failures if u['unit_id'] not in repair['parent_unit_ids']]
        if any(t.get('verdict') in ('fail', 'render_failed') for t in entry.get('preview_trials', [])
               if t['id'] not in repair['parent_trial_ids']):
            raise ValueError('Local material repair failed; preserve the fixed attempt and diagnose before further work')
        failures = [t for t in failures if t['id'] not in repair['parent_trial_ids']]
    mode = case['execution_policy']['mode']
    method_failed = raw_failures or any(i['category'] == 'model_internal' for i in issues)
    if (method_failed or len(failed_candidates) >= 2 or
            (mode == 'method_trial' and (failures or issues or failed_candidates)) or
            (failures and (len(failures) >= 2 or any(t.get('failure_scope', 'method') == 'method' for t in failures)))):
        message = ('Provider return or candidate failed; preserve the failure and diagnose a scoped tool correction before proceeding'
                   if provider_first(case) else
                   'method research trial consumed; preserve the failure, diagnose a new candidate in explicit development scope'
                   if mode == 'method_trial' else
                   'Production method failed; preserve stage results and hand the counterexample to method research')
        raise ValueError(message)


def guard_method(case, file_id, operation_id, provider_spec=None, plan=None, entry=None):
    if not controlled(case):
        return
    if entry is None:
        state = project.read_json(workflow.state_path(case)); workflow.verify_state_case(case, state)
        entry = state['files'][file_id]
    route = verify_route(case, file_id, entry, required=True)
    if provider_first(case):
        if provider_spec is not None:
            from runninghub_adapter import model_for
            model = model_for(provider_spec['model_id'], case['batch_id'])
            if not model or model.get('mode') not in ('kling-video-edit', 'rhart-video-edit', 'reference-to-video'):
                raise ValueError('Provider-first editing does not run segmentation or local mask reconstruction')
            correction = interval_route(case, file_id, operation_id, entry)
            if correction and correction.get('parent_failure'):
                adapted = provider_spec.get('editing_input', provider_spec)
                if (provider_spec['model_id'] != correction['model_id']
                        or adapted.get('prompt_format') != correction['prompt_format']
                        or adapted.get('context', adapted.get('prompt')) != correction['prompt']):
                    raise ValueError('Consume the fixed diagnosed provider correction; old requests cannot resubmit')
        if plan is not None:
            from local_material_repair import verify as verify_repair, verify_plan
            repair = verify_repair(case, file_id, entry)
            if repair:
                verify_plan(case, file_id, entry, plan)
            elif (plan.get('composition_mode') != 'aigc_regional'
                    or any(p.get('blend_mode') != 'native_replace' for p in plan.get('parts', []))):
                raise ValueError('Use reviewed native intervals; local repair requires exact diagnosed parent bindings')
        check_method_failure(case, file_id, operation_id)
        return
    method = route['methods'].get(operation_id)
    if not method:
        raise ValueError('Frozen method missing for operation')
    if provider_spec is not None:
        expected = method.get('provider') or {}
        defaults = dict(parameters={}, prompt=None, image_references=[])
        recipe=provider_spec.get('method_recipe',provider_spec)
        if recipe.get('model_id') != provider_spec.get('model_id'):
            raise ValueError('Frozen method model differs from prepared request')
        actual = {k:recipe.get(k, defaults.get(k)) for k in ('model_id','parameters','prompt','image_references')}
        frozen = {k:expected.get(k, defaults.get(k)) for k in actual}
        if actual != frozen:
            raise ValueError('Frozen method differs from provider request; production cannot try another model or parameters')
    if plan is not None:
        from local_material_repair import verify_plan
        verify_plan(case, file_id, entry, plan)
        actual = {k:plan.get(k) for k in ('composition_mode', 'segmentation_backend')}
        if actual != method.get('local'):
            raise ValueError('Frozen method differs from compositor; preserve materials and use method research')
    check_method_failure(case, file_id, operation_id)


def route_for(operation, case=None):
    if case and provider_first(case):
        return "native_video_edit"
    from production_effort import source_preserving_recolor
    return ('source_preserving_recolor' if source_preserving_recolor(operation) else
            'native_video_edit' if operation.get('kind') == 'modify' and operation.get('surface_motion') == 'deformable'
            else 'existing_asset_or_regional_edit')


def local_edit_blockers(case, item):
    """Report implemented provider dependencies, without choosing a new method."""
    if not case.get('execution_policy', {}).get('local_only'):
        return []
    blockers=[]
    for op in item['operations']:
        route=route_for(op, case)
        if route=='source_preserving_recolor':
            blockers.append(op['id']+': Local-only order conflicts with the fixed SAM3 provider route. '
                'No local SAM3 runner is wired here; require compatible verified masks or an explicitly diagnosed local method. '
                'Use diagnose-source-frames for bounded cached-frame diagnosis; a localization result is not an executable edit.')
        elif route=='native_video_edit':
            blockers.append(op['id']+': Local-only order conflicts with native remote video editing; no local editing method is bound.')
    return blockers


def verify_route(case, file_id, entry, required=False, recovery=False):
    handoff = entry.get('route_handoff')
    if not handoff:
        if required and case.get('execution_policy', {}).get('stage_handoff_required'):
            raise ValueError('Freeze source observation and route before preparation, preview or submit')
        return None
    item = workflow.item_for(case, file_id)
    if handoff['dependency'] != workflow.requirement_binding(case, item):
        raise ValueError('Frozen route inputs changed')
    if project.digest(project.project_path(handoff['observation']['path'])) != handoff['observation']['sha256']:
        raise ValueError('Frozen source observation changed')
    if handoff['routes'] != {op['id']: route_for(op, case) for op in item['operations']}:
        raise ValueError('Frozen route changed; diagnose the affected operation first')
    from source_observation import require_receipt
    require_receipt(case, file_id, entry, handoff['observation'])
    if controlled(case) and not provider_first(case):
        if set(handoff.get('methods', {})) != {o['id'] for o in item['operations']}:
            raise ValueError('Frozen methods must cover all operations')
        if recovery:
            # Reading immutable results grants no new prepare/render permission.
            # Those consumers still require the current implementation/proof.
            return handoff
        from local_material_repair import verify as verify_local_repair
        repair = verify_local_repair(case, file_id, entry)
        if handoff['implementation'] != implementation_binding():
            # Once exact local materials exist, producer-only maintenance must
            # not send the order upstream again. Consumer code is independently
            # bound by the ordinary preview fingerprint; every input stays fixed.
            fixed = entry.get('material_handoff') if repair else None
            if not fixed or any(project.digest(project.project_path(p)) != sha
                                for p, sha in fixed['artifacts'].items()):
                raise ValueError('Frozen method implementation changed; explicit maintenance required')
        if case['execution_policy']['mode'] == 'production' and not repair:
            observed = project.read_json(project.project_path(handoff['observation']['path']))
            for op_id, method in handoff['methods'].items():
                verify_method_proof(method, item, op_id, observed)
    return handoff


def freeze_route(case, spec):
    """A -> B handoff, before any unit exists; semantic observation remains Agent work."""
    if not (controlled(case) and case['execution_policy']['mode'] == 'method_trial'):
        require_production(case)
    fid = spec['file_id']; item = workflow.item_for(case, fid)
    from source_observation import required, require_receipt
    saved = project.read_json(workflow.state_path(case))
    workflow.verify_state_case(case, saved)
    observation = spec.get('observation') or saved['files'][fid].get('source_observation', {}).get('receipt')
    if not observation:
        raise ValueError('Complete full source observation before freeze-route')
    if required(case):
        require_receipt(case, fid, saved['files'][fid], observation)
    path = project.project_path(observation['path'])
    if project.digest(path) != observation['sha256']:
        raise ValueError('Source observation hash changed')
    observed = project.read_json(path)
    operations = {op['id'] for op in item['operations']}
    if observed.get('source_sha256') != item['source_sha256'] or set(observed.get('operations', {})) != operations:
        raise ValueError('Source observation must cover this source and every operation')
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case, state)
        entry = state['files'][fid]
        require_receipt(case, fid, entry, observation)
        for op in observed['operations'].values():
            for field in ('appearance_ranges_half_open', 'risk_ranges_half_open'):
                ranges = op.get(field)
                if not isinstance(ranges, list) or (field.startswith('appearance') and not ranges):
                    raise ValueError('Freeze full appearances and risk ranges before production')
                for r in ranges:
                    if (not isinstance(r, list) or len(r) != 2 or any(type(n) is not int for n in r)
                            or not 0 <= r[0] < r[1] <= entry['source_frame_count']):
                        raise ValueError('Invalid source observation range')
            quality = op.get('quality_requirements')
            if not isinstance(quality, list) or not quality or not all(isinstance(q, str) and q.strip() for q in quality):
                raise ValueError('Freeze operation quality requirements with the route')
        handoff = dict(dependency=workflow.requirement_binding(case, item), observation=observation,
                       routes={op['id']: route_for(op, case) for op in item['operations']})
        if controlled(case) and not provider_first(case):
            methods = spec.get('methods', {})
            if set(methods) != operations or any(not m.get('id') or not m.get('local') for m in methods.values()):
                raise ValueError('Freeze a specific method for every operation before preparing materials')
            for method in methods.values():
                method_recipe(method)
            if case['execution_policy']['mode'] == 'production':
                for op_id, method in methods.items():
                    verify_method_proof(method, item, op_id, observed)
            handoff.update(methods=methods, implementation=implementation_binding())
        if entry.get('route_handoff'):
            if entry['route_handoff'] != handoff:
                raise ValueError('Route is already fixed; preserve original observation and diagnose changes')
            verify_route(case, fid, entry)
        else:
            if entry.get('execution_units') or entry.get('material_handoff') or entry.get('current_candidate_id'):
                raise ValueError('Cannot freeze a new route after production has started')
            entry['route_handoff'] = handoff
    return dict(status='route_fixed', file_id=fid, routes=handoff['routes'], new_model_calls=0)


def current_issue(case, file_id, entry, issue):
    if (issue.get('status') != 'open' or issue.get('operation_id') not in
            {op['id'] for op in workflow.item_for(case, file_id)['operations']} or
            issue.get('dependency') != workflow.requirement_binding(case, workflow.item_for(case, file_id)) or
            issue.get('candidate_id') != entry.get('current_candidate_id')):
        return False
    candidate_id = entry.get('current_candidate_id')
    expected = (workflow.verify_candidate(case, file_id, workflow.candidate_for(entry, candidate_id))['output_sha256']
                if candidate_id else workflow.item_for(case, file_id)['source_sha256'])
    return issue.get('candidate_sha256') == expected


def guard_upstream(case, file_id=None, operation_id=None, entry=None):
    """A finished material stage cannot silently become a new model experiment."""
    require_production(case)
    path = workflow.state_path(case)
    if not path.is_file():
        return
    if entry is None:
        state = project.read_json(path)
        workflow.verify_state_case(case, state)
        entries = {file_id: state['files'][file_id]} if file_id else state['files']
    else:
        entries = {file_id: entry}
    for fid, e in entries.items():
        from production_effort import require_reviewed_repair
        require_reviewed_repair(case, fid, [operation_id] if operation_id else
                                [o['id'] for o in workflow.item_for(case, fid)['operations']])
        verify_route(case, fid, e, required=True)
        if controlled(case):
            check_method_failure(case, fid, operation_id)
        repair = next((i for i in e['issues'] if i['id'] == e.get('material_repair_issue')
                       and current_issue(case, fid, e, i)), None)
        if e.get('material_handoff') or (e.get('current_candidate_id') and not repair):
            raise ValueError('Material inputs are fixed; consume the handoff. Reopen only the affected file with an open issue')
        if repair and repair['operation_id'] != operation_id:
            raise ValueError('Only the diagnosed operation may reopen upstream work')


def pin_materials(case, manifest, artifacts, entry=None):
    """First real preview freezes the already validated material dependencies."""
    require_production(case)
    fid = manifest['file_id']
    if not workflow.state_path(case).is_file():
        return
    binding = dict(manifest=manifest, artifacts=artifacts)
    fingerprint = workflow.hash_json(binding)
    def pin(entry):
        repair = active_manifest_repair(case, fid, entry)
        if repair and manifest != project.read_json(project.project_path(repair['manifest']['path'])):
            raise ValueError('Preview must consume the exact saved complete repair manifest')
        route = verify_route(case, fid, entry, required=True)
        if route:
            observed = project.read_json(project.project_path(route['observation']['path']))
            risks = {n for op in observed['operations'].values()
                     for lo, hi in op['risk_ranges_half_open'] for n in range(lo, hi)}
            declared = {n for lo, hi in manifest.get('risk_preview_ranges_half_open', []) for n in range(lo, hi)}
            if not risks <= declared:
                raise ValueError('Frozen risks must remain in the material preview and acceptance path')
        if controlled(case):
            from execution_plan import require_frozen_appearances
            require_frozen_appearances(case, manifest)
            for op in manifest['operations']:
                if op.get('plan'):
                    plan = project.read_json(project.project_path(op['plan']))
                    guard_method(case, fid, op['operation_id'], plan=plan, entry=entry)
        old = entry.get('material_handoff')
        if old and old['fingerprint'] == fingerprint:
            return
        if not old and entry.get('material_repair_issue'):
            issue = next((i for i in entry['issues'] if i['id'] == entry['material_repair_issue']
                          and current_issue(case, fid, entry, i)), None)
            if not issue:
                raise ValueError('Material repair requires the current candidate issue')
            previous = entry.get('material_history', [])
            if previous:
                old_ops = {o['operation_id']: o for o in previous[-1]['manifest']['operations']}
                if any(o != old_ops.get(o['operation_id']) and o['operation_id'] != issue['operation_id']
                       for o in manifest['operations']):
                    raise ValueError('Material repair changed an unaffected operation')
        if old:
            failed = [t for t in entry.get('preview_trials', [])
                      if t.get('handoff_fingerprint') == old['fingerprint']
                      and t.get('verdict') in ('fail', 'render_failed') and t.get('diagnosis')]
            issues = [i for i in entry['issues'] if i['category'] != 'packaging' and current_issue(case, fid, entry, i)]
            if not failed and not issues:
                raise ValueError('Fixed materials changed without a diagnosed preview failure or open issue')
            affected = {i['operation_id'] for i in issues}
            for trial in failed:
                affected.update(trial.get('operation_ids', []))
            old_ops = {o['operation_id']: o for o in old['manifest']['operations']}
            if any(o != old_ops.get(o['operation_id']) and o['operation_id'] not in affected
                   for o in manifest['operations']):
                raise ValueError('Material repair changed an unaffected operation')
            entry.setdefault('material_history', []).append(old)
        entry['material_handoff'] = dict(binding, fingerprint=fingerprint, fixed_at=workflow.now())
    if entry is not None:
        pin(entry)
    else:
        with workflow.locked_state(case) as state:
            workflow.verify_state_case(case, state)
            pin(state['files'][fid])


def reopen(case, spec):
    require_production(case)
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case, state)
        entry = state['files'][spec['file_id']]
        if any(j['status'] in ('running', 'queued') for j in entry['jobs']):
            raise ValueError('Recover original local job before reopening materials')
        if any(str(u.get('status', '')).upper() not in ('PREPARED', 'FAILED', 'RAW_READY')
               for u in entry.get('execution_units', [])):
            raise ValueError('Reconcile original provider task before reopening materials')
        issue = next((i for i in entry['issues'] if i['id'] == spec['issue_id'] and i['status'] == 'open'), None)
        if not issue or issue['category'] not in ('localization', 'protection', 'model_internal'):
            raise ValueError('Upstream repair requires an open localization/protection/model_internal issue; asset/composition/packaging stay downstream')
        if not current_issue(case, spec['file_id'], entry, issue):
            raise ValueError('Repair issue must bind the current candidate, operation and inputs')
        if controlled(case) and issue['category'] == 'model_internal':
            raise ValueError('Method failure belongs to method research; production materials remain frozen')
        if controlled(case):
            check_method_failure(case, spec['file_id'], issue['operation_id'])
        old = entry.pop('material_handoff', None)
        if old:
            entry.setdefault('material_history', []).append(dict(old, reopened_for=issue['id']))
        entry['material_repair_issue'] = issue['id']
    return dict(status='affected_materials_reopened', file_id=spec['file_id'], issue_id=issue['id'],
                reuse=['source', 'other files', 'unchanged raw returns and tracks'], new_model_calls=0)


def active_manifest_repair(case, fid, entry):
    repair = entry.get('manifest_repair')
    if not repair:
        return None
    issue = next((i for i in entry['issues'] if i['id'] == repair['issue_id']), None)
    if issue and issue['status'] in ('pending_recheck', 'closed'):
        return None  # The child, rather than the parent handoff, now needs review.
    if not issue or not current_issue(case, fid, entry, issue):
        raise ValueError('Saved manifest repair no longer binds the current open issue')
    binding = {k:v for k,v in repair.items() if k != 'fingerprint'}
    if workflow.hash_json(binding) != repair['fingerprint']:
        raise ValueError('Saved manifest repair binding changed')
    for path, expected in repair['artifacts'].items():
        if project.digest(project.project_path(path)) != expected:
            raise ValueError('Saved manifest repair material changed: '+path)
    return repair


def prepare_manifest_repair(case, spec):
    """Merge one diagnosed plan into its complete parent; never rebuild other operations."""
    require_production(case)
    fid = spec['file_id']
    from production_effort import require_reviewed_repair
    require_reviewed_repair(case, fid, [o['id'] for o in workflow.item_for(case, fid)['operations']])
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case, state)
        entry = state['files'][fid]
        if any(j['status'] in ('running', 'queued') for j in entry['jobs']):
            raise ValueError('Recover original local job before preparing a repair')
        issue = next((i for i in entry['issues'] if i['id'] == spec['issue_id']), None)
        if not issue or not current_issue(case, fid, entry, issue):
            raise ValueError('Repair requires the current exact candidate and open issue')
        if issue['category'] not in ('localization', 'protection', 'asset', 'composition'):
            raise ValueError('Pixel repair cannot fix model-internal or packaging failures')
        if issue['category'] in ('localization', 'protection') and entry.get('material_repair_issue') != issue['id']:
            raise ValueError('Reopen the diagnosed affected materials before preparing this repair')
        parent = workflow.verify_candidate(case, fid, workflow.candidate_for(entry, issue['candidate_id']))
        manifest = project.read_json(project.project_path(parent['plan']))
        if manifest.get('schema') != 'execution_manifest_v1':
            raise ValueError('Manifest repair requires a complete unified parent')
        ref = spec['operation_plan']
        if project.digest(project.project_path(ref['path'])) != ref['sha256']:
            raise ValueError('Repair operation plan hash mismatch')
        op = next(o for o in manifest['operations'] if o['operation_id'] == issue['operation_id'])
        if op.get('disposition') != 'ready':
            raise ValueError('Blocked parent operation needs a separately diagnosed material path')
        previous = project.read_json(project.project_path(op['plan']))
        changed = project.read_json(project.project_path(ref['path']))
        if changed == previous:
            raise ValueError('Unchanged repair plan; reuse and review the parent')
        if any(changed.get(k) != previous.get(k) for k in ('composition_mode', 'segmentation_backend')):
            raise ValueError('A local repair cannot change the frozen method')
        op.update(plan=ref['path'], plan_sha256=ref['sha256'])
        manifest['parent_output_sha256'] = parent['output_sha256']
        from execution_plan import validate
        artifacts = validate(case, manifest)[3]  # All appearances, methods and other plans stay checked.
        old = active_manifest_repair(case, fid, entry)
        if old and project.read_json(project.project_path(old['manifest']['path'])) != manifest:
            fixed = entry.get('material_handoff', {})
            failed = any(t.get('handoff_fingerprint') == fixed.get('fingerprint')
                         and t.get('verdict') in ('fail', 'render_failed') and t.get('diagnosis')
                         and issue['operation_id'] in t.get('operation_ids', [])
                         for t in entry.get('preview_trials', []))
            if fixed.get('manifest') != project.read_json(project.project_path(old['manifest']['path'])) or not failed:
                raise ValueError('Repair materials are fixed; diagnose the saved preview before another correction')
        key = workflow.hash_json(dict(manifest=manifest, parent_id=parent['id'], issue_id=issue['id']))
        path = workflow.state_path(case).parent/'plans'/('repair-'+key+'.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if project.read_json(path) != manifest:
                raise ValueError('Saved complete repair manifest changed')
        else:
            with path.open('x') as stream:
                __import__('json').dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
        relative = str(path.relative_to(project.ROOT))
        artifacts[relative] = project.digest(path)
        binding = dict(issue_id=issue['id'], operation_id=issue['operation_id'], parent_id=parent['id'],
                       parent_sha256=parent['output_sha256'], manifest=dict(path=relative, sha256=artifacts[relative]),
                       artifacts=artifacts)
        repair = dict(binding, fingerprint=workflow.hash_json(binding))
        if old != repair:
            if entry.get('manifest_repair'):
                entry.setdefault('manifest_repair_history', []).append(entry['manifest_repair'])
            entry['manifest_repair'] = repair
    return dict(status='repair_materials_fixed', **repair, new_model_calls=0, new_renders=0)


def failure_handoff(item, entry):
    """Carry the saved counterexample forward instead of asking the host to rediscover it.

    This is diagnostic input, not a method proof, approval or permission to retry.
    """
    failed = [t for t in entry.get('preview_trials', []) if t.get('verdict') in ('fail', 'render_failed')]
    context = dict(source=item['source'], source_sha256=item['source_sha256'],
        failed_preview_count=len(failed),
        fixed_material_fingerprint=entry.get('material_handoff', {}).get('fingerprint'),
        observation=entry.get('route_handoff', {}).get('observation'),
        verification_scope='Saved failure and bound report; no new visual review or trial')
    if failed:
        trial = failed[-1]
        context['failed_preview'] = {k:trial[k] for k in (
            'id','verdict','failure_scope','diagnosis','next_change','operation_ids',
            'checks','observed_source_frames') if k in trial}
        if trial.get('report') and trial.get('report_sha256'):
            if project.digest(project.project_path(trial['report'])) != trial['report_sha256']:
                raise ValueError('Saved failure report changed; cannot hand off another counterexample')
            context['failed_preview']['report'] = dict(path=trial['report'], sha256=trial['report_sha256'])
    repair = entry.get('local_material_repair')
    if repair:
        context['local_repair'] = {k:repair[k] for k in (
            'request_fingerprint','diagnosis','parent_unit_ids','parent_trial_ids','manifest') if k in repair}
    return context


def status(case):
    """Read-only recovery: exact artifacts first, no decode/render/model request."""
    state = project.read_json(workflow.state_path(case))
    workflow.verify_state_case(case, state)
    files = []
    for item in case['files']:
        fid = item['id']; entry = state['files'][fid]
        handoff = verify_route(case, fid, entry, recovery=True)
        routes = handoff['routes'] if handoff else {o['id']: route_for(o, case) for o in item['operations']}
        if not handoff and fid in state.get('preservation_fallbacks', {}):
            from production_effort import preservation_fallback
            if preservation_fallback(case, state, fid):
                routes[state['preservation_fallbacks'][fid]['operation_id']] = 'source_preserving_recolor'
        result = dict(file_id=fid, phase='materials', next_action='prepare_frozen_route', route=list(routes.values()),
                      file_state_token=workflow.hash_json(entry), expected_revision=state['revision'])
        if handoff:
            result['source_observation'] = handoff['observation']
        if not handoff and case.get('execution_policy', {}).get('stage_handoff_required'):
            result.update(phase='intake', next_action='freeze_observed_route')
            from source_observation import required, progress
            if required(case):
                observation = progress(case, fid, entry)
                result.update(next_action=observation['next_action'], source_observation=observation)
        if project.digest(project.project_path(item['source'])) != item['source_sha256']:
            raise ValueError('Actual source changed')
        jobs = [j for j in entry['jobs'] if j['status'] in ('running', 'queued')]
        repair = active_manifest_repair(case, fid, entry)
        repair_manifest = project.read_json(project.project_path(repair['manifest']['path'])) if repair else None
        fixed_repair = repair and entry.get('material_handoff', {}).get('manifest') == repair_manifest
        if repair:
            result['repair_handoff'] = repair
        if jobs:
            result.update(next_action='recover_original_local_job', jobs=[j['id'] for j in jobs])
        elif repair and not fixed_repair:
            result.update(phase='repair_preview', next_action='preview_fixed_repair',
                          fixed_materials=dict(manifest=repair_manifest, artifacts=repair['artifacts']))
        elif entry['current_candidate_id'] and not fixed_repair:
            candidate = workflow.verify_candidate(case, fid, workflow.candidate_for(entry, entry['current_candidate_id']))
            qualified = (candidate['status'] == 'qualified' and workflow.quality_reviews_pass(candidate)
                         and not any(i['status'] != 'closed' for i in entry['issues']))
            if qualified:
                try:
                    workflow.require_candidate_coverage(case, candidate)
                except (ValueError,OSError,KeyError) as exc:
                    qualified = False
                    result['coverage_error'] = str(exc)
            result.update(phase='delivery' if qualified else 'review',
                          next_action='reuse_qualified_candidate' if qualified else 'review_existing_candidate',
                          candidate_id=candidate['id'], output=candidate['output'], output_sha256=candidate['output_sha256'])
            result['repair_requests'] = []
            for i in entry['issues']:
                if i['status'] != 'open' or i['candidate_id'] != candidate['id']:
                    continue
                request = dict(issue_id=i['id'], operation_id=i['operation_id'],
                category=i['category'], next_action=('reopen-materials' if i['category'] in ('localization','protection')
                    and entry.get('material_repair_issue') != i['id'] else 'prepare-repair')
                if i['category'] in ('localization','protection','asset','composition') else
                    'remux-source-audio' if i['category'] == 'packaging' else 'diagnose_separate_dependency')
                failed = next((j for j in reversed(entry['jobs']) if j.get('action') == 'remux-source-audio'
                               and j.get('issue_id') == i['id'] and j['status'] == 'failed'), None)
                if failed:
                    request.update(next_action='diagnose_saved_packaging_failure',
                                   job={k:failed[k] for k in ('id','status','error','output') if k in failed})
                if provider_first(case) and i['category'] != 'packaging':
                    request['next_action'] = 'diagnose_defect_and_choose_existing_tool'
                result['repair_requests'].append(request)
        elif entry.get('material_handoff'):
            h = entry['material_handoff']
            if workflow.hash_json(dict(manifest=h['manifest'], artifacts=h['artifacts'])) != h['fingerprint']:
                raise ValueError('Fixed material handoff binding changed')
            for p, expected in h['artifacts'].items():
                if project.digest(project.project_path(p)) != expected:
                    raise ValueError('Fixed material changed: '+p)
            result.update(phase='finish', next_action='preview_or_export_fixed_materials',
                          handoff_fingerprint=h['fingerprint'],
                          fixed_materials=dict(manifest=h['manifest'], artifacts=h['artifacts']),
                          resume_request=dict(action='resume', file_id=fid))
            from production_effort import recover_preview
            previews = []
            for trial in entry.get('preview_trials', []):
                if trial.get('handoff_fingerprint') != h['fingerprint']:
                    continue
                saved = {k:trial[k] for k in ('verdict','diagnosis','next_change','operation_ids',
                         'checks','observed_source_frames','failure_scope') if k in trial}
                saved['trial_id'] = trial['id']
                saved['render_result_complete'] = bool(trial.get('report') and trial.get('report_sha256'))
                if trial.get('report'):
                    receipt = recover_preview(case, h['manifest'], trial['preview'], trial['requested_source_frames'])
                    saved.update(report=dict(path=receipt['report'], sha256=receipt['report_sha256']),
                                 output=receipt['output'], output_sha256=receipt['output_sha256'],
                                 source_frames_half_open=receipt['source_frames_half_open'])
                previews.append(saved)
            if previews:
                result['saved_previews'] = previews
                if any(t['verdict'] in ('fail','render_failed') for t in previews):
                    result.update(phase='material_repair', next_action='diagnose_saved_preview_failure')
                elif any(not t['render_result_complete'] for t in previews):
                    result.update(phase='preview_recovery', next_action='reconcile_original_preview')
                elif any(t['verdict'] == 'pending' for t in previews):
                    result.update(phase='preview_review', next_action='review_existing_preview')
                else:
                    result['next_action'] = 'export_fixed_materials_after_current_checks'
        elif entry.get('execution_units'):
            units = []
            from runninghub_adapter import verify_unit
            for u in entry['execution_units']:
                verify_unit(case, u)
                if u.get('raw') and project.digest(project.project_path(u['raw']['path'])) != u['raw']['sha256']:
                    raise ValueError('Original provider return changed')
                units.append(dict(unit_id=u['unit_id'], task_id=u.get('provider_task_id'), status=u['status'],
                                  next_action='review_original_raw' if u.get('raw') else
                                  ('query_original_task' if u.get('provider_task_id') else
                                   ('submit_prepared_unit' if u['status'] == 'prepared' else 'reconcile_unknown_submission'))))
            result.update(next_action='consume_existing_units', units=units)
        if result['phase'] != 'delivery' and not jobs:
            from production_effort import require_reviewed_repair, RepairLimitReached, CandidateReviewRequired
            try:
                require_reviewed_repair(case, fid, [o['id'] for o in item['operations']])
            except RepairLimitReached as exc:
                result.update(phase='method_research', next_action='preserve_results_and_diagnose_method', reason=str(exc))
            except CandidateReviewRequired:
                result.update(phase='review', next_action='review_existing_candidate')
        if provider_first(case):
            result['editing_strategy'] = 'provider_first_v1'
            result['repair_policy'] = 'Existing editor tools only; no source colour separation, hand masks or local recolor development'
        if controlled(case) and result['phase'] != 'delivery' and not jobs:
            try:
                check_method_failure(case, fid)
            except ValueError as exc:
                result.update(phase='needs_review' if provider_first(case) else 'method_research',
                              next_action='report_defect_and_choose_scoped_action' if provider_first(case) else 'preserve_results_and_diagnose_method', blocked=str(exc))
        if provider_first(case) and entry.get('local_material_repair') and result['phase'] != 'delivery' and not jobs:
            result.update(phase='preview_review', next_action='review_saved_result_only',
                          blocked='Complex local repair is disabled; preserve and review the saved result without rebuilding')
        if result['phase'] == 'method_research':
            result['blocking_context'] = failure_handoff(item, entry)
        files.append(result)
    config = project.ROOT / 'config/models.json'
    return dict(batch_id=case['batch_id'], revision=state['revision'], files=files, accepted=state['accepted'],
                execution_mode=project.read_json(config).get('execution_mode', 'production') if config.is_file() else 'production',
                new_model_calls=0, new_renders=0, state_mutated=False)


def resume(case, spec):
    """Deterministic consumption; only unfinished export executes, never upstream."""
    current = status(case)
    selected = next(f for f in current['files'] if f['file_id'] == spec['file_id'])
    if selected['phase'] in ('review', 'delivery') or selected['next_action'] == 'recover_original_local_job':
        result = dict(selected, execution_reused=True, new_model_calls=0, new_renders=0)
        if spec.get('delivery_out'):
            if selected['phase'] != 'delivery':
                raise ValueError('Only a qualified candidate can reuse a normal delivery')
            if not project.project_path(spec['delivery_out']).exists():
                require_production(case)
            from delivery_package import build
            result['delivery'] = build(case, spec['delivery_out'], full_comparison=spec.get('full_comparison', False))
        return result
    if selected['phase'] != 'finish':
        return selected  # Host observes/reconciles the indicated original artifacts.
    require_production(case)
    entry = project.read_json(workflow.state_path(case))['files'][spec['file_id']]
    manifest = entry['material_handoff']['manifest']
    from production_effort import require_preview
    require_preview(case, manifest)  # Same risk/quality requirements as ordinary execute.
    plan = spec.get('plan')
    if plan is None:
        # The saved handoff is the canonical input. Materialize it once for the
        # existing execute entry instead of asking the host to reconstruct JSON.
        fixed = entry['material_handoff']
        path = workflow.state_path(case).parent/'plans'/('fixed-'+fixed['fingerprint']+'.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('x') as stream:
                __import__('json').dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
        plan = str(path.relative_to(project.ROOT))
    if project.read_json(project.project_path(plan)) != manifest:
        raise ValueError('Resume must consume the exact fixed manifest')
    candidate_id = spec.get('candidate_id') or 'fixed-'+entry['material_handoff']['fingerprint'][:24]
    repair = active_manifest_repair(case, spec['file_id'], entry)
    if repair and any(spec.get(k, repair[k]) != repair[k] for k in ('parent_id', 'issue_id')):
        raise ValueError('Resume must consume the exact saved repair parent and issue')
    return workflow.execute_plan(case, plan, candidate_id,
        repair['parent_id'] if repair else spec.get('parent_id'),
        repair['issue_id'] if repair else spec.get('issue_id'))


def resume_batch(case, spec):
    """A bounded pass over the existing chain; never create models or retry failures."""
    before = status(case)
    requests = spec.get('files', {})
    if set(requests) - {f['file_id'] for f in before['files']}:
        raise ValueError('Batch resume contains an unknown file')
    results = []; renders = 0
    for item in before['files']:
        fid = item['file_id']
        if item['phase'] == 'finish' and fid in requests:
            request = dict(requests[fid], file_id=fid)
            if request.get('delivery_out'):
                raise ValueError('Batch delivery is built once after every file qualifies')
            try:
                candidate = resume(case, request)
                renders += candidate.get('new_renders', 0 if candidate.get('execution_reused') else 1)
                results.append(dict(file_id=fid, candidate=candidate))
            except ValueError as exc:
                results.append(dict(item, blocked=str(exc)))
        else:
            results.append(dict(item, execution_reused=True))
    after = status(case)
    complete = all(f['phase'] == 'delivery' for f in after['files'])
    result = dict(batch_id=case['batch_id'], status='qualified_batch' if complete else 'awaiting_stage_inputs',
                  files=results, next=after['files'], new_model_calls=0, new_renders=renders, accepted=after['accepted'])
    if spec.get('delivery_out'):
        if not complete:
            result['delivery_blocked'] = 'Complete batch requires every full output to qualify'
        else:
            from delivery_package import build
            result['delivery'] = build(case, spec['delivery_out'], full_comparison=True)
    return result


def dispatch(case, spec):
    if spec['action'] == 'repair-local-materials':
        from local_material_repair import prepare
        return prepare(case, spec)
    if case.get('execution_policy', {}).get('reuse_only') is True:
        if spec['action'] not in ('status', 'reuse-candidate', 'resume', 'resume-batch'):
            raise ValueError('Reuse-only batch cannot reopen production or method research')
    if spec['action'] == 'reuse-candidate':
        return workflow.reuse_candidate(case, spec)
    if spec['action'] == 'repair-semantic-coordinate-order':
        from semantic_scan import repair_coordinate_order
        return repair_coordinate_order(case, spec)
    if spec['action'] == 'scan-source':
        from semantic_scan import scan
        return scan(case, spec)
    if spec['action'] == 'diagnose-source-frames':
        from semantic_diagnostics import diagnose
        return diagnose(case,spec)
    if spec['action'] == 'plan-source-scan':
        from semantic_scan import scan_plan
        return scan_plan(case, spec)
    if spec['action'] == 'trial-source-tracking':
        from semantic_tracking import trial
        return trial(case, spec)
    if spec['action'] == 'recover-semantic-journal':
        from semantic_scan import recover
        return recover(case, spec)
    if spec['action'] == 'repair-semantic-runtime':
        from semantic_scan import repair_runtime
        return repair_runtime(case, spec)
    if spec['action'] in ('prepare-source-observation', 'record-source-observation'):
        from source_observation import prepare, record
        return (prepare if spec['action'] == 'prepare-source-observation' else record)(case, spec)
    if spec['action'] == 'prepare-repair':
        return prepare_manifest_repair(case, spec)
    if spec['action'] == 'remux-source-audio':
        from packaging_repair import remux
        return remux(case, spec)
    if spec['action'] in ('freeze-component-probe', 'component-probe', 'finish-maintenance'):
        require_research(case)
    if spec['action'] == 'resume-batch':
        return resume_batch(case, spec)
    if spec['action'] in ('freeze-review-completion', 'review-completion'):
        from review_completion import freeze, run
        return (freeze if spec['action'] == 'freeze-review-completion' else run)(case, spec)
    if spec['action'] in ('freeze-component-probe', 'component-probe'):
        from component_probe import dispatch as probe
        return probe(case, spec)
    if spec['action'] == 'configure-interval-edit':
        return configure_interval_edit(case, spec)
    if spec['action'] == 'finish-maintenance':
        return finish_maintenance(case, spec)
    if spec['action'] == 'status':
        return status(case)
    if spec['action'] == 'resume':
        return resume(case, spec)
    if spec['action'] == 'freeze-route':
        return freeze_route(case, spec)
    if spec['action'] == 'reopen-materials':
        return reopen(case, spec)
    raise ValueError('Unknown pipeline action')
