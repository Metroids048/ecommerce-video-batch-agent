"""Regress 013 version loops and 015 method changes without business-run writes."""
import copy
import json
import unittest
from unittest.mock import patch
import production_effort as effort
import production_pipeline as pipeline
import project
import runninghub_adapter as provider
import workflow
import test_production_pipeline as fixtures

class Remediation(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.PipelineTests();self.fixture.setUp()
        self.root,self.case,self.manifest=self.fixture.root,self.fixture.case,self.fixture.manifest
        (self.root/'config').mkdir(exist_ok=True)
        (self.root/'config/models.json').write_text(json.dumps(dict(execution_mode='production')))
        (self.root/'scripts').mkdir(exist_ok=True)
        (self.root/'scripts/method.py').write_text('# frozen local fixture implementation')
    def tearDown(self): self.fixture.tearDown()
    def policy(self,mode='production'):
        self.case['execution_policy']=dict(version=3,mode=mode,stage_handoff_required=True,max_failed_previews_per_method=2)
        (self.root/'case.json').write_text(json.dumps(self.case))
        with workflow.locked_state(self.case) as s:s['case_binding']=workflow.case_binding(self.case)
    def freeze(self):
        observed=dict(source_sha256=self.case['files'][0]['source_sha256'],operations={
            o['id']:dict(appearance_ranges_half_open=[[0,6]],risk_ranges_half_open=[],quality_requirements=['Full output; preserve audio and detail'])
            for o in self.case['files'][0]['operations']})
        p=self.root/'observation.json';p.write_text(json.dumps(observed))
        return pipeline.freeze_route(self.case,dict(file_id='one',observation=dict(path=p.name,sha256=project.digest(p)),methods={
            o['id']:dict(id='fixed-surface-v1',local=dict(composition_mode='exact_asset_on_source_label',segmentation_backend=None),
                        provider=dict(model_id='fixed-model',parameters={'strength':1})) for o in self.case['files'][0]['operations']}))
    def test_new_intake_enforces_production_and_research_separation(self):
        self.assertEqual(effort.new_policy()['version'],3)
        self.assertEqual(effort.new_policy()['mode'],'production')
        self.policy();before=workflow.state_path(self.case).read_bytes()
        for action in ('configure-interval-edit', 'freeze-review-completion', 'review-completion'):
            with self.assertRaisesRegex(ValueError,'method research'):
                pipeline.dispatch(self.case,dict(action=action))
        self.assertEqual(before,workflow.state_path(self.case).read_bytes())
    def test_freeze_requires_consumable_method_evidence_before_preparation(self):
        self.policy()
        with self.assertRaisesRegex(ValueError,'method proof'):self.freeze()
        self.assertNotIn('route_handoff',project.read_json(workflow.state_path(self.case))['files']['one'])

    def test_recovery_reads_fixed_results_after_code_change_without_granting_new_work(self):
        self.policy('method_trial'); self.freeze()
        before = workflow.state_path(self.case).read_bytes()
        (self.root/'scripts/method.py').write_text('# changed implementation')
        self.assertEqual(pipeline.status(self.case)['files'][0]['next_action'], 'prepare_frozen_route')
        self.assertEqual(before, workflow.state_path(self.case).read_bytes())
        with self.assertRaisesRegex(ValueError, 'implementation changed'):
            pipeline.guard_upstream(self.case, 'one', operation_id='a')

    def test_test_file_change_does_not_change_method_implementation(self):
        before = pipeline.implementation_binding()
        (self.root/'scripts/test_unrelated.py').write_text('# offline regression only')
        self.assertEqual(before, pipeline.implementation_binding())
    def test_method_is_fixed_before_first_preview_and_direct_submit(self):
        self.policy('method_trial');self.freeze()
        with patch.object(provider,'model_for') as model,patch.object(provider.source_timeline,'load_for_item') as decode:
            with self.assertRaisesRegex(ValueError,'Frozen method'):
                provider.prepare_unit(self.case,dict(file_id='one',operation_id='a',model_id='other-model',parameters={}))
            model.assert_not_called();decode.assert_not_called()
        with patch.object(provider,'require_remote'),patch.object(provider,'verify_unit',return_value=dict(model_id='other-model',operation_id='a',parameters={})),patch.object(provider,'unit_for',return_value=dict(operation_id='a')):
            with self.assertRaisesRegex(ValueError,'Frozen method'):
                provider.remote_action(self.case,dict(action='submit',file_id='one',unit_id='u',authorization='missing'),transport=lambda p:self.fail('no transport'))
    def test_failed_method_cannot_reset_failure_count_by_renaming(self):
        self.policy('method_trial');self.freeze()
        with workflow.locked_state(self.case) as s:
            s['files']['one']['preview_trials']=[dict(id='013-v1',method_key='old-a',verdict='fail',diagnosis='geometry corrupt',operation_ids=['a']),dict(id='013-v39',method_key='new-b',verdict='fail',diagnosis='same failure',operation_ids=['a'])]
        with self.assertRaisesRegex(ValueError,'method research.*consumed'):effort.check(self.case,'one',start_method=True,method_family='renamed-again')
        pipeline.guard_upstream(self.case,'one',operation_id='b')
        with self.assertRaisesRegex(ValueError,'method research.*consumed'):pipeline.guard_upstream(self.case,'one',operation_id='a')
    def test_maintenance_exception_does_not_open_unrelated_intake(self):
        (self.root/'config').mkdir(exist_ok=True);(self.root/'config/models.json').write_text(json.dumps(dict(execution_mode='maintenance')))
        with patch('component_probe.active',return_value=dict(batch_id='015',file_id='bag')),patch('review_completion.active',return_value=None):
            with self.assertRaisesRegex(ValueError,'maintenance'):pipeline.require_production()
    def test_batch_resume_consumes_all_files_once_without_upstream_or_revision(self):
        self.case['files'].append(dict(copy.deepcopy(self.case['files'][0]),id='two'))
        workflow.state_path(self.case).unlink();workflow.initialize(self.case,'case.json')
        before=workflow.state_path(self.case).read_bytes();result=pipeline.dispatch(self.case,dict(action='resume-batch'))
        self.assertEqual({f['file_id'] for f in result['files']},{'one','two'})
        self.assertEqual(result['status'],'awaiting_stage_inputs');self.assertEqual(result['new_model_calls'],0);self.assertEqual(result['new_renders'],0)
        self.assertEqual(before,workflow.state_path(self.case).read_bytes())

class CompletePath(Remediation):
    def preview_pass(self, frames=6):
        import execution_plan
        self.policy('method_trial');self.freeze()
        report=execution_plan.preview(self.case,self.manifest,self.root/'trial.mp4',0,frames/30)
        (self.root/'observed.txt').write_text(f'{frames} synthetic fixture frames observed; not a production quality claim')
        effort.review_preview(self.case,dict(file_id='one',trial_id=report['trial_id'],observed_source_frames=list(range(frames)),
            checks={k:dict(status='pass',scope=f'all {frames} fixture frames') for k in ('target','protection','temporal')},
            evidence=[dict(path='observed.txt',sha256=project.digest(self.root/'observed.txt'))]))
        return report

    def test_actual_trial_proof_can_be_consumed_and_tampering_rejects_export(self):
        self.preview_pass()
        parent=copy.deepcopy(self.case)
        research=project.read_json(workflow.state_path(parent))['files']['one']
        methods=copy.deepcopy(research['route_handoff']['methods'])
        for op,method in methods.items():
            method['proof']=dict(case=dict(path='case.json',sha256=project.digest(self.root/'case.json')),
                                 file_id='one',operation_id=op,trial_id=research['preview_trials'][0]['id'])
        self.case=copy.deepcopy(parent);self.case['batch_id']='production'
        self.case['execution_policy']['mode']='production'
        (self.root/'production.json').write_text(json.dumps(self.case));workflow.initialize(self.case,'production.json')
        spec=dict(file_id='one',observation=research['route_handoff']['observation'],methods=methods)
        pipeline.freeze_route(self.case,spec)
        self.assertEqual(pipeline.status(self.case)['files'][0]['next_action'],'prepare_frozen_route')
        (self.root/'observed.txt').write_text('tampered observation')
        with self.assertRaisesRegex(ValueError,'proof artifact changed'):pipeline.guard_upstream(self.case,'one',operation_id='a')

    def test_simple_trial_does_not_certify_added_protection_requirements(self):
        self.preview_pass()
        research = project.read_json(workflow.state_path(self.case))['files']['one']
        methods = copy.deepcopy(research['route_handoff']['methods'])
        for op, method in methods.items():
            method['proof'] = dict(case=dict(path='case.json', sha256=project.digest(self.root/'case.json')),
                                  file_id='one', operation_id=op, trial_id=research['preview_trials'][0]['id'])
        self.case = copy.deepcopy(self.case); self.case['batch_id'] = 'harder-production'
        self.case['execution_policy']['mode'] = 'production'
        self.case['files'][0]['operations'][0]['protect'].append('thin straps and reflective zipper')
        (self.root/'harder.json').write_text(json.dumps(self.case)); workflow.initialize(self.case, 'harder.json')
        with self.assertRaisesRegex(ValueError, 'applicability'):
            pipeline.freeze_route(self.case, dict(file_id='one', observation=research['route_handoff']['observation'], methods=methods))

    def test_new_risks_and_quality_requirements_use_the_requested_observation(self):
        self.preview_pass(frames=3)
        research = project.read_json(workflow.state_path(self.case))['files']['one']
        methods = copy.deepcopy(research['route_handoff']['methods'])
        for op, method in methods.items():
            method['proof'] = dict(case=dict(path='case.json', sha256=project.digest(self.root/'case.json')),
                                  file_id='one', operation_id=op, trial_id=research['preview_trials'][0]['id'])
        self.case = copy.deepcopy(self.case); self.case['batch_id'] = 'new-observation-production'
        self.case['execution_policy']['mode'] = 'production'
        (self.root/'production.json').write_text(json.dumps(self.case)); workflow.initialize(self.case, 'production.json')
        original = project.read_json(self.root/'observation.json')
        baseline = workflow.state_path(self.case).read_bytes()
        for changed in ('quality', 'risk'):
            with self.subTest(changed=changed):
                workflow.state_path(self.case).write_bytes(baseline)  # Independent isolated negative inputs.
                observed = copy.deepcopy(original)
                if changed == 'quality':
                    observed['operations']['a']['quality_requirements'].append('Reflective zipper remains unchanged')
                else:
                    observed['operations']['a']['risk_ranges_half_open'] = [[4, 5]]
                path = self.root/(changed+'.json'); path.write_text(json.dumps(observed))
                with self.assertRaisesRegex(ValueError, 'applicability'):
                    pipeline.freeze_route(self.case, dict(file_id='one', methods=methods,
                        observation=dict(path=path.name, sha256=project.digest(path))))
                self.assertNotIn('route_handoff', project.read_json(workflow.state_path(self.case))['files']['one'])

    def test_frozen_appearances_cannot_shrink_to_only_middle_interval(self):
        import execution_plan
        self.policy('method_trial');self.freeze()
        plan=project.read_json(self.root/'a.json')
        for n in [0,1,5]:
            plan['review']['frame_states'][n]='absent';plan['review']['records'][n]['state']='absent'
        (self.root/'a.json').write_text(json.dumps(plan));self.manifest['operations'][0]['plan_sha256']=project.digest(self.root/'a.json')
        with self.assertRaisesRegex(ValueError,'Frozen appearances'):
            execution_plan.validate(self.case,self.manifest)

    def test_method_proof_rejects_changed_source_observation(self):
        self.preview_pass()
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        method=copy.deepcopy(entry['route_handoff']['methods']['a'])
        method['proof']=dict(case=dict(path='case.json',sha256=project.digest(self.root/'case.json')),
            file_id='one',operation_id='a',trial_id=entry['preview_trials'][0]['id'])
        observation=project.project_path(entry['route_handoff']['observation']['path'])
        changed=project.read_json(observation);changed['operations']['a']['risk_ranges_half_open']=[[0,1]]
        observation.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'proof source observation changed'):
            pipeline.verify_method_proof(method)

    def test_method_proof_must_originate_in_method_trial(self):
        self.preview_pass()
        self.case['execution_policy']['mode']='production'
        (self.root/'case.json').write_text(json.dumps(self.case))
        with workflow.locked_state(self.case) as state:
            state['case_binding']=workflow.case_binding(self.case)
            entry=state['files']['one']
            method=copy.deepcopy(entry['route_handoff']['methods']['a'])
            method['proof']=dict(case=dict(path='case.json',sha256=project.digest(self.root/'case.json')),
                file_id='one',operation_id='a',trial_id=entry['preview_trials'][0]['id'])
        with self.assertRaisesRegex(ValueError,'explicit method trial'):
            pipeline.verify_method_proof(method)

    def test_method_trial_cannot_become_normal_batch_delivery(self):
        import delivery_package
        self.policy('method_trial')
        with self.assertRaisesRegex(ValueError,'method research'):
            delivery_package.build(self.case,'research-delivery',full_comparison=True)

    def test_full_export_rechecks_the_frozen_method(self):
        self.preview_pass()
        with workflow.locked_state(self.case) as state:
            state['files']['one']['route_handoff']['methods']['a']['local']['composition_mode']='other'
        with self.assertRaisesRegex(ValueError,'Frozen method'):
            effort.require_preview(self.case,self.manifest)

    def test_015_actual_failure_metadata_blocks_renamed_method(self):
        from pathlib import Path
        failure=json.loads((Path(__file__).parent/'testdata/workflow_failures/015-failed-previews.json').read_text())
        self.policy('method_trial');self.freeze()
        with workflow.locked_state(self.case) as state:
            state['files']['one']['preview_trials']=failure['trials']
        with self.assertRaisesRegex(ValueError,'method research.*consumed'):
            effort.check(self.case,'one',start_method=True,method_family='renamed-015')

    def test_013_repeated_failed_candidates_stop_upstream(self):
        from pathlib import Path
        failure=json.loads((Path(__file__).parent/'testdata/workflow_failures/013-failed-localization.json').read_text())
        self.policy()
        with workflow.locked_state(self.case) as state:
            state['files']['one']['issues']=failure['issues']
        with self.assertRaisesRegex(ValueError,'Production method failed'):
            pipeline.check_method_failure(self.case,'one','backpack-edit')
        pipeline.check_method_failure(self.case,'one','unaffected-operation')

    def test_research_cannot_prepare_a_second_unit_before_preview(self):
        self.policy('method_trial');self.freeze()
        with workflow.locked_state(self.case) as state:
            state['files']['one']['execution_units']=[dict(unit_id='original',operation_id='a',status='prepared')]
        with patch.object(provider,'model_for') as model:
            with self.assertRaisesRegex(ValueError,'original method trial'):
                provider.prepare_unit(self.case,dict(file_id='one',operation_id='a',unit_id='renamed',model_id='fixed-model',parameters={'strength':1}))
            model.assert_not_called()


    def test_prepared_normalization_keeps_the_frozen_recipe(self):
        self.policy('method_trial');self.freeze()
        pipeline.guard_method(self.case,'one','a',provider_spec=dict(model_id='fixed-model',parameters={'strength':1,'providerWorkflowId':'derived'},
            method_recipe=dict(model_id='fixed-model',parameters={'strength':1})))
        with self.assertRaisesRegex(ValueError,'Frozen method'):
            pipeline.guard_method(self.case,'one','a',provider_spec=dict(model_id='fixed-model',parameters={'strength':1},
                method_recipe=dict(model_id='fixed-model',parameters={'strength':2})))

    def test_method_reuses_input_dimensions_and_text_without_research(self):
        method=dict(id='sam3-fixed',local={'composition_mode':'source_recolor','segmentation_backend':'sam3-video-semantic'},
            provider=dict(model_id='sam3-video-semantic',parameters=dict(width=1280,height=720,text_prompt='bag',strength=1),
                          input_fields=['parameters.width','parameters.height','parameters.text_prompt']))
        other=copy.deepcopy(method);other['provider']['parameters'].update(width=720,height=1280,text_prompt='fabric')
        self.assertEqual(pipeline.method_recipe(method),pipeline.method_recipe(other))
        other['provider']['parameters']['strength']=2
        self.assertNotEqual(pipeline.method_recipe(method),pipeline.method_recipe(other))

class DefaultEntry(unittest.TestCase):
    """Use the ordinary CLI dispatcher, with real local media and isolated state."""
    def setUp(self):
        self.fixture = fixtures.PipelineTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        (self.root/'config').mkdir()
        (self.root/'config/models.json').write_text(json.dumps(dict(execution_mode='production')))

    def tearDown(self):
        self.fixture.tearDown()

    def cli(self, *args):
        import contextlib
        import io
        output = io.StringIO()
        with patch('sys.argv', ['project.py', *args]), contextlib.redirect_stdout(output):
            code = project.main()
        return code, json.loads(output.getvalue())

    def intake(self):
        request = project.read_json(self.root/'request.json')
        request['batch_id'] = 'new-default-entry'
        request['execution_policy'] = dict(version=2)  # An old request cannot select the old default.
        request['files'][0]['operations'][0]['id'] = 'backpack-edit'
        (self.root/'new-request.json').write_text(json.dumps(request))
        code, result = self.cli('intake', '--case', 'new/case.json', '--request', 'new-request.json')
        self.assertEqual(code, 0, result)
        case = project.read_json(self.root/'new/case.json')
        self.assertEqual(case['execution_policy']['version'], 3)
        self.assertEqual(case['execution_policy']['mode'], 'production')
        self.assertEqual(len(case['files']), len(request['files']))
        workflow.initialize(case, 'new/case.json')
        return case

    def test_default_intake_does_not_need_an_opt_in_and_rejects_research(self):
        case = self.intake()
        (self.root/'research.json').write_text(json.dumps(dict(action='configure-interval-edit')))
        before = workflow.state_path(case).read_bytes()
        code, result = self.cli('pipeline', '--case', 'new/case.json', '--spec', 'research.json')
        self.assertEqual(code, 2)
        self.assertIn('method research', result['error'])
        self.assertEqual(workflow.state_path(case).read_bytes(), before)

    def test_default_batch_resume_preserves_013_failure_instead_of_retrying(self):
        from pathlib import Path
        case = self.intake()
        failure = project.read_json(Path(__file__).parent/'testdata/workflow_failures/013-failed-localization.json')
        with workflow.locked_state(case) as state:
            state['files']['one']['issues'] = failure['issues']
        (self.root/'resume.json').write_text(json.dumps(dict(action='resume-batch')))
        before = workflow.state_path(case).read_bytes()
        code, result = self.cli('pipeline', '--case', 'new/case.json', '--spec', 'resume.json')
        self.assertEqual(code, 0, result)
        self.assertEqual(result['files'][0]['next_action'], 'preserve_results_and_diagnose_method')
        self.assertEqual(result['new_model_calls'], 0)
        self.assertEqual(result['new_renders'], 0)
        self.assertEqual(workflow.state_path(case).read_bytes(), before)


if __name__=='__main__':unittest.main()
