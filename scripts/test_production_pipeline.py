"""Reproduce stage regression and needless re-execution on isolated local fixtures."""
import copy
import json
import unittest
from unittest.mock import patch

import delivery_package
import execution_plan
import production_pipeline as pipeline
import project
import runninghub_adapter
import test_execution_plan as fixtures
import workflow


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.UnifiedExecution()
        self.fixture.setUp()
        self.root, self.case, self.manifest = self.fixture.root, self.fixture.case, self.fixture.manifest
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        workflow.initialize(self.case, 'case.json')

    def tearDown(self):
        self.fixture.tearDown()

    def pin(self):
        artifacts = execution_plan.validate(self.case, self.manifest)[3]
        pipeline.pin_materials(self.case, self.manifest, artifacts)
        return artifacts

    def issue(self, category='protection'):
        issue = dict(id='observed-defect', status='open', category=category,
                     operation_id='a', candidate_id=None,
                     candidate_sha256=self.case['files'][0]['source_sha256'],
                     dependency=workflow.requirement_binding(self.case, self.case['files'][0]))
        with workflow.locked_state(self.case) as state:
            state['files']['one']['issues'].append(issue)
        return issue

    def test_new_production_requires_route_before_prepare_and_preview(self):
        self.case['execution_policy'] = dict(version=2, stage_handoff_required=True)
        with workflow.locked_state(self.case) as state:
            state['case_binding'] = workflow.case_binding(self.case)
        with self.assertRaisesRegex(ValueError, 'Freeze source observation'):
            pipeline.guard_upstream(self.case, 'one', operation_id='a')
        with self.assertRaisesRegex(ValueError, 'Freeze source observation'):
            self.pin()
        observation = dict(source_sha256=self.case['files'][0]['source_sha256'], operations={
            o['id']:dict(appearance_ranges_half_open=[[0,6]],risk_ranges_half_open=[[0,6]],
                        quality_requirements=['Preserve source details and review complete output'])
            for o in self.case['files'][0]['operations']})
        path=self.root/'source-observation.json';path.write_text(json.dumps(observation))
        spec=dict(file_id='one',observation=dict(path='source-observation.json',sha256=project.digest(path)))
        result=pipeline.freeze_route(self.case,spec)
        self.assertEqual(result['status'],'route_fixed')
        before=workflow.state_path(self.case).read_bytes()
        pipeline.freeze_route(self.case,spec)
        pipeline.guard_upstream(self.case,'one',operation_id='a')
        self.assertEqual(workflow.state_path(self.case).read_bytes(),before)
        with self.assertRaisesRegex(ValueError, 'Frozen risks'):
            self.pin()
        artifacts = execution_plan.validate(self.case, self.manifest)[3]
        pipeline.pin_materials(self.case, dict(self.manifest, risk_preview_ranges_half_open=[[0, 6]]), artifacts)
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'observation changed'):
            pipeline.guard_upstream(self.case,'one',operation_id='a')

    def test_submit_of_prepared_unit_cannot_bypass_material_handoff(self):
        self.pin()
        with patch.object(runninghub_adapter,'require_remote'), \
             patch.object(runninghub_adapter,'verify_unit',return_value=dict(model_id='unused',operation_id='a')), \
             patch.object(runninghub_adapter,'unit_for',return_value=dict(operation_id='a')), \
             patch.object(runninghub_adapter,'model_for',return_value={}), \
             patch('production_effort.check'):
            with self.assertRaisesRegex(ValueError,'inputs are fixed'):
                runninghub_adapter.remote_action(self.case,dict(action='submit',file_id='one',unit_id='u',authorization='absent.json'),
                    transport=lambda p:self.fail('must reject before transport'))

    def test_old_candidate_issue_cannot_reopen_current_materials(self):
        self.pin();self.issue()
        with workflow.locked_state(self.case) as state:
            state['files']['one']['current_candidate_id']='newer'
        with self.assertRaisesRegex(ValueError,'current candidate'):
            pipeline.reopen(self.case,dict(file_id='one',issue_id='observed-defect'))

    def test_fixed_handoff_blocks_upstream_and_unexplained_material_change(self):
        artifacts = self.pin()
        before = workflow.state_path(self.case).read_bytes()
        pipeline.pin_materials(self.case, self.manifest, artifacts)
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)
        with patch.object(runninghub_adapter, 'model_for', return_value=None), patch.object(
                runninghub_adapter.source_timeline, 'load_for_item') as decode:
            with self.assertRaisesRegex(ValueError, 'inputs are fixed'):
                runninghub_adapter.prepare_unit(self.case, dict(file_id='one', model_id='unused'))
            decode.assert_not_called()
        changed = dict(self.manifest, risk_preview_ranges_half_open=[[0, 3]])
        with self.assertRaisesRegex(ValueError, 'without a diagnosed'):
            pipeline.pin_materials(self.case, changed, artifacts)
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)
        self.issue('packaging')
        with self.assertRaisesRegex(ValueError, 'without a diagnosed'):
            pipeline.pin_materials(self.case, changed, artifacts)

    def test_diagnosed_failure_can_repair_materials_without_repeating_source_stage(self):
        artifacts = self.pin()
        with workflow.locked_state(self.case) as state:
            entry = state['files']['one']
            entry['preview_trials'] = [dict(verdict='fail', diagnosis='observed edge spill',
                handoff_fingerprint=entry['material_handoff']['fingerprint'])]
        changed = dict(self.manifest, risk_preview_ranges_half_open=[[0, 3]])
        pipeline.pin_materials(self.case, changed, artifacts)
        entry = project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(len(entry['material_history']), 1)
        self.assertEqual(entry['material_handoff']['manifest'], changed)

    def test_reopen_requires_exact_open_issue_and_reconciled_jobs(self):
        self.pin()
        spec = dict(file_id='one', issue_id='observed-defect')
        with self.assertRaisesRegex(ValueError, 'requires an open'):
            pipeline.reopen(self.case, spec)
        self.issue()
        with workflow.locked_state(self.case) as state:
            state['files']['one']['jobs'].append(dict(id='original', status='running'))
        with self.assertRaisesRegex(ValueError, 'original local job'):
            pipeline.reopen(self.case, spec)
        with workflow.locked_state(self.case) as state:
            state['files']['one']['jobs'][0]['status'] = 'complete'
            state['files']['one']['execution_units'] = [dict(status='submission_unknown')]
        with self.assertRaisesRegex(ValueError, 'original provider task'):
            pipeline.reopen(self.case, spec)
        with workflow.locked_state(self.case) as state:
            state['files']['one']['execution_units'] = []
        pipeline.reopen(self.case, spec)
        entry = project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertNotIn('material_handoff', entry)
        self.assertEqual(entry['material_repair_issue'], 'observed-defect')
        pipeline.guard_upstream(self.case, 'one', operation_id='a')
        changed = copy.deepcopy(self.manifest)
        untouched = self.root/changed['operations'][1]['plan']
        untouched.write_text(untouched.read_text()+'\n')
        changed['operations'][1]['plan_sha256'] = project.digest(untouched)
        artifacts = execution_plan.validate(self.case, changed)[3]
        with self.assertRaisesRegex(ValueError, 'unaffected operation'):
            pipeline.pin_materials(self.case, changed, artifacts)

    def test_recovery_returns_exact_candidate_even_when_current_code_requires_new_preview(self):
        candidate = workflow.execute_plan(self.case, 'manifest.json', 'v1')
        before = workflow.state_path(self.case).read_bytes()
        with patch('production_effort.require_preview', side_effect=AssertionError('unneeded preview')), \
             patch('regional_composite.render', side_effect=AssertionError('unneeded render')):
            reused = workflow.execute_plan(self.case, 'manifest.json', 'v1')
            for _ in range(3):
                recovered = pipeline.resume(self.case, dict(file_id='one'))
                self.assertEqual(recovered['output_sha256'], candidate['output_sha256'])
                self.assertEqual(recovered['new_renders'], 0)
        self.assertTrue(reused['execution_reused'])
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)
        (self.root/candidate['output']).write_bytes(b'changed output')
        with self.assertRaisesRegex(ValueError, 'output hash changed'):
            pipeline.status(self.case)

    def test_fixed_dependency_tamper_and_changed_candidate_plan_are_rejected(self):
        self.pin()
        (self.root/'a.npz').write_bytes(b'changed mask')
        with self.assertRaisesRegex(ValueError, 'Fixed material changed'):
            pipeline.status(self.case)

    def test_unknown_submission_never_recovers_as_a_new_submit(self):
        for unit, expected in [
                (dict(status='submission_unknown'), 'reconcile_unknown_submission'),
                (dict(status='RUNNING', provider_task_id='original'), 'query_original_task'),
                (dict(status='prepared'), 'submit_prepared_unit')]:
            with workflow.locked_state(self.case) as state:
                state['files']['one']['execution_units'] = [dict(unit, unit_id='original')]
            before = workflow.state_path(self.case).read_bytes()
            with patch.object(runninghub_adapter, 'verify_unit', return_value={}):
                current = pipeline.status(self.case)
            self.assertEqual(current['files'][0]['units'][0]['next_action'], expected)
            self.assertEqual(workflow.state_path(self.case).read_bytes(), before)

    def test_maintenance_blocks_cli_and_transport_before_credentials_or_preparation(self):
        (self.root/'config').mkdir()
        (self.root/'config/models.json').write_text(json.dumps(dict(execution_mode='maintenance')))
        with patch('sys.argv', ['project.py', 'prepare', '--case', 'case.json']), \
             patch.object(project, 'prepare') as prepare, patch('builtins.print'):
            self.assertEqual(project.main(), 2)
            prepare.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'production is stopped'):
            runninghub_adapter.require_remote('submit', self.case['batch_id'])
        before = workflow.state_path(self.case).read_bytes()
        self.assertEqual(pipeline.status(self.case)['execution_mode'], 'maintenance')
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)

    def test_read_only_transaction_does_not_advance_revision(self):
        before = workflow.state_path(self.case).read_bytes()
        with workflow.locked_state(self.case) as state:
            workflow.verify_state_case(self.case, state)
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)

    def test_direct_provider_cannot_bypass_fixed_preservation_parameters(self):
        self.case['files'][0]['operations'][0].update(kind='modify',surface_motion='deformable',
            desired_change='Change the fabric to blue, preserve all source details',
            recolor_mode='preserve_source_details',target_color_rgb=[0,71,171],semantic_label='bag')
        with workflow.locked_state(self.case) as state:
            state['case_binding'] = workflow.case_binding(self.case)
        with patch.object(runninghub_adapter, 'model_for', return_value=None), \
             self.assertRaisesRegex(ValueError, 'fixed full-source SAM3'):
            runninghub_adapter.prepare_unit(self.case, dict(file_id='one',operation_id='a',unit_id='trial',
                model_id='sam3-video-semantic',source_frames_half_open=[0,6],
                parameters=dict(width=384,height=256,text_prompt='bag')))

    def test_same_delivery_reuses_files_but_corruption_is_rejected(self):
        report = dict(revision=1, files=[], original_goal_complete=False)
        state = project.read_json(workflow.state_path(self.case))
        report['revision'] = state['revision']
        with patch.object(workflow, 'delivery_report', return_value=report):
            delivery_package.build(self.case, 'handoff', diagnostic=True)
            with patch.object(delivery_package.shutil, 'copy2') as copy_file:
                reused = delivery_package.build(self.case, 'handoff', diagnostic=True)
                self.assertTrue(reused['execution_reused'])
                copy_file.assert_not_called()
            (self.root/'handoff/用户输入.txt').write_text('corrupted')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                delivery_package.build(self.case, 'handoff', diagnostic=True)


if __name__ == '__main__':
    unittest.main()
