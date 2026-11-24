"""Reproduce version-renaming loops at real preview/export/provider entry points."""
import copy
from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import patch

import execution_plan
import production_effort as effort
import project
import runninghub_adapter
import source_matte
import workflow
from test_execution_plan import UnifiedExecution


class ProductionEffort(unittest.TestCase):
    def setUp(self):
        self.fixture = UnifiedExecution()
        self.fixture.setUp()
        self.root, self.case, self.manifest = self.fixture.root, self.fixture.case, self.fixture.manifest
        self.case['execution_policy'] = dict(effort.new_policy(), version=2)
        # These are frozen v2 method-route regressions, not new provider-first orders.
        self.case['execution_policy'].pop('editing_strategy', None)
        (self.root/'case.json').write_text(json.dumps(self.case))
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        workflow.initialize(self.case, 'case.json')
        self.freeze_fixture_route()

    def freeze_fixture_route(self):
        # Exercise the same A -> B handoff now required by new production intake.
        import production_pipeline
        import source_observation
        dep=workflow.requirement_binding(self.case,self.case['files'][0])
        queue=source_observation.prepare(self.case,dict(file_id='one',out='source-'+dep['requirements_sha256'][:16]))['queue']
        images=project.read_json(project.project_path(queue['path']))['evidence']
        evidence=[{k:e[k] for k in ('path','sha256')} for e in images]
        operations={}
        for op,x in [('a',5),('b',60)]:
            operations[op]=dict(quality_requirements=['Review all fixture frames and protect source details'],
                records=[dict(frame=n,state='visible',instances=[dict(id=op,bbox=[x,20,30,16])],
                    all_instances_observed=True,reason='Synthetic regression fixture only',evidence=evidence) for n in range(6)])
        source_observation.record(self.case,dict(file_id='one',queue_sha256=queue['sha256'],
            dependency=workflow.requirement_binding(self.case,self.case['files'][0]),
            observer='synthetic-regression',operations=operations))
        production_pipeline.freeze_route(self.case,dict(file_id='one'))

    def tearDown(self):
        self.fixture.tearDown()

    def preview(self, name, start='0', end='0.2'):
        output = self.root/(name+'.mp4')
        with patch('sys.argv', ['project.py', 'preview', '--case', 'case.json', '--plan',
                'manifest.json', '--out', output.name, '--start', start, '--end', end]):
            self.assertEqual(project.main(), 0)
        state = project.read_json(workflow.state_path(self.case))
        return state['files']['one']['preview_trials'][-1]

    def review(self, trial, verdict, expected_exit=0, observed=None, retract=False):
        (self.root/'observed.txt').write_text('Fixture observation, not production visual acceptance')
        spec = dict(file_id='one', trial_id=trial['id'], observed_source_frames=list(range(6)) if observed is None else observed,
            checks={k:dict(status=verdict,scope='all six fixture frames') for k in ('target','protection','temporal')},
            evidence=[dict(path='observed.txt',sha256=project.digest(self.root/'observed.txt'))],
            diagnosis='fixture mask spills',next_change='measure foreground rather than tune hue',
            retract_previous_pass=retract,operation_ids=['a'])
        (self.root/'review.json').write_text(json.dumps(spec))
        with patch('sys.argv', ['project.py','preview-review','--case','case.json','--spec','review.json']):
            self.assertEqual(project.main(),expected_exit)

    def test_full_export_needs_actual_preview_review_and_invalidates_on_change(self):
        with self.assertRaisesRegex(ValueError,'passing preview'):
            workflow.execute_plan(self.case,'manifest.json','v1')
        t=self.preview('risk')
        with self.assertRaisesRegex(ValueError,'previous preview'):
            execution_plan.preview(self.case,self.manifest,self.root/'other.mp4',0,.2)
        with self.assertRaisesRegex(ValueError,'passing preview'):
            execution_plan.render(self.case,self.manifest,self.root/'full.mp4')
        self.review(t,'pass')
        result=workflow.execute_plan(self.case,'manifest.json','v1')
        self.assertTrue(project.project_path(result['output']).is_file())
        self.assertFalse(workflow.quality_reviews_pass(result))
        changed=copy.deepcopy(self.manifest);changed['parent_output_sha256']='different'
        with self.assertRaisesRegex(ValueError,'passing preview'):
            effort.require_preview(self.case,changed)
        (self.root/'risk.mp4').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'media changed'):
            effort.require_preview(self.case,self.manifest)

    def change_plan(self, opacity):
        plan = project.read_json(self.root/'a.json')
        plan['label_style']['ink_opacity'] = opacity
        (self.root/'a.json').write_text(json.dumps(plan))
        self.manifest['operations'][0]['plan_sha256'] = project.digest(self.root/'a.json')
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))

    def test_same_method_parameter_loop_is_blocked_even_with_a_fake_backend_label(self):
        self.review(self.preview('v1'),'fail')
        with self.assertRaisesRegex(ValueError,'Identical preview'):
            execution_plan.preview(self.case,self.manifest,self.root/'renamed.mp4',0,.2)
        self.change_plan(.9)
        self.review(self.preview('v2'),'fail')
        self.change_plan(.8)
        with self.assertRaisesRegex(ValueError,'Same method failed twice'):
            execution_plan.preview(self.case,self.manifest,self.root/'v39.mp4',0,.2)
        changed = copy.deepcopy(self.manifest)
        plan = project.read_json(self.root/'a.json')
        plan['segmentation_backend'] = 'different-actual-backend-fixture'
        (self.root/'new-method.json').write_text(json.dumps(plan))
        changed['operations'][0].update(plan='new-method.json',plan_sha256=project.digest(self.root/'new-method.json'))
        with self.assertRaisesRegex(ValueError,'Same method failed twice'):
            effort.begin_preview(self.case,changed,'different.mp4')
        self.assertFalse((self.root/'v39.mp4').exists())

    def test_failed_semantic_strategy_blocks_new_segmentation_and_matting_before_preparation(self):
        self.review(self.preview('semantic-bad-1'),'fail')
        self.change_plan(.9)
        self.review(self.preview('semantic-bad-2'),'fail')
        # Mark the method family on the actual fixture trials. Production derives
        # legacy families from hash-checked shared preview reports.
        with workflow.locked_state(self.case) as state:
            for trial in state['files']['one']['preview_trials']:
                trial['method_families']=['local_recolor']
        for mode in ('video-segmentation','video-matting'):
            with patch.object(runninghub_adapter,'model_for',return_value=dict(mode=mode)):
                with self.assertRaisesRegex(ValueError,'Semantic strategy failed twice'):
                    runninghub_adapter.prepare_unit(self.case,dict(file_id='one',model_id='changed-trimap'))
        # Restoring/download of existing work and a native edit route remain usable.
        effort.check(self.case,'one')
        effort.check(self.case,'one',start_method=True,method_family='native_video_edit')

    def test_first_deformable_modification_rejects_segmentation_without_waiting_for_failure(self):
        self.case['files'][0]['operations'][0].update(kind='modify', surface_motion='deformable')
        with workflow.locked_state(self.case) as state:
            state['case_binding'] = workflow.case_binding(self.case)
        for mode in ('video-segmentation', 'video-matting'):
            with patch.object(runninghub_adapter, 'model_for', return_value=dict(mode=mode)):
                with self.assertRaisesRegex(ValueError, 'starts with native video editing'):
                    runninghub_adapter.prepare_unit(self.case, dict(file_id='one', model_id='first-probe'))
        with self.assertRaisesRegex(ValueError, 'starts with native video editing'):
            source_matte.prepare(self.case, dict(file_id='one', method='vitmatte_input'), 'unused.npz')
        effort.check(self.case, 'one', method_family='native_video_edit')
        self.assertEqual(project.read_json(workflow.state_path(self.case))['files']['one'].get('preview_trials', []), [])

    def test_preservation_switch_requires_hash_bound_native_texture_failure(self):
        body=dict(checks=[dict(kind='texture_geometry',status='fail')])
        path=self.root/'failed-native.json';path.write_text(json.dumps(body))
        unit=dict(raw_review=dict(path=path.name,sha256=project.digest(path)))
        with workflow.locked_state(self.case) as state:
            state['preservation_fallbacks']={'one':dict(failed_unit_id='native',operation_id='a',
                review_sha256=project.digest(path),user_instruction='preserve original print')}
        state=project.read_json(workflow.state_path(self.case))
        with patch.object(runninghub_adapter,'unit_for',return_value=unit),patch.object(
                runninghub_adapter,'verify_unit',return_value=dict(operation_id='a',model_id='kling-o3-std-edit')):
            self.assertTrue(effort.preservation_fallback(self.case,state,'one'))
            path.write_text(json.dumps(dict(checks=[dict(kind='texture_geometry',status='pass')])))
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                effort.preservation_fallback(self.case,state,'one')

    def test_pure_color_preservation_routes_before_any_paid_native_failure(self):
        for op in self.case['files'][0]['operations']:
            op.update(kind='modify',surface_motion='deformable',recolor_mode='preserve_source_details',
                      target_color_rgb=[0,71,171],semantic_label='backpack',
                      desired_change='Change only fabric color; preserve source details')
        with workflow.locked_state(self.case) as state:
            state['case_binding']=workflow.case_binding(self.case)
            state['files']['one'].pop('route_handoff')
            # This synthetic fixture changes its original requirements before
            # any unit/render. Re-observe against that identity; never reuse the
            # previous target's receipt or relax the new source gate.
            state['files']['one'].pop('source_observation')
        self.freeze_fixture_route()
        effort.check(self.case,'one',method_family='local_recolor')
        with self.assertRaisesRegex(ValueError,'not native reconstruction'):
            runninghub_adapter.prepare_native_recolor(self.case,dict(file_id='one',operation_id='a'))
        with patch.object(runninghub_adapter,'model_for',return_value=dict(mode='kling-video-edit')):
            with self.assertRaisesRegex(ValueError,'fixed SAM3'):
                runninghub_adapter.prepare_unit(self.case,dict(file_id='one',operation_id='a',model_id='kling-o3-std-edit'))
        self.case['files'][0]['operations'][0]['target_color_rgb']=[0,71,True]
        with self.assertRaisesRegex(ValueError,'requires pure color'):
            effort.source_preserving_recolor(self.case['files'][0]['operations'][0])

    def test_passing_risk_probe_separates_nonconsecutive_failures(self):
        self.review(self.preview('bad-edge'),'fail')
        self.change_plan(.9)
        self.review(self.preview('corrected-edge'),'pass')
        # A later, separately observed defect authorizes only a scoped repair.
        with workflow.locked_state(self.case) as state:
            state['files']['one']['issues'].append(dict(id='later-defect', status='open',
                category='composition', operation_id='a', candidate_id=None,
                candidate_sha256=self.case['files'][0]['source_sha256'],
                dependency=workflow.requirement_binding(self.case,self.case['files'][0])))
        self.change_plan(.8)
        self.review(self.preview('different-trim-defect'),'fail')
        self.change_plan(.7)
        corrected=self.preview('trim-repair')
        self.review(corrected,'fail')
        self.change_plan(.6)
        with self.assertRaisesRegex(ValueError,'Same method failed twice'):
            execution_plan.preview(self.case,self.manifest,self.root/'loop.mp4',0,.2)

    def test_unmodified_source_preview_cannot_qualify_a_method(self):
        blocked=copy.deepcopy(self.manifest)
        blocked['operations']=[dict(operation_id=op['operation_id'],disposition='blocked',
            blocked_reason='method not exercised') for op in blocked['operations']]
        (self.root/'manifest.json').write_text(json.dumps(blocked))
        trial=self.preview('unchanged')
        self.review(trial,'pass',expected_exit=2)
        self.assertEqual(project.read_json(workflow.state_path(self.case))['files']['one']['preview_trials'][-1]['verdict'],'pending')

    def test_fail_fast_accepts_counterexample_but_partial_review_cannot_pass(self):
        trial = self.preview('partial')
        self.review(trial, 'pass', expected_exit=2, observed=[0])
        self.review(trial, 'fail', observed=[0])

    def test_later_counterexample_retracts_pass_without_erasing_review_or_reopening_fail(self):
        trial=self.preview('overlooked-sleeve')
        self.review(trial,'pass')
        effort.require_preview(self.case,self.manifest)
        self.review(trial,'fail',observed=[2],retract=True)
        reviewed=project.read_json(workflow.state_path(self.case))['files']['one']['preview_trials'][-1]
        self.assertEqual(reviewed['verdict'],'fail')
        self.assertEqual(reviewed['review_history'][0]['verdict'],'pass')
        with self.assertRaisesRegex(ValueError,'passing preview'):
            effort.require_preview(self.case,self.manifest)
        self.review(trial,'pass',expected_exit=2,retract=True)

    def test_elapsed_time_does_not_end_quality_work_and_policy_cannot_be_removed(self):
        self.case['execution_policy']['started_at']=(datetime.now(timezone.utc)-timedelta(hours=5)).isoformat()
        with workflow.locked_state(self.case) as state:
            state['case_binding']=workflow.case_binding(self.case)
            state['files']['one']['method_started_at']=(datetime.now(timezone.utc)-timedelta(hours=5)).isoformat()
        (self.root/'case.json').write_text(json.dumps(self.case))
        effort.check(self.case,'one',start_method=True)
        self.preview('after45minutes')
        changed=copy.deepcopy(self.case);changed.pop('execution_policy')
        with self.assertRaisesRegex(ValueError,'binding changed'):
            effort.check(changed,'one')

    def test_document_changes_keep_pixels_but_compositor_changes_invalidate(self):
        scripts=self.root/'scripts';scripts.mkdir()
        (scripts/'surface_text.py').write_text('# compositor v1')
        baseline=effort.fingerprint(self.manifest)
        (self.root/'SKILL.md').write_text('clarified guidance')
        (scripts/'unrelated_provider.py').write_text('# unrelated')
        self.assertEqual(effort.fingerprint(self.manifest),baseline)
        (scripts/'surface_text.py').write_text('# compositor v2')
        self.assertNotEqual(effort.fingerprint(self.manifest),baseline)

    def test_opening_preview_cannot_replace_later_motion_risk_and_same_pixels_can_preview_another_range(self):
        self.manifest['risk_preview_ranges_half_open'] = [[0,3],[3,6]]
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        self.review(self.preview('opening',end='0.1'),'pass',observed=[0,1,2])
        with self.assertRaisesRegex(ValueError,'Risk preview ranges'):
            effort.require_preview(self.case,self.manifest)
        self.review(self.preview('later',start='0.1'),'pass',observed=[3,4,5])
        effort.require_preview(self.case,self.manifest)


if __name__ == '__main__':
    unittest.main()
