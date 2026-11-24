"""Regress the actual export/repair/resume seams, without remote work or old media edits."""
import argparse
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import execution_plan
import delivery_package
import production_effort
import production_pipeline as pipeline
import project
import source_recolor
import video_segmentation
import workflow
import runninghub_adapter as provider
from test_production_pipeline import PipelineTests


class ConvergenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = PipelineTests()
        self.fixture.setUp()
        self.root, self.case, self.manifest = self.fixture.root, self.fixture.case, self.fixture.manifest

    def tearDown(self):
        self.fixture.tearDown()

    def strict_operation(self):
        op = self.case['files'][0]['operations'][0]
        op.clear()
        op.update(id='a', kind='modify', target='cloth', protect=['print', 'straps'],
                  surface_motion='deformable', recolor_mode='preserve_source_details',
                  target_color_rgb=[210, 50, 40], semantic_label='backpack fabric')
        self.manifest['dependency'] = workflow.requirement_binding(self.case, self.case['files'][0])
        with workflow.locked_state(self.case) as state:
            state['case_binding'] = workflow.case_binding(self.case)

    def mask_unit(self):
        item = self.case['files'][0]
        request = dict(model_id='sam3-video-semantic', operation_id='a',
                       source_frames_half_open=[0, 6], parameters=dict(width=96, height=64))
        unit = dict(unit_id='seg', operation_id='a', request_fingerprint='frozen', status='raw_ready',
                    raw=dict(path=item['source'], sha256=item['source_sha256']))
        with workflow.locked_state(self.case) as state:
            state['files']['one']['execution_units'] = [unit]
        return request

    def test_semantic_repair_consumes_original_failure_without_unlocking_grid(self):
        self.strict_operation()
        review_path=self.root/'failed-raw.json'
        review=dict(checks=[dict(kind='coverage',status='fail')],diagnosis='Zero detections',
                    semantic_repair=dict(target='backpack',straps='backpack shoulder straps'))
        review_path.write_text(json.dumps(review))
        prior=dict(unit_id='failed',operation_id='a',raw=dict(sha256='raw'),
                   raw_review=dict(path='failed-raw.json',sha256=project.digest(review_path),
                                   status='fail_or_limited',raw_sha256='raw'))
        with workflow.locked_state(self.case) as state:
            state['files']['one']['execution_units']=[prior]
        spec=dict(file_id='one',semantic_repair=dict(failed_unit_id='failed',role='target'))
        operation=self.case['files'][0]['operations'][0]
        request=dict(model_id='sam3-video-semantic')
        with patch.object(provider,'verify_unit',return_value=request):
            fixed,deps=provider.semantic_repair_parameters(self.case,spec,operation)
            original=video_segmentation.fixed_recolor_parameters(self.case['files'][0],operation)
            self.assertEqual(fixed,dict(original,text_prompt='backpack'))
            self.assertEqual(deps,{'failed-raw.json':project.digest(review_path)})
            with self.assertRaisesRegex(ValueError,'frozen'):
                provider.semantic_repair_parameters(self.case,dict(spec,semantic_repair=dict(failed_unit_id='failed',role='unplanned')),operation)
            request['semantic_repair']=spec['semantic_repair']
            with self.assertRaisesRegex(ValueError,'trial chains'):
                provider.semantic_repair_parameters(self.case,spec,operation)
            part=dict(spec,semantic_repair=dict(failed_unit_id='failed',role='straps'))
            fixed,_=provider.semantic_repair_parameters(self.case,part,operation)
            self.assertEqual(fixed['text_prompt'],'backpack shoulder straps')
            review.update(observed_frames=[3],semantic_anchor=dict(frame=3,text_prompt='backpack'))
            review_path.write_text(json.dumps(review))
            with workflow.locked_state(self.case) as state:
                state['files']['one']['execution_units'][0]['raw_review']['sha256']=project.digest(review_path)
            request['parameters']=dict(text_prompt='backpack')
            anchored=dict(spec,semantic_repair=dict(failed_unit_id='failed',role='target',anchor_frame=3))
            fixed,deps=provider.semantic_repair_parameters(self.case,anchored,operation)
            self.assertEqual(fixed,dict(original,text_prompt='backpack',anchor_frame=3))
            with self.assertRaisesRegex(ValueError,'observed visible frame'):
                provider.semantic_repair_parameters(self.case,dict(anchored,semantic_repair=dict(
                    failed_unit_id='failed',role='target',anchor_frame=4)),operation)
            request['parameters']['anchor_frame']=3
            with self.assertRaisesRegex(ValueError,'trial chains'):
                provider.semantic_repair_parameters(self.case,anchored,operation)
            point_anchor=dict(frame=3,point_prompt=dict(positive=[dict(x=20,y=20)],negative=[]))
            review['instance_point_anchors']={'strap':point_anchor}
            review_path.write_text(json.dumps(review))
            with workflow.locked_state(self.case) as state:
                state['files']['one']['execution_units'][0]['raw_review']['sha256']=project.digest(review_path)
            point_spec=dict(spec,semantic_repair=dict(failed_unit_id='failed',role='strap',conditioning='instance_points'))
            fixed,_=provider.semantic_repair_parameters(self.case,point_spec,operation)
            self.assertEqual(fixed['point_prompt'],point_anchor['point_prompt'])
            self.assertEqual(fixed['anchor_frame'],3)
            request['parameters']['point_prompt']=point_anchor['point_prompt']
            with self.assertRaisesRegex(ValueError,'trial chains'):
                provider.semantic_repair_parameters(self.case,point_spec,operation)
            request['parameters'].pop('point_prompt')
            request['parameters'].pop('anchor_frame')
            request.pop('semantic_repair')
            review_path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                provider.semantic_repair_parameters(self.case,spec,operation)

    def test_common_export_cannot_switch_frozen_preserving_route(self):
        self.strict_operation()
        self.case['files'][0]['operations'][0]['replacement_asset']='asset.png'
        self.manifest['dependency']=workflow.requirement_binding(self.case,self.case['files'][0])
        with workflow.locked_state(self.case) as state:
            state['case_binding']=workflow.case_binding(self.case)
        with self.assertRaisesRegex(ValueError, 'source-preserving route'):
            execution_plan.validate(self.case, self.manifest)

    def strict_plan(self):
        self.strict_operation()
        item = self.case['files'][0]
        (self.root/'observation.txt').write_text('Observed six source frames')
        mask = self.root/'semantic.npz'
        np.savez(mask, alpha=np.full((6, 64, 96), 255, np.uint8), protected=np.zeros((6,64,96),np.uint8),
                 source_sha256=np.array(item['source_sha256']), source_dimensions=[96,64],
                 mask_dimensions=[96,64], frames_half_open=[0,6])
        return dict(operation_id='a', dependency=workflow.requirement_binding(self.case,item),
                    composition_mode='source_recolor', color_rgb=[210,50,40], color_asset=None,
                    recolor_mode='hsv_hue', segmentation_backend='sam3-video-semantic',
                    full_appearance_required=True, source_frame_count=6, source_frames_half_open=[0,6],
                    edit_frames_half_open=[0,6], masks='semantic.npz',
                    artifact_hashes={'semantic.npz':project.digest(mask)},
                    review=dict(frame_states=['visible']*6, records=[dict(frame=n,state='visible') for n in range(6)],
                                evidence=[dict(path='observation.txt',sha256=project.digest(self.root/'observation.txt'))]))

    def test_shared_recolor_validation_pins_color_backend_mode_coverage_and_evidence(self):
        plan = self.strict_plan(); item = self.case['files'][0]
        # Import's source binding is exercised separately with a real decoded fixture.
        with patch.object(video_segmentation,'validate_import'):
            self.assertIn('observation.txt', source_recolor.validate(self.case,item,plan))
        for change in (dict(color_rgb=[0,0,255]),dict(recolor_mode='hsv_chroma'),
                       dict(segmentation_backend='sam2-video-segmentation'),dict(full_appearance_required=False)):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError,'Source-preserving'):
                source_recolor.validate(self.case,item,dict(plan,**change))
        (self.root/'observation.txt').write_text('different observation')
        with self.assertRaisesRegex(ValueError,'observation changed'):
            source_recolor.validate(self.case,item,plan)

    def test_failed_a_preview_cannot_reopen_b_material(self):
        self.fixture.pin()
        with workflow.locked_state(self.case) as state:
            e=state['files']['one']
            e['preview_trials']=[dict(verdict='fail',diagnosis='a edge spill',operation_ids=['a'],
                                     handoff_fingerprint=e['material_handoff']['fingerprint'])]
        changed=copy.deepcopy(self.manifest)
        p=self.root/'b.json';p.write_text(p.read_text()+'\n')
        changed['operations'][1]['plan_sha256']=project.digest(p)
        artifacts=execution_plan.validate(self.case,changed)[3]
        with self.assertRaisesRegex(ValueError,'unaffected operation'):
            pipeline.pin_materials(self.case,changed,artifacts)

    def test_reopened_a_requires_an_explicit_operation_id(self):
        self.fixture.pin();self.fixture.issue()
        pipeline.reopen(self.case,dict(file_id='one',issue_id='observed-defect'))
        for op in (None,'b'):
            with self.subTest(op=op), self.assertRaisesRegex(ValueError,'diagnosed operation'):
                pipeline.guard_upstream(self.case,'one',operation_id=op)
        pipeline.guard_upstream(self.case,'one',operation_id='a')

    def test_real_preview_failure_must_scope_its_operations(self):
        # Use the actual common compositor and its report, not a manufactured pass.
        self.case['execution_policy']=dict(version=2,max_failed_previews_per_method=2)
        with workflow.locked_state(self.case) as state:
            state['case_binding']=workflow.case_binding(self.case)
        execution_plan.preview(self.case,self.manifest,self.root/'trial.mp4',0,.2)
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        spec=dict(file_id='one',trial_id=entry['preview_trials'][-1]['id'],observed_source_frames=[0],
                  checks={k:dict(status='fail',scope='a edge') for k in ('target','protection','temporal')},
                  diagnosis='a edge spill',next_change='correct a mask',
                  evidence=[dict(path='input.txt',sha256=project.digest(self.root/'input.txt'))])
        with self.assertRaisesRegex(ValueError,'affected operation_ids'):
            production_effort.review_preview(self.case,spec)
        production_effort.review_preview(self.case,dict(spec,operation_ids=['a']))
        reviewed=project.read_json(workflow.state_path(self.case))['files']['one']['preview_trials'][-1]
        self.assertEqual(reviewed['operation_ids'],['a'])

    def test_mask_resume_reuses_bound_import_without_decoding_again(self):
        request=self.mask_unit()
        with patch.object(video_segmentation.provider,'verify_unit',return_value=request):
            first=video_segmentation.import_masks(self.case,dict(file_id='one',unit_id='seg'),'import.npz')
            original=(self.root/'import.npz').read_bytes()
            with patch.object(video_segmentation.cv2,'VideoCapture',side_effect=AssertionError('repeat decode')):
                second=video_segmentation.import_masks(self.case,dict(file_id='one',unit_id='seg'),'import.npz')
            self.assertTrue(second['execution_reused'])
            self.assertEqual(first['masks_sha256'],second['masks_sha256'])
            self.assertEqual(original,(self.root/'import.npz').read_bytes())
            side=self.root/'import.npz.json';report=project.read_json(side)
            report['source_sha256']='wrong';side.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'source-bound'):
                video_segmentation.import_masks(self.case,dict(file_id='one',unit_id='seg'),'import.npz')

    def test_exact_recolor_prepare_reuses_plan_and_rejects_changed_requirements(self):
        self.strict_operation();request=self.mask_unit()
        with patch.object(video_segmentation.provider,'verify_unit',return_value=request):
            video_segmentation.import_masks(self.case,dict(file_id='one',unit_id='seg'),'import.npz')
            spec=dict(file_id='one',operation_id='a',segmentation_import='import.npz.json',
                      frame_states=['visible']*6,evidence=[dict(path='input.txt',sha256=project.digest(self.root/'input.txt'))],
                      color_rgb=[210,50,40],out='recolor.json',risk_preview_ranges_half_open=[[0,6]],
                      editable_source_hsv=dict(hue_range=[90,140],minimum_saturation=50))
            source_recolor.prepare_plan(self.case,spec)
            self.assertEqual(project.read_json(self.root/'recolor-pixels.json')['editable_source_hsv'],spec['editable_source_hsv'])
            before=[(self.root/p).read_bytes() for p in ('recolor.json','recolor-pixels.json')]
            self.assertTrue(source_recolor.prepare_plan(self.case,spec)['execution_reused'])
            self.assertEqual(before,[(self.root/p).read_bytes() for p in ('recolor.json','recolor-pixels.json')])
            with self.assertRaisesRegex(ValueError,'frozen RGB'):
                source_recolor.prepare_plan(self.case,dict(spec,color_rgb=[1,2,3]))
            with self.assertRaisesRegex(ValueError,'Material colour gate'):
                source_recolor.prepare_plan(self.case,dict(spec,editable_source_hsv=dict(hue_range=[140,90],minimum_saturation=50)))

    def test_timing_summary_unions_overlap_and_exposes_unrecorded_time(self):
        journal=self.root/'runs/unified';journal.mkdir(exist_ok=True)
        t=lambda n:f'2026-11-14T00:00:{n:02d}+00:00'
        commands=[dict(command='provider',provider_action='query',started_at=t(5),completed_at=t(15),
                       work_class='per_video',exit_code=2)]
        stages=[dict(event='phase_start',event_id='a',phase='engineering',work_class='one_time_engineering',at=t(0)),
                dict(event='phase_end',event_id='a',phase='engineering',work_class='one_time_engineering',at=t(10)),
                dict(event='phase_start',event_id='b',phase='final_review',work_class='per_video',at=t(20)),
                dict(event='phase_end',event_id='b',phase='final_review',work_class='per_video',at=t(25))]
        for name,rows in [('timings.jsonl',commands),('stage-events.jsonl',stages)]:
            (journal/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        summary=project.timing_summary(self.case,until=t(30))
        self.assertEqual(summary['observed_wall_seconds'],30)
        self.assertEqual(summary['recorded_activity_union_seconds'],20)
        self.assertEqual(summary['unattributed_wall_seconds'],10)
        self.assertEqual(summary['failed_commands'],1)
        self.assertEqual(summary['work_class_union_seconds']['one_time_engineering'],10)
        self.assertEqual(summary['work_class_union_seconds']['per_video'],15)

    def test_delivery_carries_stage_measurements_and_exact_snapshot(self):
        started=project.record_stage_timing(self.case,dict(action='start',phase='engineering'))
        project.record_stage_timing(self.case,dict(action='end',event_id=started['event_id']))
        state=project.read_json(workflow.state_path(self.case))
        report=dict(revision=state['revision'],files=[],original_goal_complete=False)
        with patch.object(workflow,'delivery_report',return_value=report):
            delivery_package.build(self.case,'handoff',diagnostic=True)
            metadata=project.read_json(self.root/'handoff/package.json')
            self.assertIn('one_time_engineering',metadata['timing_summary']['work_class_union_seconds'])
            self.assertEqual((self.root/'handoff/evidence/stage-events.jsonl').read_bytes(),
                             (self.root/'runs/unified/stage-events.jsonl').read_bytes())
            before=(self.root/'handoff/package.json').read_bytes()
            self.assertTrue(delivery_package.build(self.case,'handoff',diagnostic=True)['execution_reused'])
            self.assertEqual(before,(self.root/'handoff/package.json').read_bytes())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report');args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ConvergenceTests)
    for module in ('test_production_pipeline','test_source_recolor','test_stage_timing','test_production_effort','test_video_segmentation','test_runninghub_adapter'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    baseline=project.read_json(project.ROOT/'evidence/project-convergence-20261114/production-baseline.json')
    changed=[p for p,h in baseline['files'].items() if project.digest(project.project_path(p))!=h]
    if args.report:
        report=dict(status='pass' if result.wasSuccessful() and not changed else 'fail',
                    tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                    protected_files=len(baseline['files']),changed_production_files=changed,
                    scope='Isolated export/repair/resume/timing regressions; no new business video or remote call',
                    native_video_quality='NOT_RUN',fresh_video_latency='NOT_RUN',accepted=False)
        project.project_path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return 0 if result.wasSuccessful() and not changed else 1


if __name__=='__main__':
    raise SystemExit(main())
