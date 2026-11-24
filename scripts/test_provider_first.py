"""Synthetic six-frame tests of the real new-order path; no business or network work."""
import copy
import io
import json
import shutil
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import project
import workflow
import runninghub_adapter as provider
import production_pipeline as pipeline
import execution_plan
import aigc_composite
import test_execution_plan
import test_source_observation


class ProviderFirstTests(unittest.TestCase):
    def setUp(self):
        self.base = test_execution_plan.UnifiedExecution(); self.base.setUp()
        self.root = self.base.root
        code = Path(__file__).resolve().parent
        (self.root/'config').mkdir(); (self.root/'scripts').mkdir()
        shutil.copy2(code.parent/'config/models.json', self.root/'config/models.json')
        for name in ('aigc_composite.py','execution_plan.py'):
            shutil.copy2(code/name,self.root/'scripts'/name)
        req = project.read_json(self.root/'request.json');req['batch_id']='provider-first'
        req['files'][0]['operations'] = [dict(id='a',kind='modify',target='original dark bag',
            desired_change='make it olive green',surface_motion='deformable',
            recolor_mode='preserve_source_details',target_color_rgb=[60,90,40],semantic_label='bag',
            protect=['hands','print','straps'])]
        if self._testMethodName == 'test_bound_asset_reaches_actual_saved_request':
            req['files'][0]['operations'][0]['reference_asset']='asset.png'
        (self.root/'new-request.json').write_text(json.dumps(req))
        project.intake('new/case.json','new-request.json')
        self.case=project.read_json(self.root/'new/case.json')
        self.case['files'][0].update({k:self.base.case['files'][0][k] for k in ('source_timeline','source_timeline_sha256')})
        (self.root/'case.json').write_text(json.dumps(self.case))
        workflow.initialize(self.case,'case.json')
        self.obs=test_source_observation.SourceObservationTests();self.obs.root=self.root;self.obs.case=self.case
        self.obs.prepare();packet=self.obs.packet(second=self._testMethodName == 'test_multiple_instances_still_require_explicit_coverage');packet['operations'].pop('b')
        self.obs.cli(packet)
        self.obs.cli(dict(action='freeze-route',file_id='one'))

    def tearDown(self):
        self.base.tearDown()

    def test_pipeline_rejects_new_colour_repair_without_mutating_workflow(self):
        state_path = workflow.state_path(self.case)
        before = state_path.read_bytes()
        spec = self.root/'blocked-colour-repair.json'
        spec.write_text(json.dumps(dict(action='repair-local-materials', file_id='one',
                                        kind='observed_colour_regions')))
        output = io.StringIO()
        with patch('sys.argv', ['project.py', 'pipeline', '--case', 'case.json', '--spec', str(spec)]):
            with redirect_stdout(output), redirect_stderr(output):
                code = project.main()
        self.assertEqual(code, 2, output.getvalue())
        self.assertIn('existing editor tools', output.getvalue())
        self.assertEqual(state_path.read_bytes(), before)

    def test_scoped_remote_intake_uses_only_matching_batch_authorization(self):
        config = project.read_json(self.root/'config/models.json')
        config.update(execution_mode='production', remote_calls_allowed=False)
        config['scoped_remote_batches'] = {'authorized': {'remote_calls_allowed': True}}
        (self.root/'config/models.json').write_text(json.dumps(config))
        for batch in ('authorized', 'unauthorized'):
            req = project.read_json(self.root/'new-request.json')
            req['batch_id'] = batch
            (self.root/'scope-request.json').write_text(json.dumps(req))
            project.intake(batch+'/case.json', 'scope-request.json')
            saved = project.read_json(self.root/batch/'case.json')
            self.assertEqual(saved['execution_policy'].get('local_only',False), batch != 'authorized')

    def test_new_order_observation_does_not_launch_local_model(self):
        import source_observation
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        entry['source_observation'].pop('receipt',None)
        with patch('semantic_scan.progress',side_effect=AssertionError('No default local model')):
            result=source_observation.progress(self.case,'one',entry)
        self.assertEqual(result['next_action'],'record-source-observation')

    def test_retired_batch_blocks_prepare_and_remote_before_credentials(self):
        (workflow.state_path(self.case).parent/'retired-after-cleanup.json').write_text('{}')
        result=self.cli_provider(dict(action='prepare-native-edit',file_id='one',operation_id='a',unit_prefix='retired'),expected=2)
        self.assertIn('retired after user cleanup',str(result))
        with workflow.locked_state(self.case) as state:
            state['case_binding']['execution_policy'].pop('local_only',None)
        with self.assertRaisesRegex(ValueError,'retired after user cleanup'):
            provider.require_remote('submit',batch_id=self.case['batch_id'])

    def cli_provider(self,spec,expected=0):
        (self.root/'provider.json').write_text(json.dumps(spec))
        out=io.StringIO()
        with patch('sys.argv',['project.py','provider','--case','case.json','--spec','provider.json']),redirect_stdout(out),redirect_stderr(out):
            code=project.main()
        self.assertEqual(code,expected,out.getvalue())
        return json.loads(out.getvalue())

    def prepare(self):
        result=self.cli_provider(dict(action='prepare-native-edit',file_id='one',operation_id='a',unit_prefix='edit',
            historical_replay=True,reference_clip='source.mp4',prompt='Keep the original shot.'))
        self.unit=result['units'][0]
        state=project.read_json(workflow.state_path(self.case))
        self.unit=provider.unit_for(state,'one',self.unit['unit_id'])
        return provider.verify_unit(self.case,self.unit)

    def raw(self,fail=False,partial=False):
        request=self.prepare()
        receipt=dict(route=request['route'],source_sha256=self.case['files'][0]['source_sha256'],
            reference_clip_sha256=project.digest(self.root/'source.mp4'),source_frames_half_open=[0,6],
            body=dict(request['parameters'],prompt=request['prompt']),task_id='synthetic-old-task',status='SUCCESS',
            request_sha256='synthetic',download={'sha256':project.digest(self.root/'source.mp4')})
        (self.root/'receipt.json').write_text(json.dumps(receipt))
        self.cli_provider(dict(action='replay',file_id='one',unit_id=self.unit['unit_id'],receipt='receipt.json',
            raw='source.mp4',provider_task_id='synthetic-old-task',historical_request_sha256='synthetic'))
        (self.root/'observed.txt').write_text('Six synthetic frames: test contract only, no production quality claim.')
        self.evidence=[dict(path='observed.txt',sha256=project.digest(self.root/'observed.txt'))]
        review=dict(raw_sha256=receipt['download']['sha256'],request_fingerprint=self.unit['request_fingerprint'],
            observed_frames=[0,5] if partial else list(range(6)),evidence=self.evidence,
            checks=[dict(kind=k,status='fail' if fail and k=='texture_geometry' else 'pass',scope='synthetic fixture') for k in provider.RAW_CHECKS],
            defects=[dict(source_time=0,category='model_internal',text='Synthetic geometry failure')] if fail else [])
        (self.root/'raw-review.json').write_text(json.dumps(review))
        self.cli_provider(dict(action='review-raw',file_id='one',unit_id=self.unit['unit_id'],evidence='raw-review.json'))
        return dict(action='plan-native-edit',file_id='one',operation_id='a',unit_ids=[self.unit['unit_id']],
            frame_states=['visible']*6,observation_evidence=self.evidence,risk_preview_ranges_half_open=[[0,6]],out='native.json')

    def test_intake_prompt_and_assets_reach_saved_request(self):
        request=self.prepare()
        self.assertEqual(self.case['execution_policy']['editing_strategy'],'provider_first_v1')
        self.assertEqual(self.case['execution_policy']['source_semantic_execution'],'host_agent')
        self.assertEqual(pipeline.route_for(self.case['files'][0]['operations'][0],self.case),'native_video_edit')
        for word in ('original dark bag','olive green','hands','print','straps','60','preserve_source_details'):
            self.assertIn(word,request['prompt'])
        self.assertEqual(request['image_references'],[])
        # Separate real prepared unit tests exact asset consumption, without
        # pretending the legacy no-image receipt proves image provenance.
        asset_request = dict(action='prepare',file_id='one',operation_id='a',unit_id='with-asset',
            model_id='wan-3.0-standard',source_frames_half_open=[0,6],historical_replay=True,
            reference_clip='source.mp4',image_references=['asset.png'])
        # The source fixture intake already binds asset.png only when requested.
        with self.assertRaisesRegex(ValueError, 'not bound'):
            provider.prepare_unit(self.case, asset_request)
        asset_prompt, images = provider.editing_input(self.case['files'][0],
            dict(self.case['files'][0]['operations'][0],replacement_asset='asset.png'),
            dict(source_frames_half_open=[0,6]))
        self.assertEqual(images,['asset.png']); self.assertIn('asset.png',asset_prompt)
        self.assertNotIn('masks',request['prompt'])
        self.assertEqual(request['source_frames_half_open'],[0,6])
        request['prompt']='omit protection'
        path=self.root/self.unit['request_path'];path.write_text(json.dumps(request))
        self.unit['request_fingerprint']=workflow.hash_json(request)
        with self.assertRaisesRegex(ValueError,'lost its target'):
            provider.verify_unit(self.case,self.unit)

    def test_mask_free_refill_and_idempotent_recovery(self):
        spec=self.raw();self.cli_provider(spec)
        manifest=project.read_json(self.root/'native.json')
        execution_plan.validate(self.case,manifest)
        plan=project.read_json(self.root/'native-pixels.json')
        self.assertFalse(any('masks' in p for p in plan['parts']))
        self.assertTrue(all(p['blend_mode']=='native_replace' for p in plan['parts']))
        report=dict(modified_source_frames=[],modified_instance_count=0)
        layer=aigc_composite.Layer(plan,report)
        try:
            out,touched=layer.compose(np.zeros((64,96,3),np.uint8),0)
            self.assertGreater(out.mean(),200);self.assertTrue(touched.all())
        finally:layer.close()
        before=(self.root/'native-pixels.json').read_bytes()
        with patch.object(aigc_composite,'prepare_plan',side_effect=AssertionError('must reuse')):
            recovered=self.cli_provider(spec)
        self.assertTrue(recovered['assembly_reused'])
        self.assertEqual(before,(self.root/'native-pixels.json').read_bytes())

    def test_bound_asset_reaches_actual_saved_request(self):
        request=self.prepare()
        self.assertEqual(request['image_references'],['asset.png'])
        self.assertIn('asset.png',request['prompt'])
        with self.assertRaisesRegex(ValueError,'Identical request'):
            provider.prepare_unit(self.case,dict(file_id='one',operation_id='a',unit_id='renamed-asset',
                model_id='wan-3.0-standard',source_frames_half_open=[0,6],historical_replay=True,
                reference_clip='source.mp4',prompt='Keep the original shot.'))

    def test_multiple_instances_still_require_explicit_coverage(self):
        spec=self.raw()
        spec['frame_records']=[dict(frame=n,state='visible',semantic_instance_ids=['a-front','a-rear']) for n in range(6)]
        self.cli_provider(spec)
        manifest=project.read_json(self.root/'native.json');execution_plan.validate(self.case,manifest)
        plan=project.read_json(self.root/'native-pixels.json')
        plan['review']['records'][3]['semantic_instance_ids']=['a-front']
        (self.root/'native-pixels.json').write_text(json.dumps(plan))
        manifest['operations'][0]['plan_sha256']=project.digest(self.root/'native-pixels.json')
        with self.assertRaisesRegex(ValueError,'explicit per-frame plan coverage'):
            execution_plan.validate(self.case,manifest)

    def test_normal_preview_export_and_candidate_recovery(self):
        spec=self.raw();self.cli_provider(spec)
        with self.assertRaisesRegex(ValueError,'passing preview'):
            workflow.execute_plan(self.case,'native.json','full')
        with patch('sys.argv',['project.py','preview','--case','case.json','--plan','native.json',
                '--out','preview.mp4','--start','0','--end','0.2']),redirect_stdout(io.StringIO()):
            self.assertEqual(project.main(),0)
        trial=project.read_json(workflow.state_path(self.case))['files']['one']['preview_trials'][-1]
        review=dict(file_id='one',trial_id=trial['id'],observed_source_frames=list(range(6)),
            checks={k:dict(status='pass',scope='six synthetic white frames only') for k in ('target','protection','temporal')},
            evidence=self.evidence,diagnosis='Synthetic transport fixture only',next_change='none',
            retract_previous_pass=False,operation_ids=['a'])
        (self.root/'preview-review.json').write_text(json.dumps(review))
        with patch('sys.argv',['project.py','preview-review','--case','case.json','--spec','preview-review.json']),redirect_stdout(io.StringIO()):
            self.assertEqual(project.main(),0)
        result=workflow.execute_plan(self.case,'native.json','full')
        self.assertFalse(workflow.quality_reviews_pass(result))
        sidecar=project.read_json(project.project_path(result['sidecar']))
        self.assertTrue(sidecar['timeline_exact']);self.assertTrue(sidecar['audio_packets_pts_payload_exact'])
        self.assertFalse(result['accepted'])
        with patch.object(execution_plan,'render',side_effect=AssertionError('must reuse original')):
            self.assertTrue(workflow.execute_plan(self.case,'native.json','full')['execution_reused'])

    def test_failed_raw_not_refilled_or_retried_under_new_name(self):
        spec=self.raw(fail=True)
        self.cli_provider(spec,expected=2)
        self.assertFalse((self.root/'native-pixels.json').exists())
        rejected=self.cli_provider(dict(action='prepare',unit_id='renamed',file_id='one',operation_id='a',
            model_id='wan-3.0-standard',source_frames_half_open=[0,6],historical_replay=True,reference_clip='source.mp4'),expected=2)
        self.assertIn('failed',str(rejected).lower())
        status=pipeline.status(self.case)
        self.assertEqual(status['files'][0]['next_action'],'report_defect_and_choose_scoped_action')

    def test_sampled_raw_review_is_not_full_review(self):
        spec=self.raw(partial=True)
        self.cli_provider(spec,expected=2)
        self.assertFalse((self.root/'native-pixels.json').exists())

    def test_direct_prompt_preserves_old_requests_and_checks_new_request(self):
        old = self.prepare()
        new = self.cli_provider(dict(action='prepare',file_id='one',operation_id='a',unit_id='direct',
            model_id='wan-3.0-standard',source_frames_half_open=[0,6],historical_replay=True,
            reference_clip='source.mp4',prompt_format=2,prompt='Change the dark bag to olive green.'))
        state=project.read_json(workflow.state_path(self.case))
        request=provider.verify_unit(self.case,provider.unit_for(state,'one','direct'))
        self.assertTrue(request['prompt'].startswith('Change the dark bag to olive green.'))
        for word in ('original dark bag','olive green','hands','print','straps','60'):
            self.assertIn(word,request['prompt'])
        self.assertEqual(provider.verify_unit(self.case,provider.unit_for(state,'one',self.unit['unit_id'])),old)

    def test_diagnosed_correction_consumes_only_exact_failed_parent(self):
        self.raw(fail=True)
        self.case['execution_policy']['version']=3
        (self.root/'case.json').write_text(json.dumps(self.case))
        with workflow.locked_state(self.case) as state:
            state['case_binding']=workflow.case_binding(self.case)
        config=project.read_json(self.root/'config/models.json')
        config['execution_mode']='maintenance'
        (self.root/'config/models.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,'needs the exact failed raw'):
            pipeline.configure_interval_edit(self.case,dict(file_id='one',operation_id='a',user_instruction='Fix original failure.'))
        with workflow.locked_state(self.case) as state:
            unit=state['files']['one']['execution_units'][0]
            state['files']['one']['interval_edit_route']=dict(operation_id='a',
                dependency=workflow.requirement_binding(self.case,self.case['files'][0]),
                parent_failure=dict(unit_id=unit['unit_id'],request_fingerprint=unit['request_fingerprint'],
                    raw_sha256=unit['raw']['sha256'],review_sha256=unit['raw_review']['sha256']))
        pipeline.check_method_failure(self.case,'one','a')
        pipeline.check_method_failure(self.case,'one')
        with workflow.locked_state(self.case) as state:
            other=copy.deepcopy(state['files']['one']['execution_units'][0]);other['unit_id']='new-failed-correction'
            state['files']['one']['execution_units'].append(other)
        with self.assertRaisesRegex(ValueError,'Provider return or candidate failed'):
            pipeline.check_method_failure(self.case,'one','a')

    def test_mask_path_removed_for_new_orders(self):
        spec=self.raw();spec['target_matte']='not-needed.npz'
        error=self.cli_provider(spec,expected=2)
        self.assertIn('does not use masks',str(error))


if __name__=='__main__':unittest.main()
