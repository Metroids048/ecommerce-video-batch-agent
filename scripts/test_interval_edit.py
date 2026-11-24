"""Offline regressions for the actual prepare/submit/compose phase boundaries."""
import copy
import json
from fractions import Fraction
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import project
import workflow
import source_timeline
import runninghub_adapter as provider
import production_pipeline as pipeline
import aigc_composite


class IntervalEdit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.patch = patch.object(project,'ROOT',self.root); self.patch.start()
        (self.root/'config').mkdir(); (self.root/'scripts').mkdir()
        (self.root/'scripts/implementation.py').write_text('original')
        self.config = dict(execution_mode='production',primary=dict(id='kling-o3-std-edit',
            route='/openapi/v2/kling-video-o3-std/video-edit',mode='kling-video-edit',limits=dict(
            max_reference_video_seconds_conservative=15,max_input_plus_output_seconds=30,max_output_seconds=15)))
        self.save_config()
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=96x64:r=359/12',
                        '-frames:v','500','-c:v','libx264',str(self.root/'source.mp4')],check=True,capture_output=True)
        (self.root/'input.txt').write_text('Recolor outer blue to green; preserve print, straps, hands, texture, action and audio.')
        request = dict(batch_id='isolated',user_input='input.txt',files=[dict(id='one',source='source.mp4',operations=[dict(
            id='a',kind='modify',surface_motion='deformable',target='outer blue',desired_change='Grass green RGB 85,160,70',protect=['print','straps','hands','texture','action','audio'],
            recolor_mode='preserve_source_details',target_color_rgb=[85,160,70],semantic_label='outer fabric')])])
        self.write('request.json',request); project.intake('case.json','request.json')
        self.case = project.read_json(self.root/'case.json')
        self.case['execution_policy']['version'] = 2
        # This class covers the historical remote-capable v2 interval gates.
        # Local-only intake/default replay has its own real-entry regressions.
        self.case['execution_policy'].pop('editing_strategy',None)
        self.case['execution_policy']['source_semantic_review_required']=False
        self.case['execution_policy'].pop('local_only',None)
        self.write('case.json', self.case)
        workflow.initialize(self.case,'case.json')
        item=self.case['files'][0]
        self.write('observation.json',dict(source_sha256=item['source_sha256'],operations=dict(a=dict(
            appearance_ranges_half_open=[[0,500]],risk_ranges_half_open=[[0,500]],quality_requirements=['All protection requirements']))))
        pipeline.freeze_route(self.case,dict(file_id='one',observation=dict(path='observation.json',sha256=project.digest(self.root/'observation.json'))))
        with workflow.locked_state(self.case) as state:
            state['files']['one']['preview_trials']=[dict(verdict='fail',diagnosis='Observed straps and lining spill')]
        self.original_binding = workflow.case_binding(self.case)
        self.config['execution_mode']='maintenance';self.save_config()
        pipeline.configure_interval_edit(self.case,dict(file_id='one',operation_id='a',probe_frames_half_open=[100,453],
            user_instruction='Fix mainchain offline; then test hardest interval first without reducing protection.'))
        self.config['execution_mode']='production';self.save_config()

    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def write(self,name,value):
        (self.root/name).write_text(json.dumps(value))

    def save_config(self):
        self.write('config/models.json',self.config)

    def spec(self,span=[100,453],uid='hard'):
        return dict(file_id='one',operation_id='a',unit_id=uid,model_id='kling-o3-std-edit',
                    source_frames_half_open=span,parameters=dict(keepOriginalSound=False),prompt='Recolor outer blue only; preserve all protected details.')

    def test_original_requirements_and_failure_are_retained(self):
        self.assertEqual(workflow.case_binding(self.case),self.original_binding)
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(entry['route_handoff']['routes']['a'],'source_preserving_recolor')
        self.assertEqual(entry['preview_trials'][0]['verdict'],'fail')
        with self.assertRaisesRegex(ValueError,'maintenance'):
            pipeline.configure_interval_edit(self.case,{})

    def test_maintenance_cli_blocks_prepare_and_submit_before_transport(self):
        self.config['execution_mode']='maintenance';self.save_config()
        self.write('prepare.json',dict(action='prepare',**self.spec()))
        with patch.object(provider,'remote_action') as remote,patch('sys.argv',[
                'project.py','provider','--case','case.json','--spec','prepare.json']):
            self.assertEqual(project.main(),2)
            remote.assert_not_called()
        with self.assertRaisesRegex(ValueError,'maintenance'):
            provider.remote_action(self.case,dict(action='submit'))

    def test_real_359_12_source_is_sliced_one_to_one_on_model_clock(self):
        self.config['primary']['limits'].update(min_reference_width=72,min_reference_height=72,max_reference_dimension=4553)
        self.save_config()
        unit=provider.prepare_unit(self.case,self.spec())
        request=provider.verify_unit(self.case,unit)
        ref=source_timeline.build(self.root/request['reference_clip'])
        source=source_timeline.load_for_item(self.case['files'][0])
        self.assertEqual(Fraction(source['fps']),Fraction(359,12))
        self.assertEqual(len(ref['pts']),353);self.assertEqual(Fraction(ref['fps']),24)
        self.assertEqual(request['source_frames_half_open'],[100,453])
        self.assertEqual(request['reference_duration_seconds'],353/24)
        self.assertEqual(request['reference_dimensions'],[108,72])
        metadata=project.probe(self.root/request['reference_clip'])
        self.assertEqual([metadata['width'],metadata['height']],[108,72])
        provider.verify_submission_dimensions(self.case,request,self.config['primary'])
        with self.assertRaisesRegex(ValueError,'dimension validation'):
            provider.verify_submission_dimensions(self.case,dict(request,reference_dimensions=[96,64]),self.config['primary'])
        binding=dict(fixture='explicit offline input/output correspondence')
        audit=dict(method='one_to_one_source_pts',binding=binding,observed_source_frames=list(range(100,453)),
                   reference_sha256=ref['source_sha256'],frame_correspondence='one_to_one_reference_frames',evidence=['synthetic fixture'])
        aigc_composite.validate_frame_index_mapping(source,ref,[100,453],audit,binding,ref)
        with self.assertRaisesRegex(ValueError,'padding/retiming'):
            aigc_composite.validate_frame_index_mapping(source,dict(ref,pts=ref['pts'][:-1]),[100,453],audit,binding,ref)
        with self.assertRaisesRegex(ValueError,'Rate difference'):
            aigc_composite.validate_frame_index_mapping(source,dict(ref,fps='25'),[100,453],audit,binding,ref)

    def test_provider_rejection_dimensions_are_adapted_without_crop(self):
        model=dict(limits=dict(min_reference_width=720,min_reference_height=720,max_reference_dimension=4553))
        self.assertEqual(provider.reference_dimensions(dict(dimensions=[952,540]),model),[1270,720])
        with self.assertRaisesRegex(ValueError,'cropping is forbidden'):
            provider.reference_dimensions(dict(dimensions=[5000,540]),model)

    def test_native_request_rejects_crop_parameter_that_provider_does_not_consume(self):
        with self.assertRaisesRegex(ValueError,'Source crop is not supported'):
            provider.prepare_unit(self.case,dict(self.spec(),source_crop_xywh=[0,0,96,64]))

    def test_old_prepared_size_cannot_submit_or_contaminate_reference_cache(self):
        old=provider.prepare_unit(self.case,self.spec())
        original=provider.verify_unit(self.case,old)
        self.config['primary']['limits'].update(min_reference_width=72,min_reference_height=72)
        self.config.update(remote_calls_allowed=True,media_upload_allowed=True,paid_submission_allowed=True,
                           remote_submission_allowed=True,adapter=dict(contract_current_verified=True))
        self.save_config()
        with self.assertRaisesRegex(ValueError,'dimension validation'):
            provider.remote_action(self.case,dict(action='submit',file_id='one',unit_id='hard',authorization='missing.json'),
                                   transport=lambda p:self.fail('Invalid size reached transport'))
        corrected=provider.prepare_unit(self.case,self.spec(uid='size-corrected'))
        request=provider.verify_unit(self.case,corrected)
        self.assertNotEqual(request['reference_clip'],original['reference_clip'])
        self.assertEqual(request['reference_dimensions'],[108,72])

    def test_old_prepared_native_crop_is_rejected_before_authorization_or_transport(self):
        self.config.update(remote_calls_allowed=True,media_upload_allowed=True,paid_submission_allowed=True,
                           remote_submission_allowed=True,adapter=dict(contract_current_verified=True))
        self.save_config()
        unit=provider.prepare_unit(self.case,self.spec())
        request=provider.verify_unit(self.case,unit);request['source_crop_xywh']=[0,0,96,64]
        self.write(unit['request_path'],request)
        with workflow.locked_state(self.case) as state:
            state['files']['one']['execution_units'][0]['request_fingerprint']=workflow.hash_json(request)
        with self.assertRaisesRegex(ValueError,'Source crop is not supported'):
            provider.remote_action(self.case,dict(action='submit',file_id='one',unit_id='hard',authorization='missing.json'),
                                   transport=lambda p:self.fail('Unsupported crop reached transport'))

    def test_expansion_and_alternate_models_are_blocked_before_hard_pass(self):
        with self.assertRaisesRegex(ValueError,'Hardest interval'):
            provider.prepare_unit(self.case,self.spec([0,353],'expanded'))
        with self.assertRaisesRegex(ValueError,'model is fixed'):
            provider.prepare_unit(self.case,dict(self.spec(),model_id='sam3-video-semantic'))
        with self.assertRaisesRegex(ValueError,'Hardest interval'):
            pipeline.guard_interval_unit(self.case,'one','a','kling-o3-std-edit',[0,353])

    def test_already_prepared_submit_cannot_bypass_hard_review(self):
        self.config.update(remote_calls_allowed=True,media_upload_allowed=True,paid_submission_allowed=True,
                           remote_submission_allowed=True,adapter=dict(contract_current_verified=True))
        self.save_config()
        unit=provider.prepare_unit(self.case,self.spec())
        # Simulate an older prepared request from before this gate existed.
        request=provider.verify_unit(self.case,unit);request['source_frames_half_open']=[0,353]
        self.write(unit['request_path'],request)
        with workflow.locked_state(self.case) as state:
            state['files']['one']['execution_units'][0]['request_fingerprint']=workflow.hash_json(request)
        with self.assertRaisesRegex(ValueError,'Hardest interval'):
            provider.remote_action(self.case,dict(action='submit',file_id='one',unit_id='hard',authorization='missing.json'),
                                   transport=lambda p:self.fail('No transport before hard review'))

    def test_full_bound_hard_pass_releases_only_observed_targets(self):
        unit=provider.prepare_unit(self.case,self.spec())
        self.write('seen.json',dict(scope='synthetic offline all-frame and playback observation'))
        audit=dict(observed_frames=list(range(353)),frame_correspondence='one_to_one_reference_frames',
                   normal_speed_playback=dict(rate=1,ended=True),evidence=[dict(path='seen.json',sha256=project.digest(self.root/'seen.json'))])
        self.write('hard-review.json',audit)
        with workflow.locked_state(self.case) as state:
            u=state['files']['one']['execution_units'][0]
            u['raw']=dict(sha256='synthetic-raw');u['raw_review']=dict(status='pass',raw_sha256='synthetic-raw',
                path='hard-review.json',sha256=project.digest(self.root/'hard-review.json'))
        self.assertIsNotNone(pipeline.guard_interval_unit(self.case,'one','a','kling-o3-std-edit',[0,353]))
        with self.assertRaisesRegex(ValueError,'Only observed target'):
            pipeline.guard_interval_unit(self.case,'one','a','kling-o3-std-edit',[400,753])
        self.write('seen.json',dict(changed=True))
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            pipeline.guard_interval_unit(self.case,'one','a','kling-o3-std-edit',[0,353])

    def test_changed_production_code_or_policy_stops_execution(self):
        self.write('offline.json',dict(returncode=0,scope='synthetic regression receipt'))
        self.config['implementation_lock']=dict(binding=pipeline.implementation_binding(),verification='offline.json',sha256=project.digest(self.root/'offline.json'))
        self.save_config();pipeline.require_production()
        (self.root/'scripts/implementation.py').write_text('changed during production')
        with self.assertRaisesRegex(ValueError,'implementation changed'):
            pipeline.require_production()


if __name__ == '__main__':
    unittest.main()
