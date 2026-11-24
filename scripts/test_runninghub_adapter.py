import copy
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
import project
import workflow
import runninghub_adapter as provider
import aigc_composite
import execution_plan
import test_execution_plan as fixtures


class ProviderBranch(unittest.TestCase):
    setUp = fixtures.UnifiedExecution.setUp
    tearDown = fixtures.UnifiedExecution.tearDown

    def test_source_preservation_repair_links_only_failed_matching_raw_for_recheck(self):
        self.setup_provider();self.raw_and_plan(fail=True)
        state=project.read_json(workflow.state_path(self.case));issue=state['files']['one']['issues'][0]
        plan=dict(composition_mode='source_recolor',segmentation_backend='sam3-video-semantic',full_appearance_required=True)
        (self.root/'preserving.json').write_text(json.dumps(plan))
        (self.root/'preserving-manifest.json').write_text(json.dumps(dict(operations=[dict(operation_id='a',plan='preserving.json')])))
        candidate=dict(id='preserving',plan='preserving-manifest.json',status='pending_agent_quality_review',accepted=False)
        with workflow.locked_state(self.case) as state:state['files']['one']['candidates'].append(candidate)
        spec=dict(action='link-preservation-repair',file_id='one',candidate_id='preserving',issue_ids=[issue['id']])
        with patch('production_effort.preservation_fallback',return_value=True), \
             patch('source_recolor.validate'), \
             patch.object(workflow,'verify_candidate',side_effect=lambda c,f,x:x), \
             patch.object(provider,'verify_unit',return_value=dict(operation_id='a',model_id='kling-o3-std-edit')):
            with self.assertRaisesRegex(ValueError,'Unrelated'):
                provider.dispatch(self.case,dict(spec,issue_ids=['unrelated']))
            result=provider.dispatch(self.case,spec)
            self.assertEqual(result['status'],'pending_actual_candidate_recheck')
            saved=project.read_json(workflow.state_path(self.case))['files']['one']
            self.assertEqual(saved['issues'][0]['status'],'pending_recheck')
            self.assertEqual(saved['candidates'][0]['status'],'pending_agent_quality_review')
            self.assertEqual(saved['execution_units'][0]['raw_review']['status'],'fail_or_limited')
            (self.root/'raw-review.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                provider.dispatch(self.case,spec)

    def test_collect_never_submits_when_original_task_id_missing(self):
        self.setup_provider()
        with patch.object(provider,'remote_action') as remote:
            with self.assertRaisesRegex(ValueError,'Missing original task'):
                provider.dispatch(self.case,dict(action='collect-native-recolor',file_id='one',unit_ids=['u1']))
            remote.assert_not_called()

    def test_collect_nonblocking_queries_independent_units_concurrently(self):
        import threading
        self.setup_provider()
        with workflow.locked_state(self.case) as state:
            original=state['files']['one']['execution_units'][0]
            original.update(provider_task_id='original-1',status='RUNNING')
            state['files']['one']['execution_units'].append(dict(original,unit_id='u2',provider_task_id='original-2'))
        barrier=threading.Barrier(2)
        def query(case,spec):
            self.assertEqual(spec['action'],'query')
            barrier.wait(timeout=2)
            return dict(status='RUNNING')
        with patch.object(provider,'remote_action',side_effect=query) as remote, \
             patch('time.sleep',side_effect=AssertionError('nonblocking collection must return')):
            result=provider.dispatch(self.case,dict(action='collect-native-recolor',file_id='one',unit_ids=['u1','u2'],wait=False))
        self.assertEqual(result['pending'],['u1','u2'])
        self.assertEqual(result['status'],'waiting_original_tasks')
        self.assertEqual(remote.call_count,2)

    def test_collect_preflights_all_original_ids_before_any_remote_work(self):
        self.setup_provider()
        with workflow.locked_state(self.case) as state:
            original=state['files']['one']['execution_units'][0]
            original.update(provider_task_id='original-1',status='RUNNING')
            state['files']['one']['execution_units'].append(dict(original,unit_id='u2',provider_task_id=None))
        with patch.object(provider,'remote_action') as remote:
            with self.assertRaisesRegex(ValueError,'Missing original task'):
                provider.dispatch(self.case,dict(action='collect-native-recolor',file_id='one',unit_ids=['u1','u2'],wait=False))
            remote.assert_not_called()

    def test_collect_creates_observation_immediately_and_reuses_bound_evidence(self):
        import review_queue
        self.setup_provider()
        provider.replay(self.case,self.replay)
        spec=dict(action='collect-native-recolor',file_id='one',unit_ids=['u1'],observation_out='observations')
        with patch.object(provider,'remote_action') as remote, patch('builtins.print'):
            result=provider.dispatch(self.case,spec)
            self.assertEqual(result['collected'],['u1'])
            remote.assert_not_called()
            inventory=project.read_json(self.root/'observations/u1/inventory.json')
            self.assertEqual(inventory['paired_frames'],list(range(6)))
            self.assertEqual(inventory['status'],'pending_actual_agent_observation')
            with patch.object(review_queue,'build_pairs') as build:
                provider.dispatch(self.case,spec)
                build.assert_not_called()
            (self.root/inventory['evidence'][0]['path']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                provider.dispatch(self.case,spec)

    def test_native_contexts_cover_full_source_without_padding_or_segmentation(self):
        for count in (81, 90, 353, 354, 690, 756, 891, 1440):
            windows = provider.native_contexts(count, '24000/1001')
            covered = set()
            for lo, hi in windows:
                self.assertGreaterEqual(lo, 0)
                self.assertLessEqual(hi, count)
                self.assertEqual((hi-lo-1) % 16, 0)
                self.assertLessEqual((hi-lo)*1001/24000, 15)
                covered.update(range(lo, hi))
            self.assertEqual(covered, set(range(count)))
        self.assertEqual(provider.native_contexts(690, '24000/1001'), [[0,353],[337,690]])
        with self.assertRaisesRegex(ValueError, 'no frame conversion'):
            provider.native_contexts(690, '25')

    def test_native_recolor_dispatch_prepares_only_native_source_units(self):
        self.case['files'][0]['operations'][0].update(kind='modify', surface_motion='deformable')
        with patch.object(provider.source_timeline, 'load_for_item', return_value={'fps':'24000/1001','pts':list(range(690))}), \
             patch.object(provider.source_timeline, 'require_cfr', side_effect=lambda x:x), \
             patch.object(provider, 'prepare_unit', side_effect=lambda c,s:s) as prepare:
            result = provider.dispatch(self.case, dict(action='prepare-native-recolor',
                file_id='one', operation_id='a', unit_prefix='native', prompt='Recolor the fabric'))
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual({u['model_id'] for u in result['units']}, {'kling-o3-std-edit'})
        self.assertEqual(result['source_windows'], [[0,353],[337,690]])

    def test_preserve_recolor_reuses_fixed_full_source_semantic_graph_without_trials(self):
        self.case['files'][0]['dimensions']=[1080,1920]
        self.case['files'][0]['operations'][0].update(kind='modify',surface_motion='deformable',
            recolor_mode='preserve_source_details',target_color_rgb=[0,71,171],semantic_label='backpack')
        with patch.object(provider.source_timeline,'load_for_item',return_value={'fps':'24','pts':list(range(799))}), \
             patch.object(provider.source_timeline,'require_cfr',side_effect=lambda x:x), \
             patch.object(provider,'prepare_unit',side_effect=lambda c,s:s) as prepare:
            result=provider.dispatch(self.case,dict(action='prepare-recolor',file_id='one',
                operation_id='a',unit_prefix='tracking'))
        self.assertEqual(prepare.call_count,1)
        unit=result['units'][0]
        self.assertEqual(unit['model_id'],'sam3-video-semantic')
        self.assertEqual(unit['source_frames_half_open'],[0,799])
        self.assertEqual(unit['parameters'],dict(width=576,height=1024,text_prompt='backpack'))
        self.assertEqual(result['new_model_calls'],0)

    def test_color_balance_removes_bright_patch_without_copying_source_word(self):
        source=np.full((64,96,3),160,np.uint8)
        source[25:31,30:55]=40
        raw=np.full_like(source,178);raw[22:24,25:27]=174
        alpha=np.zeros((64,96),np.uint8);alpha[15:45,15:75]=255
        alpha[14,15:75]=128
        corrected=aigc_composite.match_masked_flat_color(source,raw,alpha)
        self.assertTrue(np.all(corrected[27,40]==160))
        self.assertTrue(np.all(corrected[22,25]==156))
        self.assertTrue(np.all(corrected[14,40]==160))
        self.assertTrue(np.array_equal(corrected[alpha==0],raw[alpha==0]))
        output,effective=__import__('regional_composite').compose_frame(source,corrected,alpha)
        self.assertTrue(np.array_equal(output[effective==0],source[effective==0]))

    def setup_provider(self):
        code_root=Path(__file__).resolve().parents[1]
        (self.root/'config').mkdir();(self.root/'scripts').mkdir()
        config=json.loads((code_root/'config/models.json').read_text())
        config['execution_mode']='production'  # Isolated synthetic fixture, no live transport.
        config.pop('implementation_lock',None)  # Real project hashes/receipt do not describe this temporary fixture.
        (self.root/'config/models.json').write_text(json.dumps(config))
        for name in ('aigc_composite.py','execution_plan.py'):
            shutil.copy2(code_root/'scripts'/name,self.root/'scripts'/name)
        workflow.initialize(self.case,'case.json')
        self.spec=dict(action='prepare',unit_id='u1',file_id='one',operation_id='a',
            model_id='wan-3.0-standard',source_frames_half_open=[0,6],
            historical_replay=True,reference_clip='source.mp4',prompt='Change only a')
        self.unit=provider.prepare_unit(self.case,self.spec)
        request=project.read_json(project.project_path(self.unit['request_path']))
        self.receipt=dict(route=request['route'],source_sha256=self.case['files'][0]['source_sha256'],
            reference_clip_sha256=project.digest(self.root/'source.mp4'),source_frames_half_open=[0,6],
            body=dict(request['parameters'],prompt=request['prompt']),task_id='old-task',status='SUCCESS',
            request_sha256='historical',download={'sha256':project.digest(self.root/'source.mp4')})
        (self.root/'receipt.json').write_text(json.dumps(self.receipt))
        self.replay=dict(action='replay',file_id='one',unit_id='u1',receipt='receipt.json',raw='source.mp4',
            provider_task_id='old-task',historical_request_sha256='historical')

    def raw_and_plan(self,fail=False):
        provider.replay(self.case,self.replay)
        (self.root/'observation.txt').write_text('Synthetic test observation, not business acceptance')
        evidence=[dict(path='observation.txt',sha256=project.digest(self.root/'observation.txt'))]
        review=dict(raw_sha256=self.receipt['download']['sha256'],request_fingerprint=self.unit['request_fingerprint'],
            observed_frames=[0,5],evidence=evidence,checks=[dict(kind=k,status='fail' if fail else 'pass',scope='test fixture') for k in provider.RAW_CHECKS],
            defects=[dict(source_time=0,category='model_internal',text='fixture failure')] if fail else [])
        (self.root/'raw-review.json').write_text(json.dumps(review))
        provider.review_raw(self.case,dict(file_id='one',unit_id='u1',evidence='raw-review.json'))
        alpha=np.zeros((6,64,96),np.uint8);alpha[:,5:15,5:15]=255
        np.savez(self.root/'raw-mask.npz',alpha=alpha,raw_to_source=np.repeat(np.eye(3)[None],6,0))
        spec=dict(file_id='one',operation_id='a',parts=[dict(unit_id='u1',masks='raw-mask.npz',masks_sha256=project.digest(self.root/'raw-mask.npz'))],
            frame_states=['visible']*6,observation_evidence=evidence,diagnostic=fail,out='aigc.json')
        aigc_composite.prepare_plan(self.case,spec)
        manifest=copy.deepcopy(self.manifest)
        manifest['operations'][0].update(plan='aigc.json',plan_sha256=project.digest(self.root/'aigc.json'))
        (self.root/'aigc-manifest.json').write_text(json.dumps(manifest))
        return manifest,spec

    def test_nominal_rate_rounding_needs_bound_full_frame_audit(self):
        source=dict(fps='24000/1001',time_base='1/24000',pts=[n*1001 for n in range(353)])
        raw=dict(fps='24',time_base='1/24',pts=list(range(353)))
        binding=dict(source_sha256='source',raw_sha256='raw',request_fingerprint='request',source_frames_half_open=[0,353])
        mapping=dict(method='one_to_one_source_pts',binding=binding,
                     observed_source_frames=list(range(353)),evidence=[dict(path='actual-observation.jpg',sha256='fixture')])
        aigc_composite.validate_frame_index_mapping(source,raw,[0,353],mapping,binding)
        for changed in [dict(mapping,observed_source_frames=[0,352]),dict(mapping,binding={}),dict(mapping,evidence=[])]:
            with self.assertRaisesRegex(ValueError,'correspondence audit'):
                aigc_composite.validate_frame_index_mapping(source,raw,[0,353],changed,binding)
        # The original 119 -> 113 provider failure stays rejected, even with an audit.
        with self.assertRaisesRegex(ValueError,'padding/retiming forbidden'):
            aigc_composite.validate_frame_index_mapping(source,dict(raw,pts=list(range(347))),[0,353],mapping,binding)
        with self.assertRaisesRegex(ValueError,'Rate difference'):
            aigc_composite.validate_frame_index_mapping(source,dict(raw,fps='25'),[0,353],mapping,binding)
        with self.assertRaisesRegex(ValueError,'half a source frame'):
            aigc_composite.validate_frame_index_mapping(source,dict(raw,pts=[n+1 for n in range(353)]),[0,353],mapping,binding)

    def test_mask_cache_loads_only_active_part_and_supports_backward_preview(self):
        self.setup_provider();self.raw_and_plan()
        plan=project.read_json(self.root/'aigc.json');part=plan['parts'][0]
        part['write_frames_half_open']=[0,3]
        second=copy.deepcopy(part);second['write_frames_half_open']=[3,6]
        shutil.copy2(self.root/'raw-mask.npz',self.root/'second-mask.npz');second['masks']='second-mask.npz'
        plan['parts'].append(second)
        report=dict(modified_source_frames=[],modified_instance_count=0)
        layer=aigc_composite.Layer(plan,report)
        try:
            self.assertIsNone(layer.mask_cache)
            frame=np.zeros((64,96,3),np.uint8)
            for n,path in [(0,'raw-mask.npz'),(3,'second-mask.npz'),(1,'raw-mask.npz')]:
                result,union=layer.compose(frame,n)
                self.assertIn(path,layer.mask_cache)
                self.assertLessEqual(len(layer.mask_cache),2)
                self.assertEqual(int(union.sum()),100)
                self.assertFalse(np.any(result[~union]!=frame[~union]))
        finally:layer.close()
        self.assertIsNone(layer.mask_cache)

    def test_all_remote_actions_block_before_transport_or_credential(self):
        self.setup_provider()
        for action in ('submit','query','download','quote','upload'):
            with patch.object(provider,'remote_action', wraps=provider.remote_action):
                with self.assertRaisesRegex(ValueError,'All remote calls blocked'):
                    provider.remote_action(self.case,dict(action=action),transport=lambda _:self.fail('network touched'))

    def test_exact_replay_rejects_wrong_source_parameters_task_or_raw(self):
        self.setup_provider()
        for field,value in [('source_sha256','wrong'),('task_id','other'),('source_frames_half_open',[1,7])]:
            bad=dict(self.receipt,**{field:value});(self.root/'receipt.json').write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'binding mismatch'):provider.replay(self.case,self.replay)
        (self.root/'receipt.json').write_text(json.dumps(self.receipt))
        provider.replay(self.case,self.replay)
        with self.assertRaisesRegex(ValueError,'fresh'):provider.replay(self.case,self.replay)
        state=project.read_json(workflow.state_path(self.case));u=state['files']['one']['execution_units'][0]
        self.assertEqual(u['new_model_calls'],0);self.assertEqual(u['mode'],'historical_offline_replay')

    def test_unreviewed_raw_cannot_enter_compositor(self):
        self.setup_provider();provider.replay(self.case,self.replay)
        with self.assertRaisesRegex(ValueError,'reviewed before'):
            aigc_composite.prepare_plan(self.case,dict(file_id='one',operation_id='a',parts=[dict(unit_id='u1')]))

    def test_explicit_diagnostic_export_in_maintenance_preserves_failed_verdict(self):
        self.setup_provider();manifest,_=self.raw_and_plan(fail=True)
        manifest['operations'][1] = dict(operation_id='b', disposition='blocked', blocked_reason='Not covered by cached return')
        (self.root/'diagnostic-manifest.json').write_text(json.dumps(manifest))
        (self.root/'config/models.json').write_text(json.dumps(dict(execution_mode='maintenance')))
        with self.assertRaisesRegex(ValueError,'maintenance'):
            workflow.execute_plan(self.case,'diagnostic-manifest.json','ordinary')
        # Production preview rejection must remain active without blocking an
        # explicitly unqualified review return through the nested compositor.
        with patch('production_effort.require_preview', side_effect=ValueError('No passing production preview')):
            candidate=workflow.execute_plan(self.case,'diagnostic-manifest.json','review-return',diagnostic=True)
        self.assertFalse(workflow.quality_reviews_pass(candidate))
        self.assertTrue(candidate['operation_results'][0]['diagnostic'])
        self.assertFalse(candidate['operation_results'][0]['raw_reviews_pass'])
        self.assertFalse(candidate['accepted'])
        import delivery_package
        with self.assertRaisesRegex(ValueError,'Unqualified'):
            delivery_package.build(self.case,'normal')
        result=delivery_package.build(self.case,'review',diagnostic=True)
        self.assertEqual(result['status'],'failure_diagnostic')
        again=workflow.execute_plan(self.case,'diagnostic-manifest.json','review-return',diagnostic=True)
        self.assertEqual(again['new_renders'],0)

    def test_diagnostic_flag_rejects_ordinary_plans(self):
        workflow.initialize(self.case,'case.json')
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'explicitly diagnostic'):
            workflow.execute_plan(self.case,'manifest.json','review-return',diagnostic=True)

    def region_review_fixture(self, spec):
        state=project.read_json(workflow.state_path(self.case))
        unit=provider.unit_for(state,'one','u1');part=spec['parts'][0]
        report=dict(binding=dict(file_id='one',operation_id='a',source_sha256=self.case['files'][0]['source_sha256'],
            raw_sha256=unit['raw']['sha256'],request_fingerprint=unit['request_fingerprint'],
            raw_review_sha256=unit['raw_review']['sha256'],masks_sha256=part['masks_sha256'],
            write_frames_half_open=[0,6],blend_mode='alpha',color_match='none',
            composition_context_sha256=aigc_composite.region_context(state,self.case['files'][0],spec['parts'],spec['frame_states'],'none')),
            observed_source_frames=list(range(6)),evidence=spec['observation_evidence'],
            checks=[dict(kind=k,status='pass',scope='Synthetic written-region fixture only') for k in provider.RAW_CHECKS])
        (self.root/'region-review.json').write_text(json.dumps(report))
        spec['parts'][0]['write_region_review']='region-review.json'
        return report

    def test_local_write_region_review_keeps_raw_failure_and_requires_final_review(self):
        self.setup_provider();manifest,spec=self.raw_and_plan(fail=True)
        self.region_review_fixture(spec)
        spec.update(out='regional.json',diagnostic=False)
        aigc_composite.prepare_plan(self.case,spec)
        manifest['operations'][0].update(plan='regional.json',plan_sha256=project.digest(self.root/'regional.json'))
        (self.root/'regional-manifest.json').write_text(json.dumps(manifest))
        candidate=workflow.execute_plan(self.case,'regional-manifest.json','region-v1')
        self.assertFalse(workflow.quality_reviews_pass(candidate))
        self.assertFalse(candidate['operation_results'][0]['raw_reviews_pass'])
        state=project.read_json(workflow.state_path(self.case));issue=state['files']['one']['issues'][0]
        evidence=spec['observation_evidence']
        qa=dict(file_id='one',candidate_id=candidate['id'],source_sha256=candidate['source_sha256'],
            output_sha256=candidate['output_sha256'],dependency=candidate['dependency'],expected_revision=state['revision'],
            checks=[dict(kind=k,status='pass',scope='Synthetic final-review fixture') for k in ('target_coverage','visual','playback','delivery')],
            operation_reviews=[dict(operation_id=op,output_sha256=candidate['output_sha256'],dependency=candidate['dependency'],
                checks=[dict(kind=k,status='pass',scope='Synthetic final pixels only',evidence=evidence)
                    for k in ('target_coverage','visual','protection','temporal')]) for op in ('a','b')],
            resolved_issue_ids=[issue['id']])
        (self.root/'final-review.json').write_text(json.dumps(qa))
        result=workflow.record_agent_review(self.case,'final-review.json')
        self.assertEqual(result['status'],'qualified')
        state=project.read_json(workflow.state_path(self.case))
        self.assertEqual(state['files']['one']['issues'][0]['status'],'closed')
        self.assertEqual(state['files']['one']['execution_units'][0]['raw_review']['status'],'fail_or_limited')
        self.assertFalse(result['accepted'])
        state['files']['one']['execution_units'][0].update(reserved_cny=7,new_model_calls=1)
        workflow.state_path(self.case).write_text(json.dumps(state))
        import delivery_package
        delivery_package.build(self.case,'qualified-handoff')
        metadata=project.read_json(self.root/'qualified-handoff/package.json')
        self.assertEqual(metadata['packaging_only_new_cost_cny'],0)
        self.assertIsNone(metadata['batch_cost']['new_cost_cny'])
        self.assertEqual(metadata['batch_cost']['reserved_total_cny'],7)
        (self.root/'region-review.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'dependency hash changed'):
            workflow.verify_candidate(self.case,'one',state['files']['one']['candidates'][0])

    def test_write_region_review_rejects_wrong_mask_or_incomplete_observation(self):
        self.setup_provider();_,spec=self.raw_and_plan(fail=True)
        review=self.region_review_fixture(spec);spec.update(out='regional.json',diagnostic=False)
        for invalid in (dict(review,binding=dict(review['binding'],masks_sha256='other')),
                        dict(review,observed_source_frames=[0,5]),
                        dict(review,binding=dict(review['binding'],composition_context_sha256='old-neighbor-or-code'))):
            (self.root/'region-review.json').write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError,'Write-region review'):
                aigc_composite.prepare_plan(self.case,spec)

    def test_failed_raw_full_candidate_and_same_queue_remain_unqualified(self):
        self.setup_provider();manifest,_=self.raw_and_plan(fail=True)
        c=workflow.execute_plan(self.case,'aigc-manifest.json','diagnostic-v1')
        self.assertEqual(c['operation_ids'],['a','b']);self.assertFalse(workflow.quality_reviews_pass(c))
        side=project.read_json(project.project_path(c['sidecar']));self.assertEqual(side['output_frame_count'],6)
        self.assertTrue(side['timeline_exact']);self.assertEqual(side['operations'][0]['method'],'aigc_regional')
        state=project.read_json(workflow.state_path(self.case));self.assertEqual(state['files']['one']['issues'][0]['observation_basis'],'provider_raw')
        raw=workflow.snapshot(self.case)['files']['one']['raw_returns'][0]
        self.assertEqual(raw['id'],'u1');self.assertEqual(raw['source_seconds'],[0,0.2])
        self.assertEqual(raw['quality_status'],'fail_or_limited')
        from delivery_package import build
        with self.assertRaises(ValueError):build(self.case,'delivery')
        self.assertFalse((self.root/'delivery').exists())

    def test_two_methods_overlap_is_rejected(self):
        self.setup_provider();manifest,spec=self.raw_and_plan()
        plan=project.read_json(self.root/'b.json')
        for r in plan['review']['records']:r['source_quad']=[[5,5],[35,5],[35,21],[5,21]]
        (self.root/'b.json').write_text(json.dumps(plan));manifest['operations'][1]['plan_sha256']=project.digest(self.root/'b.json')
        with self.assertRaisesRegex(ValueError,'overlap'):execution_plan.render(self.case,manifest,self.root/'overlap.mp4')

    def test_raw_verdict_cannot_be_overridden_and_parent_is_verified(self):
        self.setup_provider();_,_=self.raw_and_plan(fail=True)
        p=project.read_json(self.root/'aigc.json');p['raw_reviews_pass']=True
        with self.assertRaisesRegex(ValueError,'cannot be overridden'):
            aigc_composite.validate(self.case,self.case['files'][0],p)
        with self.assertRaises(ValueError):provider.prepare_unit(self.case,dict(self.spec,unit_id='bad-parent',parent_candidate_id='nonexistent'))

    def test_raw_defects_after_export_stay_in_same_queue(self):
        self.setup_provider();self.raw_and_plan(fail=True)
        c=workflow.execute_plan(self.case,'aigc-manifest.json','diagnostic-v1')
        provider.review_raw(self.case,dict(file_id='one',unit_id='u1',evidence='raw-review.json'))
        state=project.read_json(workflow.state_path(self.case));issue=state['files']['one']['issues'][-1]
        self.assertEqual(issue['candidate_id'],c['id']);self.assertEqual(issue['observation_basis'],'provider_raw')

    def test_raw_hash_drift_invalidates_manifest(self):
        self.setup_provider();manifest,_=self.raw_and_plan()
        (self.root/'raw-review.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'artifact changed'):execution_plan.validate(self.case,manifest)

    def test_timing_mismatch_cannot_be_padded(self):
        self.setup_provider();_,spec=self.raw_and_plan()
        spec['out']='other.json'
        import source_timeline
        original=source_timeline.build
        def altered(path):
            result=original(path);result['fps']='24';return result
        with patch.object(source_timeline,'build',side_effect=altered):
            with self.assertRaisesRegex(ValueError,'retiming forbidden'):aigc_composite.prepare_plan(self.case,spec)

    def test_unknown_submit_is_reserved_and_never_retried(self):
        self.setup_provider()
        # Test-only isolated policy/contract/consent. Production policy is untouched.
        p=provider.policy();p.update(remote_calls_allowed=True,media_upload_allowed=True,paid_submission_allowed=True,remote_submission_allowed=True)
        p['adapter']['contract_current_verified']=True;(self.root/'config/models.json').write_text(json.dumps(p))
        # A live transport test needs a schema-valid integer-duration input;
        # historical replay fixtures intentionally retain their 0.2s duration.
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=white:s=96x64:r=30:d=2',
                        '-c:v','libx264',str(self.root/'live-source.mp4')],check=True,capture_output=True)
        req=project.read_json(self.root/'request.json');req['batch_id']='live-contract';req['files'][0]['source']='live-source.mp4'
        (self.root/'live-request.json').write_text(json.dumps(req));project.intake('live/case.json','live-request.json')
        self.case=project.read_json(self.root/'live/case.json')
        self.case['execution_policy']['version']=2
        # Frozen legacy transport fixture: source-observation semantics have
        # dedicated real-entry tests; this test exercises unknown submission.
        self.case['execution_policy'].pop('editing_strategy', None)
        self.case['execution_policy']['source_semantic_review_required']=False
        (self.root/'live/case.json').write_text(json.dumps(self.case))
        import source_timeline
        (self.root/'live-timeline.json').write_text(json.dumps(source_timeline.build(self.root/'live-source.mp4')))
        self.case['files'][0].update(source_timeline='live-timeline.json',source_timeline_sha256=project.digest(self.root/'live-timeline.json'))
        workflow.initialize(self.case,'live/case.json')
        observation=dict(source_sha256=self.case['files'][0]['source_sha256'],operations={
            op['id']:dict(appearance_ranges_half_open=[[0,60]],risk_ranges_half_open=[[0,60]],
                          quality_requirements=['Preserve source and verify original return'])
            for op in self.case['files'][0]['operations']})
        observed=self.root/'live-observation.json';observed.write_text(json.dumps(observation))
        from production_pipeline import freeze_route
        freeze_route(self.case,dict(file_id='one',observation=dict(path='live-observation.json',sha256=project.digest(observed))))
        spec=dict(self.spec,unit_id='live',historical_replay=False,source_frames_half_open=[0,60],prompt='A different explicitly revised request');spec.pop('reference_clip')
        provider.prepare_unit(self.case,spec)
        (self.root/'grant.json').write_text(json.dumps(dict(batch_id=self.case['batch_id'],remote_authorized=True,user_authorization_evidence='test-only',total_budget_cny=10,unit_caps_cny={'live':5},full_cost_confirmed=True)))
        call=dict(action='submit',file_id='one',unit_id='live',authorization='grant.json',maximum_charge_cny=5)
        def fail(_):raise ValueError('simulated lost reply')
        with self.assertRaisesRegex(ValueError,'lost reply'):provider.remote_action(self.case,call,transport=fail)
        state=project.read_json(workflow.state_path(self.case));u=provider.unit_for(state,'one','live')
        self.assertEqual(u['status'],'submission_unknown');self.assertEqual(u['reserved_cny'],5)
        with self.assertRaisesRegex(ValueError,'Already attempted'):provider.remote_action(self.case,call,transport=lambda _:self.fail('resubmitted'))

    def test_scoped_policy_does_not_enable_other_batches(self):
        self.setup_provider()
        provider.require_remote('submit','three-scenarios-fresh-010')
        for batch in (None, self.case['batch_id'], 'ecommerce-batch-002'):
            with self.assertRaisesRegex(ValueError,'All remote calls blocked'):
                provider.require_remote('submit',batch)

    def test_text_erasure_model_is_scoped_and_has_no_generation_parameters(self):
        self.setup_provider()
        config=provider.policy()
        scoped=config['scoped_remote_batches']['complex-batch-012']
        config['scoped_remote_batches'][self.case['batch_id']]=scoped
        (self.root/'config/models.json').write_text(json.dumps(config))
        self.assertIsNone(provider.model_for('volc-text-erase-pro'))
        self.assertIsNotNone(provider.model_for('volc-text-erase-pro',self.case['batch_id']))
        uid='text-test'
        spec=dict(self.spec,unit_id=uid,model_id='volc-text-erase-pro',parameters=dict(
            eraseType='text',encodeMode='quality',clientToken=self.case['batch_id']+'-'+uid))
        unit=provider.prepare_unit(self.case,spec)
        request=project.read_json(project.project_path(unit['request_path']))
        self.assertEqual(set(request['parameters']),{'eraseType','encodeMode','clientToken'})
        spec['unit_id']='bad-text-test';spec['parameters']['duration']='5'
        with self.assertRaisesRegex(ValueError,'erasure parameters'):
            provider.prepare_unit(self.case,spec)

    def test_video_editor_uses_its_own_schema_and_preserves_task_receipts(self):
        self.setup_provider()
        config=provider.policy()
        config['scoped_remote_batches'][self.case['batch_id']]=config['scoped_remote_batches']['complex-batch-012']
        (self.root/'config/models.json').write_text(json.dumps(config))
        self.assertIsNone(provider.model_for('rhart-video-g-edit'))
        spec=dict(self.spec,unit_id='editor',model_id='rhart-video-g-edit',parameters={'resolution':'720p'},
                  historical_replay=False,prompt='Erase only product lettering; preserve hands and displays')
        spec.pop('reference_clip')
        provider.prepare_unit(self.case,spec)
        (self.root/'grant.json').write_text(json.dumps(dict(batch_id=self.case['batch_id'],remote_authorized=True,
            user_authorization_evidence='test-only',total_budget_cny=30,unit_caps_cny={'editor':21},full_cost_confirmed=True)))
        call=dict(file_id='one',unit_id='editor',authorization='grant.json')
        provider.remote_action(self.case,dict(call,action='submit',maximum_charge_cny=21),
            transport=lambda payload:dict(status='QUEUED',task_id='new-test-task',quote_cny=2.05))
        unit=provider.remote_action(self.case,dict(call,action='query'),
            transport=lambda payload:dict(status='RUNNING',task_id='new-test-task'))
        self.assertEqual([r['action'] for r in unit['receipts']],['submit','query'])
        self.assertEqual(unit['receipts'][0]['response']['quote_cny'],2.05)
        unit=provider.remote_action(self.case,dict(call,action='download'),
            transport=lambda payload:dict(status='SUCCESS',task_id='new-test-task',raw_path=str(self.root/'source.mp4')))
        self.assertEqual(unit['status'],'raw_ready')
        unit=provider.remote_action(self.case,dict(call,action='query'),
            transport=lambda payload:dict(status='SUCCESS',task_id='new-test-task',consumeMoney='0.1'))
        self.assertEqual(unit['status'],'raw_ready')
        self.assertEqual(unit['raw']['sha256'],project.digest(self.root/'source.mp4'))
        unit=provider.remote_action(self.case,dict(call,action='query'),
            transport=lambda payload:dict(status='query_unknown',task_id='new-test-task',
                failure_stage='query',create_attempted=False,error='fixture HTTP failure'))
        self.assertEqual(unit['status'],'raw_ready')
        self.assertEqual(unit['provider_task_id'],'new-test-task')
        self.assertEqual(unit['reserved_cny'],21)
        self.assertEqual(unit['new_model_calls'],1)
        self.assertEqual(unit['last_receipt']['failure_stage'],'query')
        spec['unit_id']='bad-editor';spec['parameters']['duration']='5'
        with self.assertRaisesRegex(ValueError,'video edit parameters'):provider.prepare_unit(self.case,spec)

    def test_kling_editor_uses_native_schema_and_is_scoped_to_new_batch(self):
        self.setup_provider()
        config=provider.policy()
        config['scoped_remote_batches'][self.case['batch_id']]=config['scoped_remote_batches']['segmentation-benchmark-014']
        (self.root/'config/models.json').write_text(json.dumps(config))
        self.assertIsNone(provider.model_for('kling-o3-std-edit'))
        spec=dict(self.spec,unit_id='kling',model_id='kling-o3-std-edit',
                  parameters={'keepOriginalSound':False},historical_replay=False)
        spec.pop('reference_clip')
        unit=provider.prepare_unit(self.case,spec)
        request=project.read_json(project.project_path(unit['request_path']))
        self.assertEqual(request['parameters'],{'keepOriginalSound':False})
        (self.root/'grant.json').write_text(json.dumps(dict(batch_id=self.case['batch_id'],remote_authorized=True,
            user_authorization_evidence='test-only',total_budget_cny=10,unit_caps_cny={'kling':10},full_cost_confirmed=True)))
        submitted=provider.remote_action(self.case,dict(action='submit',file_id='one',unit_id='kling',
            authorization='grant.json',maximum_charge_cny=10),
            transport=lambda payload:dict(status='RUNNING',task_id='fixture-kling'))
        self.assertEqual(submitted['status'],'RUNNING')
        spec['unit_id']='wrong-kling';spec['parameters']['duration']=5
        with self.assertRaisesRegex(ValueError,'Kling edit parameters'):
            provider.prepare_unit(self.case,spec)

    def test_sam2_prompt_change_reuses_source_clip_and_duplicate_rejects_before_io(self):
        self.setup_provider()
        config=provider.policy()
        config['scoped_remote_batches'][self.case['batch_id']]=config['scoped_remote_batches']['segmentation-benchmark-014']
        (self.root/'config/models.json').write_text(json.dumps(config))
        spec=dict(self.spec,unit_id='sam-a',model_id='sam2-video-base-plus',historical_replay=False,
            parameters=dict(width=192,height=128,anchors=[dict(frame=0,
                positive_points=[dict(x=30,y=30)],negative_points=[])]))
        spec.pop('reference_clip')
        a=provider.prepare_unit(self.case,spec)
        spec['unit_id']='sam-b'; spec['parameters']['anchors'][0]['positive_points'][0]['x']=31
        with patch.object(provider.subprocess,'run',wraps=provider.subprocess.run) as invoked:
            b=provider.prepare_unit(self.case,spec)
        self.assertFalse(any('-vf' in call.args[0] for call in invoked.call_args_list), 'cached source clip was transcoded again')
        ra=project.read_json(project.project_path(a['request_path']))
        rb=project.read_json(project.project_path(b['request_path']))
        self.assertEqual(ra['reference_clip'],rb['reference_clip'])
        spec['unit_id']='sam-renamed'
        with self.assertRaisesRegex(ValueError,'Identical'):
            provider.prepare_unit(self.case,spec)
        self.assertFalse((self.root/'runs'/self.case['batch_id']/'units'/'one'/'sam-renamed').exists())

    def test_empty_task_receipt_stays_unknown_and_cannot_be_resubmitted(self):
        self.setup_provider()
        config=provider.policy()
        config['scoped_remote_batches'][self.case['batch_id']]=config['scoped_remote_batches']['complex-batch-012']
        (self.root/'config/models.json').write_text(json.dumps(config))
        spec=dict(self.spec,unit_id='empty-task',model_id='rhart-video-g-edit',parameters={'resolution':'720p'},
                  historical_replay=False,prompt='Erase product lettering and preserve hands')
        spec.pop('reference_clip');provider.prepare_unit(self.case,spec)
        (self.root/'grant.json').write_text(json.dumps(dict(batch_id=self.case['batch_id'],remote_authorized=True,
            user_authorization_evidence='test-only',total_budget_cny=30,unit_caps_cny={'empty-task':21},full_cost_confirmed=True)))
        call=dict(action='submit',file_id='one',unit_id='empty-task',authorization='grant.json',maximum_charge_cny=21)
        reply=dict(status='',task_id='',provider_error_code='UPSTREAM_ERROR',provider_error_message='fixture rejection')
        unit=provider.remote_action(self.case,call,transport=lambda _:reply)
        self.assertEqual(unit['status'],'submission_unknown')
        self.assertEqual(unit['reserved_cny'],21)
        self.assertEqual(unit['receipts'][0]['response'],reply)
        with self.assertRaisesRegex(ValueError,'Already attempted'):
            provider.remote_action(self.case,call,transport=lambda _:self.fail('duplicate submission'))
        with self.assertRaisesRegex(ValueError,'No original task'):
            provider.remote_action(self.case,dict(call,action='query'),transport=lambda _:self.fail('query without task'))

    def test_precreate_failure_is_saved_without_charge_or_retry(self):
        self.setup_provider()
        config=provider.policy()
        config['scoped_remote_batches'][self.case['batch_id']]=config['scoped_remote_batches']['complex-batch-012']
        (self.root/'config/models.json').write_text(json.dumps(config))
        spec=dict(self.spec,unit_id='empty-task',model_id='rhart-video-g-edit',parameters={'resolution':'720p'},
                  historical_replay=False,prompt='Erase product lettering and preserve hands')
        spec.pop('reference_clip');provider.prepare_unit(self.case,spec)
        (self.root/'grant.json').write_text(json.dumps(dict(batch_id=self.case['batch_id'],remote_authorized=True,
            user_authorization_evidence='test-only',total_budget_cny=30,unit_caps_cny={'empty-task':21},full_cost_confirmed=True)))
        call=dict(action='submit',file_id='one',unit_id='empty-task',authorization='grant.json',maximum_charge_cny=21)
        reply=dict(status='submission_not_created',create_attempted=False,failure_stage='quote',error='Full input/output cost outside reserved budget')
        unit=provider.remote_action(self.case,call,transport=lambda _:reply)
        self.assertEqual(unit['status'],'submission_not_created')
        self.assertEqual(unit['new_model_calls'],0)
        self.assertEqual(unit['reserved_cny'],0)
        self.assertEqual(unit['receipts'][0]['response'],reply)
        with self.assertRaisesRegex(ValueError,'Already attempted'):
            provider.remote_action(self.case,call,transport=lambda _:self.fail('duplicate submission'))
        with self.assertRaisesRegex(ValueError,'No original task'):
            provider.remote_action(self.case,dict(call,action='query'),transport=lambda _:self.fail('query without task'))

    def test_fractional_live_duration_blocked_before_reservation(self):
        self.setup_provider()
        p=provider.policy('three-scenarios-fresh-010');(self.root/'config/models.json').write_text(json.dumps(p))
        spec=dict(self.spec,unit_id='invalid-live',historical_replay=False,prompt='Distinct invalid live contract request');spec.pop('reference_clip')
        provider.prepare_unit(self.case,spec)
        (self.root/'grant.json').write_text(json.dumps(dict(batch_id=self.case['batch_id'],remote_authorized=True,user_authorization_evidence='test-only',total_budget_cny=10,unit_caps_cny={'invalid-live':5},full_cost_confirmed=True)))
        with self.assertRaisesRegex(ValueError,'integer seconds'):
            provider.remote_action(self.case,dict(action='submit',file_id='one',unit_id='invalid-live',authorization='grant.json',maximum_charge_cny=5),transport=lambda _:self.fail('network touched'))
        u=provider.unit_for(project.read_json(workflow.state_path(self.case)),'one','invalid-live')
        self.assertEqual(u['status'],'prepared');self.assertNotIn('reserved_cny',u)

if __name__=='__main__':unittest.main()
