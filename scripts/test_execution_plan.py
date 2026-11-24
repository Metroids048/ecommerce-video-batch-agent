import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import project
import workflow
import execution_plan
import source_timeline


class CoverageFindings(unittest.TestCase):
    def test_moving_written_plane_is_not_a_replacement_dropout(self):
        q=[[10,10],[110,10],[110,30],[10,30]]
        moved=[[10,25],[110,25],[110,45],[10,45]]
        rows=[{'frame':n,'instances':[{'state':'visible','source_quad':p}]}
              for n,p in enumerate([q,moved,q])]
        self.assertEqual(execution_plan.coverage_findings(rows)['replacement_dropout_risks'],[])

    def test_known_block_and_short_track_dropout_are_exposed(self):
        q=[[10,10],[60,10],[60,30],[10,30]]
        rows=[{'frame':0,'instances':[{'state':'visible','source_quad':q}]},
              {'frame':1,'instances':[{'state':'unresolved','word_quad':q,'blocked_reason':'bounds'}]},
              {'frame':2,'instances':[]},
              {'frame':3,'instances':[{'state':'visible','source_quad':q}]}]
        findings=execution_plan.coverage_findings(rows)
        self.assertEqual(findings['known_unedited_instances'][0]['frame'],1)
        self.assertEqual(findings['replacement_dropout_risks'],[[1,3]])

    def test_two_instances_do_not_hide_an_unedited_one(self):
        q=[[10,10],[60,10],[60,30],[10,30]]
        result=execution_plan.coverage_findings([{'frame':0,'instances':[
            {'state':'visible','source_quad':q},
            {'state':'unresolved','word_quad':q,'blocked_reason':'occlusion'}]}])
        self.assertEqual(len(result['known_unedited_instances']),1)


class UnifiedExecution(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.patch=patch.object(project,'ROOT',self.root);self.patch.start()
        (self.root/'input.txt').write_text('Replace both printed labels; keep the full clip.')
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=white:s=96x64:r=30:d=0.2','-c:v','libx264','-video_track_timescale','1000000',str(self.root/'source.mp4')],check=True,capture_output=True)
        asset=np.zeros((16,32,4),np.uint8);asset[:,:,3]=255
        cv2.imwrite(str(self.root/'asset.png'),asset)
        req={'batch_id':'unified','user_input':'input.txt','files':[{'id':'one','source':'source.mp4','operations':[{'id':x,'kind':'replace','target':x,'protect':['rest'],'replacement_asset':'asset.png'} for x in ('a','b')]}]}
        (self.root/'request.json').write_text(json.dumps(req));project.intake('case.json','request.json');self.case=project.read_json(self.root/'case.json')
        # These regressions exercise frozen pre-budget manifests. New intake
        # effort/preview enforcement has separate real-CLI tests.
        self.case.pop('execution_policy')
        (self.root/'case.json').write_text(json.dumps(self.case))
        t=source_timeline.build(self.root/'source.mp4');(self.root/'timeline.json').write_text(json.dumps(t));item=self.case['files'][0];item.update(source_timeline='timeline.json',source_timeline_sha256=project.digest(self.root/'timeline.json'))
        self.manifest={'schema':'execution_manifest_v1','file_id':'one','dependency':workflow.requirement_binding(self.case,item),'operations':[]}
        for op,x in [('a',5),('b',60)]:
            mask=op+'.npz';np.savez(self.root/mask,alpha=np.full((6,16,32),255,np.uint8),protected=np.zeros((6,16,32),np.uint8))
            plan={'file_id':'one','case_version':self.case['case_version'],'source_sha256':item['source_sha256'],'operation_id':op,'source_frame_count':6,'source_frames_half_open':[0,6],'composition_mode':'exact_asset_on_source_label','mask_space':'label_plane','masks':mask,'masks_sha256':project.digest(self.root/mask),'exact_asset':'asset.png','exact_asset_sha256':project.digest(self.root/'asset.png'),'label_style':{'plane_size':[32,16],'polarity':'dark','asset_fit':[.8,.6]},'review':{'frame_states':['visible']*6,'records':[{'frame':n,'state':'visible','source_quad':[[x,20],[x+30,20],[x+30,36],[x,36]]} for n in range(6)]}}
            (self.root/(op+'.json')).write_text(json.dumps(plan));self.manifest['operations'].append({'operation_id':op,'disposition':'ready','plan':op+'.json','plan_sha256':project.digest(self.root/(op+'.json'))})
    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def review_before_repair(self, candidate):
        # Exercise the real review entry before the existing diagnosed repair.
        cap=cv2.VideoCapture(str(self.root/candidate['output']));count=0
        while cap.read()[0]:count+=1
        cap.release();self.assertEqual(count,6)
        evidence=self.root/'fixture-observation.txt'
        evidence.write_text('Decoded six local fixture frames; operation a receives a composition correction. Synthetic workflow evidence only.')
        scope='All six local fixture frames; technical fixture observation, not production visual acceptance'
        artifact=dict(path=evidence.name,sha256=project.digest(evidence))
        review=dict(expected_revision=project.read_json(workflow.state_path(self.case))['revision'],
            file_id='one',candidate_id=candidate['id'],source_sha256=candidate['source_sha256'],
            output_sha256=candidate['output_sha256'],dependency=candidate['dependency'],
            checks=[dict(kind=k,status='reviewed_with_limits' if k=='delivery' else 'fail' if k=='visual' else 'pass',scope=scope)
                    for k in ('target_coverage','visual','playback','delivery')],
            operation_reviews=[dict(operation_id=op,output_sha256=candidate['output_sha256'],dependency=candidate['dependency'],
                checks=[dict(kind=k,status='fail' if op=='a' and k=='visual' else 'pass',scope=scope,evidence=[artifact])
                        for k in ('target_coverage','visual','protection','temporal')]) for op in ('a','b')])
        (self.root/'fixture-review.json').write_text(json.dumps(review))
        workflow.record_agent_review(self.case,'fixture-review.json')
    def test_preview_cli_preserves_encoded_video_instead_of_json(self):
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        with patch('sys.argv',['project.py','preview','--case','case.json','--plan','manifest.json',
                               '--out','trial.mp4','--start','0','--end','0.1']):
            self.assertEqual(project.main(),0)
        cap=cv2.VideoCapture(str(self.root/'trial.mp4'))
        self.assertTrue(cap.isOpened());self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),3)
        cap.release()
        self.assertTrue((self.root/'trial.mp4.json').exists())
        journal=self.root/'runs/unified/timings.jsonl'
        event=json.loads(journal.read_text().splitlines()[-1])
        self.assertEqual(event['command'],'preview');self.assertEqual(event['exit_code'],0)
        self.assertGreater(event['elapsed_seconds'],0)
        with patch('sys.argv',['project.py','preview','--case','case.json','--plan','manifest.json',
                               '--out','bad.mp4','--start','0.15','--end','0.1']):
            self.assertEqual(project.main(),2)
        self.assertEqual(json.loads(journal.read_text().splitlines()[-1])['exit_code'],2)

    def test_surface_reprepare_keeps_all_geometry_and_rejects_wrong_source(self):
        import text_localization
        (self.root/'scripts').mkdir()
        (self.root/'scripts/surface_text.py').write_bytes(Path(__file__).with_name('surface_text.py').read_bytes())
        parent=project.read_json(self.root/'a.json')
        parent['spec']={'method':'on_device_text','reference_frame':0,
                        'reference_quad':[[5,20],[35,20],[35,36],[5,36]]}
        for record in parent['review']['records']:
            q=record.pop('source_quad')
            record['instances']=[{'word_quad':q,'state':'visible'},
                                 {'word_quad':[[60,20],[90,20],[90,36],[60,36]],'state':'visible'}]
        (self.root/'scan.json').write_text('{}')
        parent.update(localization_evidence='scan.json',localization_evidence_sha256=project.digest(self.root/'scan.json'))
        (self.root/'geometry.json').write_text(json.dumps(parent))
        spec={'method':'surface_reprepare','parent_plan':'geometry.json',
              'parent_plan_sha256':project.digest(self.root/'geometry.json'),'source_sha256':'wrong'}
        with self.assertRaisesRegex(ValueError,'another source'):
            text_localization.reprepare(self.case,parent,spec,'surface.json')
        spec['source_sha256']=parent['source_sha256']
        text_localization.reprepare(self.case,parent,spec,'surface.json')
        newer=project.read_json(self.root/'surface.json')
        self.assertEqual([len(r['instances']) for r in newer['review']['records']],[2]*6)
        self.assertEqual([[i['word_quad'] for i in r['instances']] for r in parent['review']['records']],
                         [[i['word_quad'] for i in r['instances']] for r in newer['review']['records']])
        self.assertFalse(newer['review']['full_mask_reviewed'])
        self.assertEqual(newer['geometry_parent_sha256'],spec['parent_plan_sha256'])
        (self.root/'geometry.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Geometry parent changed'):
            text_localization.reprepare(self.case,parent,spec,'other.json')

    def test_missing_or_duplicate_operation_rejected(self):
        for operations in [self.manifest['operations'][:1],[self.manifest['operations'][0]]*2]:
            m=copy.deepcopy(self.manifest);m['operations']=operations
            with self.assertRaisesRegex(ValueError,'every requested'): execution_plan.validate(self.case,m)

    def test_unchanged_repair_reuses_parent_instead_of_rendering_again(self):
        workflow.initialize(self.case,'case.json')
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        candidate=workflow.execute_plan(self.case,'manifest.json','v1')
        self.review_before_repair(candidate)
        state=project.read_json(workflow.state_path(self.case))
        issue=workflow.add_feedback(self.case,dict(file_id='one',candidate_id='v1',
            candidate_sha256=candidate['output_sha256'],source_sha256=candidate['source_sha256'],
            expected_revision=state['revision'],source_time=.05,text='Check seam',operation_id='a',category='composition'))
        repair=dict(self.manifest,parent_output_sha256=candidate['output_sha256'])
        (self.root/'repair.json').write_text(json.dumps(repair))
        with patch('regional_composite.render') as render:
            with self.assertRaisesRegex(ValueError,'Unchanged render inputs'):
                workflow.execute_plan(self.case,'repair.json','v2','v1',issue['id'])
            render.assert_not_called()
        self.assertEqual(len(project.read_json(workflow.state_path(self.case))['files']['one']['jobs']),1)
    def test_two_operations_and_repair_keep_both_in_complete_output(self):
        workflow.initialize(self.case,'case.json')
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        c=workflow.execute_plan(self.case,'manifest.json','v1')
        self.review_before_repair(c)
        self.assertEqual(c['operation_ids'],['a','b']);self.assertFalse(workflow.quality_reviews_pass(c))
        cap=cv2.VideoCapture(str(self.root/c['output']));ok,f=cap.read();cap.release();self.assertTrue(ok)
        self.assertLess(float(f[23:33,10:30].mean()),245);self.assertLess(float(f[23:33,65:85].mean()),245)
        side=project.read_json(self.root/c['sidecar']);self.assertEqual(side['output_frame_count'],6);self.assertTrue(side['timeline_exact'])
        self.assertEqual([len(x['modified_source_frames']) for x in side['operations']],[6,6])
        state=project.read_json(workflow.state_path(self.case))
        issue=workflow.add_feedback(self.case,dict(file_id='one',candidate_id='v1',candidate_sha256=c['output_sha256'],source_sha256=c['source_sha256'],expected_revision=state['revision'],source_time=.05,text='Change only a',operation_id='a',category='composition'))
        changed=project.read_json(self.root/'a.json');changed['label_style']['ink_opacity']=0
        (self.root/'a-v2.json').write_text(json.dumps(changed))
        repair=copy.deepcopy(self.manifest);repair['parent_output_sha256']=c['output_sha256'];repair['operations'][0].update(plan='a-v2.json',plan_sha256=project.digest(self.root/'a-v2.json'))
        (self.root/'repair.json').write_text(json.dumps(repair))
        newer=workflow.execute_plan(self.case,'repair.json','v2','v1',issue['id'])
        cap=cv2.VideoCapture(str(self.root/newer['output']));ok,g=cap.read();cap.release();self.assertTrue(ok)
        self.assertGreater(float(g[23:33,10:30].mean()),250)
        self.assertLess(float(g[23:33,65:85].mean()),245)
        self.assertEqual(newer['operation_ids'],['a','b'])
        self.assertEqual(newer['parent_id'],'v1')
    def test_same_operation_keeps_two_simultaneous_instances(self):
        plan=project.read_json(self.root/'a.json')
        for record in plan['review']['records']:
            q=record.pop('source_quad')
            record['instances']=[{'instance_id':'left','source_quad':q,'state':'visible'},
                                 {'instance_id':'right','source_quad':[[5,42],[35,42],[35,58],[5,58]],'state':'visible'}]
        (self.root/'a.json').write_text(json.dumps(plan))
        self.manifest['operations'][0]['plan_sha256']=project.digest(self.root/'a.json')
        report=execution_plan.render(self.case,self.manifest,self.root/'instances.mp4')
        cap=cv2.VideoCapture(str(self.root/'instances.mp4'));ok,f=cap.read();cap.release()
        self.assertTrue(ok)
        self.assertLess(float(f[23:33,10:30].mean()),245)
        self.assertLess(float(f[45:55,10:30].mean()),245)
        self.assertEqual(report['operations'][0]['modified_instance_count'],12)

    def test_wrong_operation_asset_rejected(self):
        plan=project.read_json(self.root/'a.json');plan['exact_asset']='other.png';(self.root/'a.json').write_text(json.dumps(plan));self.manifest['operations'][0]['plan_sha256']=project.digest(self.root/'a.json')
        with self.assertRaisesRegex(ValueError,'exact operation'):execution_plan.validate(self.case,self.manifest)

    def test_full_states_cannot_hide_missing_duplicate_or_mismatched_frame_records(self):
        original=project.read_json(self.root/'a.json')
        changes=[original['review']['records'][:-1],
                 original['review']['records'][:-1]+[original['review']['records'][0]],
                 [dict(r,state='unresolved') if r['frame']==2 else r
                  for r in original['review']['records']]]
        for records in changes:
            with self.subTest(records=records):
                plan=copy.deepcopy(original);plan['review']['records']=records
                (self.root/'a.json').write_text(json.dumps(plan))
                self.manifest['operations'][0]['plan_sha256']=project.digest(self.root/'a.json')
                with self.assertRaisesRegex(ValueError,'Complete ordered coverage records'):
                    execution_plan.validate(self.case,self.manifest)

    def test_clipped_text_plan_cannot_bypass_preflight_at_export(self):
        workflow.initialize(self.case,'case.json')
        plan=project.read_json(self.root/'a.json')
        plan['review']['records'][0]['word_quad']=[[-5,20],[30,20],[30,36],[-5,36]]
        (self.root/'a.json').write_text(json.dumps(plan))
        self.manifest['operations'][0]['plan_sha256']=project.digest(self.root/'a.json')
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'Clipped source word'):
            workflow.execute_plan(self.case,'manifest.json','clipped')
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertIsNone(entry['current_candidate_id'])
        self.assertEqual(entry['jobs'][0]['status'],'failed')

    def test_overlapping_operations_fail_without_activating_output(self):
        workflow.initialize(self.case,'case.json')
        plan=project.read_json(self.root/'b.json')
        for record in plan['review']['records']:
            record['source_quad']=[[5,20],[35,20],[35,36],[5,36]]
        (self.root/'b.json').write_text(json.dumps(plan))
        self.manifest['operations'][1]['plan_sha256']=project.digest(self.root/'b.json')
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'operation overlap'):
            workflow.execute_plan(self.case,'manifest.json','overlap')
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertIsNone(entry['current_candidate_id'])
        self.assertEqual(entry['jobs'][0]['status'],'failed')

if __name__=='__main__': unittest.main()
