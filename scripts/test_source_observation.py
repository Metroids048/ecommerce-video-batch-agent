"""Six-frame synthetic plumbing regressions; no semantic/business acceptance."""
import copy
import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import project
import workflow
import production_pipeline as pipeline
import execution_plan
from test_execution_plan import UnifiedExecution


class SourceObservationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = UnifiedExecution(); self.fixture.setUp()
        self.root, self.case = self.fixture.root, self.fixture.case
        # v2 isolates source handoff from the independent v3 method-proof gate.
        self.case['execution_policy'] = dict(version=2, stage_handoff_required=True,
            source_semantic_review_required=True, local_only=True)
        (self.root/'case.json').write_text(json.dumps(self.case))
        workflow.initialize(self.case,'case.json')

    def tearDown(self):
        self.fixture.tearDown()

    def cli(self, spec, expected=0):
        (self.root/'action.json').write_text(json.dumps(spec))
        out = io.StringIO()
        with patch('sys.argv', ['project.py','pipeline','--case','case.json','--spec','action.json']), \
                redirect_stdout(out), redirect_stderr(out):
            code = project.main()
        self.assertEqual(code, expected, out.getvalue())
        return json.loads(out.getvalue().strip().splitlines()[-1])

    def prepare(self):
        return self.cli(dict(action='prepare-source-observation', file_id='one'))

    def packet(self, frames=range(6), state='visible', second=False):
        saved = project.read_json(workflow.state_path(self.case))['files']['one']['source_observation']
        queue = project.read_json(project.project_path(saved['queue']['path']))
        evidence = [{k:e[k] for k in ('path','sha256')} for e in queue['evidence'] if len(e['frames'])==6]
        operations = {}
        for op, x in [('a',5),('b',60)]:
            rows = []
            for n in frames:
                instances = [dict(id=op+'-front', bbox=[x,20,30,16])] if state=='visible' else []
                if second and op=='a': instances.append(dict(id='a-rear',bbox=[60,20,30,16]))
                rows.append(dict(frame=n,state=state,instances=instances,evidence=evidence,
                    all_instances_observed=state!='unresolved',reason='Synthetic fixture observation only'))
            operations[op] = dict(quality_requirements=['Keep source details'],records=rows)
        return dict(action='record-source-observation',file_id='one',queue_sha256=saved['queue']['sha256'],
                    dependency=workflow.requirement_binding(self.case,self.case['files'][0]),
                    observer='synthetic-regression',operations=operations)

    def finish(self, second=False):
        self.prepare(); return self.cli(self.packet(second=second))

    def test_ordinary_cli_prepares_all_frames_and_recovers_without_decode(self):
        result = self.prepare()
        self.assertEqual(result['next_action'],'record-source-observation')
        self.assertEqual(result['missing_frames_half_open'],{'a':[[0,6]],'b':[[0,6]]})
        saved = project.read_json(workflow.state_path(self.case))['files']['one']['source_observation']
        queue = project.read_json(project.project_path(saved['queue']['path']))
        self.assertEqual(queue['paired_frames'],list(range(6)))
        self.assertTrue(queue['source_only'])
        from PIL import Image
        with Image.open(project.project_path(queue['overview_sheets'][0])) as sheet:
            self.assertEqual(sheet.width,1536)  # source once, no duplicated comparison image
        before = workflow.state_path(self.case).read_bytes()
        with patch('review_queue.build_pairs', side_effect=AssertionError('must reuse queue')):
            again = self.prepare()
            resumed = self.cli(dict(action='resume-batch'))
        self.assertEqual(result,again)
        self.assertEqual(resumed['files'][0]['source_observation']['queue'],saved['queue'])
        self.assertEqual(workflow.state_path(self.case).read_bytes(),before)

    def test_incremental_receipts_keep_missing_and_unresolved_then_freeze(self):
        self.prepare()
        self.cli(self.packet(frames=[0,1,2]))
        result=self.cli(self.packet(frames=[3,4],state='unresolved'))
        self.assertEqual(result['missing_frames_half_open']['a'],[[5,6]])
        self.assertEqual(result['unresolved_frames_half_open']['a'],[[3,5]])
        self.cli(dict(action='freeze-route',file_id='one'),expected=2)
        result=self.cli(self.packet(frames=[3,4,5]))
        self.assertEqual(result['next_action'],'freeze-route')
        self.assertEqual(result['appearance_ranges_half_open']['a'],[[0,6]])
        before=workflow.state_path(self.case).read_bytes()
        self.cli(self.packet(frames=[3,4,5]))
        self.assertEqual(workflow.state_path(self.case).read_bytes(),before)
        self.cli(dict(action='freeze-route',file_id='one'))  # consumes saved receipt, no hand-written range JSON
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(entry['route_handoff']['observation'],entry['source_observation']['receipt'])
        execution_plan.validate(self.case,self.fixture.manifest)

    def test_manually_claimed_ranges_cannot_bypass_new_source_review(self):
        observation=dict(source_sha256=self.case['files'][0]['source_sha256'],operations={
            op:dict(appearance_ranges_half_open=[[0,6]],risk_ranges_half_open=[],quality_requirements=['Keep details'])
            for op in ('a','b')})
        path=self.root/'manual.json';path.write_text(json.dumps(observation))
        before=workflow.state_path(self.case).read_bytes()
        result=self.cli(dict(action='freeze-route',file_id='one',
            observation=dict(path=path.name,sha256=project.digest(path))),expected=2)
        self.assertIn('source observation',str(result).lower())
        self.assertEqual(workflow.state_path(self.case).read_bytes(),before)

    def test_missing_same_frame_instance_is_rejected_at_manifest_consumer(self):
        self.finish(second=True)
        self.cli(dict(action='freeze-route',file_id='one'))
        with self.assertRaisesRegex(ValueError,'instance'):
            execution_plan.validate(self.case,self.fixture.manifest)
        plan=project.read_json(self.root/'a.json')
        for row in plan['review']['records']:
            quad=row.pop('source_quad')
            row['instances']=[dict(state='visible',source_quad=quad),
                              dict(state='visible',source_quad=[[60,20],[90,20],[90,36],[60,36]])]
        (self.root/'a.json').write_text(json.dumps(plan))
        self.fixture.manifest['operations'][0]['plan_sha256']=project.digest(self.root/'a.json')
        execution_plan.validate(self.case,self.fixture.manifest)
        # Equal counts at the wrong positions still omit the observed rear label.
        for row in plan['review']['records']:row['instances'][1]=copy.deepcopy(row['instances'][0])
        (self.root/'a.json').write_text(json.dumps(plan))
        self.fixture.manifest['operations'][0]['plan_sha256']=project.digest(self.root/'a.json')
        with self.assertRaisesRegex(ValueError,'instance position'):
            execution_plan.validate(self.case,self.fixture.manifest)

    def test_absence_and_reappearance_do_not_get_inferred_from_no_match(self):
        self.prepare();self.cli(self.packet(frames=[0,1,4,5]))
        result=self.cli(self.packet(frames=[2,3],state='absent'))
        self.assertEqual(result['appearance_ranges_half_open']['a'],[[0,2],[4,6]])
        self.cli(dict(action='freeze-route',file_id='one'))
        with self.assertRaisesRegex(ValueError,'absent'):
            execution_plan.validate(self.case,self.fixture.manifest)

    def test_duplicate_frame_or_instance_and_wrong_bindings_leave_state_unchanged(self):
        self.prepare(); good=self.packet();bad=[]
        p=copy.deepcopy(good);p['operations']['a']['records'].append(p['operations']['a']['records'][0]);bad.append(p)
        p=copy.deepcopy(good);p['operations']['a']['records'][0]['instances']*=2;bad.append(p)
        p=copy.deepcopy(good);p['queue_sha256']='wrong';bad.append(p)
        p=copy.deepcopy(good);p['dependency']['requirements_sha256']='wrong';bad.append(p)
        p=copy.deepcopy(good);p['operations']['a']['records'][0]['evidence'][0]['sha256']='wrong';bad.append(p)
        p=copy.deepcopy(good);p['operations']['a']['records'][0]['instances'][0]['bbox']=[90,20,30,16];bad.append(p)
        p=copy.deepcopy(good);p['operations']['a']['records'][0]['all_instances_observed']=False;bad.append(p)
        for p in bad:
            with self.subTest(packet=p):
                before=workflow.state_path(self.case).read_bytes()
                self.cli(p,expected=2)
                self.assertEqual(workflow.state_path(self.case).read_bytes(),before)

    def test_queue_tampering_blocks_prepare_status_and_freeze(self):
        self.finish()
        saved=project.read_json(workflow.state_path(self.case))['files']['one']['source_observation']
        queue=project.read_json(project.project_path(saved['queue']['path']))
        project.project_path(queue['evidence'][0]['path']).write_bytes(b'changed')
        for action in ('prepare-source-observation','freeze-route','status'):
            self.cli(dict(action=action,file_id='one'),expected=2)

    def test_native_detail_addition_preserves_partial_observation_and_reuses_queue(self):
        self.prepare();self.cli(self.packet(frames=[0,1,2]))
        before=project.read_json(workflow.state_path(self.case))['files']['one']['source_observation']
        result=self.cli(dict(action='prepare-source-observation',file_id='one',stills=[3]))
        self.assertEqual(result['queue'],before['queue'])
        self.assertEqual(result['receipt'],before['receipt'])
        detail=project.read_json(project.project_path(result['detail_queues'][0]['path']))
        image=next(e for e in detail['images'] if e['source_frame']==3)
        packet=self.packet(frames=[3,4,5])
        for op in packet['operations'].values():op['records'][0]['evidence']=[{k:image[k] for k in ('path','sha256')}]
        self.cli(packet)
        saved=workflow.state_path(self.case).read_bytes()
        with patch('review_queue.build',side_effect=AssertionError('must reuse native detail')):
            self.cli(dict(action='prepare-source-observation',file_id='one',stills=[3]))
        self.assertEqual(workflow.state_path(self.case).read_bytes(),saved)

    def test_new_intake_requires_receipt_and_unknown_frames_remain_pending(self):
        self.assertTrue(project.read_json(self.root/'case.json')['execution_policy']['source_semantic_review_required'])
        from production_effort import new_policy
        self.assertTrue(new_policy()['source_semantic_review_required'])
        self.prepare();result=self.cli(self.packet(state='unresolved'))
        self.assertEqual(result['unresolved_frames_half_open']['a'],[[0,6]])
        self.assertEqual(result['appearance_ranges_half_open']['a'],[])
        self.cli(dict(action='freeze-route',file_id='one'),expected=2)

    def test_complete_source_observation_does_not_bypass_v3_method_proof(self):
        self.finish()
        self.case['execution_policy'].update(version=3,mode='production')
        (self.root/'case.json').write_text(json.dumps(self.case))
        with workflow.locked_state(self.case) as state:state['case_binding']=workflow.case_binding(self.case)
        result=self.cli(dict(action='freeze-route',file_id='one'),expected=2)
        self.assertIn('specific method',str(result))

    def test_material_pin_consumes_fully_occluded_frames_without_forcing_an_edit(self):
        self.prepare();self.cli(self.packet(frames=[0,1,4,5]))
        self.cli(self.packet(frames=[2,3],state='occluded'))
        # A disposable method_trial policy isolates pin consumption from the
        # separate business method-proof gate; no method trial is executed.
        self.case['execution_policy'].update(version=3,mode='method_trial')
        (self.root/'case.json').write_text(json.dumps(self.case))
        with workflow.locked_state(self.case) as state:state['case_binding']=workflow.case_binding(self.case)
        (self.root/'scripts').mkdir();(self.root/'config').mkdir()
        (self.root/'config/models.json').write_text('{}')
        methods={op:dict(id='synthetic-'+op,local=dict(composition_mode='exact_asset_on_source_label',segmentation_backend=None)) for op in ('a','b')}
        self.cli(dict(action='freeze-route',file_id='one',methods=methods))
        for op in self.fixture.manifest['operations']:
            path=self.root/op['plan'];plan=project.read_json(path)
            for n in (2,3):
                plan['review']['frame_states'][n]='absent'
                plan['review']['records'][n]['state']='absent'
            path.write_text(json.dumps(plan));op['plan_sha256']=project.digest(path)
        self.fixture.manifest['risk_preview_ranges_half_open']=[[1,5]]
        pipeline.pin_materials(self.case,self.fixture.manifest,{})
        state=project.read_json(workflow.state_path(self.case))
        self.assertEqual(state['files']['one']['material_handoff']['manifest'],self.fixture.manifest)

    def test_actual_default_intake_saves_semantic_gate_and_maintenance_still_blocks_media(self):
        project.intake('new-fixture/case.json','request.json')
        self.assertTrue(project.read_json(self.root/'new-fixture/case.json')['execution_policy']['source_semantic_review_required'])
        self.case['execution_policy'].pop('local_only')
        (self.root/'case.json').write_text(json.dumps(self.case))
        with workflow.locked_state(self.case) as state:state['case_binding']=workflow.case_binding(self.case)
        (self.root/'config').mkdir()
        (self.root/'config/models.json').write_text(json.dumps(dict(execution_mode='maintenance')))
        result=self.cli(dict(action='prepare-source-observation',file_id='one'),expected=2)
        self.assertIn('maintenance',str(result))
        self.assertNotIn('source_observation',project.read_json(workflow.state_path(self.case))['files']['one'])

    def test_legacy_routes_are_not_migrated_and_reuse_only_cannot_prepare(self):
        self.case['execution_policy'].pop('source_semantic_review_required')
        with workflow.locked_state(self.case) as state:state['case_binding']=workflow.case_binding(self.case)
        observation=dict(source_sha256=self.case['files'][0]['source_sha256'],operations={
            op:dict(appearance_ranges_half_open=[[0,6]],risk_ranges_half_open=[],quality_requirements=['Keep details'])
            for op in ('a','b')})
        path=self.root/'legacy.json';path.write_text(json.dumps(observation))
        pipeline.freeze_route(self.case,dict(file_id='one',observation=dict(path=path.name,sha256=project.digest(path))))
        self.case['execution_policy']['reuse_only']=True
        with self.assertRaisesRegex(ValueError,'Reuse-only'):
            pipeline.dispatch(self.case,dict(action='prepare-source-observation',file_id='one'))


if __name__=='__main__':unittest.main()
