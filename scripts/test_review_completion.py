"""The full-review exception must never shrink coverage or qualify a failed method."""
import json
import unittest
from unittest.mock import patch
import project
import workflow
import production_pipeline as pipeline
import review_completion as completion
import runninghub_adapter as provider
from test_interval_edit import IntervalEdit


class FullReview(unittest.TestCase):
    setUp = IntervalEdit.setUp
    tearDown = IntervalEdit.tearDown
    write = IntervalEdit.write
    save_config = IntervalEdit.save_config

    def make_contract(self):
        self.config['execution_mode'] = 'maintenance'; self.save_config()
        raw = self.root/'raw.mp4'; raw.write_bytes(b'original provider fixture')
        request = dict(file_id='one', operation_id='a', unit_id='hard', model_id='kling-o3-std-edit',
                       source_frames_half_open=[100,453], prompt='unchanged', parameters={},image_references=[])
        unit = dict(unit_id='hard',status='raw_ready',request_fingerprint='fixed',
                    raw=dict(path='raw.mp4',sha256=project.digest(raw)))
        with workflow.locked_state(self.case) as state:
            e = state['files']['one']; e['execution_units']=[unit]
            e['issues']=[dict(id='coverage',category='localization',status='open')]
        self.write('authorization.json',dict(batch_id='isolated',remote_authorized=True,full_cost_confirmed=True,
                   total_budget_cny=150,unit_caps_cny=dict(front=10,tail=10)))
        specs = [dict(request,unit_id='front',source_frames_half_open=[0,113]),
                 dict(request,unit_id='tail',source_frames_half_open=[419,500])]
        contract = dict(batch_id='isolated',file_id='one',operation_id='a',issue_id='coverage',reuse_unit_id='hard',
                        user_instruction='Process all target appearances and return defects for feedback',
                        quality_policy='review_with_limits',unit_specs=specs,unit_caps_cny=dict(front=10,tail=10),
                        authorization='authorization.json')
        return request, contract

    def test_full_union_required_and_original_failure_retained(self):
        request, contract = self.make_contract()
        with patch.object(provider,'verify_unit',return_value=request), patch.object(pipeline,'current_issue',return_value=True):
            broken = dict(contract,unit_specs=contract['unit_specs'][1:])
            self.write('contract.json',broken)
            with self.assertRaisesRegex(ValueError,'every observed target'):
                completion.freeze(self.case,dict(file_id='one',contract='contract.json'))
            self.write('contract.json',contract)
            result = completion.freeze(self.case,dict(file_id='one',contract='contract.json'))
        self.assertEqual(result['target_frames'],500); self.assertFalse(result['qualified'])
        entry = project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(entry['preview_trials'][0]['verdict'],'fail')
        self.assertEqual(workflow.case_binding(self.case),self.original_binding)
        with self.assertRaisesRegex(ValueError,'maintenance'):
            pipeline.require_production()

    def test_exception_only_accepts_frozen_intervals(self):
        request, contract = self.make_contract()
        with patch.object(completion,'_ACTIVE',contract), patch.object(provider,'verify_unit',return_value=request):
            pipeline.require_production(self.case)
            self.assertIsNotNone(pipeline.guard_interval_unit(self.case,'one','a','kling-o3-std-edit',[0,113]))
            with self.assertRaisesRegex(ValueError,'Hardest interval'):
                pipeline.guard_interval_unit(self.case,'one','a','kling-o3-std-edit',[0,81])
            with self.assertRaisesRegex(ValueError,'model is fixed'):
                pipeline.guard_interval_unit(self.case,'one','a','other',[0,113])

    def test_method_change_refused_before_any_remote_work(self):
        request, contract = self.make_contract()
        contract['unit_specs'][0]['prompt']='another trial'
        self.write('contract.json',contract)
        with patch.object(provider,'verify_unit',return_value=request), patch.object(pipeline,'current_issue',return_value=True):
            with self.assertRaisesRegex(ValueError,'unchanged API method'):
                completion.freeze(self.case,dict(file_id='one',contract='contract.json'))
