"""Local order entry must not reopen remote work or relax method checks."""
import copy
import json
import unittest
from unittest.mock import patch
import project
import production_pipeline as pipeline
import runninghub_adapter as provider
import workflow
import test_production_pipeline as fixtures

class LocalOrders(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.PipelineTests(); self.fixture.setUp()
        self.root,self.parent=self.fixture.root,self.fixture.case
        (self.root/'config').mkdir(exist_ok=True)
        self.config=self.root/'config/models.json'
        self.config.write_text(json.dumps(dict(execution_mode='maintenance',remote_calls_allowed=False)))
    def tearDown(self):self.fixture.tearDown()
    def create(self):
        request=dict(batch_id='local-new',user_input='input.txt',files=copy.deepcopy(self.parent['files']))
        (self.root/'local-request.json').write_text(json.dumps(request))
        with patch('sys.argv',['project.py','intake','--case','local/case.json','--request','local-request.json']):
            self.assertEqual(project.main(),0)
        self.case=project.read_json(self.root/'local/case.json')
        workflow.initialize(self.case,'local/case.json')
    def test_ordinary_intake_local_default_does_not_enable_global_production(self):
        before=self.config.read_bytes();self.create()
        self.assertTrue(self.case['execution_policy']['local_only'])
        self.assertEqual(self.case['execution_policy']['mode'],'production')
        pipeline.require_production(self.case)
        with self.assertRaisesRegex(ValueError,'maintenance'):pipeline.require_production(self.parent)
        self.assertEqual(before,self.config.read_bytes())
    def test_saved_local_order_denies_every_remote_action_even_after_global_grant(self):
        self.create();self.config.write_text(json.dumps(dict(execution_mode='production',remote_calls_allowed=True)))
        for action in ('submit','upload','download','query','quote'):
            with self.assertRaisesRegex(ValueError,'Local-only'):provider.require_remote(action,self.case['batch_id'])
    def test_local_order_still_requires_method_proof_and_qualified_result(self):
        self.create()
        with self.assertRaisesRegex(ValueError,'method proof'):pipeline.verify_method_proof(dict(id='unproven'))
        import delivery_package
        with self.assertRaisesRegex(ValueError,'Unqualified'):delivery_package.build(self.case,'invalid-delivery')
    def test_raw_import_rejects_final_candidate_without_original_unit(self):
        self.create()
        with self.assertRaisesRegex(ValueError,'Unknown execution unit'):
            provider.replay(self.case,dict(file_id='one',operation_id='a',unit_id='u',origin=dict(
                case=dict(path='case.json',sha256=project.digest(self.root/'case.json')),file_id='one',unit_id='final-v1')))

    def original_unit(self, status='fail_or_limited'):
        self.create()
        raw=dict(path='source.mp4',sha256=project.digest(self.root/'source.mp4'),
                 provider_task_id='existing-task',request_fingerprint='old-request')
        review=dict(raw_sha256=raw['sha256'],request_fingerprint='old-request',evidence=[])
        (self.root/'raw-review.json').write_text(json.dumps(review))
        unit=dict(unit_id='original',operation_id='a',status='raw_ready',provider_task_id='existing-task',
                  request_fingerprint='old-request',raw=raw,
                  receipts=[dict(response=dict(task_id='existing-task',status='SUCCESS'))],
                  raw_review=dict(path='raw-review.json',sha256=project.digest(self.root/'raw-review.json'),status=status))
        with workflow.locked_state(self.parent) as state:state['files']['one'].setdefault('execution_units',[]).append(unit)
        spec=dict(file_id='one',operation_id='a',unit_id='imported',origin=dict(
            case=dict(path='case.json',sha256=project.digest(self.root/'case.json')),file_id='one',unit_id='original'))
        return spec

    def test_raw_import_retains_failure_and_is_idempotent(self):
        spec=self.original_unit()
        with patch.object(provider,'verify_unit',return_value=dict(model_id='original-model')):
            provider.replay(self.case,spec)
            before=workflow.state_path(self.case).read_bytes()
            self.assertTrue(provider.replay(self.case,spec)['reused'])
            self.assertEqual(before,workflow.state_path(self.case).read_bytes())
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(entry['execution_units'][0]['raw_review']['status'],'fail_or_limited')
        self.assertEqual(entry['raw_returns'][0]['path'],'source.mp4')

    def test_invalid_original_review_creates_no_partial_import(self):
        spec=self.original_unit();(self.root/'raw-review.json').write_text('{}')
        with patch.object(provider,'verify_unit',return_value={}):
            with self.assertRaisesRegex(ValueError,'review changed'):provider.replay(self.case,spec)
        self.assertFalse((self.root/'runs/local-new/units/one/imported').exists())

    def genuine_original(self):
        spec = self.original_unit()
        request = dict(model_id='original-model', file_id='one', operation_id='a',
                       unit_id='original', artifact_hashes={},
                       dependency=workflow.requirement_binding(self.parent, self.parent['files'][0]))
        (self.root/'original-request.json').write_text(json.dumps(request))
        fingerprint = workflow.hash_json(request)
        review_path = self.root/'raw-review.json'
        review = project.read_json(review_path); review['request_fingerprint'] = fingerprint
        review_path.write_text(json.dumps(review))
        with workflow.locked_state(self.parent) as state:
            unit = state['files']['one']['execution_units'][0]
            unit.update(file_id='one', dependency=request['dependency'], request_path='original-request.json',
                        request_fingerprint=fingerprint)
            unit['raw']['request_fingerprint'] = fingerprint
            unit['raw_review']['sha256'] = project.digest(review_path)
        return spec

    def test_original_journal_append_does_not_invalidate_immutable_raw(self):
        spec = self.genuine_original()
        provider.replay(self.case, spec)
        before = workflow.state_path(self.case).read_bytes()
        with workflow.locked_state(self.parent) as state:
            state['maintenance_note'] = 'unrelated journal append'
        unit = project.read_json(workflow.state_path(self.case))['files']['one']['execution_units'][0]
        provider.verify_unit(self.case, unit)
        self.assertTrue(provider.replay(self.case, spec)['reused'])
        self.assertEqual(before, workflow.state_path(self.case).read_bytes())
        with workflow.locked_state(self.parent) as state:
            state['files']['one']['execution_units'][0]['raw_review']['status'] = 'pass'
        with self.assertRaisesRegex(ValueError, 'provenance changed'):
            provider.verify_unit(self.case, unit)

if __name__=='__main__':unittest.main()
