"""Latency entry contracts on temporary synthetic fixtures, no video production."""
import json
import unittest
from unittest.mock import patch

import project
import workflow
import production_effort
import semantic_scan
import test_source_observation
import test_semantic_scan


class LocalizationEffortTests(unittest.TestCase):
    setUp = test_source_observation.SourceObservationTests.setUp
    tearDown = test_source_observation.SourceObservationTests.tearDown
    cli = test_source_observation.SourceObservationTests.cli
    predictions = test_semantic_scan.SemanticScanTests.predictions

    def configure(self):
        self.case['execution_policy'] = dict(production_effort.new_policy('method_trial'), local_only=True)
        (self.root/'case.json').write_text(json.dumps(self.case))
        workflow.state_path(self.case).unlink()
        workflow.initialize(self.case, 'case.json')
        path = self.root/'authorization.json'
        path.write_text(json.dumps({k:True for k in ('allow_keyframe_tracking','keep_every_frame_review',
            'latency_first_quality_repair_later','no_download_install','zero_api')}))
        return dict(action='scan-source', file_id='one', authorization=semantic_scan.observation.artifact(path))

    def test_default_intake_policy_and_readonly_call_estimate(self):
        project.intake('new-fixture/case.json','request.json')
        policy=project.read_json(self.root/'new-fixture/case.json')['execution_policy']
        self.assertEqual(policy['source_localization_strategy'],'keyframe_tracking')
        self.configure()
        self.assertEqual(self.case['execution_policy'].get('source_localization_strategy'), 'keyframe_tracking')
        before=workflow.state_path(self.case).read_bytes()
        with patch('semantic_scan.engine_binding',side_effect=AssertionError('no runtime probe')), \
                patch('cv2.VideoCapture',side_effect=AssertionError('no source decode')):
            result=self.cli(dict(action='plan-source-scan',file_id='one'))
        self.assertEqual(result['strategy'],'keyframe_tracking')
        self.assertEqual(result['max_new_model_calls'],2)
        self.assertEqual(result['remaining_frames'],6)
        self.assertFalse(result['quality_verified'])
        self.assertEqual(before,workflow.state_path(self.case).read_bytes())

    def test_fresh_default_uses_only_first_last_and_keeps_tracking_unresolved(self):
        spec=self.configure()
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model, \
                patch('carrier_tracking.register',return_value=None):
            model.return_value.__enter__.return_value.predict.side_effect=self.predictions
            self.cli(spec)
            self.assertEqual([c.args[0] for c in model.return_value.__enter__.return_value.predict.call_args_list],[0,5])
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        data=semantic_scan.read_result(self.case,'one',entry)
        self.assertTrue(all(r['state']=='unresolved' for r in data['operations']['a']['records'][1:5]))
        self.cli(dict(action='freeze-route',file_id='one'),expected=2)

    def test_no_silent_independent_or_unknown_strategy(self):
        spec=self.configure()
        with patch('semantic_scan.engine_binding',side_effect=AssertionError('no model before choice')):
            for strategy in ['independent_frames','typo']:
                self.cli(dict(spec,strategy=strategy),expected=2)

    def test_failed_hybrid_requires_diagnosis_and_only_resumes_missing_anchor(self):
        spec=self.configure()
        def fail(n):
            if n==5:raise ValueError('preserved failure')
            return self.predictions(n)
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model, \
                patch('carrier_tracking.register',return_value=None):
            model.return_value.__enter__.return_value.predict.side_effect=fail
            self.cli(spec,expected=2)
        with patch('semantic_scan.engine_binding',side_effect=AssertionError('must diagnose before probe')):
            self.cli(spec,expected=2)
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=self.predictions
            self.cli(dict(spec,resume_reason='Synthetic failure diagnosed; original journal retained'))
            self.assertEqual([c.args[0] for c in model.return_value.__enter__.return_value.predict.call_args_list],[5])

    def test_low_disk_refuses_before_preparation_or_model(self):
        spec=self.configure();before=workflow.state_path(self.case).read_bytes()
        with patch('shutil.disk_usage') as disk,patch('semantic_scan.engine_binding',side_effect=AssertionError('no probe')):
            disk.return_value.free=0
            self.cli(spec,expected=2)
        self.assertEqual(before,workflow.state_path(self.case).read_bytes())

    def test_production_does_not_promote_unqualified_tracking(self):
        spec=self.configure()
        self.case['execution_policy']['mode']='production'
        (self.root/'case.json').write_text(json.dumps(self.case))
        workflow.state_path(self.case).unlink()
        workflow.initialize(self.case,'case.json')
        with patch('semantic_scan.engine_binding',side_effect=AssertionError('not a production proof')):
            self.cli(spec,expected=2)


if __name__=='__main__':unittest.main()
