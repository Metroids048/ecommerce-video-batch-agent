"""Ordinary hybrid scan behavior; synthetic data do not establish video quality."""
import copy
import json
import unittest
from unittest.mock import patch
import project
import workflow
import semantic_scan
import test_source_observation
import test_semantic_scan


class HybridTests(unittest.TestCase):
    setUp=test_source_observation.SourceObservationTests.setUp
    tearDown=test_source_observation.SourceObservationTests.tearDown
    cli=test_source_observation.SourceObservationTests.cli
    predictions=test_semantic_scan.SemanticScanTests.predictions

    def prepare(self):
        self.case['execution_policy']['mode']='method_trial'
        (self.root/'case.json').write_text(json.dumps(self.case));workflow.state_path(self.case).unlink()
        workflow.initialize(self.case,'case.json')
        def interrupted(n):
            if n==1:raise ValueError('Preserved original interruption')
            return self.predictions(n)
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=interrupted
            self.cli(dict(action='scan-source',file_id='one'),expected=2)
        path=self.root/'authorization.json'
        path.write_text(json.dumps({k:True for k in ('allow_keyframe_tracking','keep_every_frame_review','latency_first_quality_repair_later','no_download_install','zero_api')}))
        return dict(action='scan-source',file_id='one',strategy='keyframe_tracking',authorization=semantic_scan.observation.artifact(path),
                    resume_reason='Synthetic original interruption diagnosed; consume saved anchors under authorized strategy')

    def run_scan(self,spec,fail=False):
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model,patch('carrier_tracking.register',return_value=None):
            model.return_value.__enter__.return_value.predict.side_effect=ValueError('anchor interrupted') if fail else self.predictions
            result=self.cli(spec,expected=2 if fail else 0)
            return result,model.return_value.__enter__.return_value.predict.call_count

    def test_two_anchors_replace_six_independent_calls_and_preserve_pending_checks(self):
        spec=self.prepare();old=copy.deepcopy(project.read_json(workflow.state_path(self.case))['files']['one']['source_semantic_execution'])
        result,calls=self.run_scan(spec);self.assertEqual(calls,1) # frame 0 inherited, frame 5 new
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(entry['source_semantic_execution_history'][-1],old)
        data=semantic_scan.read_result(self.case,'one',entry)
        self.assertEqual(len(data['frame_artifacts']),6)
        for operation in data['operations'].values():
            self.assertTrue(all(r['state']=='unresolved' and not r['all_instances_observed'] for r in operation['records'][1:5]))
        self.assertFalse(data['accepted']);self.assertNotIn('receipt',entry['source_observation'])
        frozen=workflow.state_path(self.case).read_bytes()
        with patch('semantic_scan.engine_binding',side_effect=AssertionError('no probe')),patch('cv2.VideoCapture',side_effect=AssertionError('no decode')),patch('semantic_scan.LocalModel',side_effect=AssertionError('no model')):
            self.assertEqual(self.cli(spec),result)
        self.assertEqual(frozen,workflow.state_path(self.case).read_bytes())
        self.cli(dict(action='freeze-route',file_id='one'),expected=2)

    def test_recovery_indexes_original_uncommitted_frame_and_resumes_missing_anchor(self):
        spec=self.prepare();_,calls=self.run_scan(spec,fail=True);self.assertEqual(calls,1)
        with workflow.locked_state(self.case) as state:
            state['files']['one']['source_semantic_execution']['frame_artifacts'].pop('2')
        result,calls=self.run_scan(spec);self.assertEqual(calls,1)
        saved=project.read_json(workflow.state_path(self.case))['files']['one']['source_semantic_execution']
        self.assertEqual(saved['journal_recoveries'][-1]['frames'],[2])
        self.assertEqual(len(saved['frame_artifacts']),6)

    def test_model_change_and_changed_authorization_are_rejected(self):
        spec=self.prepare()
        with patch('semantic_scan.engine_binding',return_value={'options':{'model':'other'}}):self.cli(spec,expected=2)
        self.run_scan(spec,fail=True)
        path=self.root/'authorization2.json';path.write_text((self.root/'authorization.json').read_text())
        spec['authorization']=semantic_scan.observation.artifact(path)
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}):self.cli(spec,expected=2)


if __name__=='__main__':unittest.main()
