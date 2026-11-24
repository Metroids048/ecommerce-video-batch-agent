"""Tracking safety and ordinary-CLI plumbing, not visual acceptance."""
import json
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import project
import workflow
import semantic_tracking as tracking
import test_source_observation
import test_semantic_scan


class TrackingTests(unittest.TestCase):
    setUp=test_source_observation.SourceObservationTests.setUp
    tearDown=test_source_observation.SourceObservationTests.tearDown
    cli=test_source_observation.SourceObservationTests.cli
    predictions=test_semantic_scan.SemanticScanTests.predictions

    def prepare_trial(self):
        self.case['execution_policy']['mode']='method_trial'
        (self.root/'case.json').write_text(json.dumps(self.case))
        workflow.state_path(self.case).unlink();workflow.initialize(self.case,'case.json')
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=self.predictions
            self.cli(dict(action='scan-source',file_id='one'))
        auth=self.root/'authorization.json'
        auth.write_text(json.dumps(dict(allow_keyframe_tracking=True,keep_every_frame_review=True)))
        return dict(action='trial-source-tracking',file_id='one',frames_half_open=[0,6],anchor_frames=[0,5],
            authorization=tracking.observation.artifact(auth))

    def test_translation_and_cut_keep_loss_unresolved(self):
        rng=np.random.default_rng(17)
        image=rng.integers(0,256,(160,240),dtype=np.uint8)
        shifted=cv2.warpAffine(image,np.float32([[1,0,3],[0,1,2]]),(240,160))
        result=tracking.propagate([image,shifted,np.zeros_like(image)],0,[70,50,70,45])
        np.testing.assert_allclose(result[1]['bbox'],[73,52,70,45],atol=.3)
        self.assertNotIn(2,result)

    def test_cli_preserves_old_cache_and_all_frame_quality_pending(self):
        spec=self.prepare_trial();before=project.read_json(workflow.state_path(self.case))['files']['one']['source_semantic_execution']
        with patch('carrier_tracking.register',return_value=None),patch('semantic_scan.LocalModel',side_effect=AssertionError('no model')):
            result=self.cli(spec)
        data=tracking.observation.read_artifact(result['result'],'Test result')
        self.assertEqual(len(data['frames']),6)
        for packet in data['operations'].values():
            self.assertEqual(len(packet['refresh_frames']),6)
            self.assertTrue(all(r['state']=='unresolved' and not r['all_instances_observed'] for r in packet['records']))
        self.assertFalse(data['accepted']);self.assertEqual(data['new_model_calls'],0)
        after=project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(before,after['source_semantic_execution'])
        self.assertNotIn('receipt',after['source_observation'])
        frozen=workflow.state_path(self.case).read_bytes()
        with patch('cv2.VideoCapture',side_effect=AssertionError('must reuse result')):
            self.assertEqual(self.cli(spec),result)
        self.assertEqual(frozen,workflow.state_path(self.case).read_bytes())
        self.cli(dict(action='freeze-route',file_id='one'),expected=2)

    def test_changed_trial_and_bad_authorization_are_rejected(self):
        spec=self.prepare_trial()
        with patch('carrier_tracking.register',return_value=None):self.cli(spec)
        changed=dict(spec,frames_half_open=[1,6],anchor_frames=[1,5])
        self.cli(changed,expected=2)
        auth=self.root/'authorization.json';auth.write_text('{}')
        self.cli(spec,expected=2)

    def test_missing_heldout_cannot_trigger_model_replay(self):
        spec=self.prepare_trial()
        with workflow.locked_state(self.case) as state:
            saved=state['files']['one']['source_semantic_execution']
            path=project.project_path(saved['frame_artifacts'].pop('2')['path']);path.unlink()
        with patch('semantic_scan.LocalModel',side_effect=AssertionError('no model fallback')):
            self.cli(spec,expected=2)


if __name__=='__main__':unittest.main()
