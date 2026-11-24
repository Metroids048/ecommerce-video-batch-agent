"""Recover saved preview artifacts at the normal entry, without another trial."""
import copy
import json
import unittest
from unittest.mock import patch

import execution_plan
import production_effort as effort
import project
import workflow
import test_production_effort as fixtures


class PreviewRecovery(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProductionEffort()
        self.fixture.setUp()
        self.root, self.case, self.manifest = self.fixture.root, self.fixture.case, self.fixture.manifest
        self.trial = self.fixture.preview('original')

    def tearDown(self):
        self.fixture.tearDown()

    def recover(self):
        with patch.object(execution_plan, '_preview', side_effect=AssertionError('must not render again')):
            return execution_plan.preview(self.case, self.manifest, self.root/'original.mp4', 0, .2)

    def test_pending_pass_and_fail_reuse_same_result_without_state_change(self):
        before = workflow.state_path(self.case).read_bytes()
        result = self.recover()
        self.assertEqual(result['trial_id'], self.trial['id'])
        self.assertEqual(result['preview_verdict'], 'pending')
        self.assertEqual(result['new_renders'], 0)
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)
        self.fixture.review(self.trial, 'pass')
        # Read-only artifact recovery is independent of current render code.
        before = workflow.state_path(self.case).read_bytes()
        with patch.object(effort, 'fingerprint', return_value='changed-code'):
            self.assertEqual(self.recover()['preview_verdict'], 'pass')
            with self.assertRaisesRegex(ValueError, 'passing preview'):
                effort.require_preview(self.case, self.manifest)
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)
        self.fixture.review(self.trial, 'fail', retract=True)
        before = workflow.state_path(self.case).read_bytes()
        self.assertEqual(self.recover()['preview_verdict'], 'fail')
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'passing preview'):
            effort.require_preview(self.case, self.manifest)

    def test_normal_cli_recovers_original_and_never_starts_new_trial(self):
        before = workflow.state_path(self.case).read_bytes()
        with patch('sys.argv', ['project.py','preview','--case','case.json','--plan','manifest.json',
                               '--out','original.mp4','--start','0','--end','0.2']), \
             patch.object(execution_plan, '_preview', side_effect=AssertionError('no render')):
            self.assertEqual(project.main(), 0)
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)

    def test_changed_dependencies_and_incomplete_result_never_recover(self):
        for name in ('source.mp4', 'a.npz', 'original.mp4', 'original.mp4.json'):
            path = self.root/name; saved = path.read_bytes()
            path.write_bytes(saved+b'changed')
            with self.assertRaises(ValueError):
                self.recover()
            path.write_bytes(saved)
        original = copy.deepcopy(self.manifest)
        self.manifest['risk_preview_ranges_half_open'] = [[0, 3]]
        with self.assertRaises(ValueError):
            self.recover()
        self.manifest = original
        with workflow.locked_state(self.case) as state:
            del state['files']['one']['preview_trials'][0]['report']
        with self.assertRaisesRegex(ValueError, 'incomplete|original preview'):
            self.recover()


if __name__ == '__main__':
    unittest.main()
