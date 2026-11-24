"""A fixed stage must supply its own inputs to the next ordinary CLI action."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import execution_plan
import production_pipeline as pipeline
import project
import test_production_effort as fixtures
import workflow


class StageConsumption(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProductionEffort()
        self.fixture.setUp()
        self.root, self.case = self.fixture.root, self.fixture.case
        with contextlib.redirect_stdout(io.StringIO()):
            self.trial = self.fixture.preview('original')
        (self.root/'resume.json').write_text(json.dumps(dict(action='resume',file_id='one')))

    def tearDown(self):
        self.fixture.tearDown()

    def resume_cli(self):
        output = io.StringIO()
        with patch('sys.argv',['project.py','pipeline','--case','case.json','--spec','resume.json']), \
             contextlib.redirect_stdout(output):
            code = project.main()
        return code, json.loads(output.getvalue())

    def test_pending_trial_is_next_action_instead_of_upstream_or_export(self):
        before = workflow.state_path(self.case).read_bytes()
        with patch.object(execution_plan, 'render', side_effect=AssertionError('pending review cannot render')):
            code, result = self.resume_cli()
        self.assertEqual(code, 0, result)
        self.assertEqual(result['next_action'], 'review_existing_preview')
        self.assertEqual(result['saved_previews'][0]['trial_id'], self.trial['id'])
        self.assertEqual(result['fixed_materials']['manifest'], self.fixture.manifest)
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)

    def test_failed_trial_returns_bound_counterexample_and_does_not_try_export(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.fixture.review(self.trial, 'fail')
        before = workflow.state_path(self.case).read_bytes()
        with patch.object(execution_plan, 'render', side_effect=AssertionError('failed preview cannot render')):
            code, result = self.resume_cli()
        self.assertEqual(code, 0, result)
        self.assertEqual(result['next_action'], 'diagnose_saved_preview_failure')
        self.assertEqual(result['saved_previews'][0]['verdict'], 'fail')
        self.assertEqual(result['saved_previews'][0]['operation_ids'], ['a'])
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)

    def test_incomplete_original_trial_is_reconciled_before_any_review_or_retry(self):
        with workflow.locked_state(self.case) as state:
            trial = state['files']['one']['preview_trials'][0]
            del trial['report']; del trial['report_sha256']
        before = workflow.state_path(self.case).read_bytes()
        with patch.object(execution_plan, 'render', side_effect=AssertionError('must reconcile original trial')):
            code, result = self.resume_cli()
        self.assertEqual(code, 0, result)
        self.assertEqual(result['next_action'], 'reconcile_original_preview')
        self.assertEqual(result['saved_previews'][0]['trial_id'], self.trial['id'])
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)

    def test_pass_consumes_fixed_manifest_without_rebuilding_plan_or_renaming_candidate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.fixture.review(self.trial, 'pass')
        code, candidate = self.resume_cli()
        self.assertEqual(code, 0, candidate)
        self.assertTrue((self.root/candidate['output']).is_file())
        self.assertFalse(workflow.quality_reviews_pass(candidate))
        self.assertEqual(project.read_json(self.root/candidate['plan']), self.fixture.manifest)
        before = workflow.state_path(self.case).read_bytes()
        with patch.object(execution_plan, 'render', side_effect=AssertionError('must reuse original candidate')):
            code, recovered = self.resume_cli()
        self.assertEqual(code, 0, recovered)
        self.assertEqual(recovered['output_sha256'], candidate['output_sha256'])
        self.assertEqual(recovered['new_renders'], 0)
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
