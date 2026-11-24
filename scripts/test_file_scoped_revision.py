"""Unrelated batch progress must not discard bound observations for another file."""
import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import project
import test_operation_reviews as fixtures
import workflow


class FileScopedRevision(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OperationReviews()
        self.fixture.setUp()
        self.root, self.case, self.state = self.fixture.root, self.fixture.case, self.fixture.state
        one = self.case['files'][0]
        one.update(duration_seconds=10, fps_container_average=30)
        self.case['files'].append(dict(copy.deepcopy(one), id='two'))
        entry = self.state['files']['one']
        entry.update(current_candidate_id='v1', raw_returns=[], execution_units=[], jobs=[])
        entry['candidates'][0]['status'] = 'pending_review'
        self.state['files']['two'] = dict(candidates=[],current_candidate_id=None,issues=[],jobs=[],raw_returns=[],execution_units=[])
        self.state['case_binding'] = workflow.case_binding(self.case)
        self.fixture.path.write_text(json.dumps(self.state))
        (self.root/'case.json').write_text(json.dumps(self.case))
        self.token = workflow.hash_json(self.state['files']['one'])
        self.fixture.review['expected_file_state'] = self.token

    def tearDown(self):
        self.fixture.tearDown()

    def change(self, fid):
        with workflow.locked_state(self.case) as state:
            state['files'][fid]['jobs'].append(dict(id='unrelated-progress',status='complete'))

    def test_snapshot_supplies_original_file_token_before_derived_view_fields(self):
        self.assertEqual(workflow.snapshot(self.case)['files']['one']['file_state_token'], self.token)

    def test_bound_agent_review_survives_other_file_progress_at_normal_cli(self):
        self.change('two')
        (self.root/'qa.json').write_text(json.dumps(self.fixture.review))
        output = io.StringIO()
        with patch('sys.argv',['project.py','agent-review','--case','case.json','--request','qa.json']), \
             contextlib.redirect_stdout(output):
            code = project.main()
        self.assertEqual(code, 0, output.getvalue())
        self.assertEqual(json.loads(output.getvalue())['status'], 'qualified')
        self.assertEqual(len(project.read_json(self.fixture.path)['files']['two']['jobs']), 1)

    def test_same_file_change_and_legacy_stale_revision_still_reject(self):
        self.change('one')
        with self.assertRaisesRegex(ValueError, 'conflict'):
            self.fixture.record()
        self.fixture.review.pop('expected_file_state')
        with self.assertRaisesRegex(ValueError, 'conflict'):
            self.fixture.record()

    def test_current_global_revision_never_allows_a_wrong_file_token(self):
        self.fixture.review['expected_file_state'] = 'wrong-file'
        with self.assertRaisesRegex(ValueError, 'conflict'):
            self.fixture.record()

    def test_other_file_progress_never_excuses_changed_output_or_observation(self):
        self.change('two')
        for name in ('output.mp4','observation.jpg'):
            path = self.root/name; saved = path.read_bytes(); path.write_bytes(saved+b'changed')
            with self.assertRaises(ValueError):
                self.fixture.record()
            path.write_bytes(saved)

    def test_feedback_keeps_original_observation_when_only_other_file_changed(self):
        self.change('two')
        body = dict(file_id='one',candidate_id='v1',candidate_sha256=self.fixture.candidate['output_sha256'],
            source_sha256=self.fixture.candidate['source_sha256'],expected_revision=1,
            expected_file_state=self.token,operation_id='a',source_time=1,text='Observed original edge defect')
        issue = workflow.add_feedback(self.case, body)
        self.assertEqual(issue['operation_id'], 'a')
        self.assertEqual(issue['candidate_id'], 'v1')
        with self.assertRaisesRegex(ValueError, 'conflict'):
            workflow.add_feedback(self.case, body)


if __name__ == '__main__':
    unittest.main()
