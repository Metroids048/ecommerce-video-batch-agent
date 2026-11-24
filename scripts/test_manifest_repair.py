"""Exercise a bound multi-operation repair through the ordinary CLI, using six synthetic frames."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import cv2
import execution_plan
import project
import test_production_effort as fixtures
import workflow


class ManifestRepair(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProductionEffort()
        self.fixture.setUp()
        self.root, self.case = self.fixture.root, self.fixture.case
        with contextlib.redirect_stdout(io.StringIO()):
            trial = self.fixture.preview('original')
            self.fixture.review(trial, 'pass')
            self.parent = workflow.execute_plan(self.case, 'manifest.json', 'parent')
            self.fixture.fixture.review_before_repair(self.parent)
        state = project.read_json(workflow.state_path(self.case))
        self.issue = workflow.add_feedback(self.case, dict(file_id='one', candidate_id='parent',
            candidate_sha256=self.parent['output_sha256'], source_sha256=self.parent['source_sha256'],
            expected_revision=state['revision'], source_time=.05, text='Correct only operation a',
            operation_id='a', category='composition'))
        plan = project.read_json(self.root/'a.json')
        plan['label_style']['ink_opacity'] = 0
        (self.root/'a-repair.json').write_text(json.dumps(plan))
        self.spec = dict(action='prepare-repair', file_id='one', issue_id=self.issue['id'],
            operation_plan=dict(path='a-repair.json', sha256=project.digest(self.root/'a-repair.json')))

    def tearDown(self):
        self.fixture.tearDown()

    def cli(self, spec):
        (self.root/'pipeline-request.json').write_text(json.dumps(spec))
        output = io.StringIO()
        with patch('sys.argv', ['project.py','pipeline','--case','case.json','--spec','pipeline-request.json']), \
             contextlib.redirect_stdout(output):
            code = project.main()
        return code, json.loads(output.getvalue())

    def prepare(self):
        code, result = self.cli(self.spec)
        self.assertEqual(code, 0, result)
        return result

    def test_prepare_keeps_unaffected_operation_and_reuses_exact_handoff(self):
        prepared = self.prepare()
        manifest = project.read_json(self.root/prepared['manifest']['path'])
        self.assertEqual(manifest['operations'][1], self.fixture.manifest['operations'][1])
        self.assertEqual(manifest['operations'][0]['plan'], 'a-repair.json')
        self.assertEqual(manifest['parent_output_sha256'], self.parent['output_sha256'])
        before = workflow.state_path(self.case).read_bytes()
        with patch.object(execution_plan, 'render', side_effect=AssertionError('prepare cannot render')):
            self.assertEqual(self.prepare()['manifest'], prepared['manifest'])
            code, current = self.cli(dict(action='resume',file_id='one'))
        self.assertEqual(code, 0, current)
        self.assertEqual(current['next_action'], 'preview_fixed_repair')
        self.assertEqual(current['repair_handoff']['issue_id'], self.issue['id'])
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)

    def test_preview_resume_and_review_keep_parent_issue_and_complete_output(self):
        prepared = self.prepare()
        with patch('sys.argv', ['project.py','preview','--case','case.json','--plan',prepared['manifest']['path'],
                  '--out','repair-preview.mp4','--start','0','--end','0.2']), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(project.main(), 0)
        code, waiting = self.cli(dict(action='resume',file_id='one'))
        self.assertEqual(waiting['next_action'], 'review_existing_preview')
        trial = project.read_json(workflow.state_path(self.case))['files']['one']['preview_trials'][-1]
        with contextlib.redirect_stdout(io.StringIO()):
            self.fixture.review(trial, 'pass')
        code, child = self.cli(dict(action='resume',file_id='one'))
        self.assertEqual(code, 0, child)
        self.assertEqual(child['parent_id'], 'parent')
        self.assertEqual(child['operation_ids'], ['a','b'])
        frames = []
        for candidate in (self.parent, child):
            cap = cv2.VideoCapture(str(self.root/candidate['output']))
            ok, frame = cap.read(); cap.release()
            self.assertTrue(ok); frames.append(frame)
        self.assertGreater(float(frames[1][23:33,10:30].mean()), 250)
        self.assertLess(float(frames[1][23:33,65:85].mean()), 245)
        self.assertFalse(workflow.quality_reviews_pass(child))
        entry = project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(entry['issues'][0]['status'], 'pending_recheck')
        self.assertEqual(entry['issues'][0]['result_candidate_id'], child['id'])
        before = workflow.state_path(self.case).read_bytes()
        with patch.object(execution_plan, 'render', side_effect=AssertionError('child already exists')):
            _, recovered = self.cli(dict(action='resume',file_id='one'))
        self.assertEqual(recovered['output_sha256'], child['output_sha256'])
        self.assertEqual(recovered['new_renders'], 0)
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)

        # Review the exact child, retaining the failed parent's separate evidence.
        review = project.read_json(self.root/'fixture-review.json')
        review.update(expected_revision=project.read_json(workflow.state_path(self.case))['revision'],
                      candidate_id=child['id'], output_sha256=child['output_sha256'],
                      resolved_issue_ids=[self.issue['id']])
        for check in review['checks']:
            check['status'] = 'pass'
        for operation in review['operation_reviews']:
            operation['output_sha256'] = child['output_sha256']
            for check in operation['checks']:
                check['status'] = 'pass'
        (self.root/'child-review.json').write_text(json.dumps(review))
        with patch('sys.argv',['project.py','agent-review','--case','case.json','--request','child-review.json']), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(project.main(),0)
        _, delivered = self.cli(dict(action='resume',file_id='one',delivery_out='fixture-delivery'))
        self.assertEqual(delivered['delivery']['status'],'qualified_delivery')
        self.assertFalse(delivered['delivery']['accepted'])
        final = project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(final['issues'][0]['status'],'closed')
        self.assertFalse(workflow.quality_reviews_pass(workflow.candidate_for(final,'parent')))

    def test_tampered_or_wrong_operation_material_cannot_prepare(self):
        self.spec['operation_plan']['sha256'] = '0'*64
        self.assertEqual(self.cli(self.spec)[0], 2)
        self.spec['operation_plan'] = dict(path='b.json',sha256=project.digest(self.root/'b.json'))
        self.assertEqual(self.cli(self.spec)[0], 2)

    def test_saved_repair_and_other_operation_tamper_are_rejected(self):
        prepared = self.prepare()
        (self.root/'b.json').write_text('{}')
        self.assertEqual(self.cli(dict(action='resume',file_id='one'))[0],2)

    def test_unreviewed_correction_leaves_no_new_manifest_or_revision(self):
        self.prepare()
        before = workflow.state_path(self.case).read_bytes()
        plans = set((workflow.state_path(self.case).parent/'plans').glob('*.json'))
        changed = project.read_json(self.root/'a-repair.json')
        changed['label_style']['ink_opacity'] = .5
        (self.root/'a-next.json').write_text(json.dumps(changed))
        self.spec['operation_plan'] = dict(path='a-next.json',sha256=project.digest(self.root/'a-next.json'))
        self.assertEqual(self.cli(self.spec)[0],2)
        self.assertEqual(workflow.state_path(self.case).read_bytes(),before)
        self.assertEqual(set((workflow.state_path(self.case).parent/'plans').glob('*.json')),plans)

    def test_prepare_cannot_switch_method_or_bypass_mode_or_review(self):
        plan = project.read_json(self.root/'a-repair.json')
        plan['composition_mode'] = 'source_recolor'
        (self.root/'a-repair.json').write_text(json.dumps(plan))
        self.spec['operation_plan']['sha256'] = project.digest(self.root/'a-repair.json')
        self.assertEqual(self.cli(self.spec)[0],2)
        (self.root/'config').mkdir()
        (self.root/'config/models.json').write_text(json.dumps(dict(execution_mode='maintenance')))
        self.assertEqual(self.cli(self.spec)[0],2)

    def test_failed_repair_preview_returns_failure_instead_of_old_parent_or_export(self):
        prepared = self.prepare()
        manifest = project.read_json(self.root/prepared['manifest']['path'])
        with contextlib.redirect_stdout(io.StringIO()):
            execution_plan.preview(self.case,manifest,self.root/'failed-repair.mp4',0,.2)
            trial = project.read_json(workflow.state_path(self.case))['files']['one']['preview_trials'][-1]
            self.fixture.review(trial,'fail')
        before = workflow.state_path(self.case).read_bytes()
        with patch.object(execution_plan, 'render', side_effect=AssertionError('failure cannot export')):
            code, result = self.cli(dict(action='resume',file_id='one'))
        self.assertEqual(code, 0, result)
        self.assertEqual(result['next_action'], 'diagnose_saved_preview_failure')
        self.assertEqual(result['saved_previews'][-1]['verdict'], 'fail')
        self.assertEqual(workflow.state_path(self.case).read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
