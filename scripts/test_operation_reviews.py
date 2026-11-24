"""Operation-level review binding regressions; fixtures are not visual acceptance."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import project
import workflow


class OperationReviews(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.patcher = patch.object(project, 'ROOT', self.root)
        self.patcher.start()
        for name in ('source.mp4', 'output.mp4', 'input.txt', 'observation.jpg'):
            (self.root/name).write_bytes(name.encode())
        self.case = {'batch_id': 'test', 'case_version': 'v1', 'user_input': 'input.txt', 'files': [
            {'id': 'one', 'source': 'source.mp4', 'source_sha256': project.digest(self.root/'source.mp4'),
             'operations': [{'id': op, 'kind': 'remove', 'target': op, 'protect': ['rest']}
                            for op in ('a', 'b')]}]}
        (self.root/'manifest.json').write_text(json.dumps({'file_id': 'one', 'operations': []}))
        self.candidate = {'id': 'v1', 'file_id': 'one', 'output': 'output.mp4', 'plan': 'manifest.json',
            'source_sha256': self.case['files'][0]['source_sha256'],
            'output_sha256': project.digest(self.root/'output.mp4'),
            'dependency': workflow.requirement_binding(self.case, self.case['files'][0]),
            'execution_manifest': True, 'operation_ids': ['a', 'b'], 'parent_id': None,
            'reviews': [{'kind': kind, 'status': 'pass', 'output_sha256': project.digest(self.root/'output.mp4')}
                        for kind in ('technical', 'audio_preservation')]}
        self.state = {'revision': 1, 'case_version': 'v1', 'case_binding': workflow.case_binding(self.case),
            'files': {'one': {'candidates': [self.candidate], 'issues': []}}}
        self.path = workflow.state_path(self.case)
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps(self.state))
        self.review = {'expected_revision': 1, 'file_id': 'one', 'candidate_id': 'v1',
            'source_sha256': self.candidate['source_sha256'],
            'output_sha256': self.candidate['output_sha256'], 'dependency': self.candidate['dependency'],
            'checks': [{'kind': kind, 'status': 'pass', 'scope': 'fixture observation'}
                       for kind in ('target_coverage', 'visual', 'playback', 'delivery')],
            'operation_reviews': [self.operation(op) for op in ('a', 'b')]}

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def operation(self, op):
        return {'operation_id': op, 'output_sha256': self.candidate['output_sha256'],
            'dependency': self.candidate['dependency'], 'checks': [
                {'kind': kind, 'status': 'pass', 'scope': 'fixture, not business quality',
                 'evidence': [{'path': 'observation.jpg', 'sha256': project.digest(self.root/'observation.jpg')}]}
                for kind in ('target_coverage', 'visual', 'protection', 'temporal')]}

    def record(self):
        (self.root/'qa.json').write_text(json.dumps(self.review))
        return workflow.record_agent_review(self.case, 'qa.json')

    def test_file_level_pass_does_not_replace_missing_operation(self):
        self.review['operation_reviews'].pop()
        with self.assertRaisesRegex(ValueError, 'operation'):
            self.record()
        self.assertEqual(project.read_json(self.path)['revision'], 1)

    def test_stale_operation_cannot_be_imported(self):
        self.review['operation_reviews'][1]['output_sha256'] = 'old'
        with self.assertRaisesRegex(ValueError, 'binding'):
            self.record()

    def test_real_evidence_hash_required(self):
        self.review['operation_reviews'][1]['checks'][0]['evidence'][0]['sha256'] = 'old'
        with self.assertRaisesRegex(ValueError, 'evidence'):
            self.record()

    def test_limits_remain_limits_even_if_file_checks_pass(self):
        self.review['operation_reviews'][1]['checks'][0]['status'] = 'reviewed_with_limits'
        self.assertEqual(self.record()['status'], 'review_copy_with_limits')

    def test_unedited_instance_or_dropout_cannot_be_overridden_by_pass(self):
        self.record()
        candidate=project.read_json(self.path)['files']['one']['candidates'][0]
        for finding in ({'known_unedited_instances':[{'frame':1}]},
                        {'replacement_dropout_risks':[[1,2]]}):
            candidate['operation_results']=[dict(operation_id='a',**finding)]
            self.assertFalse(workflow.quality_reviews_pass(candidate))

    def test_complete_bound_operations_can_qualify_and_evidence_is_immutable(self):
        self.assertEqual(self.record()['status'], 'qualified')
        candidate = project.read_json(self.path)['files']['one']['candidates'][0]
        self.assertTrue(workflow.quality_reviews_pass(candidate))
        (self.root/'observation.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'dependency hash changed'):
            workflow.verify_candidate(self.case, 'one', candidate)

    def test_fixed_operation_closes_without_hiding_other_operation_limits(self):
        state = project.read_json(self.path)
        state['files']['one']['issues'] = [{'id': 'fixed-a', 'operation_id': 'a',
            'status': 'pending_recheck', 'result_candidate_id': 'v1'}]
        self.path.write_text(json.dumps(state))
        self.review['operation_reviews'][1]['checks'][0]['status'] = 'reviewed_with_limits'
        self.review['resolved_issue_ids'] = ['fixed-a']
        self.assertEqual(self.record()['status'], 'review_copy_with_limits')
        self.assertEqual(project.read_json(self.path)['files']['one']['issues'][0]['status'], 'closed')

    def fixed_local_repair(self):
        self.case['execution_policy'] = {'local_only': True}
        state = project.read_json(self.path)
        state['case_binding'] = workflow.case_binding(self.case)
        entry = state['files']['one']; candidate = entry['candidates'][0]
        pixel = {'file_id': 'one', 'operation_id': 'a', 'composition_mode': 'source_recolor'}
        (self.root/'pixels.json').write_text(json.dumps(pixel))
        manifest = {'file_id': 'one', 'dependency': candidate['dependency'], 'operations': [
            {'operation_id': 'a', 'plan': 'pixels.json',
             'plan_sha256': project.digest(self.root/'pixels.json')}]}
        (self.root/'manifest.json').write_text(json.dumps(manifest))
        (self.root/'raw-review.json').write_text('{"status":"fail"}')
        raw_review = {'path': 'raw-review.json', 'sha256': project.digest(self.root/'raw-review.json'),
                      'status': 'fail', 'raw_sha256': 'original-raw'}
        artifacts = {p: project.digest(self.root/p) for p in
                     ('pixels.json', 'manifest.json', 'raw-review.json')}
        entry['execution_units'] = [{'unit_id': 'original-unit', 'raw_review': raw_review}]
        entry['issues'] = [{'id': 'original-failure', 'operation_id': 'a', 'status': 'open',
            'observation_basis': 'provider_raw', 'unit_id': 'original-unit',
            'raw_sha256': 'original-raw', 'source_sha256': candidate['source_sha256'],
            'dependency': candidate['dependency']}]
        entry['local_material_repair'] = dict(dependency=candidate['dependency'],
            artifacts=artifacts, manifest='manifest.json', pixel_plan_fingerprint=workflow.hash_json(pixel),
            parent_unit_ids=['original-unit'], parent_issue_ids=['original-failure'],
            parent_reviews={'original-unit': raw_review})
        candidate.update(plan='manifest.json', artifact_hashes=artifacts.copy(), operation_results=[
            {'operation_id': 'a', 'method': 'source_recolor', 'regenerated_source_geometry': False}])
        self.review['resolved_issue_ids'] = ['original-failure']
        self.path.write_text(json.dumps(state))
        return state

    def test_fixed_local_material_provenance_can_close_actual_parent_issue(self):
        self.fixed_local_repair()
        # This fixture proves lineage only; it has no full media/coverage trial.
        self.review['checks'][-1]['status'] = 'reviewed_with_limits'
        self.assertEqual(self.record()['status'], 'review_copy_with_limits')
        entry = project.read_json(self.path)['files']['one']
        self.assertEqual(entry['issues'][0]['status'], 'closed')
        self.assertEqual(entry['execution_units'][0]['raw_review']['status'], 'fail')
        self.assertFalse(entry['candidates'][0]['accepted'])

    def test_fixed_local_material_requires_each_original_parent_exactly_once(self):
        from local_material_repair import verify
        for duplicate in (False,True):
            with self.subTest(duplicate=duplicate):
                state=self.fixed_local_repair();entry=state['files']['one']
                parents=entry['execution_units']
                entry['execution_units']=parents*2 if duplicate else []
                with self.assertRaisesRegex(ValueError,'original parent units'):
                    verify(self.case,'one',entry)

    def test_fixed_local_material_rechecks_bound_parent_without_binding_unrelated_journal(self):
        from local_material_repair import verify
        from runninghub_adapter import raw_origin_binding
        state=self.fixed_local_repair();entry=state['files']['one'];unit=entry['execution_units'][0]
        entry['local_material_repair']['parent_unit_bindings']={unit['unit_id']:raw_origin_binding(unit)}
        unit['status_note']='unrelated journal update'
        self.assertIsNotNone(verify(self.case,'one',entry))
        unit['provider_task_id']='different-original'
        with self.assertRaisesRegex(ValueError,'parent unit binding changed'):
            verify(self.case,'one',entry)

    def test_fixed_local_material_provenance_does_not_waive_operation_quality(self):
        for status in ('fail', 'reviewed_with_limits'):
            with self.subTest(status=status):
                self.fixed_local_repair()
                self.review['operation_reviews'][0]['checks'][2]['status'] = status
                with self.assertRaisesRegex(ValueError, 'Unverified operation'):
                    self.record()
                self.assertEqual(project.read_json(self.path)['files']['one']['issues'][0]['status'], 'open')

    def test_fixed_local_material_cannot_close_unrelated_or_unbound_issue(self):
        changes = [lambda e, c: e['issues'][0].update(raw_sha256='different'),
                   lambda e, c: e['issues'][0].update(unit_id='different'),
                   lambda e, c: e['issues'][0].update(operation_id='b'),
                   lambda e, c: e['issues'][0].update(source_sha256='different'),
                   lambda e, c: e['issues'][0].update(dependency={'different': True}),
                   lambda e, c: c.update(plan='different.json'),
                   lambda e, c: c['artifact_hashes'].pop('pixels.json'),
                   lambda e, c: c['operation_results'][0].update(regenerated_source_geometry=True)]
        for change in changes:
            with self.subTest(change=change):
                state = self.fixed_local_repair(); entry = state['files']['one']
                change(entry, entry['candidates'][0]); self.path.write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError, 'unrelated'):
                    self.record()

    def test_quality_rechecks_operation_binding(self):
        self.candidate['reviews'] += copy.deepcopy(self.review['checks'])
        for check in self.candidate['reviews']:
            check['output_sha256'] = self.candidate['output_sha256']
        self.candidate['operation_reviews'] = copy.deepcopy(self.review['operation_reviews'])
        self.candidate['operation_reviews'][1]['dependency'] = {'old': True}
        self.assertFalse(workflow.quality_reviews_pass(self.candidate))


if __name__ == '__main__':
    unittest.main()
