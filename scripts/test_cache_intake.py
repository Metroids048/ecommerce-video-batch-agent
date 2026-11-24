"""Exact-cache intake regressions; synthetic media is not business acceptance."""
import copy
import json
import unittest
from unittest.mock import patch

import delivery_package
import production_pipeline as pipeline
import project
import workflow
import test_production_pipeline as fixtures


class CacheIntake(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PipelineTests(); self.fixture.setUp()
        self.root, self.parent = self.fixture.root, self.fixture.case
        candidate = workflow.execute_plan(self.parent, 'manifest.json', 'v1')
        with workflow.locked_state(self.parent) as state:
            candidate = state['files']['one']['candidates'][0]
            for review in candidate['reviews']:
                if review['kind'] == 'visual': review['status'] = 'pass'
            for kind in ('target_coverage', 'visual', 'playback', 'delivery'):
                if kind in {r['kind'] for r in candidate['reviews']}: continue
                candidate['reviews'].append(dict(kind=kind, status='pass',
                    output_sha256=candidate['output_sha256'], scope='six synthetic frames'))
            candidate['operation_reviews'] = [dict(operation_id=op, output_sha256=candidate['output_sha256'],
                dependency=candidate['dependency'], checks=[dict(kind=k, status='pass', scope='synthetic',
                    evidence=[dict(path='input.txt', sha256=project.digest(self.root/'input.txt'))])
                    for k in ('target_coverage', 'visual', 'protection', 'temporal')]) for op in ('a', 'b')]
            candidate['status'] = 'qualified'
        self.request = dict(batch_id='cache-new', user_input='input.txt', reuse_only=True,
                            files=copy.deepcopy(self.parent['files']))
        (self.root/'request-cache.json').write_text(json.dumps(self.request))
        self.spec = dict(action='reuse-candidate', file_id='one', origin=dict(
            case='case.json', case_sha256=project.digest(self.root/'case.json'), file_id='one', candidate_id='v1'))

    def tearDown(self): self.fixture.tearDown()

    def create(self):
        project.intake('cache-new/case.json', 'request-cache.json')
        self.case = project.read_json(self.root/'cache-new/case.json')
        workflow.initialize(self.case, 'cache-new/case.json')

    def test_cache_enters_pending_review_and_recovery_is_idempotent(self):
        self.create(); before = workflow.state_path(self.parent).read_bytes()
        result = pipeline.dispatch(self.case, self.spec)
        self.assertEqual(result['status'], 'cached_candidate_pending_review')
        self.assertEqual(workflow.snapshot(self.case)['files']['one']['delivery_status'], 'pending_review')
        with self.assertRaisesRegex(ValueError, 'Unqualified'):
            delivery_package.build(self.case, 'cannot-deliver')
        current = workflow.state_path(self.case).read_bytes()
        pipeline.dispatch(self.case, self.spec)
        self.assertEqual(current, workflow.state_path(self.case).read_bytes())
        self.assertEqual(before, workflow.state_path(self.parent).read_bytes())

    def test_changed_requirements_or_assets_are_cache_misses(self):
        for change in ('target', 'protect', 'source_sha256', 'asset'):
            with self.subTest(change=change):
                self.create()
                if change == 'asset': (self.root/'asset.png').write_bytes(b'changed')
                else:
                    item = self.case['files'][0]
                    if change == 'source_sha256': item[change] = '0'*64
                    else: item['operations'][0][change] = 'changed'
                    with workflow.locked_state(self.case) as state:
                        state['case_binding'] = workflow.case_binding(self.case)
                with self.assertRaises(ValueError): pipeline.dispatch(self.case, self.spec)
                workflow.state_path(self.case).unlink(); (self.root/'cache-new/case.json').unlink()
                (self.root/'cache-new/one-timeline.json').unlink()

    def test_cache_origin_tampering_invalidates_candidate(self):
        self.create(); pipeline.dispatch(self.case, self.spec)
        (self.root/'case.json').write_text('{}')
        self.assertEqual(workflow.snapshot(self.case)['files']['one']['delivery_status'], 'blocked')

    def test_reuse_only_rejects_production_and_research_without_global_toggle(self):
        self.create()
        (self.root/'config').mkdir(exist_ok=True)
        for mode in ('production', 'maintenance'):
            (self.root/'config/models.json').write_text(json.dumps(dict(execution_mode=mode)))
            with self.assertRaisesRegex(ValueError, 'Reuse-only'):
                pipeline.require_production(self.case)
            with self.assertRaisesRegex(ValueError, 'Reuse-only'):
                pipeline.require_production(batch_id=self.case['batch_id'])
            with self.assertRaisesRegex(ValueError, 'Reuse-only'):
                pipeline.dispatch(self.case, dict(action='freeze-route'))
            with patch('regional_composite.render', side_effect=AssertionError('no render')):
                self.assertEqual(pipeline.dispatch(self.case, dict(action='resume-batch'))['new_renders'], 0)


if __name__ == '__main__': unittest.main()
