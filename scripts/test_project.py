"""Regressions for actual intake failures; no generation or modified source media."""
import copy
import unittest

import project


class InputAndRepairRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = project.read_json(project.ROOT / 'cases/beauty-batch-001/case.json')

    def test_replacement_without_new_asset_is_rejected(self):
        case = copy.deepcopy(self.case)
        case['files'][0]['operations'][0]['kind'] = 'replace'
        with self.assertRaisesRegex(ValueError, 'Replacement asset missing'):
            project.validate_case(case, verify_hash=False)

    def test_silent_source_cannot_count_as_audio_preservation(self):
        case = copy.deepcopy(self.case)
        silent = next(x for x in project.read_json(project.ROOT / 'cases/beauty-batch-001/source-probes.json') if x['id'] == 'ugc-01')
        case['files'][0]['source'] = silent['path']
        with self.assertRaisesRegex(ValueError, 'Original audio required but absent'):
            project.validate_case(case, verify_hash=False)

    def test_changed_source_cannot_reuse_an_old_identity(self):
        case = copy.deepcopy(self.case)
        case['files'][0]['source_sha256'] = 'invalid-source-hash'
        with self.assertRaisesRegex(ValueError, 'Source hash changed'):
            project.validate_case(case)

    def test_repair_only_invalidates_selected_file_interval(self):
        case = project.read_json(project.ROOT / 'cases/ecommerce-batch-002/case.json')
        with self.assertRaisesRegex(ValueError, 'Unverified parent'):
            project.repair_plan(self.case, 'ugc-18', 94, 99, 'historical-minimax-v1')
        result = project.repair_plan(case, 'makeup-08', 23, 24,
            'evidence/full-erase-20261112/candidates/makeup-08-full-v7.mp4')
        self.assertEqual(result['source_writeback_seconds'], [23, 24])
        self.assertEqual(set(result['reuse_other_file_ids']), {'fashion-aboutyou', 'makeup-03'})
        self.assertEqual(result['reuse_same_file_outside_seconds'][0], [0, 23])
        self.assertFalse(result['actual_repair_executed'])
        with self.assertRaisesRegex(ValueError, 'valid source interval'):
            project.repair_plan(self.case, 'ugc-18', 94, 999, 'historical-minimax-v1')


if __name__ == '__main__':
    unittest.main()
