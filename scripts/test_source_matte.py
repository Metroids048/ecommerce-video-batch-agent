import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import project


class SourceMatte(unittest.TestCase):
    def test_cli_unions_overlaps_scales_coordinates_and_protects_after_feather(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project, 'ROOT', Path(directory).resolve()):
            root = Path(directory).resolve()
            (root/'source.bin').write_bytes(b'source identity')
            source_hash = project.digest(root/'source.bin')
            case = {'batch_id': 'matte', 'files': [{'id': 'one', 'source': 'source.bin',
                'source_sha256': source_hash, 'dimensions': [80, 60],
                'source_frame_count_inventory': 2, 'operations': [{'id': 'cloth'}]}]}
            (root/'case.json').write_text(json.dumps(case))
            spec = dict(file_id='one', operation_id='cloth', source_sha256=source_hash,
                coordinate_size=[40, 30], mask_size=[80, 60], frames_half_open=[0, 2],
                feather_sigma=1, records=[{'frame': 0, 'state': 'visible',
                    'target_polygons': [[[2, 2], [20, 2], [20, 20], [2, 20]],
                                        [[10, 10], [28, 10], [28, 26], [10, 26]]],
                    'protected_polygons': [[[13, 13], [16, 13], [16, 16], [13, 16]]]},
                    {'frame': 1, 'state': 'absent', 'target_polygons': [], 'protected_polygons': []}])
            (root/'spec.json').write_text(json.dumps(spec))
            with patch('sys.argv', ['project.py', 'matte', '--case', 'case.json',
                                   '--spec', 'spec.json', '--out', 'mask.npz']):
                self.assertEqual(project.main(), 0)
            with np.load(root/'mask.npz', allow_pickle=False) as z:
                self.assertEqual(z['alpha'][0, 23, 23], 255)  # overlap cannot become an even-odd hole
                self.assertEqual(z['alpha'][0, 46, 50], 255)  # explicit 2x coordinate conversion
                self.assertEqual(z['alpha'][0, 28, 28], 0)    # protected after blur
                self.assertEqual(z['protected'][0, 28, 28], 255)
                self.assertFalse(z['alpha'][1].any())
            report = json.loads((root/'mask.npz.json').read_text())
            self.assertFalse(report['accepted'])
            self.assertEqual(report['status'], 'prepared_pending_visual_review')
            event = json.loads((root/'runs/matte/timings.jsonl').read_text().splitlines()[-1])
            self.assertEqual(event['file_id'], 'one')
            self.assertEqual(event['operation_id'], 'cloth')
            from source_matte import prepare, validate_metadata
            with np.load(root/'mask.npz', allow_pickle=False) as arrays:
                validate_metadata(arrays, source_hash, [80, 60], [0, 2])
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    validate_metadata(arrays, 'another-source', [80, 60], [0, 2])
            spec['source_sha256'] = 'another-source'
            with self.assertRaisesRegex(ValueError, 'another source'):
                prepare(case, spec, 'wrong.npz')
            self.assertFalse((root/'wrong.npz').exists())

    def test_missing_frame_wrong_source_and_coordinate_scale_are_rejected(self):
        from source_matte import rasterize
        base = dict(coordinate_size=[40, 30], mask_size=[80, 60], frames_half_open=[0, 2],
                    records=[dict(frame=0, state='absent', target_polygons=[], protected_polygons=[])])
        with self.assertRaisesRegex(ValueError, 'every frame'):
            rasterize(base, [80, 60])
        base['records'].append(dict(frame=1, state='absent', target_polygons=[], protected_polygons=[]))
        base['coordinate_size'] = [40, 40]
        with self.assertRaisesRegex(ValueError, 'aspect'):
            rasterize(base, [80, 60])
        base['coordinate_size'] = [40, 30]
        base['records'][0] = dict(frame=0, state='visible',
            target_polygons=[[[0, 0], [41, 0], [2, 5]]], protected_polygons=[])
        with self.assertRaisesRegex(ValueError, 'coordinate'):
            rasterize(base, [80, 60])


if __name__ == '__main__':
    unittest.main()
