"""Regressions from the actual compressed source and supervised product boundary."""
import unittest
from pathlib import Path

import cv2
import numpy as np

from track_clothing_logo import ink_mask


class ClothingMaskTests(unittest.TestCase):
    def test_white_print_with_compressed_skin_chroma_is_editable(self):
        root = Path(__file__).resolve().parents[1]
        cap = cv2.VideoCapture(str(root / 'cases/ecommerce-batch-002/inputs/makeup-08.mp4'))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 781)
            ok, frame = cap.read()
            self.assertTrue(ok)
            alpha, protected, _ = ink_mask(frame, [160, 762, 102, 28])
            # This bright A pixel was incorrectly retained as a finger in v5.
            self.assertGreater(alpha[777, 255], 200)
            self.assertEqual(protected[777, 255], 0)
            self.assertEqual(alpha[775, 240], 0)  # actual finger interior
        finally:
            cap.release()

    def test_actual_full_plan_preserves_occluding_compact(self):
        import json
        root = Path(__file__).resolve().parents[1]
        plan = json.loads((root / 'evidence/full-erase-20261112/track-plan-v6.json').read_text())
        with np.load(root / plan['masks']) as stored:
            alpha = stored['alpha']
            # v7 painted a dark rectangle over this fully opaque Dior compact.
            self.assertFalse(alpha[712].any())
            # Preserve the silver rim, while still removing the visible print.
            self.assertEqual(alpha[1135, 778, 276], 0)
            self.assertGreater(alpha[1135, 770, 230], 200)

    def test_product_exclusion_remains_hard_boundary(self):
        frame = np.full((40, 60, 3), 210, np.uint8)
        polygon = [[10, 10], [24, 10], [20, 30], [10, 30]]
        alpha, _, _ = ink_mask(frame, [10, 10, 40, 20], [polygon])
        self.assertTrue(alpha[15:25, 12:18].any())
        self.assertFalse(alpha[:, 26:].any())


if __name__ == '__main__':
    unittest.main()
