"""Regressions for the observed full-frame redraw/subtitle-loss failure."""
import unittest
import numpy as np


class RegionalWritebackTests(unittest.TestCase):
    def test_instance_plan_cannot_fall_back_to_legacy_single_carrier(self):
        import regional_composite as rc
        with self.assertRaisesRegex(ValueError,'unified execution manifest'):
            rc.render({}, {'file_id':'one','review':{'records':[{'instances':[{},{}]}]}}, 'unused.mp4')

    def test_color_change_is_limited_to_source_keyframe(self):
        import regional_composite as rc
        plan={'color_match':'context_median','color_match_overrides':{'712':'none'}}
        modes=rc.color_modes_for(plan,710,715)
        self.assertEqual(modes,['context_median','context_median','none','context_median','context_median'])
        with self.assertRaisesRegex(ValueError,'outside'):
            rc.color_modes_for(plan,713,715)

    def test_real_occluded_cloth_has_usable_color_context(self):
        import cv2
        import json
        from pathlib import Path
        import regional_composite as rc
        root = Path(__file__).resolve().parents[1]
        # Keep the exact mask that exposed the missing same-frame color context.
        plan = json.loads((root / 'evidence/full-erase-20261112/track-plan-v4.json').read_text())
        with np.load(root / plan['masks']) as stored:
            masks, warps = stored['alpha'], stored['raw_to_source']
        source = cv2.VideoCapture(str(root / 'cases/ecommerce-batch-002/inputs/makeup-08.mp4'))
        donor = cv2.VideoCapture(str(root / plan['cloud']))
        try:
            ok, raw = donor.read()
            self.assertTrue(ok)
            raw = cv2.resize(raw, (576, 1024))
            samples = np.full((1234, 3), np.nan, np.float32)
            hard_frames = (711, 761, 762, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 789, 846, 847, 848, 850, 851, 962)
            editable = np.zeros(1234, bool)
            nearby = sorted({n for center in hard_frames for n in range(center-12, center+13)})
            frames = {}
            for frame_number in nearby:
                source.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = source.read()
                self.assertTrue(ok)
                if not masks[frame_number].any():
                    continue
                editable[frame_number] = True
                patch = cv2.warpPerspective(raw, warps[frame_number], (576, 1024))
                try:
                    _, samples[frame_number] = rc.match_context_color(frame, patch, masks[frame_number], True)
                except ValueError:
                    pass
                frames[frame_number] = (frame, patch)
            calibrated, borrowed = rc.resolve_color_context(samples, editable)
            self.assertTrue(np.isfinite(samples[776]).all())
            self.assertTrue(np.isfinite(samples[781]).all())
            self.assertTrue(np.max(np.abs(samples[776])) < 45)
            for frame_number in hard_frames:
                with self.subTest(frame=frame_number):
                    frame, patch = frames[frame_number]
                    offset = calibrated[frame_number]
                    matched = np.clip(patch.astype(np.float32)+offset, 0, 255).astype(np.uint8)
                    self.assertTrue(np.isfinite(offset).all())
                    result, effective = rc.compose_frame(frame, matched, masks[frame_number])
                    np.testing.assert_array_equal(result[effective == 0], frame[effective == 0])
        finally:
            source.release()
            donor.release()

    def test_unmeasured_long_context_gap_is_rejected(self):
        import regional_composite as rc
        samples = np.full((20, 3), np.nan, np.float32)
        samples[0] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, 'No measured cloth context'):
            rc.resolve_color_context(samples, np.ones(20, bool))

    def test_partial_plan_cannot_request_full_delivery(self):
        import regional_composite as rc
        with self.assertRaisesRegex(ValueError, 'every source frame'):
            rc.validate_full_coverage([390, 540], 1234, ['visible'] * 150, np.ones(150, bool))
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            rc.validate_full_coverage([0, 3], 3, ['visible', 'unresolved', 'occluded'], [True, False, False])
        with self.assertRaisesRegex(ValueError, 'visible target'):
            rc.validate_full_coverage([0, 3], 3, ['visible', 'visible', 'occluded'], [True, False, False])
        rc.validate_full_coverage([0, 3], 3, ['visible', 'partial', 'occluded'], [True, True, False])

    def test_cached_fabric_donor_mapping_is_explicit_and_bounded(self):
        import regional_composite as rc
        rc.validate_donor_map(np.array([0, 0, 12], np.int32), 3, 150)
        for values in (np.array([0, 150, 0]), np.array([0, -1, 0]), np.array([0, 0]), np.array([0., 0., 0.])):
            with self.assertRaises(ValueError):
                rc.validate_donor_map(values, 3, 150)

    def test_color_match_uses_surrounding_fabric_not_original_letters(self):
        import regional_composite as rc
        source = np.full((40, 50, 3), 60, np.uint8)
        cloud = np.full_like(source, 40)
        alpha = np.zeros(source.shape[:2], np.uint8)
        alpha[15:25, 18:32] = 255
        source[17:22, 20:29] = 240  # removed letters must not influence color match
        cloud[18:20, 20:23] = 43  # actual provider texture remains after color adjustment
        matched, offset = rc.match_context_color(source, cloud, alpha)
        np.testing.assert_array_equal(offset, [20, 20, 20])
        np.testing.assert_array_equal(matched[18, 20], [63, 63, 63])
        actual, effective = rc.compose_frame(source, matched, alpha)
        np.testing.assert_array_equal(actual[effective == 0], source[effective == 0])

    def test_non_target_subtitles_and_protected_foreground_survive(self):
        import regional_composite as rc
        source = np.full((16, 20, 3), 77, np.uint8)
        source[13:] = 220  # subtitle pixels absent from the provider result
        cloud = np.full_like(source, 12)
        alpha = np.zeros(source.shape[:2], np.uint8)
        alpha[3:11, 5:14] = 255
        protected = np.zeros_like(alpha)
        protected[6:9, 10:16] = 255  # foreground crossing target
        actual, effective = rc.compose_frame(source, cloud, alpha, protected)
        np.testing.assert_array_equal(actual[effective == 0], source[effective == 0])
        np.testing.assert_array_equal(actual[effective == 255], cloud[effective == 255])
        self.assertGreater(np.count_nonzero(effective), 0)
        np.testing.assert_array_equal(actual[13:], source[13:])

    def test_missing_or_extra_cloud_frames_cannot_be_silently_retimed(self):
        import regional_composite as rc
        for count in (149, 151):
            with self.assertRaisesRegex(ValueError, 'frame count'):
                rc.validate_timing(30, 150, 30, count, 150)
        with self.assertRaisesRegex(ValueError, 'frame rate'):
            rc.validate_timing(30, 150, 25, 150, 150)
        rc.validate_timing(30, 150, 30, 150, 150)


if __name__ == '__main__':
    unittest.main()
