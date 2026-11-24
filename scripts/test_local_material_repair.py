import unittest
import numpy as np
from local_material_repair import native_region, material_components, observed_region, repair_scope, prepare


class LocalMaterialRepairTests(unittest.TestCase):
    def test_soft_red_retains_desaturated_fabric_without_brown_or_skin(self):
        import cv2
        hsv=np.array([[[178,60,170],[178,100,160],[12,70,180],[6,120,160],[90,120,160]]],np.uint8)
        frame=cv2.cvtColor(np.repeat(hsv,3,axis=0),cv2.COLOR_HSV2BGR)
        record=dict(regions=[[[0,0],[5,0],[5,3],[0,3]]],protected=[],
            material_gate=dict(soft_red=dict(inner_hue=3,outer_hue=8,saturation_ramp=[25,65])))
        mask=observed_region(frame,record,[5,3])
        self.assertGreater(mask[1,0],140)
        self.assertLess(mask[1,4],10)

    def test_observed_colour_region_preserves_skin_and_unrelated_red(self):
        import cv2
        hsv = np.array([[[178,200,180],[2,180,180],[8,75,180],[2,200,180],[2,200,180]]],np.uint8)
        frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        record = dict(regions=[[[0,0],[3,0],[3,1],[0,1]]],
                      protected=[[[1,0],[1.1,0],[1.1,1],[1,1]]],
                      material_gate=dict(hue_ranges=[[160,179],[0,8]],minimum_saturation=90))
        mask = observed_region(frame,record,[5,1])
        self.assertEqual(mask[0,0],255)
        self.assertEqual(mask[0,1],0)
        self.assertEqual(mask[0,2],0)
        self.assertEqual(mask[0,4],0)

    def test_historical_provider_first_repair_binding_remains_readable(self):
        # Reading old artifacts does not authorize creation of another repair.
        case=dict(execution_policy=dict(editing_strategy='provider_first_v1'))
        self.assertTrue(repair_scope(case,'observed_colour_regions'))
        self.assertFalse(repair_scope(case,'semantic_material_components'))
        self.assertFalse(repair_scope(case,'native_colour_region'))

    def test_provider_first_blocks_new_colour_repair_before_source_or_state_read(self):
        from unittest.mock import patch
        import production_pipeline
        case = dict(execution_policy=dict(editing_strategy='provider_first_v1', local_only=True))
        with patch.object(production_pipeline, 'require_production') as production:
            with self.assertRaisesRegex(ValueError, 'existing editor tools'):
                prepare(case, dict(kind='observed_colour_regions'))
            production.assert_not_called()

    def test_provider_first_blocks_direct_segmentation_recolor_prepare(self):
        import source_recolor
        case = dict(execution_policy=dict(editing_strategy='provider_first_v1'))
        with self.assertRaisesRegex(ValueError, 'existing editor tools'):
            source_recolor.prepare_plan(case, {})

    def test_provider_first_blocks_direct_matte_before_source_read(self):
        import source_matte
        case = dict(execution_policy=dict(editing_strategy='provider_first_v1'))
        with self.assertRaisesRegex(ValueError, 'existing editor tools'):
            source_matte.prepare(case, {'file_id': 'one'}, 'unused.npz')

    def test_provider_first_blocks_local_method_before_component_probe(self):
        from unittest.mock import patch
        from production_effort import check
        case = dict(execution_policy=dict(editing_strategy='provider_first_v1'))
        with patch('component_probe.active', side_effect=AssertionError('Must reject before probe')):
            with self.assertRaisesRegex(ValueError, 'existing editor tools'):
                check(case, 'one', start_method=True, method_family='local_recolor')

    def test_native_blue_change_excludes_existing_blue_background(self):
        source = np.array([[[25,25,25], [180,80,20], [130,170,220]]], np.uint8)
        raw = source.copy(); raw[0,0] = [171,71,0]
        mask = native_region(source, raw, [0,71,171])
        np.testing.assert_array_equal(mask, [[255,0,0]])

    def test_material_rejects_neutral_spill_without_deleting_small_fabric(self):
        source = np.full((12,12,3), 90, np.uint8)
        source[1:4,1:4] = [20,35,70]
        source[8,8] = [20,35,70]  # A real small fabric component stays editable.
        mask = np.zeros((12,12), np.uint8)
        mask[1:4,1:4] = 255; mask[8,8] = 255; mask[6:8,1:3] = 255
        out = material_components(source, mask, dict(hue_range=[0,25],minimum_saturation=65,
                                                     minimum_component_fraction=.5))
        self.assertEqual(out[8,8],255)
        self.assertFalse(out[6:8,1:3].any())
        self.assertTrue(out[1:4,1:4].all())

    def test_thin_fabric_connected_to_spill_is_preserved_by_original_guide(self):
        source = np.full((12,12,3), 90, np.uint8)
        source[2:10,2] = [20,35,70]
        source[8,8] = [20,35,70]  # Same chroma alone could be skin.
        mask = np.zeros((12,12),np.uint8); mask[2:10,2:10] = 255
        guide = np.zeros_like(mask); guide[2:10,2] = 255
        out = material_components(source,mask,dict(hue_range=[0,25],minimum_saturation=65,
                                  minimum_component_fraction=.5),guide)
        np.testing.assert_array_equal(out, guide)


if __name__ == '__main__': unittest.main()
