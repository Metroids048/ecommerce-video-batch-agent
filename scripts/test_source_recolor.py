import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import project
import workflow
from source_recolor import validate
from source_recolor import Layer


class SourceRecolor(unittest.TestCase):
    def test_luma_chroma_retains_shadows_and_outside_pixels(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project,'ROOT',Path(directory).resolve()):
            alpha=np.zeros((1,4,4),np.uint8);alpha[:,:,:2]=255
            np.savez(Path(directory)/'m.npz',alpha=alpha)
            frame=np.full((4,4,3),[90,90,210],np.uint8);frame[2:,:,:]=[20,10,90]
            plan=dict(masks='m.npz',color_rgb=[91,113,67],recolor_mode='luma_chroma',edit_frames_half_open=[0,1])
            result,touched=Layer(plan,dict(modified_source_frames=[],modified_instance_count=0)).compose(frame,0)
            np.testing.assert_array_equal(result[~touched],frame[~touched])
            np.testing.assert_allclose(np.dot(result[touched],[.114,.587,.299]),
                np.dot(frame[touched],[.114,.587,.299]),atol=1)
            self.assertLess(result[3,0,1],result[0,0,1])

    def test_source_blue_material_gate_preserves_orange_and_skin_inside_sam3_object(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project,'ROOT',Path(directory).resolve()):
            root=Path(directory).resolve()
            alpha=np.full((1,8,8),255,np.uint8);alpha[:,:,6:]=0
            np.savez(root/'m.npz',alpha=alpha,protected=np.zeros_like(alpha),source_dimensions=[8,8])
            frame=np.full((8,8,3),[170,70,30],np.uint8)
            frame[1:3,1:3]=[25,90,230]  # orange original logo
            frame[4:6,1:3]=[100,150,210]  # foreground skin
            plan=dict(masks='m.npz',color_rgb=[85,160,70],edit_frames_half_open=[0,1],
                recolor_mode='hsv_hue',editable_source_hsv=dict(hue_range=[90,140],minimum_saturation=50))
            result,touched=Layer(plan,dict(modified_source_frames=[],modified_instance_count=0)).compose(frame,0)
            np.testing.assert_array_equal(result[1:3,1:3],frame[1:3,1:3])
            np.testing.assert_array_equal(result[4:6,1:3],frame[4:6,1:3])
            np.testing.assert_array_equal(result[:,:,][~touched],frame[~touched])
            self.assertTrue(touched[0,0]);self.assertFalse(touched[:,6:].any())

    def test_hue_only_keeps_ink_texture_value_and_all_pixels_outside_tracked_object(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project,'ROOT',Path(directory).resolve()):
            root=Path(directory).resolve()
            alpha=np.zeros((1,8,8),np.uint8);alpha[:,:,2:6]=255
            np.savez(root/'m.npz',alpha=alpha,protected=np.zeros_like(alpha),source_dimensions=[8,8])
            frame=np.full((8,8,3),[20,30,180],np.uint8)
            frame[2:4,3:5]=[10,10,10]  # original black print
            frame[4:6,3:5]=[180,180,180]  # original neutral print
            layer=Layer(dict(masks='m.npz',color_rgb=[0,71,171],edit_frames_half_open=[0,1],
                recolor_mode='hsv_hue'),dict(modified_source_frames=[],modified_instance_count=0))
            result,touched=layer.compose(frame,0)
            np.testing.assert_array_equal(result[~touched],frame[~touched])
            np.testing.assert_array_equal(result[2:6,3:5],frame[2:6,3:5])
            original=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            changed=cv2.cvtColor(result,cv2.COLOR_BGR2HSV)
            np.testing.assert_array_equal(original[:,:,1:],changed[:,:,1:])

    def test_green_trim_highlights_stay_protected_without_keying_neutral_fabric(self):
        from source_recolor import green_material_mask
        hsv=np.full((12,20,3),[50,80,110],np.uint8)
        hsv[4:8,4:8]=[50,150,160]
        hsv[5:7,7:10]=[50,90,160]  # sampled low-chroma cord highlight
        mask=green_material_mask(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR))
        self.assertTrue(np.all(mask[5:7,7:10]==255))
        self.assertFalse(mask[:,15:].any())  # similar hue is not object identity

    def test_white_print_protection_does_not_preserve_carrier_rectangle(self):
        from source_recolor import material_protection
        frame=np.full((8,8,3),70,np.uint8);frame[3:5,3:5]=230
        roi=np.zeros((8,8),np.uint8);roi[2:6,2:6]=255
        result=material_protection(frame,roi,'white_ink')
        self.assertEqual(np.count_nonzero(result),4)
        self.assertEqual(result[2,2],0)

    def test_learned_alpha_preserves_soft_guard_interior_and_keeps_edge_fractional(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project,'ROOT',Path(directory).resolve()):
            root=Path(directory).resolve()
            np.savez(root/'alpha.npz',alpha=np.full((1,8,8),128,np.uint8))
            guard=np.zeros((1,8,8),np.uint8);guard[:,2:7,2:7]=255
            np.savez(root/'guard.npz',alpha=guard)
            envelope=np.full((1,8,8),255,np.uint8);envelope[:,:,0]=0
            np.savez(root/'envelope.npz',alpha=envelope)
            cv2.imwrite(str(root/'blue.png'),np.full((2,2,3),[105,57,29],np.uint8))
            plan=dict(masks='alpha.npz',color_asset='blue.png',edit_frames_half_open=[0,1],
                segmentation_backend='vitmatte-video-local',segmentation_protection=[dict(
                    masks='guard.npz',edit_frames_half_open=[0,1],matting_boundary='soft')])
            alpha=Layer(plan,{}).mask(np.full((8,8,3),100,np.uint8),0)
            self.assertTrue(np.all(alpha[3:6,3:6]==0))
            self.assertEqual(alpha[2,3],128)
            self.assertEqual(alpha[0,7],128)
            plan['alpha_envelope']=dict(masks='envelope.npz',edit_frames_half_open=[0,1])
            alpha=Layer(plan,{}).mask(np.full((8,8,3),100,np.uint8),0)
            self.assertTrue(np.all(alpha[:,0]==0))
            self.assertTrue(np.all(alpha[3:6,3:6]==0))
            self.assertEqual(alpha[2,3],128)

    def test_occluder_envelope_keeps_soft_edge_without_opening_background(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project,'ROOT',Path(directory).resolve()):
            root=Path(directory).resolve()
            np.savez(root/'alpha.npz',alpha=np.full((1,4,4),128,np.uint8))
            target=np.zeros((1,4,4),np.uint8);target[:,:,2:]=255
            foreground=np.zeros_like(target);foreground[:,:,1]=255
            np.savez(root/'target.npz',alpha=target)
            np.savez(root/'foreground.npz',alpha=foreground)
            cv2.imwrite(str(root/'blue.png'),np.full((2,2,3),[105,57,29],np.uint8))
            plan=dict(masks='alpha.npz',color_asset='blue.png',edit_frames_half_open=[5,6],
                segmentation_backend='vitmatte-video-local',alpha_envelope=dict(
                    masks='target.npz',edit_frames_half_open=[5,6],foreground_occluders=[dict(
                        masks='foreground.npz',edit_frames_half_open=[5,6])]))
            alpha=Layer(plan,{}).mask(np.full((4,4,3),100,np.uint8),5)
            self.assertTrue(np.all(alpha[:,0]==0))
            self.assertTrue(np.all(alpha[:,1:]==128))

    def test_refined_alpha_cannot_erase_thin_protected_materials(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project, 'ROOT', Path(directory).resolve()):
            root = Path(directory).resolve()
            np.savez(root/'base.npz', alpha=np.full((1,8,8),255,np.uint8))
            np.savez(root/'refined.npz', alpha=np.full((1,8,8),255,np.uint8))
            logo=np.zeros((1,8,8),np.uint8); logo[:,2,2:4]=255
            np.savez(root/'logo.npz',alpha=logo)
            cv2.imwrite(str(root/'blue.png'),np.full((2,2,3),[105,57,29],np.uint8))
            plan=dict(masks='base.npz',color_asset='blue.png',edit_frames_half_open=[0,1],
                recolor_mode='hsv_chroma',preserve_green_and_white=True,
                alpha_refinement=dict(masks='refined.npz'),
                segmentation_protection=[dict(masks='logo.npz',edit_frames_half_open=[0,1])])
            frame=np.full((16,16,3),180,np.uint8); frame[10,2:14]=[0,180,0]
            result,touched=Layer(plan,dict(modified_source_frames=[],modified_instance_count=0)).compose(frame,0)
            np.testing.assert_array_equal(result[4:6,4:8],frame[4:6,4:8])
            np.testing.assert_array_equal(result[10,2:14],frame[10,2:14])
            self.assertFalse(touched[4:6,4:8].any())
            self.assertTrue(touched[0,0])

    def test_new_execution_rejects_unbound_grid_and_accepts_source_matte_artifact(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project, 'ROOT', Path(directory).resolve()):
            root = Path(directory).resolve()
            (root/'input.txt').write_text('Change the fabric to the supplied blue, preserve the hand.')
            (root/'source.bin').write_bytes(b'identity')
            cv2.imwrite(str(root/'color.png'), np.full((2, 2, 3), [115, 70, 34], np.uint8))
            item = dict(id='one', source='source.bin', source_sha256=project.digest(root/'source.bin'),
                dimensions=[8, 8], source_frame_count_inventory=1,
                operations=[dict(id='cloth', reference_asset='color.png')])
            case = dict(case_version='v1', user_input='input.txt', files=[item])
            plan = dict(operation_id='cloth', dependency=workflow.requirement_binding(case, item),
                color_asset='color.png', masks='legacy.npz', edit_frames_half_open=[0, 1], source_frame_count=1)
            np.savez(root/'legacy.npz', alpha=np.full((1, 4, 4), 255, np.uint8))
            plan['artifact_hashes'] = {'legacy.npz': project.digest(root/'legacy.npz')}
            with self.assertRaisesRegex(ValueError, 'metadata'):
                validate(case, item, plan)
            from source_matte import prepare
            spec = dict(file_id='one', operation_id='cloth', source_sha256=item['source_sha256'],
                coordinate_size=[8, 8], mask_size=[4, 4], frames_half_open=[0, 1],
                records=[dict(frame=0, state='visible', target_polygons=[[[0,0],[8,0],[8,8],[0,8]]],
                              protected_polygons=[[[2,2],[4,2],[4,4],[2,4]]])])
            prepare(case, spec, 'bound.npz')
            plan.update(masks='bound.npz', artifact_hashes={'bound.npz': project.digest(root/'bound.npz')})
            validate(case, item, plan)
            plan['skin_guard_red_minus_blue'] = 28
            with self.assertRaisesRegex(ValueError, 'global skin colour threshold'):
                validate(case, item, plan)

    def test_explicit_protection_wins_after_upscaling_even_when_target_and_skin_share_red(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project, 'ROOT', Path(directory).resolve()):
            root = Path(directory).resolve()
            alpha = np.full((1, 4, 4), 255, np.uint8)
            protected = np.zeros_like(alpha); protected[:, 1:3, 1:3] = 255
            np.savez(root/'mask.npz', alpha=alpha, protected=protected, source_dimensions=[8, 8])
            cv2.imwrite(str(root/'color.png'), np.full((2, 2, 3), [115, 70, 34], np.uint8))
            plan = dict(masks='mask.npz', color_asset='color.png', edit_frames_half_open=[0, 1],
                        recolor_mode='hsv_reference', reference_value=180, reference_saturation=220)
            frame = np.full((8, 8, 3), [40, 40, 180], np.uint8)
            layer = Layer(plan, dict(modified_source_frames=[], modified_instance_count=0))
            result, touched = layer.compose(frame, 0)
            np.testing.assert_array_equal(result[2:6, 2:6], frame[2:6, 2:6])
            self.assertFalse(touched[2:6, 2:6].any())
            self.assertTrue(touched[0, 0])  # same red remains editable cloth outside measured guard
            self.assertGreater(result[0, 0, 0], result[0, 0, 2])
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                layer.compose(np.zeros((16, 16, 3), np.uint8), 0)

    def test_semantic_envelopes_close_object_holes_without_joining_separate_buckles(self):
        from source_recolor import protection_mask
        mask = np.zeros((20, 20), np.uint8)
        mask[2:8, 2:8] = 255; mask[4:6, 4:6] = 0
        mask[12:18, 12:18] = 255
        components = protection_mask(mask, 'convex_components')
        self.assertEqual(components[4, 4], 255)
        self.assertEqual(components[10, 10], 0)
        self.assertEqual(protection_mask(mask, 'convex_object')[10, 10], 255)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            protection_mask(mask, 'guess')

    def test_semantic_guard_range_and_neutral_fabric_reflections(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project, 'ROOT', Path(directory).resolve()):
            root = Path(directory).resolve()
            np.savez(root/'base.npz', alpha=np.full((3,4,4),255,np.uint8))
            guard = np.zeros((1,4,4),np.uint8); guard[:,1:3,1:3]=255
            np.savez(root/'guard.npz',alpha=guard)
            cv2.imwrite(str(root/'blue.png'),np.full((2,2,3),[105,57,29],np.uint8))
            plan = dict(masks='base.npz',color_asset='blue.png',edit_frames_half_open=[0,3],
                recolor_mode='hsv_chroma',preserve_green_and_white=True,
                segmentation_protection=[dict(masks='guard.npz',edit_frames_half_open=[1,2])])
            layer=Layer(plan,dict(modified_source_frames=[],modified_instance_count=0))
            frame=np.full((8,8,3),180,np.uint8)
            a,_=layer.compose(frame,0); b,touched=layer.compose(frame,1); c,_=layer.compose(frame,2)
            self.assertGreater(a[3,3,0],a[3,3,2])
            np.testing.assert_array_equal(b[2:6,2:6],frame[2:6,2:6])
            self.assertFalse(touched[2:6,2:6].any())
            np.testing.assert_array_equal(a,c)

    def test_red_fabric_uses_reference_hue_without_losing_shading(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project, 'ROOT', Path(directory).resolve()):
            root = Path(directory).resolve()
            alpha = np.zeros((1, 8, 8), np.uint8)
            alpha[:, 2:6, 2:6] = 255
            np.savez(root/'mask.npz', alpha=alpha)
            cv2.imwrite(str(root/'color.png'), np.full((2, 2, 3), [115, 70, 34], np.uint8))
            plan = dict(masks='mask.npz', color_asset='color.png', edit_frames_half_open=[0, 1],
                        recolor_mode='hsv_reference', reference_value=180, reference_saturation=220,
                        skin_guard_red_minus_blue=256)
            layer = Layer(plan, dict(modified_source_frames=[], modified_instance_count=0))
            frame = np.full((8, 8, 3), [40, 40, 180], np.uint8)
            frame[4:6, 2:6] //= 2
            result, touched = layer.compose(frame, 0)
            np.testing.assert_array_equal(result[~touched], frame[~touched])
            self.assertGreater(int(result[2, 2, 0]), int(result[2, 2, 2])*2)
            self.assertAlmostEqual(float(result[2, 2, 0])/result[4, 2, 0], 2, delta=.05)

    def test_source_motion_skin_background_and_unedited_time_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project, 'ROOT', Path(directory).resolve()):
            root = Path(directory).resolve()
            alpha = np.zeros((2, 20, 20), np.uint8)
            alpha[:, 3:17, 3:17] = 255
            np.savez(root/'mask.npz', alpha=alpha)
            cv2.imwrite(str(root/'color.png'), np.full((2, 2, 3), [93, 54, 23], np.uint8))
            plan = dict(masks='mask.npz', color_asset='color.png', edit_frames_half_open=[1, 3],
                        reference_white=238, skin_guard_red_minus_blue=28)
            report = dict(modified_source_frames=[], modified_instance_count=0)
            layer = Layer(plan, report)
            for n in range(4):
                frame = np.full((20, 20, 3), 238, np.uint8)
                frame[7:11, n+5:n+9] = [170, 195, 220]
                result, touched = layer.compose(frame, n)
                if n in (0, 3):
                    np.testing.assert_array_equal(frame, result)
                else:
                    np.testing.assert_array_equal(frame[7:11, n+5:n+9], result[7:11, n+5:n+9])
                    np.testing.assert_array_equal(frame[~touched], result[~touched])
                    np.testing.assert_array_equal(result[4, 4], [93, 54, 23])
                    self.assertTrue(np.any(touched))


if __name__ == '__main__':
    unittest.main()
