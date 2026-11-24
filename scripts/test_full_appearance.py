"""Regression for the actual fixed-trial-window/full-delivery scope error."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import project
import source_recolor
import cached_clean_plate


class FullAppearance(unittest.TestCase):
    def test_recolor_visible_frames_outside_trial_are_rejected(self):
        with tempfile.TemporaryDirectory() as d, patch.object(project,'ROOT',Path(d).resolve()), patch('workflow.requirement_binding',return_value={}):
            p=Path(d);np.savez(p/'m.npz',alpha=np.full((150,3,3),255,np.uint8),
                protected=np.zeros((150,3,3),np.uint8),source_sha256='fixture-source',
                source_dimensions=[3,3],mask_dimensions=[3,3],frames_half_open=[0,150])
            plan=dict(dependency={},operation_id='color',color_asset='blue.png',artifact_hashes={},masks='m.npz',edit_frames_half_open=[0,150],source_frame_count=380,source_frames_half_open=[0,380],full_appearance_required=True,review=dict(frame_states=['visible']*380))
            item=dict(id='one',source_sha256='fixture-source',dimensions=[3,3],operations=[dict(id='color',reference_asset='blue.png')])
            with self.assertRaisesRegex(ValueError,'Unprocessed visible target'):
                source_recolor.validate({},item,plan)
            plan['review']['frame_states'][150:]=['absent']*230
            source_recolor.validate({},item,plan)

    def test_clean_plate_tracks_source_without_changing_outside(self):
        with tempfile.TemporaryDirectory() as d, patch.object(project,'ROOT',Path(d).resolve()):
            p=Path(d);plate=np.full((20,20,3),100,np.uint8);plate[5:10,5:10]=[20,90,10];cv2.imwrite(str(p/'plate.png'),plate)
            alpha=np.zeros((3,20,20),np.uint8);alpha[:,6:9,6:9]=255;H=np.repeat(np.eye(3)[None],3,axis=0);H[2,0,2]=1
            np.savez(p/'m.npz',alpha=alpha,plate_to_source=H)
            plan=dict(plate='plate.png',masks='m.npz',plate_origin={},review=dict(frame_states=['visible']*3))
            report=dict(modified_source_frames=[],modified_instance_count=0);layer=cached_clean_plate.Layer(plan,report)
            for n in range(3):
                src=np.full((20,20,3),220,np.uint8);out,touched=layer.compose(src,n)
                np.testing.assert_array_equal(src[~touched],out[~touched]);self.assertTrue(touched.any())
            self.assertEqual(report['modified_source_frames'],[0,1,2])
            layer.alpha[2]=0
            with self.assertRaisesRegex(ValueError,'Visible target unwritten'):layer.compose(src,2)

if __name__=='__main__':unittest.main()
