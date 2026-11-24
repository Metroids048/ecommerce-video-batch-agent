import unittest
import cv2
import numpy as np
from print_blur import fit_psf,gaussian_kernel

class PrintBlurTests(unittest.TestCase):
    def test_direction_is_measured_from_same_registered_glyphs(self):
        ref=np.zeros((80,240),np.uint8)
        cv2.putText(ref,'BEAUTY',(4,56),0,1.2,180,3)
        for angle in (0,60,120):
            observed=cv2.filter2D(ref,-1,gaussian_kernel(6,.7,angle))
            found=fit_psf(ref,observed)
            self.assertTrue(found['direction_supported'])
            self.assertLessEqual(abs(found['angle_degrees']-angle),30)
            self.assertGreater(found['score'],.9)

    def test_defocus_and_wrong_identity_do_not_authorize_direction(self):
        ref=np.zeros((80,240),np.uint8);cv2.putText(ref,'BEAUTY',(4,56),0,1.2,180,3)
        result=fit_psf(ref,cv2.GaussianBlur(ref,(0,0),3))
        self.assertFalse(result['direction_supported'])
        self.assertIsNone(fit_psf(ref,np.ones_like(ref)))
