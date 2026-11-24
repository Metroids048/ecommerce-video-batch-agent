import unittest
import cv2
import numpy as np
from print_registration import freeze_profile,register_extent,fit_word_blur


class CompletePrintRegistration(unittest.TestCase):
    def setUp(self):
        self.image=np.full((240,600,3),125,np.uint8)
        cv2.putText(self.image,'BEAUTY',(120,140),0,1.6,(235,235,235),4)
        self.quad=np.float32([[115,100],[325,100],[325,150],[115,150]])

    def test_shifted_tight_box_recovers_full_word_at_nondefault_profile_size(self):
        profile=freeze_profile(self.image,self.quad,(210,50))
        found=register_extent(profile,self.image,self.quad+[23,0],'light')
        self.assertTrue(found['supported'])
        self.assertLess(np.linalg.norm(np.float32(found['word_quad']).mean(0)-self.quad.mean(0)),5)

    def test_blank_carrier_cannot_authorize_a_full_print(self):
        profile=freeze_profile(self.image,self.quad)
        found=register_extent(profile,np.full_like(self.image,125),self.quad,'light')
        self.assertFalse(found['supported'])

    def test_registered_directional_blur_is_measured_from_actual_glyphs(self):
        from print_blur import gaussian_kernel
        profile=freeze_profile(self.image,self.quad)
        blurred=cv2.filter2D(self.image,-1,gaussian_kernel(3.2,.4,0))
        fitted=fit_word_blur(profile,blurred,self.quad,'light')
        self.assertTrue(fitted['direction_supported'])
        self.assertLess(min(fitted['angle_degrees'],180-fitted['angle_degrees']),31)

    def test_defocus_retains_a_supported_isotropic_estimate(self):
        profile=freeze_profile(self.image,self.quad)
        blurred=cv2.GaussianBlur(self.image,(0,0),3)
        fitted=fit_word_blur(profile,blurred,self.quad,'light')
        self.assertGreaterEqual(fitted['score'],.7)
        self.assertFalse(fitted['direction_supported'])
        self.assertGreater(fitted['isotropic_sigma'],2)
