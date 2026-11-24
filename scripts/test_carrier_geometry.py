import unittest
import cv2
import numpy as np
from carrier_geometry import register_carrier,relocate_rectangles,dark_surface_proposals
from print_registration import freeze_profile


class IndependentCarrier(unittest.TestCase):
    def test_word_is_excluded_and_border_retains_translation(self):
        rng=np.random.default_rng(4)
        image=cv2.GaussianBlur(rng.integers(80,180,(200,400),dtype=np.uint8),(3,3),0)
        c=np.float32([[40,40],[350,40],[350,170],[40,170]])
        w=np.float32([[90,80],[290,80],[290,130],[90,130]])
        moved=cv2.warpAffine(image,np.float32([[1,0,5],[0,1,3]]),(400,200),borderValue=120)
        moved[83:133,95:295]=220
        found=register_carrier(image,moved,c,w)
        self.assertIsNotNone(found)
        self.assertTrue(np.allclose(found['word_quad'],w+[5,3],atol=1.5))
        self.assertIsNone(register_carrier(image,np.full_like(image,120),c,w))

    def test_full_frame_dark_carrier_proposal_does_not_need_old_track_or_ocr(self):
        image=np.full((600,720,3),190,np.uint8)
        c=np.float32([[210,250],[510,250],[510,390],[210,390]])
        cv2.rectangle(image,(210,250),(510,390),(95,95,95),-1)
        cv2.putText(image,'BEAUTY',(245,335),0,1.5,(235,235,235),4)
        w=np.float32([[238,295],[445,295],[445,345],[238,345]])
        profile=freeze_profile(image,w)
        moved=cv2.warpAffine(image,np.float32([[1,0,-110],[0,1,-140]]),(720,600),borderValue=(190,190,190))
        found=relocate_rectangles(moved,dark_surface_proposals(moved),c,w,profile,'light')
        self.assertTrue(found)
        self.assertLess(np.linalg.norm(np.float32(found[0]['word_quad']).mean(0)-(w+[-110,-140]).mean(0)),12)
        self.assertEqual(relocate_rectangles(np.full_like(moved,190),dark_surface_proposals(moved),c,w,profile,'light'),[])
