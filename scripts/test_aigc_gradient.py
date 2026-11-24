import unittest
import cv2
import numpy as np
from aigc_composite import match_source_boundary_gradient,match_gray_surface_plane
from regional_composite import compose_frame

class SourceBoundaryTests(unittest.TestCase):
    def test_bright_provider_surface_removes_dark_letters_without_touching_hand(self):
        source=np.full((120,160,3),65,np.uint8)
        cv2.putText(source,'K',(57,76),0,1.1,(12,12,12),3)
        source[60:94,105:145]=[100,150,200]
        raw=np.full_like(source,145)
        raw[45:48,70:74]=151
        alpha=np.zeros(source.shape[:2],np.uint8);alpha[25:100,35:125]=255
        alpha[60:94,105:145]=0
        matched=match_source_boundary_gradient(source,raw,alpha)
        result,effective=compose_frame(source,matched,alpha)
        np.testing.assert_array_equal(result[effective==0],source[effective==0])
        self.assertGreater(np.min(result[53:75,60:70]),55)
        self.assertLess(np.max(result[35:50,45:55]),75)
        self.assertGreater(np.mean(result[45:48,70:74]),np.mean(result[40:43,70:74])+2)

    def test_gray_surface_matches_sloping_light_without_a_flat_slab(self):
        y,x=np.mgrid[:160,:220];level=90+.17*x+.08*y+.0008*y*y
        clean=np.repeat(np.rint(level).astype(np.uint8)[:,:,None],3,axis=2)
        source=clean.copy();cv2.putText(source,'UGREEN',(55,90),0,.6,(30,30,30),2)
        alpha=np.zeros((160,220),np.uint8);alpha[55:110,45:150]=255
        raw=np.full_like(source,140);raw[65:68,75:80]+=3
        matched=match_gray_surface_plane(source,raw,alpha,np.full_like(alpha,255))
        result,effective=compose_frame(source,matched,alpha)
        np.testing.assert_array_equal(result[effective==0],source[effective==0])
        self.assertLess(np.mean(abs(result[alpha>0].astype(float)-clean[alpha>0])),1)
        self.assertGreater(np.mean(result[65:68,75:80]-clean[65:68,75:80]),2)
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            match_gray_surface_plane(source,raw,alpha,np.zeros_like(alpha))

if __name__=='__main__':unittest.main()
