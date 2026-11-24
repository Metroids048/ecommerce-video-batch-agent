import unittest
import cv2
import numpy as np
from exact_asset import compose_label, compose_planar_text
from regional_composite import compose_frame


class ExactAssetTests(unittest.TestCase):
    def test_original_border_survives_bounded_exact_asset_writeback(self):
        source=np.full((160,320,3),120,np.uint8)
        cv2.rectangle(source,(20,20),(299,139),(70,80,90),3)
        cv2.putText(source,'OLD',(100,92),cv2.FONT_HERSHEY_SIMPLEX,1,(240,240,240),2)
        asset=np.zeros((30,100,4),np.uint8);asset[5:25,10:90]=[10,20,30,255]
        replacement=compose_label(source,[[20,20],[299,20],[299,139],[20,139]],asset)
        mask=np.zeros((160,320),np.uint8);mask[45:115,45:275]=255
        out,effective=compose_frame(source,replacement,mask)
        self.assertTrue(np.array_equal(out[effective==0],source[effective==0]))
        self.assertGreater(np.count_nonzero(out!=source),100)

    def test_empty_asset_and_missing_carrier_are_rejected(self):
        source=np.zeros((100,100,3),np.uint8)
        with self.assertRaisesRegex(ValueError,'carrier'):
            compose_label(source,[],np.zeros((5,5,4),np.uint8))
        with self.assertRaisesRegex(ValueError,'RGBA'):
            compose_label(source,[[0,0],[99,0],[99,99],[0,99]],np.zeros((5,5,4),np.uint8))


class PaperCleanupTests(unittest.TestCase):
    def test_complete_broad_glyphs_removed_without_changing_other_print(self):
        source=np.full((180,500,3),180,np.uint8)
        cv2.putText(source,'NUDE',(45,105),cv2.FONT_HERSHEY_SIMPLEX,2.5,(120,115,145),12)
        cv2.putText(source,'KEEP',(15,165),cv2.FONT_HERSHEY_SIMPLEX,1,(20,20,20),2)
        asset=np.zeros((30,80,4),np.uint8);asset[10:20,20:60]=[0,0,0,255]
        style={'plane_size':[400,120],'polarity':'dark','asset_fit':[.4,.4],'paper_cleanup':'robust_border'}
        raw,mask=compose_planar_text(source,[[20,10],[419,10],[419,129],[20,129]],asset,style,return_mask=True)
        result,_=compose_frame(source,raw,mask)
        old=np.all(source==[120,115,145],axis=2)
        self.assertFalse(np.any(np.all(result[old]==[120,115,145],axis=1)))
        self.assertTrue(np.array_equal(result[140:],source[140:]))

    def test_print_cut_by_plane_boundary_is_rejected(self):
        source=np.full((100,300,3),190,np.uint8)
        cv2.putText(source,'NUDE',(-20,70),cv2.FONT_HERSHEY_SIMPLEX,2,(100,100,140),8)
        asset=np.full((20,80,4),255,np.uint8)
        with self.assertRaisesRegex(ValueError,'boundary'):
            compose_planar_text(source,[[0,0],[299,0],[299,99],[0,99]],asset,
                {'plane_size':[300,100],'polarity':'dark','asset_fit':[.8,.6],'paper_cleanup':'robust_border'},return_mask=True)

    def test_neighboring_row_is_not_old_print_or_cleanup_boundary(self):
        source=np.full((160,500,3),190,np.uint8)
        cv2.putText(source,'KEEP',(30,10),cv2.FONT_HERSHEY_SIMPLEX,1,(70,70,70),2)
        cv2.putText(source,'NUDE',(45,113),cv2.FONT_HERSHEY_SIMPLEX,2.5,(120,115,145),12)
        asset=np.zeros((20,80,4),np.uint8);asset[5:15,20:60]=[0,0,0,255]
        raw,mask=compose_planar_text(source,[[0,0],[499,0],[499,159],[0,159]],asset,
            {'plane_size':[500,160],'polarity':'dark','asset_fit':[.4,.4],'paper_cleanup':'robust_border'},return_mask=True)
        result,_=compose_frame(source,raw,mask)
        self.assertTrue(np.array_equal(result[:15],source[:15]))

class PaperOcclusionTests(unittest.TestCase):
    def test_different_color_foreground_is_preserved_above_new_print(self):
        source=np.full((180,400,3),220,np.uint8)
        cv2.putText(source,'NUDE',(30,120),cv2.FONT_HERSHEY_SIMPLEX,2.2,(190,170,215),8)
        source[80:95,20:110]=(150,175,230)
        asset=np.zeros((30,100,4),np.uint8);asset[:]=(0,0,0,255)
        style={'plane_size':[360,140],'polarity':'dark','asset_fit':[.85,.6],'paper_cleanup':'robust_border'}
        raw,mask=compose_planar_text(source,[[20,20],[379,20],[379,159],[20,159]],asset,style,True)
        final=np.rint(source*(1-mask[:,:,None]/255)+raw*(mask[:,:,None]/255)).astype(np.uint8)
        self.assertTrue(np.array_equal(final[83:92,25:105],source[83:92,25:105]))

class RealClothRegression(unittest.TestCase):
    def test_complete_word_on_nonuniform_cloth_has_clear_cleanup_boundary(self):
        from pathlib import Path
        fixture=Path(__file__).resolve().parent.parent/'evidence/localization-repair-20261112/bag-plane-1325-0.jpg'
        source=cv2.imread(str(fixture))
        self.assertIsNotNone(source)
        asset=np.zeros((20,80,4),np.uint8);asset[5:15,20:60]=[0,0,0,255]
        raw,mask=compose_planar_text(source,[[0,0],[511,0],[511,159],[0,159]],asset,
            {'plane_size':[512,160],'polarity':'light','asset_fit':[.4,.4],'paper_cleanup':'robust_border'},True)
        self.assertGreater(np.count_nonzero(mask),1000)
