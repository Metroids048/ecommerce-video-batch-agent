import unittest
import cv2
import numpy as np
import surface_text


class SourcePrintColorTests(unittest.TestCase):
    def test_exact_light_asset_can_follow_observed_dark_print(self):
        source=np.full((80,240,3),160,np.uint8)
        asset=np.zeros((20,80,4),np.uint8);asset[4:16,8:72]=[225,225,225,255]
        original=asset.copy()
        result=surface_text.compose_planar_text(source,[[0,0],[239,0],[239,79],[0,79]],asset,
            {'plane_size':[240,80],'polarity':'dark','asset_fit':[.7,.5],
             'paper_cleanup':'edge_interpolation','asset_ink_bgr':[35,35,35]})
        self.assertLess(int(result[40,120,0]),45)
        self.assertTrue(np.array_equal(asset,original))

class SurfaceTextTests(unittest.TestCase):
    def test_partial_track_uses_observed_print_and_rejects_blank(self):
        from carrier_tracking import appearance_score
        source=np.full((100,300),140,np.uint8)
        cv2.putText(source,'BEAUTY',(10,70),0,1.7,225,3)
        q=np.float32([[5,20],[260,20],[260,80],[5,80]])
        moved=cv2.warpAffine(source,np.float32([[1,0,130],[0,1,0]]),(300,100),borderValue=140)
        self.assertGreater(appearance_score(source,moved,q,q+[130,0]),.9)
        self.assertLess(appearance_score(source,np.full_like(source,140),q,q+[130,0]),0)

    def test_white_asset_inherits_observed_print_illumination(self):
        from surface_text import replace
        asset=np.full((40,200,4),255,np.uint8)
        peaks=[]
        for ink in (175,225):
            source=np.full((160,512,3),130,np.uint8)
            cv2.putText(source,'BEAUTY',(15,115),0,2.4,(ink,ink,ink),6)
            out,_,_=replace(source,asset,{'polarity':'light','asset_fit':[.85,.6]})
            peaks.append(int(out.max()))
            self.assertLessEqual(out.max(),ink+1)
        self.assertGreater(peaks[1]-peaks[0],40)

    def test_achromatic_dark_sleeve_over_light_print_is_retained(self):
        from surface_text import replace
        source=np.full((160,512,3),130,np.uint8)
        cv2.putText(source,'BEAUTY',(15,115),cv2.FONT_HERSHEY_SIMPLEX,2.4,(230,230,230),6)
        cv2.rectangle(source,(320,80),(511,159),(35,35,35),-1)
        asset=np.full((40,200,4),255,np.uint8)
        out,_,_=replace(source,asset,{'polarity':'light','asset_fit':[.85,.6]})
        self.assertTrue(np.array_equal(out[95:145,345:500],source[95:145,345:500]))

    def test_shadow_is_not_a_foreground_occluder(self):
        from surface_text import foreground_protection
        source=np.full((160,512,3),(180,180,200),np.uint8)
        cv2.putText(source,'NUDE',(30,115),cv2.FONT_HERSHEY_SIMPLEX,3,(140,140,175),9)
        source[:,250:]=np.uint8(source[:,250:].astype(float)*.5)
        self.assertLess(np.count_nonzero(foreground_protection(source,'dark')),source.shape[0]*source.shape[1]*.05)

    def test_source_print_template_covers_clipped_broad_glyph(self):
        from surface_text import replace,source_ink_template
        ref=np.full((100,400,3),150,np.uint8)
        ref[20:80,20:200]=230
        ref[20:80,280:300]=230
        q=np.float32([[0,0],[399,0],[399,99],[0,99]])
        template=source_ink_template(ref,q,[400,100],'light')
        # Only the right portion of the glyph is observed. Its left edge is
        # offscreen; an image-only local top-hat cannot recover that edge.
        valid=np.ones((100,400),bool);valid[:,:160]=False
        observed=ref.copy();observed[:,:160]=observed[:,160:161]
        asset=np.zeros((10,30,4),np.uint8);asset[4:6,14:16]=255
        _,mask,_=replace(observed,asset,dict(polarity='light',asset_fit=[.1,.1],
            cleanup_word_box=[0,0,1,1],source_ink_template=template),valid)
        self.assertNotEqual(mask[40,170],0)
        self.assertFalse(mask[~valid].any())

    def test_partial_carrier_identity_ignores_shadow_and_padding(self):
        from carrier_tracking import visible_print_match
        ref=np.full((120,360,3),220,np.uint8)
        cv2.putText(ref,'NUDE',(30,85),cv2.FONT_HERSHEY_SIMPLEX,2.5,(145,145,180),6)
        q=np.float32([[20,20],[325,20],[325,100],[20,100]])
        current=ref[:,60:].copy()
        current[:,130:]=np.uint8(current[:,130:].astype(float)*.55)
        self.assertTrue(visible_print_match(ref,q,current,q-[60,0]))
        wrong=np.full_like(current,200)
        cv2.circle(wrong,(110,60),28,(80,80,140),10)
        self.assertFalse(visible_print_match(ref,q,wrong,q-[60,0]))

    def test_partial_surface_never_writes_unknown_pixels(self):
        from surface_text import replace
        source=np.full((100,300,3),120,np.uint8)
        cv2.putText(source,'OLD',(20,75),cv2.FONT_HERSHEY_SIMPLEX,2,(230,230,230),5)
        valid=np.ones((100,300),bool);valid[:,:70]=False
        asset=np.full((20,70,4),240,np.uint8)
        out,mask,_=replace(source,asset,{'polarity':'light','asset_fit':[.7,.5]},valid)
        self.assertFalse(mask[~valid].any())
        self.assertTrue(np.array_equal(out[~valid],source[~valid]))

    def test_pale_nail_inside_coloured_occluder_is_retained(self):
        from surface_text import replace
        source=np.full((160,512,3),235,np.uint8)
        cv2.putText(source,'OLD',(130,110),cv2.FONT_HERSHEY_SIMPLEX,3,(150,150,180),8)
        cv2.ellipse(source,(15,80),(100,24),0,0,360,(135,165,220),-1)
        cv2.ellipse(source,(60,80),(25,12),0,0,360,(240,240,240),-1)
        asset=np.full((40,200,4),(0,0,0,255),np.uint8)
        out,_,_=replace(source,asset,{'polarity':'dark','asset_fit':[.85,.6]})
        self.assertTrue(np.array_equal(out[75:85,45:75],source[75:85,45:75]))

    def test_measured_blur_increases_with_defocus(self):
        from surface_text import measured_blur
        gray=np.full((100,300),90,np.uint8)
        cv2.putText(gray,'BEAUTY',(12,72),cv2.FONT_HERSHEY_SIMPLEX,1.5,230,3)
        self.assertGreater(measured_blur(cv2.GaussianBlur(gray,(0,0),3),'light'), measured_blur(gray,'light')+1)

    def test_context_tracking_retains_plane_with_missing_word_corners(self):
        from carrier_tracking import register
        rng=np.random.default_rng(27)
        gray=cv2.GaussianBlur(rng.integers(80,190,(180,360),dtype=np.uint8),(5,5),0)
        cv2.putText(gray,'BEAUTY',(80,110),cv2.FONT_HERSHEY_SIMPLEX,1,230,2)
        moved=cv2.warpAffine(gray,np.float32([[1,0,5],[0,1,3]]),(360,180),borderValue=120)
        moved=cv2.GaussianBlur(moved,(0,0),1.4)
        q=np.float32([[75,80],[205,80],[205,115],[75,115]])
        result=register(gray,moved,q)
        self.assertIsNotNone(result)
        self.assertTrue(np.allclose(result['word_quad'],q+[5,3],atol=2))
        self.assertIsNone(register(gray,np.full_like(gray,120),q))

    def test_local_texture_cleanup_keeps_margin_and_removes_broad_light_print(self):
        from surface_text import replace
        yy,xx=np.mgrid[:100,:300]
        cloth=np.clip(110+xx*.12+np.sin(xx/6)*3+np.sin(yy/4)*2,0,255).astype(np.uint8)
        source=np.repeat(cloth[:,:,None],3,axis=2)
        cv2.putText(source,'OLD',(50,70),cv2.FONT_HERSHEY_SIMPLEX,2,(230,230,230),5)
        asset=np.zeros((20,70,4),np.uint8);asset[8:12,30:40]=(240,240,240,255)
        out,mask,metrics=replace(source,asset,{'polarity':'light','asset_fit':[.5,.4],'cleanup_word_box':[.1,.1,.9,.9]})
        self.assertTrue(np.array_equal(out[mask==0],source[mask==0]))
        self.assertLess(out[45:65,50:90].max(),180)
        self.assertGreater(metrics['measured_blur_sigma'],0)
