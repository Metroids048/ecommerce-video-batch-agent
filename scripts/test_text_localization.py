import unittest
import numpy as np

class TextCandidates(unittest.TestCase):
    def test_all_instances_kept_and_other_words_rejected(self):
        import text_localization as t
        q=[[.1,.8],[.3,.8],[.3,.7],[.1,.7]]
        rows=[{'text':'NUDE','confidence':1,'quad':q},
              {'text':'NUDE','confidence':1,'quad':[[x+.5,y] for x,y in q]},
              {'text':'LIPS','confidence':1,'quad':q}]
        found=t.select_instances([{'orientation':'up','rows':rows}], 'NUDE',1000,1000)
        self.assertEqual(len(found),2)
        self.assertTrue(np.allclose(found[0]['word_quad'][0],[100,200]))

    def test_mirror_transform_restores_source_and_partial_word_is_not_authorized(self):
        import text_localization as t
        rows=[{'text':'IUDE','confidence':1,'quad':[[.1,.8],[.3,.8],[.3,.7],[.1,.7]]},
              {'text':'UDE','confidence':1,'quad':[[.5,.8],[.7,.8],[.7,.7],[.5,.7]]}]
        found=t.select_instances([{'orientation':'rightMirrored','rows':rows}],'NUDE',1000,2000)
        self.assertEqual(len(found),1)
        self.assertTrue(np.allclose(found[0]['word_quad'][0],[800,1800]))
        self.assertTrue(found[0]['mirrored'])

    def test_opposite_orientation_duplicates_are_one_instance(self):
        import text_localization as t
        q=[[.1,.8],[.3,.8],[.3,.7],[.1,.7]]
        found=t.select_instances([{'orientation':'up','rows':[{'text':'BEAUTY.','confidence':1,'quad':q}]}]*2,'BEAUTY',1000,1000)
        self.assertEqual(len(found),1)

    def test_print_identity_preserves_same_word_on_other_product(self):
        import cv2,text_localization as t
        reference=np.full((70,240,3),210,np.uint8)
        cv2.putText(reference,'NUDE',(5,53),cv2.FONT_HERSHEY_SIMPLEX,1.8,(150,140,190),4)
        other=255-reference
        q=[[0,0],[239,0],[239,69],[0,69]]
        self.assertTrue(t.match_reference(reference,q,reference,q))
        self.assertFalse(t.match_reference(reference,q,other,q))

    def test_direct_flow_recovers_translation_and_refuses_cut(self):
        import cv2,text_localization as t
        image=np.full((160,320),200,np.uint8)
        cv2.putText(image,'BEAUTY',(30,90),cv2.FONT_HERSHEY_SIMPLEX,1.1,40,2)
        moved=cv2.warpAffine(image,np.float32([[1,0,4],[0,1,3]]),(320,160),borderValue=200)
        q=np.float32([[20,50],[180,50],[180,100],[20,100]])
        found=t.propagate(image,moved,q)
        self.assertIsNotNone(found)
        self.assertTrue(np.allclose(found,q+[4,3],atol=1))
        self.assertIsNone(t.propagate(image,np.full_like(image,200),q))

    def test_output_old_word_is_bound_to_requested_source_plane(self):
        import text_localization as t
        q=[[10,10],[90,10],[90,40],[10,40]]
        protected=[[110,10],[190,10],[190,40],[110,40]]
        found=t.old_print_findings([{'word_quad':q},{'word_quad':protected}],
                                  {'instances':[{'word_quad':q}]})
        self.assertEqual(len(found),1)
        self.assertEqual(found[0]['word_quad'],q)

    def test_clipped_word_cannot_use_black_warp_padding_as_cleanup_context(self):
        import text_localization as t
        frame=np.full((80,160,3),180,np.uint8)
        instance={'word_quad':[[-10,20],[130,20],[130,50],[-10,50]]}
        with self.assertRaisesRegex(ValueError,'Clipped source word'):
            t.prepare_instance(frame,instance,None,{'asset_fit':[.84,.65]})
