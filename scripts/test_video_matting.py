import unittest
from video_matting import graph, require_trimap_foreground
import numpy as np

class VideoMatting(unittest.TestCase):
    def test_thin_target_rejects_eroded_empty_trimap_before_paid_submit(self):
        mask=np.zeros((100,100),np.uint8);mask[20:80,45:55]=255
        with self.assertRaisesRegex(ValueError,'confident foreground'):
            require_trimap_foreground(mask,32)
        require_trimap_foreground(mask,1)
        mask[0:35,0:35]=255
        with self.assertRaisesRegex(ValueError,'confident foreground'):
            require_trimap_foreground(mask,6)

    def test_pretrained_local_only_dual_input_and_soft_mask_output(self):
        g=graph(dict(width=960,height=540,edge_erode=32,edge_dilate=32),24000/1001)
        self.assertEqual(g['4']['inputs']['method'],'VITMatte(local)')
        self.assertEqual(g['4']['inputs']['image'],['1',0])
        self.assertEqual(g['7'],dict(class_type='ThresholdMask',inputs=dict(mask=['3',0],value=.5)))
        self.assertEqual(g['4']['inputs']['mask'],['7',0])
        self.assertEqual(g['5']['inputs']['mask'],['4',1])
        self.assertEqual(g['6']['inputs']['frame_rate'],24000/1001)
        self.assertEqual(g['6']['inputs']['crf'],0)
        self.assertEqual(g['1']['inputs']['force_rate'],0)

if __name__=='__main__':unittest.main()
