import copy
import unittest
import video_segmentation as segmentation

class VideoSegmentation(unittest.TestCase):
    def test_dense_masks_reach_tracker_without_point_conversion_and_keep_source_order(self):
        import numpy as np
        spec=dict(width=952,height=540,parts=[dict(name='strap',seed_frame=2,
            frames_half_open=[0,5],atlas_index=0),dict(name='lining',seed_frame=3,
            frames_half_open=[2,5],atlas_index=1)])
        p=segmentation.dense_parameters(spec,[952,540],359/12,5)
        g=segmentation.dense_graph(p)
        trackers=[n for n in g.values() if n['class_type']=='SAM3_VideoTrack']
        self.assertEqual(len(trackers),4)
        self.assertTrue(all('initial_mask' in n['inputs'] and 'conditioning' not in n['inputs'] for n in trackers))
        selectors=[n['inputs']['indexes'] for n in g.values() if n['class_type']=='VHS_SelectImages']
        self.assertIn('2,1,0',selectors);self.assertIn('2,3,4',selectors);self.assertIn('2,1',selectors)
        self.assertEqual(g['1']['inputs']['format'],'None')
        small=dict(width=2,height=1,parts=spec['parts'])
        masks=[np.full((1,2),n,np.uint8) for n in [1,2,3,4,5,6,7,8]]
        result=segmentation.unpack_dense_masks(masks,small,5)
        self.assertEqual(result[:,0,0].tolist(),[1,2,6,7,8])
        with self.assertRaisesRegex(ValueError,'incomplete'):
            segmentation.unpack_dense_masks(masks[:-1],small,5)
        with self.assertRaisesRegex(ValueError,'exact source frames'):
            segmentation.dense_parameters(dict(spec,parts=[dict(spec['parts'][0],seed_frame=5)]),[952,540],30,5)

    def test_sam3_instance_points_disable_text_and_keep_full_source(self):
        points=dict(positive=[dict(x=335,y=200)],negative=[dict(x=390,y=220)])
        spec=dict(width=952,height=540,text_prompt='backpack shoulder straps',anchor_frame=480,
                  point_prompt=points)
        graph=segmentation.sam3_graph(segmentation.sam3_parameters(spec,[952,540],359/12))
        node=graph['3']['inputs']
        self.assertEqual(node['prompt'],'')
        self.assertEqual(__import__('json').loads(node['positive_coords']),points['positive'])
        self.assertEqual((node['start_frame_index'],node['propagation_direction'],node['max_frames_to_track']),
                         (480,'both',-1))
        self.assertEqual(graph['1']['inputs']['skip_first_frames'],0)
        with self.assertRaisesRegex(ValueError,'outside'):
            segmentation.sam3_parameters(dict(spec,point_prompt=dict(positive=[dict(x=952,y=0)],negative=[])),[952,540],30)

    def test_sam3_visible_anchor_propagates_both_directions_without_trimming(self):
        p=segmentation.sam3_parameters(dict(width=952,height=540,text_prompt='backpack shoulder straps',
                                           anchor_frame=470),[952,540],359/12)
        graph=segmentation.sam3_graph(p)
        self.assertEqual(graph['3']['inputs']['frame_index'],470)
        self.assertEqual(graph['3']['inputs']['start_frame_index'],470)
        self.assertEqual(graph['3']['inputs']['propagation_direction'],'both')
        self.assertEqual(graph['1']['inputs']['skip_first_frames'],0)
        self.assertEqual(graph['3']['inputs']['max_frames_to_track'],-1)
        with self.assertRaises(ValueError):
            segmentation.sam3_parameters(dict(width=952,height=540,text_prompt='straps',anchor_frame=-1),[952,540],359/12)

    def test_supplier_source_preserves_exact_952_by_540_grid(self):
        spec=segmentation.fixed_recolor_parameters(dict(dimensions=[952,540]),
            dict(semantic_label='blue outer fabric panels of backpacks'))
        self.assertEqual((spec['width'],spec['height']),(952,540))
        p=segmentation.sam3_parameters(spec,[952,540],359/12)
        graph=segmentation.sam3_graph(p)
        # VHS AnimateDiff rounds 540 to 544; None has no spatial rounding.
        self.assertEqual(graph['1']['inputs']['format'],'None')
        self.assertEqual(graph['1']['inputs']['frame_load_cap'],0)
        self.assertEqual(graph['69']['inputs']['frame_rate'],359/12)
        with self.assertRaises(ValueError):
            segmentation.sam3_parameters(dict(spec,height=544),[952,540],359/12)

    def test_current_sam3_tracks_source_at_native_rate_without_generating_pixels(self):
        p=segmentation.sam3_parameters(dict(width=576,height=1024,text_prompt='backpack'),[1080,1920],24)
        graph=segmentation.sam3_graph(p)
        self.assertEqual(graph['3']['class_type'],'easy sam3VideoSegmentation')
        self.assertEqual(graph['3']['inputs']['max_frames_to_track'],-1)
        self.assertEqual(graph['1']['inputs']['force_rate'],0)
        self.assertEqual(graph['69']['inputs']['frame_rate'],24)
        self.assertEqual(graph['69']['inputs']['crf'],0)
        self.assertFalse(graph['69']['inputs']['no_preview'])  # RH otherwise reports SUCCESS without a downloadable artifact
        self.assertNotIn('SAM3InitVideoSession',{n['class_type'] for n in graph.values()})
        with self.assertRaises(ValueError):
            segmentation.sam3_parameters(dict(width=512,height=1024,text_prompt='backpack'),[1080,1920],24)

    def test_grid_sparse_anchors_native_fps_and_object_union_graph(self):
        spec=dict(width=1024,height=576,anchors=[dict(frame=0,object_id=1,
            positive_points=[dict(x=300,y=200)],negative_points=[dict(x=600,y=400)]),
            dict(frame=0,object_id=2,positive_points=[dict(x=700,y=200)],negative_points=[])])
        p=segmentation.parameters(spec,[1920,1080],24000/1001,multiobject=True)
        graph=segmentation.graph(p)
        self.assertEqual(graph['1']['inputs']['force_rate'],0)
        self.assertEqual(graph['2']['inputs']['model'],'sam2_hiera_large.safetensors')
        self.assertEqual(graph['4']['inputs']['object_index'],2)
        self.assertEqual(graph['7']['inputs']['frame_rate'],24000/1001)
        for change in [dict(width=768),dict(anchors=[dict(spec['anchors'][0],frame=1)]),
                       dict(anchors=[dict(spec['anchors'][0],object_id=5)]),
                       dict(anchors=spec['anchors']*5)]:
            invalid=copy.deepcopy(spec);invalid.update(change)
            with self.assertRaises(ValueError):segmentation.parameters(invalid,[1920,1080],24,multiobject=True)
        invalid=copy.deepcopy(spec);invalid['anchors'][0]['positive_points'][0]['x']=1024
        with self.assertRaisesRegex(ValueError,'outside'):segmentation.parameters(invalid,[1920,1080],24,multiobject=True)

    def test_sam2_supplier_grid_and_fractional_clock_are_not_rounded(self):
        p=segmentation.parameters(dict(width=952,height=540,anchors=[dict(frame=0,object_id=1,
            positive_points=[dict(x=850,y=140)],negative_points=[])]),[952,540],359/12,multiobject=True)
        g=segmentation.graph(p)
        self.assertEqual((g['1']['inputs']['custom_width'],g['1']['inputs']['custom_height']),(952,540))
        self.assertEqual(g['1']['inputs']['format'],'None')
        self.assertEqual(g['6']['inputs']['frame_rate'],359/12)
        self.assertFalse(g['6']['inputs']['no_preview'])

if __name__=='__main__': unittest.main()
