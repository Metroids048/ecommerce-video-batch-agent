import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from aigc_composite import require_disjoint_parts, match_source_boundary_gradient, Layer, source_chroma, require_source_chroma
import project
from regional_composite import compose_frame

class ActualReturnCompositionTests(unittest.TestCase):
    def test_source_chroma_preserves_texture_protection_and_overlap(self):
        class Cap:
            def get(self,k):return {1:0,3:30,4:20}.get(k,0)
            def read(self):return True,np.full((20,30,3),9,np.uint8)
        source=np.zeros((20,30,3),np.uint8)
        source[:]=[90,45,25];source[::2,:,0]=150
        with tempfile.TemporaryDirectory(dir=project.ROOT) as folder:
            layer=object.__new__(Layer);layer.parts=[];layer.mask_cache=None
            layer.plan={'review':{'frame_states':['visible']},'color_match':'none'}
            layer.report={'modified_source_frames':[],'modified_instance_count':0}
            for k,weight in enumerate((128,127)):
                a=np.zeros((1,20,30),np.uint8);a[:,3:17,4:26]=weight
                guard=np.zeros_like(a);guard[:,8:12,10:20]=255
                p=Path(folder)/f'{k}.npz'
                np.savez_compressed(p,alpha=a,protected=guard,raw_to_source=np.eye(3)[None])
                layer.parts.append((dict(masks=str(p),source_frames_half_open=[0,1],write_frames_half_open=[0,1],blend_mode='source_chroma',target_rgb=[85,160,70]),Cap()))
            result,mask=layer.compose(source,0)
            np.testing.assert_array_equal(result[~mask],source[~mask])
            np.testing.assert_array_equal(result[mask],source_chroma(source,[85,160,70])[mask])
            np.testing.assert_array_equal(cv2.cvtColor(result,cv2.COLOR_BGR2HSV)[:,:,2],cv2.cvtColor(source,cv2.COLOR_BGR2HSV)[:,:,2])

    def test_source_chroma_requires_requested_preservation_and_rgb(self):
        item={'operations':[dict(id='fabric',recolor_mode='preserve_source_details',target_color_rgb=[85,160,70])]}
        require_source_chroma(item,'fabric',dict(target_rgb=[85,160,70]))
        with self.assertRaises(ValueError):require_source_chroma(item,'fabric',dict(target_rgb=[70,160,85]))
    def test_opaque_native_fast_path_preserves_pixels_and_protected_source(self):
        class Cap:
            def get(self,k):return {1:0,3:30,4:20}.get(k,0)
            def read(self):return True,np.full((20,30,3),100,np.uint8)
        with tempfile.TemporaryDirectory(dir=project.ROOT) as folder:
            for protection in (False,True):
                alpha=np.full((1,20,30),255,np.uint8)
                protected=np.zeros_like(alpha)
                if protection:protected[:,5:10,7:13]=255
                path=Path(folder)/f'{protection}.npz'
                np.savez_compressed(path,alpha=alpha,protected=protected,raw_to_source=np.eye(3)[None])
                layer=object.__new__(Layer);layer.mask_cache=None
                layer.plan={'review':{'frame_states':['visible']}}
                layer.report={'modified_source_frames':[],'modified_instance_count':0}
                layer.parts=[(dict(masks=str(path),source_frames_half_open=[0,1],write_frames_half_open=[0,1],blend_mode='temporal_crossfade'),Cap())]
                source=np.full((20,30,3),30,np.uint8);result,mask=layer.compose(source,0)
                expected=np.full_like(source,100)
                if protection:expected[5:10,7:13]=30
                np.testing.assert_array_equal(result,expected)
                np.testing.assert_array_equal(mask,protected[0]==0)

    def parts(self, folder, overlap=False, protect=False):
        parts=[]
        for k in range(2):
            alpha=np.zeros((3,20,30),np.uint8)
            alpha[:,:,2:15 if k==0 else 27]=255
            if k:alpha[:,:,:14 if overlap else 15]=0
            protected=np.zeros_like(alpha)
            if protect and k:protected[:,:,14:15]=255
            p=Path(folder)/f'{k}.npz'
            np.savez_compressed(p,alpha=alpha,protected=protected)
            parts.append(dict(masks=str(p),source_frames_half_open=[10+k,13+k],write_frames_half_open=[10+k,13+k]))
        return parts
    def test_same_frame_separate_actual_pixels_allowed(self):
        with tempfile.TemporaryDirectory(dir=project.ROOT) as folder:require_disjoint_parts(self.parts(folder))
    def test_one_column_overlap_rejected(self):
        with tempfile.TemporaryDirectory(dir=project.ROOT) as folder:
            with self.assertRaisesRegex(ValueError,'Overlapping provider pixels'):
                require_disjoint_parts(self.parts(folder,overlap=True))
    def test_protected_source_pixels_do_not_conflict(self):
        with tempfile.TemporaryDirectory(dir=project.ROOT) as folder:require_disjoint_parts(self.parts(folder,overlap=True,protect=True))
    def test_explicit_temporal_crossfade_requires_bounded_weights(self):
        with tempfile.TemporaryDirectory(dir=project.ROOT) as folder:
            parts=self.parts(folder,overlap=True)
            for part,weight in zip(parts,(128,127)):
                path=Path(part['masks'])
                with np.load(path) as z:alpha=z['alpha'].copy()
                alpha[alpha>0]=weight
                np.savez_compressed(path,alpha=alpha)
                part['blend_mode']='temporal_crossfade'
            require_disjoint_parts(parts)
            with np.load(parts[1]['masks']) as z:alpha=z['alpha'].copy()
            alpha[alpha>0]=128
            np.savez_compressed(parts[1]['masks'],alpha=alpha)
            with self.assertRaisesRegex(ValueError,'Crossfade alpha'):
                require_disjoint_parts(parts)

    def test_temporal_crossfade_blends_donors_once_and_preserves_source(self):
        class Cap:
            def __init__(self,value):self.value=value
            def get(self,k):return {1:0,3:30,4:20}.get(k,0)
            def read(self):return True,np.full((20,30,3),self.value,np.uint8)
        with tempfile.TemporaryDirectory(dir=project.ROOT) as folder:
            layer=object.__new__(Layer);layer.parts=[];layer.mask_cache=None
            layer.plan={'review':{'frame_states':['visible']},'color_match':'none'}
            layer.report={'modified_source_frames':[],'modified_instance_count':0}
            for k,weight in enumerate((128,127)):
                alpha=np.zeros((1,20,30),np.uint8);alpha[:,3:17,4:26]=weight
                path=Path(folder)/f'{k}.npz';np.savez_compressed(path,alpha=alpha,raw_to_source=np.eye(3)[None])
                layer.parts.append((dict(masks=str(path),source_frames_half_open=[0,1],write_frames_half_open=[0,1],blend_mode='temporal_crossfade'),Cap(100+100*k)))
            source=np.full((20,30,3),30,np.uint8);result,mask=layer.compose(source,0)
            np.testing.assert_array_equal(result[~mask],source[~mask])
            np.testing.assert_array_equal(result[mask],150)

    def test_luma_removes_provider_green_stain_and_source_letters(self):
        source=np.full((120,160,3),65,np.uint8)
        cv2.putText(source,'K',(57,76),0,1.1,(12,12,12),3)
        raw=np.full_like(source,(80,145,80));raw[45:48,70:74]=(91,160,91)
        alpha=np.zeros((120,160),np.uint8);alpha[25:100,35:125]=255
        matched=match_source_boundary_gradient(source,raw,alpha,monochrome=True)
        result,effective=compose_frame(source,matched,alpha)
        np.testing.assert_array_equal(result[effective==0],source[effective==0])
        self.assertLess(np.max(np.ptp(result[35:90,45:115].astype('int16'),axis=2)),3)
        self.assertGreater(np.min(result[53:75,60:70]),55)
        self.assertGreater(np.mean(result[45:48,70:74]),np.mean(result[40:43,70:74])+2)

    def test_internal_donor_boundary_does_not_keep_source_brand(self):
        class Cap:
            def __init__(self,value):self.value=value
            def get(self,k):return {1:0,3:160,4:120}.get(k,0)
            def read(self):return True,np.full((120,160,3),self.value,np.uint8)
        source=np.full((120,160,3),65,np.uint8)
        cv2.putText(source,'K',(57,76),0,1.1,(12,12,12),3)
        with tempfile.TemporaryDirectory(dir=project.ROOT) as folder:
            layer=object.__new__(Layer);layer.parts=[];layer.mask_cache=None
            layer.plan={'review':{'frame_states':['visible']},'color_match':'masked_flat_median'};layer.report={'modified_source_frames':[],'modified_instance_count':0}
            for k in range(2):
                a=np.zeros((1,120,160),np.uint8);a[:,25:100,35:125]=255
                if k:a[:,:63]=0
                else:a[:,63:]=0
                p=Path(folder)/f'{k}.npz';np.savez_compressed(p,alpha=a,raw_to_source=np.eye(3)[None])
                layer.parts.append((dict(masks=str(p),source_frames_half_open=[0,1],write_frames_half_open=[0,1],blend_mode='source_boundary_gradient'),Cap(145 if k==0 else 115)))
            result,mask=layer.compose(source,0)
            self.assertGreater(np.min(result[58:68,60:70]),55)
            self.assertLess(abs(float(np.mean(result[59:62,50:100]))-float(np.mean(result[64:67,50:100]))),3)
            np.testing.assert_array_equal(result[~mask],source[~mask])

if __name__=='__main__':unittest.main()
