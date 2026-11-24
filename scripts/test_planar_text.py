"""Regressions for an observed stale-occlusion claim after lost tracking."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import planar_text
import project


class TrackingStateTests(unittest.TestCase):
    def run_lost_track(self, existing=False):
        class Capture:
            def __init__(self, _):
                self.count = 0
            def get(self, _):
                return 64
            def read(self):
                self.count += 1
                return (True, np.zeros((64, 64, 3), np.uint8)) if self.count <= 2 else (False, None)
            def release(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            np.savez(root/'mask.npz', alpha=np.full((2, 10, 20), 255 if existing else 0, np.uint8),
                     protected=np.zeros((2, 10, 20), np.uint8))
            parent = {'file_id': 'clip', 'masks': 'mask.npz', 'label_style': {},
                      'review': {'method': 'test', 'records': [
                          {'frame': 0, 'state': 'occluded', 'agent_observation': 'old'},
                          {'frame': 1, 'state': 'occluded'}]}}
            if existing:
                for record in parent['review']['records']:
                    record.update(state='visible', source_quad=[[1,1],[20,1],[20,10],[1,10]])
            case = {'files': [{'id': 'clip', 'source': 'source.mp4', 'source_sha256': 'sha'}]}
            spec = {'source_sha256': 'sha', 'tracked_segments': [{
                'frames_half_open': [0, 2], 'seed_frame': 1, 'replace_existing': existing,
                'label_quad': [[1, 1], [20, 1], [20, 10], [1, 10]],
                'carrier_polygon': [[0, 0], [30, 0], [30, 30], [0, 30]]}]}
            with patch.object(project, 'ROOT', root), patch.object(planar_text.cv2, 'VideoCapture', Capture):
                result = planar_text.refine(case, parent, spec, 'new.json')
            records = json.loads((root/'new.json').read_text())['review']['records']
            self.assertEqual(result['unresolved_frames'], 1)
            self.assertEqual(records[0]['state'], 'unresolved')
            self.assertNotIn('agent_observation', records[0])
            self.assertEqual(records[1]['state'], 'visible')
            if existing:
                self.assertNotIn('source_quad', records[0])
                with np.load(root/'new.npz') as arrays:
                    self.assertFalse(arrays['alpha'][0].any())

    def test_lost_new_track_invalidates_previous_occlusion(self):
        self.run_lost_track()

    def test_lost_replacement_track_clears_previous_quad_and_pixels(self):
        self.run_lost_track(existing=True)

    def test_long_track_requires_new_anchor_and_clears_old_writeback(self):
        texture=np.random.RandomState(3).randint(0,256,(64,64,3),dtype=np.uint8)
        class Capture:
            def __init__(self,_): self.n=0
            def get(self,_): return 64
            def read(self):
                self.n+=1
                return (True,texture.copy()) if self.n<=65 else (False,None)
            def release(self): pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            np.savez(root/'mask.npz',alpha=np.full((65,10,20),255,np.uint8),protected=np.zeros((65,10,20),np.uint8))
            quad=[[10,10],[30,10],[30,20],[10,20]]
            parent={'file_id':'clip','masks':'mask.npz','label_style':{},'review':{'method':'test','records':[{'frame':n,'state':'visible','source_quad':quad} for n in range(65)]}}
            case={'files':[{'id':'clip','source':'source.mp4','source_sha256':'sha'}]}
            spec={'source_sha256':'sha','tracked_segments':[{'frames_half_open':[0,65],'seed_frame':32,'label_quad':quad,'carrier_polygon':[[0,0],[63,0],[63,63],[0,63]],'replace_existing':True}]}
            with patch.object(project,'ROOT',root),patch.object(planar_text.cv2,'VideoCapture',Capture):
                planar_text.refine(case,parent,spec,'new.json')
            with np.load(root/'new.npz') as arrays:
                self.assertFalse(arrays['alpha'][0].any())
                self.assertFalse(arrays['alpha'][64].any())
                self.assertTrue(arrays['alpha'][32].any())
            records=json.loads((root/'new.json').read_text())['review']['records']
            self.assertEqual(records[32]['track_method'],'anchor_forward_backward_LK')
            self.assertEqual(records[0]['state'],'unresolved')


class MultiSeedTests(unittest.TestCase):
    def test_two_carrier_appearances_and_nonmatch_remains_unknown(self):
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root/'source.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (256, 256))
            rng = np.random.RandomState(4)
            for _ in range(2):
                writer.write(rng.randint(0, 256, (256, 256, 3), dtype=np.uint8))
            writer.write(np.zeros((256, 256, 3), np.uint8))
            writer.release()
            (root/'asset.png').write_bytes(b'asset-fixture')
            case = {'case_version': 'v1', 'files': [{'id': 'clip', 'source': 'source.avi',
                    'source_sha256': project.digest(path),
                    'operations': [{'id': 'brand', 'replacement_asset': 'asset.png'}]}]}
            spec = {'file_id': 'clip', 'operation_id': 'brand',
                    'source_sha256': case['files'][0]['source_sha256'], 'seeds': [
                        {'seed_frame': n, 'label_quad': [[60,100],[150,100],[150,130],[60,130]],
                         'carrier_polygon': [[10,10],[240,10],[240,240],[10,240]]} for n in (0,1)]}
            with patch.object(project, 'ROOT', root), patch.object(project, 'validate_case'):
                planar_text.localize(case, spec, 'plan.json')
            plan = json.loads((root/'plan.json').read_text())
            self.assertEqual(plan['review']['frame_states'], ['visible', 'visible', 'unresolved'])
            self.assertEqual([r['seed_frame'] for r in plan['review']['records'][:2]], [0, 1])


if __name__ == '__main__':
    unittest.main()

class ConcurrentCarrierTests(unittest.TestCase):
    def test_single_instance_plan_cannot_use_legacy_refinement(self):
        parent={'file_id':'clip','review':{'records':[{'instances':[{}]}]}}
        with self.assertRaisesRegex(ValueError,'cannot overwrite instance plans'):
            planar_text.refine({'files':[{'id':'clip'}]},parent,{},'unused.json')

    def test_repeated_carrier_instances_are_not_collapsed_to_best(self):
        import cv2
        rng=np.random.RandomState(23)
        tile=rng.randint(0,256,(160,160),dtype=np.uint8)
        current=np.full((200,380),128,np.uint8)
        current[20:180,10:170]=tile;current[20:180,210:370]=tile
        sift=cv2.SIFT_create(nfeatures=1800,contrastThreshold=.025)
        refkeys,refdesc=sift.detectAndCompute(tile,None)
        keys,desc=sift.detectAndCompute(current,None)
        quad=np.float32([[30,60],[130,60],[130,100],[30,100]])
        found=planar_text.carrier_instances(refkeys,refdesc,quad,keys,desc,160*160,380,200,.30)
        self.assertEqual(len(found),2)
        centers=sorted(np.mean(i['source_quad'],axis=0)[0]for i in found)
        self.assertTrue(np.allclose(centers,[90,290],atol=2))

    def test_residual_homography_on_same_print_is_suppressed(self):
        a={'source_quad':[[10,10],[210,10],[210,30],[10,30]],'inliers':170}
        b={'source_quad':[[12,22],[212,22],[212,42],[12,42]],'inliers':19}
        c={'source_quad':[[260,10],[460,10],[460,30],[260,30]],'inliers':45}
        self.assertEqual(len(planar_text.distinct_instances([a,b,c])),2)
