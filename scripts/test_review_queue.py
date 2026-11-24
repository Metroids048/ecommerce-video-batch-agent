import tempfile
import unittest
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import project
import review_queue


class QueueTests(unittest.TestCase):
    def test_compact_pairs_use_bound_source_offset_and_preserve_pending_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'source.avi';raw=root/'raw.avi'
            for path,values in [(source,[0,30,60,90,120]),(raw,[61,91,121])]:
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,64))
                for value in values:writer.write(np.full((64,64,3),value,np.uint8))
                writer.release()
            bound=({'source_sha256':project.digest(source)},source,raw,2,project.digest(raw))
            with patch.object(project,'ROOT',root),patch.object(review_queue,'pair_inputs',return_value=bound):
                result=review_queue.build_pairs({},'clip','pairs',{'stills':[2]})
                inventory=json.loads((root/'pairs/inventory.json').read_text())
                self.assertEqual(inventory['paired_frames'],[2,3,4])
                self.assertEqual(result['status'],'pending_actual_agent_observation')
                pair=cv2.imread(str(root/'pairs/native-0002.jpg'))
                self.assertAlmostEqual(float(pair[:,:64].mean()),60,delta=2)
                with self.assertRaises(FileExistsError):review_queue.build_pairs({},'clip','pairs',{})

    def test_observe_cli_out_is_a_directory_and_returns_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root/'source.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
            writer.write(np.full((64, 64, 3), 80, np.uint8))
            writer.release()
            case = {'case_version': 'v1', 'files': [{'id': 'clip', 'source': 'source.avi',
                    'source_sha256': project.digest(source)}]}
            (root/'case.json').write_text(json.dumps(case))
            with patch.object(project, 'ROOT', root), patch('sys.argv', [
                    'project.py', 'observe', '--case', 'case.json', '--file', 'clip', '--out', 'observation'
                    ]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(project.main(), 0)
            self.assertTrue((root/'observation/queue.json').is_file())

    def test_unknown_ranges_have_real_bound_frames_but_no_review_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root/'source.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
            for n in range(10):
                writer.write(np.full((64, 64, 3), n*20, np.uint8))
            writer.release()
            sha = project.digest(source)
            case = {'case_version': 'v1', 'files': [{'id': 'clip', 'source': 'source.avi', 'source_sha256': sha}]}
            plan = {'file_id': 'clip', 'source_sha256': sha, 'operation_id': 'brand',
                    'review': {'frame_states': ['absent']*3+['unresolved']*4+['visible']*3}}
            with patch.object(project, 'ROOT', root):
                manifest = review_queue.build(case, 'clip', 'observation', plan=plan, stride=30)
                self.assertEqual(manifest['unresolved_source_frame_ranges'], [[3, 7]])
                self.assertFalse(manifest['full_visual_reviewed'])
                self.assertEqual(manifest['status'], 'pending_agent_observation')
                self.assertIn(5, [image['source_frame'] for image in manifest['images']])
                for image in manifest['images']:
                    self.assertEqual(project.digest(root/image['path']), image['sha256'])
                with self.assertRaisesRegex(ValueError, 'Preserve'):
                    review_queue.build(case, 'clip', 'observation')
