"""Real local media regressions; synthetic clips do not prove business quality."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import project
import workflow


class BatchExportTests(unittest.TestCase):
    def run_batch(self, with_audio):
        with tempfile.TemporaryDirectory() as directory, patch.object(project, 'ROOT', Path(directory).resolve()):
            root = Path(directory).resolve()
            (root/'input.txt').write_text('Remove the marked patch, keep everything else.')
            args = ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=32x32:r=30:d=0.2']
            if with_audio:
                args += ['-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.2', '-c:a', 'aac']
            subprocess.run(args + ['-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-video_track_timescale', '15360', str(root/'source.mp4')], check=True, capture_output=True)
            request = {'batch_id': 'batch', 'user_input': 'input.txt', 'files': [
                {'id': fid, 'source': 'source.mp4', 'operations': [
                    {'id': 'remove', 'kind': 'remove', 'target': 'test patch', 'protect': ['rest']}]
                 } for fid in ('one', 'two')]}
            (root/'request.json').write_text(json.dumps(request))
            project.intake('case.json', 'request.json')
            case = project.read_json(root/'case.json')
            # Preserve coverage of the archived single-plan/audio export path.
            # New production uses the unified reviewed-preview contract.
            case.pop('execution_policy')
            workflow.initialize(case, 'case.json')
            np.savez(root/'mask.npz', alpha=np.zeros((6,32,32), np.uint8),
                     raw_to_source=np.tile(np.eye(3), (6,1,1)))
            outputs = []
            for item in case['files']:
                plan = {'case_version': case['case_version'], 'file_id': item['id'],
                    'source_sha256': item['source_sha256'], 'source_frame_count': 6,
                    'source_frames_half_open': [0,6], 'cloud': 'source.mp4',
                    'cloud_sha256': item['source_sha256'], 'masks': 'mask.npz',
                    'masks_sha256': project.digest(root/'mask.npz'), 'parent_version': None,
                    'provider_task_id': None, 'review': {'records': []}}
                relative = item['id']+'.json'
                (root/relative).write_text(json.dumps(plan))
                candidate = workflow.execute_plan(case, relative, 'v1')
                outputs.append(candidate['output'])
                sidecar = project.read_json(root/candidate['sidecar'])
                self.assertTrue(sidecar['timeline_exact'])
                self.assertTrue(sidecar['audio_packets_pts_payload_exact'])
                self.assertEqual(sidecar['source_frame_count'], 6)
                self.assertEqual(sidecar['output_frame_count'], 6)
                self.assertEqual(bool(sidecar['output_audio_packets']), with_audio)
                self.assertNotEqual(candidate['status'], 'qualified')
            self.assertEqual(len(set(outputs)), 2)
            before = project.digest(root/outputs[0])
            with self.assertRaisesRegex(ValueError, 'exists'):
                workflow.execute_plan(case, 'one.json', 'v1')
            self.assertEqual(project.digest(root/outputs[0]), before)

    def test_two_files_same_version_keep_audio(self):
        self.run_batch(True)

    def test_silent_sources_remain_silent(self):
        self.run_batch(False)


if __name__ == '__main__':
    unittest.main()
