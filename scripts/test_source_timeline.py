"""PTS mapping tests, including actual locally encoded VFR media."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import source_timeline


class SourceTimelineTests(unittest.TestCase):
    def test_zero_frame_duration_requires_uniform_pts_and_matching_rate(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.mp4';path.write_bytes(b'fixture')
            def stderr(pts,rate='24000/1001'):
                return 'config in time_base: 1/24000, frame_rate: '+rate+'\n'+''.join(
                    ' n: %d pts: %d pts_time:0 duration: 0\n'%(n,p) for n,p in enumerate(pts))
            with patch('subprocess.run',return_value=SimpleNamespace(stderr=stderr([0,1001,2002]))):
                result=source_timeline.build(path)
                self.assertEqual(result['durations'],[1001]*3)
                self.assertEqual(result['duration_basis'],'uniform_decoded_pts_with_matching_declared_rate')
            for text in [stderr([0,1001,3003]),stderr([0,1001,2002], '25/1')]:
                with patch('subprocess.run',return_value=SimpleNamespace(stderr=text)):
                    with self.assertRaisesRegex(ValueError,'invalid source presentation'):
                        source_timeline.build(path)

    def test_displayed_pts_float_roundtrip_keeps_the_exact_frame(self):
        from fractions import Fraction
        timeline = {'time_base': '1/15360', 'pts': [512*n for n in range(1348)],
                    'durations': [512]*1348}
        for n, pts in enumerate(timeline['pts']):
            seconds = float(pts*Fraction(timeline['time_base']))
            self.assertEqual(source_timeline.frame_at_time(timeline, seconds), n)
        self.assertEqual(source_timeline.frame_at_time(timeline, 1/30-1e-8), 0)

    def test_vfr_mapping_uses_presentation_times_and_last_frame(self):
        timeline = {'time_base': '1/1000', 'pts': [500, 533, 610, 650],
                    'durations': [33, 77, 40, 40]}
        self.assertEqual(source_timeline.frame_at_time(timeline, .09), 1)
        self.assertEqual(source_timeline.frame_at_time(timeline, .11), 2)
        self.assertEqual(source_timeline.frame_at_time(timeline, .189), 3)
        with self.assertRaises(ValueError):
            source_timeline.frame_at_time(timeline, .19)
        with self.assertRaisesRegex(ValueError, 'Variable'):
            source_timeline.require_cfr(timeline)

    def test_cfr_timestamp_rounding_one_tick_is_allowed(self):
        timeline = {'time_base': '1/1000000', 'pts': [0, 33333, 66667, 100000],
                    'durations': [33333] * 4}
        source_timeline.require_cfr(timeline)
        self.assertEqual(source_timeline.frame_at_time(timeline, .066667), 2)

    def test_feedback_uses_bound_index_and_rejects_index_drift(self):
        from unittest.mock import patch
        import project
        import workflow
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source = root/'source.mp4'
            source.write_bytes(b'source binding fixture, not media')
            index = root/'timeline.json'
            timeline = {'source_sha256': project.digest(source), 'time_base': '1/1000',
                        'pts': [0, 100, 300, 500], 'durations': [100, 200, 200, 100]}
            index.write_text(json.dumps(timeline))
            item = {'id': 'one', 'source': 'source.mp4', 'source_sha256': project.digest(source),
                    'source_timeline': 'timeline.json', 'source_timeline_sha256': project.digest(index),
                    'fps_container_average': 10, 'duration_seconds': .6,
                    'operations': [{'id': 'logo', 'kind': 'remove', 'target': 'logo', 'protect': []}]}
            case = {'files': [item], 'case_version': 'test', 'user_input': 'source.mp4'}
            state = {'revision': 1, 'files': {'one': {'candidates': []}}}
            body = {'file_id': 'one', 'candidate_id': None, 'source_sha256': item['source_sha256'],
                    'candidate_sha256': item['source_sha256'], 'expected_revision': 1,
                    'source_time': .25, 'text': 'Wrong label'}
            with patch.object(project, 'ROOT', root):
                issue = workflow.bind_feedback(case, state, body)
                self.assertEqual(issue['source_frame'], 1)
                self.assertEqual(issue['source_pts'], 100)
                self.assertEqual(issue['time_mapping_basis'], 'decoded_source_pts')
                index.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'index hash changed'):
                    workflow.bind_feedback(case, state, body)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'local ffmpeg required')
    def test_actual_vfr_is_preserved_in_index_and_rejected_for_cfr_export(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'vfr.mp4'
            subprocess.run([shutil.which('ffmpeg'), '-v', 'error', '-f', 'lavfi',
                            '-i', 'testsrc2=size=32x32:rate=10:duration=0.5',
                            '-vf', 'setpts=if(lt(N\,2)\,N\,2*N-1)',
                            '-fps_mode', 'vfr', '-c:v', 'libx264', '-bf', '0', '-video_track_timescale', '1000', str(path)],
                           check=True, capture_output=True)
            timeline = source_timeline.build(path)
            self.assertEqual(len(timeline['pts']), 4)
            self.assertEqual(source_timeline.frame_at_time(timeline, .25), 1)
            self.assertEqual(source_timeline.frame_at_time(timeline, .3), 2)
            with self.assertRaisesRegex(ValueError, 'Variable'):
                source_timeline.require_cfr(timeline)


if __name__ == '__main__':
    unittest.main()
