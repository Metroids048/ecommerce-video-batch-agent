"""Real stream-copy repair on six synthetic frames and a generated AAC tone; no business case."""
import contextlib
import io
import json
import subprocess
import unittest
from unittest.mock import patch

import project
import production_pipeline
import source_timeline
import test_execution_plan as fixtures
import workflow


class PackagingRepair(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.UnifiedExecution(); self.fixture.setUp()
        self.root, self.case = self.fixture.root, self.fixture.case
        subprocess.run(['ffmpeg','-v','error','-i',str(self.root/'source.mp4'),'-f','lavfi','-i',
            'sine=frequency=440:sample_rate=48000:duration=0.2','-map','0:v','-map','1:a',
            '-c:v','copy','-c:a','aac',str(self.root/'tone.mp4')],check=True,capture_output=True)
        item = self.case['files'][0]
        item.update(source='tone.mp4',source_sha256=project.digest(self.root/'tone.mp4'),audio_present=True)
        (self.root/'timeline.json').write_text(json.dumps(source_timeline.build(self.root/'tone.mp4')))
        item['source_timeline_sha256'] = project.digest(self.root/'timeline.json')
        for operation in self.fixture.manifest['operations']:
            path = self.root/operation['plan']; plan = project.read_json(path)
            plan['source_sha256'] = item['source_sha256']; path.write_text(json.dumps(plan))
            operation['plan_sha256'] = project.digest(path)
        self.fixture.manifest['dependency'] = workflow.requirement_binding(self.case,item)
        (self.root/'manifest.json').write_text(json.dumps(self.fixture.manifest))
        (self.root/'case.json').write_text(json.dumps(self.case))
        workflow.initialize(self.case,'case.json')
        original = workflow.execute_plan(self.case,'manifest.json','original')
        subprocess.run(['ffmpeg','-v','error','-i',str(self.root/original['output']),'-map','0:v',
                        '-c','copy',str(self.root/'missing-audio.mp4')],check=True,capture_output=True)
        report = project.read_json(self.root/original['sidecar'])
        report.update(output='missing-audio.mp4',output_sha256=project.digest(self.root/'missing-audio.mp4'),
                      audio_packets_pts_payload_exact=False, output_audio_packets=0)
        (self.root/'missing-audio.mp4.json').write_text(json.dumps(report))
        self.parent = workflow.register_existing(self.case,'one','missing-audio','missing-audio.mp4',
            'manifest.json','original','pending_agent_quality_review','Synthetic missing-audio counterexample')
        with workflow.locked_state(self.case) as state:
            state['files']['one']['candidates'].append(self.parent)
            state['files']['one']['current_candidate_id'] = self.parent['id']
        state = project.read_json(workflow.state_path(self.case))
        self.issue = workflow.add_feedback(self.case,dict(file_id='one',candidate_id=self.parent['id'],
            candidate_sha256=self.parent['output_sha256'],source_sha256=item['source_sha256'],
            expected_revision=state['revision'],source_time=.05,operation_id='a',category='packaging',
            text='Restore the exact source audio; retain the complete edited video'))
        self.spec = dict(action='remux-source-audio',file_id='one',issue_id=self.issue['id'])

    def tearDown(self):
        self.fixture.tearDown()

    def cli(self):
        (self.root/'remux-request.json').write_text(json.dumps(self.spec))
        output = io.StringIO()
        with patch('sys.argv',['project.py','pipeline','--case','case.json','--spec','remux-request.json']), \
             contextlib.redirect_stdout(output):
            code = project.main()
        return code,json.loads(output.getvalue())

    def test_restore_source_audio_without_pixel_render_or_automatic_quality_pass(self):
        planned = workflow.plan_feedback(self.case,dict(file_id='one',issue_id=self.issue['id']))
        self.assertEqual(planned['pipeline_request'],self.spec)
        with patch('regional_composite.render',side_effect=AssertionError('Do not render pixels')):
            code, child = self.cli()
        self.assertEqual(code,0,child)
        report = project.read_json(self.root/child['sidecar'])
        self.assertTrue(report['audio_packets_pts_payload_exact'])
        self.assertTrue(report['packaging_repair']['parent_video_decoded_exact'])
        self.assertEqual(report['output_frame_count'],6)
        self.assertGreater(report['source_audio_packets'],0)
        self.assertFalse(workflow.quality_reviews_pass(child))
        self.assertEqual(child['parent_id'],self.parent['id'])
        state = workflow.state_path(self.case).read_bytes()
        with patch('subprocess.run',side_effect=AssertionError('Repeat must reuse the same child')):
            code, repeated = self.cli()
        self.assertEqual(code,0,repeated)
        self.assertEqual(repeated['output_sha256'],child['output_sha256'])
        self.assertEqual(repeated['new_renders'],0)
        self.assertEqual(workflow.state_path(self.case).read_bytes(),state)

    def test_wrong_issue_and_maintenance_refuse_before_media_processing(self):
        self.spec['issue_id'] = 'unknown'
        self.assertEqual(self.cli()[0],2)
        (self.root/'config').mkdir()
        (self.root/'config/models.json').write_text(json.dumps(dict(execution_mode='maintenance')))
        self.spec['issue_id'] = self.issue['id']
        with patch('subprocess.run',side_effect=AssertionError('Maintenance must stop before FFmpeg')):
            self.assertEqual(self.cli()[0],2)

    def test_archived_parent_code_remains_reusable_after_maintenance(self):
        scripts = self.root/'scripts'; scripts.mkdir()
        old = scripts/'producer.py'; old.write_text('# frozen parent producer\n')
        report = project.read_json(self.root/self.parent['sidecar'])
        report['artifact_hashes']['scripts/producer.py'] = project.digest(old)
        (self.root/self.parent['sidecar']).write_text(json.dumps(report))
        self.parent['artifact_hashes'].update(report['artifact_hashes'])
        self.parent['artifact_hashes'][self.parent['sidecar']] = project.digest(self.root/self.parent['sidecar'])
        workflow.archive_candidate_code(self.parent)
        with workflow.locked_state(self.case) as state:
            state['files']['one']['candidates'][-1] = self.parent
        old.write_text('# maintained producer; do not rerender the parent\n')
        code, child = self.cli()
        self.assertEqual(code,0,child)
        self.assertEqual(child['parent_id'],self.parent['id'])
        workflow.verify_candidate(self.case,'one',child)
        self.assertIn(self.parent['sidecar'],child['artifact_hashes'])

    def test_multiple_track_header_is_detected_and_refused_without_partial_audio_pass(self):
        multi = self.root/'multi-track.mp4'
        subprocess.run(['ffmpeg','-v','error','-i',str(self.root/'tone.mp4'),
            '-map','0:v','-map','0:a','-map','0:a','-c','copy',str(multi)],check=True,capture_output=True)
        metadata = project.probe(multi)
        self.assertEqual(metadata['audio_stream_count'],2)
        # Inject the measured header at the negative gate, without claiming a
        # multi-track business export or altering this fixture's frozen source.
        with patch('project.probe',return_value=metadata):
            code, result = self.cli()
        self.assertEqual(code,2,result)
        self.assertIn('multiple audio tracks',str(result))
        entry = project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(entry['current_candidate_id'],self.parent['id'])

    def test_actual_cut_tail_is_preserved_as_failure_and_never_retried(self):
        subprocess.run(['ffmpeg','-v','error','-i',str(self.root/self.parent['output']),
            '-frames:v','3','-c:v','libx264','-video_track_timescale','1000000',str(self.root/'cut.mp4')],
            check=True,capture_output=True)
        report = project.read_json(self.root/self.parent['sidecar'])
        report.update(output='cut.mp4',output_sha256=project.digest(self.root/'cut.mp4'),
            timeline_exact=False,output_frame_count=3)
        (self.root/'cut.mp4.json').write_text(json.dumps(report))
        parent = workflow.register_existing(self.case,'one','cut','cut.mp4','manifest.json',
            self.parent['id'],'pending_agent_quality_review','Actual synthetic missing-tail counterexample')
        with workflow.locked_state(self.case) as state:
            entry = state['files']['one']; entry['candidates'].append(parent)
            entry['current_candidate_id'] = parent['id']
        state = project.read_json(workflow.state_path(self.case))
        issue = workflow.add_feedback(self.case,dict(file_id='one',candidate_id=parent['id'],
            candidate_sha256=parent['output_sha256'],source_sha256=parent['source_sha256'],
            expected_revision=state['revision'],source_time=.05,operation_id='a',category='packaging',
            text='Audio missing, but the video tail is also cut; remux must refuse'))
        self.spec['issue_id'] = issue['id']
        code, failed = self.cli()
        self.assertEqual(code,2,failed)
        self.assertIn('cut video tail',str(failed))
        entry = project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(entry['current_candidate_id'],parent['id'])
        self.assertEqual(entry['jobs'][-1]['status'],'failed')
        with patch('subprocess.run',side_effect=AssertionError('Failed original job cannot be retried')):
            self.assertEqual(self.cli()[0],2)
        request = next(r for r in production_pipeline.status(self.case)['files'][0]['repair_requests']
                       if r['issue_id']==issue['id'])
        self.assertEqual(request['next_action'],'diagnose_saved_packaging_failure')
        self.assertEqual(request['job']['id'],entry['jobs'][-1]['id'])


if __name__ == '__main__':
    unittest.main()
