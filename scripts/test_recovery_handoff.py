"""Read the existing failed batch: recovery must carry its actual counterexample."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import project
import production_pipeline
import workflow


class ExistingFailureHandoff(unittest.TestCase):
    def test_changed_saved_report_is_not_returned_as_the_original_counterexample(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project, 'ROOT', Path(directory).resolve()):
            report = Path(directory)/'report.json'
            report.write_text('{"output":"old"}')
            trial = dict(id='fixed-failure', verdict='fail', report='report.json', report_sha256=project.digest(report))
            entry = dict(preview_trials=[trial])
            item = dict(source='source.mp4', source_sha256='unchanged')
            with project.command_hashes():
                production_pipeline.failure_handoff(item, entry)
                report.write_text('{"output":"new"}')
                with self.assertRaisesRegex(ValueError, 'Saved failure report changed'):
                    production_pipeline.failure_handoff(item, entry)

    def test_failed_office_resume_contains_saved_diagnosis_without_reopening_any_stage(self):
        case = project.read_json(project.ROOT/'cases/raw-api-default-validation-20261115/case.json')
        path = workflow.state_path(case)
        before = path.read_bytes()
        entry = project.read_json(path)['files']['office']
        failed = next(t for t in reversed(entry['preview_trials']) if t['verdict'] == 'fail')
        with patch('subprocess.Popen', side_effect=AssertionError('recovery must not start media or provider work')):
            with project.command_hashes():
                result = production_pipeline.dispatch(case, dict(action='resume-batch'))
        office = next(f for f in result['files'] if f['file_id'] == 'office')
        self.assertEqual(office['phase'], 'method_research')
        context = office['blocking_context']
        self.assertEqual(context['failed_preview']['id'], failed['id'])
        self.assertEqual(context['failed_preview']['diagnosis'], failed['diagnosis'])
        self.assertEqual(context['failed_preview']['next_change'], failed['next_change'])
        self.assertEqual(context['failed_preview']['checks'], failed['checks'])
        self.assertTrue({240,540,689} <= set(context['failed_preview']['observed_source_frames']))
        self.assertEqual(context['failed_preview']['report']['sha256'], failed['report_sha256'])
        self.assertEqual(context['source_sha256'], case['files'][0]['source_sha256'])
        self.assertEqual(context['fixed_material_fingerprint'], entry['material_handoff']['fingerprint'])
        self.assertEqual(result['new_renders'], 0)
        self.assertEqual(result['new_model_calls'], 0)
        self.assertEqual(before, path.read_bytes())


if __name__ == '__main__':
    unittest.main()
