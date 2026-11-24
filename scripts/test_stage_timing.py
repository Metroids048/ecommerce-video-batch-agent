"""Cover the previously unmeasured host-observation interval, without a production run."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import project


class StageTiming(unittest.TestCase):
    def test_cli_records_bound_host_interval_without_state_or_request_content(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(project, 'ROOT', Path(directory).resolve()):
            root = Path(directory).resolve()
            case = dict(batch_id='timing', case_version='v1', files=[dict(id='one',
                source_sha256='source', operations=[dict(id='cloth')])])
            (root/'case.json').write_text(json.dumps(case))
            spec = dict(action='start', phase='matte', file_id='one', operation_id='cloth',
                        prompt='not part of a timing record', at='2000-01-01T00:00:00Z')
            (root/'spec.json').write_text(json.dumps(spec))
            with patch('sys.argv', ['project.py', 'stage', '--case', 'case.json', '--spec', 'spec.json']):
                self.assertEqual(project.main(), 0)
            journal = root/'runs/timing/stage-events.jsonl'
            start = json.loads(journal.read_text())
            self.assertEqual(start['source_sha256'], 'source')
            self.assertEqual(start['operation_id'], 'cloth')
            self.assertNotIn('prompt', start)
            self.assertNotEqual(start['at'], spec['at'])
            end_spec = dict(action='end', event_id=start['event_id'])
            end = project.record_stage_timing(case, end_spec)
            self.assertGreaterEqual(end['elapsed_seconds'], 0)
            with self.assertRaisesRegex(ValueError, 'already ended'):
                project.record_stage_timing(case, end_spec)
            self.assertEqual(len(journal.read_text().splitlines()), 2)
            self.assertFalse((root/'runs/timing/workflow.json').exists())
            engineering=project.record_stage_timing(case,dict(action='start',phase='engineering'))
            self.assertEqual(engineering['work_class'],'one_time_engineering')
            project.record_stage_timing(case,dict(action='end',event_id=engineering['event_id']))
            selection=project.record_stage_timing(case,dict(action='start',phase='source_selection'))
            self.assertEqual(selection['work_class'],'research_selection')
            with self.assertRaisesRegex(ValueError, 'does not belong'):
                project.record_stage_timing(case, dict(spec, file_id='another'))


if __name__ == '__main__':
    unittest.main()
