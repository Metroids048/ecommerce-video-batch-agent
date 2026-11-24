"""Repeated CLI verification must avoid duplicate reads without trusting stale bytes."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import project
import production_pipeline
import workflow


class CommandHashReuse(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.root_patch = patch.object(project, 'ROOT', self.root)
        self.root_patch.start()
        self.source = self.root/'source.mp4'
        self.source.write_bytes(b'original source bytes')
        (self.root/'input.txt').write_text('remove old logo')
        self.case = dict(batch_id='hash-recovery', case_version='v1', user_input='input.txt', files=[
            dict(id='one', source='source.mp4', source_sha256=project.digest(self.source),
                 operations=[dict(id='erase', kind='remove', target='logo', protect=['hands'])])])
        (self.root/'case.json').write_text(json.dumps(self.case))
        # Hash/state recovery does not decode video. Byte fixtures exercise this
        # entry only; they are never supplied to a media processor or qualified.
        saved = workflow.state_path(self.case)
        saved.parent.mkdir(parents=True)
        saved.write_text(json.dumps(dict(batch_id=self.case['batch_id'], case_version='v1',
            case_binding=workflow.case_binding(self.case), revision=1, accepted=False, files=dict(one=dict(
                jobs=[], current_candidate_id=None, candidates=[], issues=[], execution_units=[])))))
        (self.root/'resume.json').write_text(json.dumps(dict(action='resume-batch')))

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def cli(self):
        output = io.StringIO()
        with patch('sys.argv', ['project.py','pipeline','--case','case.json','--spec','resume.json']), \
             contextlib.redirect_stdout(output):
            code = project.main()
        return code, json.loads(output.getvalue())

    def test_same_source_is_read_once_per_cli_and_rechecked_next_cli(self):
        opened = []
        actual_open = Path.open
        def track(path, *args, **kwargs):
            if path == self.source and args and args[0] == 'rb':
                opened.append(path)
            return actual_open(path, *args, **kwargs)
        with patch.object(Path, 'open', track):
            first = self.cli()
            self.assertEqual(first[0], 0, first)
            self.assertEqual(len(opened), 1, 'one recovery reread the unchanged complete source')
            self.assertEqual(self.cli(), first)
            self.assertEqual(len(opened), 2, 'a separate command must rehash its actual input')

    def test_same_size_rewrite_with_restored_mtime_still_rejects(self):
        original_status = production_pipeline.status
        original = self.source.stat()
        calls = []
        def changed_between_checks(case):
            result = original_status(case)
            if not calls:
                calls.append(True)
                self.source.write_bytes(b'altered! source bytes')
                self.assertEqual(self.source.stat().st_size, original.st_size)
                os.utime(self.source, ns=(original.st_atime_ns,original.st_mtime_ns))
            return result
        with patch.object(production_pipeline, 'status', changed_between_checks):
            code, result = self.cli()
        self.assertEqual(code, 2, result)
        self.assertIn('Actual source changed', result['error'])

    def test_replaced_inode_and_deleted_input_cannot_reuse_success(self):
        self.assertEqual(self.cli()[0], 0)
        replacement = self.root/'replacement'
        replacement.write_bytes(b'altered! source bytes')
        replacement.replace(self.source)
        self.assertEqual(self.cli()[0], 2)
        self.source.unlink()
        self.assertEqual(self.cli()[0], 2)


if __name__ == '__main__':
    unittest.main()
