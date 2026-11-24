import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import delivery_package
import project


class DeliveryTests(unittest.TestCase):
    def test_structured_and_legacy_provenance_survive_diagnostic_export(self):
        for provenance in ({'purpose': '内部研究', 'source_urls': ['https://example.test/video']},
                           '内部研究原片'):
            with self.subTest(provenance=provenance), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                state = root/'runs/batch/workflow.json'
                state.parent.mkdir(parents=True)
                state.write_text(json.dumps({'revision': 1, 'files': {},
                    'created_at': '2026-11-13T11:00:00+00:00'}))
                (root/'input.txt').write_text('修改要求')
                with patch.object(project, 'ROOT', root), patch.object(
                        delivery_package.workflow, 'delivery_report', return_value={
                            'revision': 1, 'files': [], 'original_goal_complete': False}):
                    result = delivery_package.build({'batch_id': 'batch',
                        'user_input': 'input.txt', 'provenance': provenance}, 'handoff', diagnostic=True)
                self.assertEqual(result['status'], 'failure_diagnostic')
                metadata = json.loads((root/'handoff/package.json').read_text())
                self.assertEqual(metadata['case_provenance'], provenance)
                self.assertIn('内部研究', (root/'handoff/index.html').read_text())
                self.assertFalse(metadata['accepted'])

    def test_queued_or_running_job_prevents_freezing_old_delivery(self):
        # Gate-only unit test; actual media copies are checked on the real batch.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root/'runs/batch/workflow.json'
            state.parent.mkdir(parents=True)
            for status in ('queued', 'running'):
                state.write_text(json.dumps({'revision': 1, 'files': {'clip': {
                    'jobs': [{'status': status}]}}}))
                with patch.object(project, 'ROOT', root), patch.object(
                        delivery_package.workflow, 'delivery_report', return_value={
                            'revision': 1, 'files': [{'candidate': None}]}):
                    with self.assertRaisesRegex(ValueError, 'existing local jobs'):
                        delivery_package.build({'batch_id': 'batch'}, 'handoff')
                self.assertFalse((root/'handoff').exists())

    def test_failed_quality_cannot_be_exported_as_normal_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root/'runs/batch/workflow.json'
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({'revision':1,'files':{'clip':{'jobs':[]}}}))
            with patch.object(project,'ROOT',root), patch.object(
                    delivery_package.workflow,'delivery_report',return_value={
                        'revision':1,'files':[{'id':'clip','candidate':None,
                            'delivery_status':'unresolved','unresolved_issues':[]}]}):
                with self.assertRaisesRegex(ValueError,'Unqualified'):
                    delivery_package.build({'batch_id':'batch'},'handoff')
            self.assertFalse((root/'handoff').exists())
