"""Replay original 015 failures through the ordinary CLI in disposable snapshots.

Media are linked read-only for recovery/hash checks. JSON/state are copied, and
new rendering/provider transport is forbidden. This is workflow verification,
not a new visual review or a repair of the original film.
"""
import contextlib
import copy
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import project
import workflow
import production_pipeline
import execution_plan

PROJECT_ROOT = project.ROOT
CASE = 'cases/fresh-preservation-015/case.json'
STATE = 'runs/fresh-preservation-015/workflow.json'
MIDDLE = 'evidence/fresh-preservation-015/review-return-20261114/consumption-fixed/manifest.json'
READ_ONLY_POPEN = subprocess.Popen


def forbid_render(command, *args, **kwargs):
    if '-hide_banner' in command and len(command) == 4 and command[-2] == '-i':
        return READ_ONLY_POPEN(command, *args, **kwargs)  # Header probe only.
    raise AssertionError('new rendering must not start: ' + str(command[:12]))


class OriginalFailureEntries(unittest.TestCase):
    receipts = []

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('ENTRY_REPAIR_RECEIPTS'):
            Path(os.environ['ENTRY_REPAIR_RECEIPTS']).write_text(json.dumps(cls.receipts, ensure_ascii=False, indent=2))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        def copy_file(source, target):
            # Metadata is independent, including candidate state and sidecars.
            return os.link(source, target) if Path(source).stat().st_size > 1024*1024 else shutil.copy2(source, target)
        for rel in ('cases/fresh-preservation-015', 'runs/fresh-preservation-015',
                    'evidence/fresh-preservation-015', 'evidence/artifact-history', 'scripts'):
            shutil.copytree(PROJECT_ROOT/rel, self.root/rel, copy_function=copy_file,
                            ignore=shutil.ignore_patterns('__pycache__'))
        (self.root/'config').mkdir()
        (self.root/'config/models.json').write_text(json.dumps({'execution_mode':'production'}))
        self.root_patch = patch.object(project, 'ROOT', self.root)
        self.root_patch.start()
        self.case = project.read_json(self.root/CASE)
        self.state = project.read_json(self.root/STATE)
        self.entry = self.state['files']['backpack']
        # Absolute provenance keys are relocated, retaining every file hash.
        for candidate in self.entry['candidates']:
            candidate['artifact_hashes'] = {
                str(Path(p).relative_to(PROJECT_ROOT)) if Path(p).is_absolute() else p: h
                for p,h in candidate.get('artifact_hashes', {}).items()}
        self.save()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def save(self):
        (self.root/STATE).write_text(json.dumps(self.state))

    def cli(self, *args):
        output = io.StringIO()
        with patch('sys.argv', ['project.py', *args]), contextlib.redirect_stdout(output):
            code = project.main()
        result = json.loads(output.getvalue())
        self.receipts.append(dict(test=self.id(), argv=list(args), exit_code=code,
            result={k:v for k,v in result.items() if k in
                    ('status','error','output_sha256','new_model_calls','new_renders','execution_reused','files')},
            isolated_snapshot=True))
        return code, result

    def replay_plan(self, relative):
        """Port only paths/code fingerprints; retain original masks/raw/state scope."""
        manifest = project.read_json(self.root/relative)
        out = self.root/'replay';out.mkdir(exist_ok=True)
        prefix = str(PROJECT_ROOT)+'/'
        def relocate(value):
            if isinstance(value, str): return value[len(prefix):] if value.startswith(prefix) else value
            if isinstance(value, list): return [relocate(v) for v in value]
            if isinstance(value, dict): return {relocate(k):relocate(v) for k,v in value.items()}
            return value
        for entry in manifest['operations']:
            plan = relocate(project.read_json(self.root/entry['plan']))
            for key in plan.get('artifact_hashes', {}):
                if key.startswith('scripts/') and key.endswith('.py'):
                    plan['artifact_hashes'][key] = project.digest(self.root/key)
            path = out/(entry['operation_id']+'.json')
            path.write_text(json.dumps(plan))
            entry['plan'] = str(path.relative_to(self.root));entry['plan_sha256'] = project.digest(path)
        path = out/'manifest.json';path.write_text(json.dumps(manifest))
        return str(path.relative_to(self.root))

    def export(self, plan, parent=None, issue=None):
        args = ['execute','--case',CASE,'--plan',plan,'--candidate','forbidden-next','--diagnostic']
        if parent: args += ['--parent',parent,'--issue',issue]
        before = (self.root/STATE).read_bytes()
        with patch('subprocess.Popen', side_effect=forbid_render) as encoder:
            code, result = self.cli(*args)
        encoder.assert_not_called()
        self.assertEqual((self.root/STATE).read_bytes(), before)
        self.assertFalse((self.root/'runs/fresh-preservation-015/backpack/forbidden-next.mp4').exists())
        return code, result

    def test_original_v1_v6_review_failures_stop_renamed_diagnostic_export_without_issues(self):
        # The failure must count even when feedback/issue persistence was skipped.
        self.entry['issues'] = []
        self.entry['preview_trials'] = []
        self.save()
        code, result = self.export(self.entry['candidates'][-1]['plan'])
        self.assertEqual(code, 2, result)
        self.assertIn('Candidate repairs failed repeatedly', result['error'])

    def test_original_pending_v5_cannot_start_v6_before_actual_review(self):
        candidates = self.entry['candidates']
        self.entry['candidates'] = [copy.deepcopy(next(c for c in candidates if c['id']=='api-post-repair-v5'))]
        self.entry['current_candidate_id'] = 'api-post-repair-v5'
        self.entry['issues'] = []
        self.entry['preview_trials'] = []
        self.save()
        code, result = self.export(candidates[-1]['plan'])
        self.assertEqual(code, 2, result)
        self.assertIn('Review current candidate before another export', result['error'])

    def test_middle_only_982_frame_manifest_rejects_before_diagnostic_render(self):
        # Restore the earlier stage in the COPY, independently of the loop guard.
        self.entry['candidates'] = []
        self.entry['current_candidate_id'] = None
        self.entry['issues'] = []
        self.entry['preview_trials'] = []
        self.save()
        code, result = self.export(self.replay_plan(MIDDLE))
        self.assertEqual(code, 2, result)
        self.assertIn('Frozen appearances', result['error'])
        self.assertIn('0', result['error'])

    def test_preview_cannot_restart_the_same_failed_candidate_repairs(self):
        self.entry['issues'] = [];self.entry['preview_trials'] = [];self.save()
        before = (self.root/STATE).read_bytes()
        with patch('subprocess.Popen', side_effect=AssertionError('preview must not render')) as encoder:
            code,result = self.cli('preview','--case',CASE,'--file','backpack',
                '--plan',self.entry['candidates'][-1]['plan'],'--out','forbidden-preview.mp4',
                '--start','0','--end','1')
        self.assertEqual(code,2,result)
        self.assertIn('Candidate repairs failed repeatedly',result['error'])
        self.assertEqual((self.root/STATE).read_bytes(),before)
        encoder.assert_not_called()

    def test_historical_partial_candidate_cannot_qualify_by_review_labels_alone(self):
        # Counterfactual labels in the disposable snapshot must not override
        # the frozen original source scope. No review of the film is asserted.
        candidate = copy.deepcopy(next(c for c in self.entry['candidates']
                                       if c['id'] == 'api-return-review-v2'))
        candidate['status'] = 'qualified'
        candidate['operation_results'] = []
        for review in candidate['reviews']:
            review['status'] = 'pass'
        for operation in candidate['operation_reviews']:
            for check in operation['checks']:
                check['status'] = 'pass'
        self.assertTrue(workflow.quality_reviews_pass(candidate))
        self.entry.update(candidates=[candidate], current_candidate_id=candidate['id'], issues=[])
        self.save()
        observed = workflow.snapshot(self.case)['files']['backpack']['candidates'][0]
        self.assertEqual(observed['binding_status'], 'valid')
        self.assertIn('Frozen appearances missing', observed['coverage_error'])
        before = (self.root/STATE).read_bytes()
        code,result = self.cli('deliver','--case',CASE,'--out','forbidden-delivery')
        self.assertEqual(code,2,result)
        self.assertFalse((self.root/'forbidden-delivery').exists())
        self.assertEqual((self.root/STATE).read_bytes(),before)

    def test_full_length_failed_v6_is_not_a_qualified_delivery(self):
        before = (self.root/STATE).read_bytes()
        with patch('subprocess.Popen', side_effect=AssertionError('delivery must not render')) as encoder:
            code,result = self.cli('deliver','--case',CASE,'--out','forbidden-delivery')
        self.assertEqual(code,2,result)
        self.assertIn('Unqualified videos cannot be delivered',result['error'])
        self.assertFalse((self.root/'forbidden-delivery').exists())
        self.assertEqual((self.root/STATE).read_bytes(),before)
        encoder.assert_not_called()

    def test_resume_batch_preserves_failures_instead_of_suggesting_more_repairs(self):
        before = (self.root/STATE).read_bytes()
        (self.root/'resume.json').write_text(json.dumps(dict(action='resume-batch')))
        with patch('subprocess.Popen', side_effect=AssertionError('resume must not render')) as encoder:
            code,result = self.cli('pipeline','--case',CASE,'--spec','resume.json')
        self.assertEqual(code,0,result)
        self.assertEqual(result['files'][0]['next_action'],'preserve_results_and_diagnose_method')
        self.assertEqual(result['new_model_calls'],0)
        self.assertEqual(result['new_renders'],0)
        self.assertEqual((self.root/STATE).read_bytes(),before)
        encoder.assert_not_called()

    def test_direct_compositor_cannot_bypass_failed_review_gate(self):
        self.entry['issues'] = []
        self.entry['preview_trials'] = []
        self.save()
        manifest = project.read_json(self.root/self.entry['candidates'][-1]['plan'])
        with patch('subprocess.Popen', side_effect=forbid_render) as encoder:
            with self.assertRaisesRegex(ValueError, 'Candidate repairs failed repeatedly'):
                execution_plan.render(self.case, manifest, self.root/'forbidden.mp4', diagnostic=True)
        encoder.assert_not_called()
        self.assertFalse((self.root/'forbidden.mp4').exists())

    def test_existing_exact_candidate_still_recovers_without_render_or_revision(self):
        candidate = self.entry['candidates'][-1]
        before = (self.root/STATE).read_bytes()
        with patch('subprocess.Popen', side_effect=AssertionError('recovery must not render')) as encoder:
            code, result = self.cli('execute','--case',CASE,'--plan',candidate['plan'],
                '--candidate',candidate['id'],'--parent',candidate['parent_id'],'--diagnostic')
        self.assertEqual(code, 0, result)
        self.assertEqual(result['output_sha256'], candidate['output_sha256'])
        self.assertEqual(result['new_renders'], 0)
        self.assertEqual((self.root/STATE).read_bytes(), before)
        encoder.assert_not_called()


if __name__ == '__main__': unittest.main()
