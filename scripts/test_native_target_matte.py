"""The real native-plan consumer must retain source-grid protection, no cloud."""
import contextlib
import io
import json
from unittest.mock import patch
import unittest
import shutil
from pathlib import Path
import numpy as np
import project
import workflow
import runninghub_adapter as provider
import test_production_pipeline as fixtures


class NativeTargetMatte(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PipelineTests(); self.fixture.setUp()
        self.root, self.case = self.fixture.root, self.fixture.case
        (self.root/'scripts').mkdir(exist_ok=True)
        for name in ('aigc_composite.py','execution_plan.py','source_matte.py','source_timeline.py'):
            shutil.copyfile(Path(__file__).with_name(name),self.root/'scripts'/name)
        self.item = self.case['files'][0]
        request = dict(file_id='one', operation_id='a', unit_id='original',
                       model_id='kling-o3-std-edit', source_frames_half_open=[0,6],
                       dependency=workflow.requirement_binding(self.case,self.item), artifact_hashes={})
        (self.root/'request-native.json').write_text(json.dumps(request))
        (self.root/'observed.txt').write_text('Synthetic six-frame geometry fixture; no semantic or production acceptance')
        review = dict(observed_frames=list(range(6)), evidence=[
            dict(path='observed.txt',sha256=project.digest(self.root/'observed.txt'))])
        (self.root/'review-native.json').write_text(json.dumps(review))
        raw = dict(path='source.mp4', sha256=self.item['source_sha256'], mode='synthetic_fixture')
        unit = dict(unit_id='original', file_id='one', operation_id='a', status='raw_ready',
                    provider_task_id='fixture-only', dependency=request['dependency'],
                    request_path='request-native.json', request_fingerprint=workflow.hash_json(request), raw=raw,
                    raw_review=dict(path='review-native.json', sha256=project.digest(self.root/'review-native.json'), status='pass'))
        with workflow.locked_state(self.case) as state:
            state['files']['one']['execution_units'] = [unit]
        self.alpha = np.zeros((6,64,96),np.uint8); self.alpha[:,10:40,10:40] = 255
        self.guard = np.zeros_like(self.alpha); self.guard[:,20:25,20:25] = 255
        (self.root/'observed.txt').write_text('Synthetic six-frame geometry fixture; no semantic or production acceptance')
        self.spec = dict(file_id='one',operation_id='a',unit_ids=['original'],target_matte='target.npz',
                         frame_states=['visible']*6, risk_preview_ranges_half_open=[[0,6]],
                         observation_evidence=[dict(path='observed.txt',sha256=project.digest(self.root/'observed.txt'))],out='native/manifest.json')

    def tearDown(self): self.fixture.tearDown()

    def matte(self, metadata=True):
        values = dict(alpha=self.alpha,protected=self.guard)
        if metadata:
            values.update(source_sha256=self.item['source_sha256'], source_dimensions=[96,64],
                          mask_dimensions=[96,64], frames_half_open=[0,6])
        np.savez(self.root/'target.npz',**values)

    def test_native_plan_consumes_protection_before_temporal_weights(self):
        self.matte(); provider.plan_native_recolor(self.case,self.spec)
        with np.load(self.root/'native/original-alpha.npz') as arrays:
            self.assertFalse(arrays['alpha'][self.guard>0].any())
            np.testing.assert_array_equal(arrays['protected'],self.guard)
        manifest = project.read_json(self.root/'native/manifest.json')
        plan = project.read_json(self.root/manifest['operations'][0]['plan'])
        self.assertEqual(plan['artifact_hashes']['target.npz'],project.digest(self.root/'target.npz'))

    def test_unbound_target_grid_is_rejected_before_any_partial_plan(self):
        self.matte(metadata=False)
        with self.assertRaisesRegex(ValueError,'source/grid/range/protection'):
            provider.plan_native_recolor(self.case,self.spec)
        self.assertFalse((self.root/'native/original-timing.json').exists())

    def test_normal_cli_recovers_exact_assembly_without_decode_or_state_change(self):
        self.matte(); provider.plan_native_recolor(self.case,self.spec)
        before = workflow.state_path(self.case).read_bytes()
        paths = list((self.root/'native').iterdir())
        hashes = {p.name:project.digest(p) for p in paths}
        request = dict(self.spec, action='plan-native-recolor')
        (self.root/'assemble.json').write_text(json.dumps(request))
        output = io.StringIO()
        with patch('sys.argv',['project.py','provider','--case','case.json','--spec','assemble.json']), \
             patch.object(provider.source_timeline,'build',side_effect=AssertionError('must not decode original raw again')), \
             contextlib.redirect_stdout(output):
            code = project.main()
        self.assertEqual(code,0,output.getvalue())
        result = json.loads(output.getvalue())
        self.assertTrue(result['assembly_reused'])
        self.assertEqual(result['manifest'],'native/manifest.json')
        self.assertEqual(result['new_renders'],0)
        self.assertEqual(workflow.state_path(self.case).read_bytes(),before)
        self.assertEqual({p.name:project.digest(p) for p in paths},hashes)

    def test_changed_request_dependencies_and_partial_assembly_never_overwrite(self):
        self.matte(); provider.plan_native_recolor(self.case,self.spec)
        manifest = self.root/'native/manifest.json'
        original = manifest.read_bytes()
        changed = dict(self.spec,diagnostic=True)
        with self.assertRaisesRegex(ValueError,'assembly.*(binding|request)|request.*changed'):
            provider.plan_native_recolor(self.case,changed)
        for name in ('target.npz','review-native.json','native/original-alpha.npz','native/manifest-pixels.json'):
            path = self.root/name; saved = path.read_bytes(); path.write_bytes(saved+b'changed')
            with self.assertRaises(ValueError):
                provider.plan_native_recolor(self.case,self.spec)
            path.write_bytes(saved)
        manifest.unlink()
        with self.assertRaisesRegex(ValueError,'[Pp]artial|incomplete|already exists'):
            provider.plan_native_recolor(self.case,self.spec)
        self.assertFalse(manifest.exists())
        manifest.write_bytes(original)

    def test_unbound_legacy_assembly_is_preserved_for_pipeline_consumption(self):
        self.matte(); provider.plan_native_recolor(self.case,self.spec)
        path = self.root/'native/manifest.json'
        manifest = project.read_json(path); del manifest['assembly_request_fingerprint']
        path.write_text(json.dumps(manifest)); before = path.read_bytes()
        with self.assertRaisesRegex(ValueError,'legacy binding absent'):
            provider.plan_native_recolor(self.case,self.spec)
        self.assertEqual(path.read_bytes(),before)
