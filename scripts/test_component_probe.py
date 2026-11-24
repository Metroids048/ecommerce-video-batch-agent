import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import component_probe as probe
import production_pipeline as pipeline


class MaintenanceProbeTests(unittest.TestCase):
    def tearDown(self):probe._ACTIVE=None

    def test_scope_rejects_other_source_model_and_range(self):
        p=dict(batch_id='015',file_id='bag',recipe=dict(provider_spec=dict(operation_id='fabric',
            model_id='sam2-video-large-multiobject',source_frames_half_open=[450,803])))
        with patch.object(probe,'active',return_value=p):
            self.assertTrue(probe.matches(dict(batch_id='015'),'bag','fabric','sam2-video-large-multiobject',[450,803]))
            for batch,fid,model,span in [('other','bag','sam2-video-large-multiobject',[450,803]),
                    ('015','other','sam2-video-large-multiobject',[450,803]),
                    ('015','bag','kling-o3-std-edit',[450,803]),
                    ('015','bag','sam2-video-large-multiobject',[0,982])]:
                with self.assertRaises(ValueError):probe.matches(dict(batch_id=batch),fid,'fabric',model,span)

    def test_code_change_and_recipe_change_rejected(self):
        probe._ACTIVE=dict(implementation=dict(code='old'))
        with patch.object(pipeline,'implementation_binding',return_value=dict(code='new')):
            with self.assertRaisesRegex(ValueError,'implementation changed'):probe.active()
        with patch.object(probe,'active',return_value=dict(recipe=dict(provider_spec=dict(points='frozen')))):
            with self.assertRaisesRegex(ValueError,'recipe changed'):probe.verify_spec({},dict(points='changed'))

    def test_probe_does_not_enable_generic_production(self):
        self.assertIsNone(probe.active())
        with patch.object(pipeline.project,'read_json',return_value=dict(execution_mode='maintenance')), \
             patch.object(pipeline.project.ROOT.__class__,'is_file',return_value=True):
            with self.assertRaisesRegex(ValueError,'production is stopped'):pipeline.require_production()

    def test_native_guide_writes_source_colour_and_preserves_unselected_pixels(self):
        import cv2
        import numpy as np
        from source_recolor import Layer
        with tempfile.TemporaryDirectory() as tmp, patch.object(probe.project,'ROOT',Path(tmp).resolve()):
            root=Path(tmp).resolve()
            np.savez_compressed(root/'object.npz',alpha=np.full((1,2,2),255,np.uint8))
            np.savez_compressed(root/'guide.npz',alpha=np.array([[[255,0],[0,255]]],np.uint8))
            plan=dict(masks='object.npz',color_rgb=[85,160,70],recolor_mode='hsv_hue',edit_frames_half_open=[0,1],
                native_colour_guide=dict(masks='guide.npz',source_frames_half_open=[0,1]))
            source=np.array([[[100,40,30],[110,50,40]],[[120,60,50],[130,70,60]]],np.uint8)
            edited,_=Layer(plan,dict(modified_source_frames=[],modified_instance_count=0)).compose(source,0)
            self.assertTrue(np.array_equal(edited[[0,1],[1,0]],source[[0,1],[1,0]]))
            self.assertTrue(np.array_equal(cv2.cvtColor(edited,cv2.COLOR_BGR2HSV)[:,:,2],
                                           cv2.cvtColor(source,cv2.COLOR_BGR2HSV)[:,:,2]))

    def test_derived_appearance_reference_tampering_is_rejected(self):
        import cv2
        import numpy as np
        from native_colour_guide import references
        with tempfile.TemporaryDirectory() as tmp, patch.object(probe.project,'ROOT',Path(tmp).resolve()), \
                patch('workflow.requirement_binding',return_value=dict(source_sha256='source')):
            root=Path(tmp).resolve();cv2.imwrite(str(root/'input.png'),np.zeros((4,6,3),np.uint8))
            manifest=dict(dependency=dict(source_sha256='source'),images=[dict(path='input.png',
                sha256=probe.project.digest(root/'input.png'),source_frame=1)])
            (root/'manifest.json').write_text(json.dumps(manifest))
            recipe=dict(reference_manifest='manifest.json',reference_manifest_sha256=probe.project.digest(root/'manifest.json'),
                provider_spec=dict(image_references=['input.png']))
            item=dict(dimensions=[6,4],source_frame_count_inventory=3)
            references({},item,recipe)
            cv2.imwrite(str(root/'input.png'),np.full((4,6,3),255,np.uint8))
            with self.assertRaisesRegex(ValueError,'reference changed'):references({},item,recipe)

    def test_replacement_keeps_only_explicit_existing_protection(self):
        old=[dict(segmentation_import=p) for p in ('wrong-straps','valid-lining','wrong-handles')]
        replacement=dict(segmentation_import='dense')
        self.assertEqual(probe.protection_layers(old,replacement,['valid-lining']),[old[1],replacement])
        self.assertEqual(probe.protection_layers(old,replacement,None),old+[replacement])
        with self.assertRaisesRegex(ValueError,'existing protection'):
            probe.protection_layers(old,replacement,['invented'])

    def test_pass_consumes_full_same_output_observations_and_failure_stays_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(probe.project,'ROOT',Path(tmp).resolve()):
            root=Path(tmp).resolve()
            def save(name,value):
                (root/name).write_text(json.dumps(value))
                return dict(path=name,sha256=probe.project.digest(root/name))
            output=save('output.mp4',{'synthetic_fixture':True})
            plan=save('pixels.json',{'synthetic_fixture':True})
            ticket=dict(recipe=dict(dependency=dict(source_sha256='source'),detail_frames=[450,452]),
                output=dict(**output,plan=plan['path'],plan_sha256=plan['sha256'],source_frames_half_open=[450,453]))
            inventory=save('inventory.json',dict(output_sha256=output['sha256'],source_sha256='source',
                paired_frames=[450,451,452],overview_sheets=['sheet.jpg'],
                evidence=[dict(path='native-0450.jpg'),dict(path='native-0452.jpg')]))
            playback=save('play.json',dict(ended=True,begin=dict(rate=1)))
            audit=dict(output_sha256=output['sha256'],source_sha256='source',verdict='pass',
                diagnosis='Synthetic fixture only',observed_frames=[450,451,452],evidence=[output,plan,inventory,playback],
                checks={k:'pass' for k in probe.REVIEW_CHECKS},inventory=inventory,playback=playback)
            probe.validate_review(ticket,audit)
            audit['observed_frames']=[450]
            with self.assertRaisesRegex(ValueError,'all paired frames'):probe.validate_review(ticket,audit)
            audit['verdict']='fail'
            probe.validate_review(ticket,audit)
            audit['verdict']='pass';audit['observed_frames']=[450,451,452];audit['checks']['protection']='fail'
            with self.assertRaisesRegex(ValueError,'six quality'):probe.validate_review(ticket,audit)
            audit['checks']['protection']='pass';(root/'pixels.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'plan changed'):probe.validate_review(ticket,audit)


if __name__=='__main__':unittest.main()
