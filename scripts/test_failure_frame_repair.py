"""Regression for cross-order prompts and late local/remote route conflicts."""
import json
import unittest
from unittest.mock import patch
import project
import semantic_scan
import workflow
from test_localization_effort import LocalizationEffortTests


class FailureRepairTests(unittest.TestCase):
    setUp=LocalizationEffortTests.setUp
    tearDown=LocalizationEffortTests.tearDown
    cli=LocalizationEffortTests.cli
    configure=LocalizationEffortTests.configure
    predictions=LocalizationEffortTests.predictions
    def test_prompt_is_per_file_and_does_not_presuppose_presence(self):
        (self.root/self.case['user_input']).write_text('OTHER_ORDER_SENTINEL: paint another bag yellow')
        item=self.case['files'][0]
        item['operations'][0].update(target='red backpack',desired_change='Keep texture; turn green')
        sent=semantic_scan.prompt_for(self.case,item,'yxyx')
        self.assertNotIn('OTHER_ORDER_SENTINEL',sent)
        self.assertIn('Keep texture; turn green',sent)
        self.assertNotIn('"all_instances":true',sent)
        self.assertIn('whether',sent)

    def strict_local(self):
        spec=self.configure()
        self.case['files'][0]['operations'][0].update(kind='modify',surface_motion='deformable',
            recolor_mode='preserve_source_details',target_color_rgb=[28,110,99],semantic_label='backpack fabric',
            desired_change='Recolor fabric, preserve original details')
        (self.root/'case.json').write_text(json.dumps(self.case))
        workflow.state_path(self.case).unlink();workflow.initialize(self.case,'case.json')
        return spec

    def test_plan_exposes_edit_blocker_before_localization(self):
        self.strict_local()
        result=self.cli(dict(action='plan-source-scan',file_id='one'))
        self.assertTrue(any('SAM3' in x for x in result['execution_blockers']))
        self.assertFalse(result['complete_delivery_estimate_available'])

    def test_fresh_full_scan_does_not_load_model_for_impossible_local_route(self):
        spec=self.strict_local()
        with patch('semantic_scan.engine_binding',side_effect=AssertionError('must stop before loading')), \
                patch('source_observation.prepare',side_effect=AssertionError('must stop before full decode')):
            self.cli(spec,expected=2)

    def test_targeted_diagnosis_preserves_scan_and_does_not_claim_review(self):
        spec=self.configure()
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=self.predictions
            self.cli(spec)
        before=project.read_json(workflow.state_path(self.case))['files']['one']['source_semantic_execution']
        diagnosis=self.root/'failure.json'
        diagnosis.write_text(json.dumps(dict(source_sha256=self.case['files'][0]['source_sha256'],
            frames=[0],reason='Synthetic preserved false target')))
        request=dict(action='diagnose-source-frames',file_id='one',frames=[0],diagnosis=semantic_scan.observation.artifact(diagnosis))
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=self.predictions
            result=self.cli(request)
            self.assertEqual(model.return_value.__enter__.return_value.predict.call_count,1)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['frames'],[0])
        self.assertEqual(before,project.read_json(workflow.state_path(self.case))['files']['one']['source_semantic_execution'])
        with patch('semantic_scan.LocalModel',side_effect=AssertionError('must reuse')):
            self.assertEqual(self.cli(request),result)
        self.cli(dict(request,frames=[0,5]),expected=2)
        self.cli(dict(action='freeze-route',file_id='one'),expected=2)


if __name__=='__main__':unittest.main()
