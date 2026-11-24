"""Synthetic mainchain regressions. Fake predictions are not semantic acceptance."""
import json
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import project
import workflow
import semantic_scan
from test_source_observation import SourceObservationTests


class SemanticScanTests(unittest.TestCase):
    setUp = SourceObservationTests.setUp
    tearDown = SourceObservationTests.tearDown
    cli = SourceObservationTests.cli
    def predictions(self, frame):
        return json.dumps({'operations':{op:dict(state='visible',all_instances=True,
            reason='Synthetic prediction',instances=[dict(box=[300,50,600,350])]) for op in ('a','b')}})

    def test_cli_model_receives_change_and_protection_without_changing_source_identity(self):
        item = self.case['files'][0]
        item['operations'][0].update(target='原红色包体和提带', desired_change='改为墨绿，保留布纹',
            protect=['手、白色标识'], target_color_rgb=[28,110,99],
            replacement_asset='asset.png')
        (self.root/'case.json').write_text(json.dumps(self.case))
        workflow.state_path(self.case).unlink()
        workflow.initialize(self.case,'case.json')
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}), \
                patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=self.predictions
            self.cli(dict(action='scan-source',file_id='one'))
        sent = model.call_args.args[1]
        operations = json.loads(sent.split('Operations: ',1)[1].split('\nOriginal user request:',1)[0])
        self.assertEqual(operations[0]['desired_change'],'改为墨绿，保留布纹')
        self.assertEqual(operations[0]['target_color_rgb'],[28,110,99])
        self.assertEqual(operations[0]['replacement_asset'],'asset.png')
        self.assertEqual(operations[0]['protect'],['手、白色标识'])
        self.assertIn('source appearance',sent)
        self.assertIn('y_min,x_min,y_max,x_max',sent)
        self.assertIn('x_min,y_min,x_max,y_max',semantic_scan.prompt_for(self.case,item))

    def test_original_coordinate_counterexample_has_explicit_order(self):
        item = dict(dimensions=[1080,1920],operations=[dict(id='fabric-blue')])
        raw = json.dumps({'operations':{'fabric-blue':dict(state='visible',all_instances=True,
            reason='Original office frame 0',instances=[dict(box=[694,147,984,324])])}})
        row = semantic_scan.parse_prediction(raw,item,0,coordinate_order='yxyx')['fabric-blue']
        self.assertEqual(row['instances'][0]['bbox'],[158.76,1332.48,191.16,556.8])
        # Legacy journals must retain their original meaning until explicit repair.
        self.assertEqual(semantic_scan.parse_prediction(raw,item,0)['fabric-blue']['instances'][0]['bbox'],
                         [749.52,282.24,313.2,339.84])

    def test_coordinate_repair_preserves_legacy_journal_and_refuses_model_change(self):
        import copy
        self.case['execution_policy']['mode']='method_trial'
        (self.root/'case.json').write_text(json.dumps(self.case))
        workflow.state_path(self.case).unlink()
        workflow.initialize(self.case,'case.json')
        self.prepare = SourceObservationTests.prepare.__get__(self)
        self.prepare()
        engine=dict(options={'model':'fixed'},code={'semantic_scan.py':'old','semantic_worker.py':'fixed'})
        item=self.case['files'][0]
        with workflow.locked_state(self.case) as state:
            entry=state['files']['one']
            binding=dict(dependency=workflow.requirement_binding(self.case,item),source_sha256=item['source_sha256'],
                queue=entry['source_observation']['queue'],engine=engine,prompt_sha256=workflow.hash_json(semantic_scan.prompt_for(self.case,item)))
            directory=self.root/'old-scan';directory.mkdir()
            path=directory/'frame-00000000.json'
            raw=self.predictions(0)
            path.write_text(json.dumps(dict(frame=0,binding_sha256=workflow.hash_json(binding),image_sha256='fixture',text=raw,
                operations=semantic_scan.parse_prediction(raw,item,0))))
            ref=semantic_scan.observation.artifact(path)
            old=dict(binding=binding,engine=engine,directory='old-scan',status='failed',frame_artifacts={'0':ref})
            entry['source_semantic_execution']=old
        original=path.read_bytes()
        diagnosis=self.root/'diagnosis.json';diagnosis.write_text(json.dumps(dict(source_sha256=item['source_sha256'],
            old_binding_sha256=workflow.hash_json(binding),confirmed_order='yxyx',reviewed_frames=[0])))
        spec=dict(action='repair-semantic-coordinate-order',file_id='one',diagnosis=semantic_scan.observation.artifact(diagnosis))
        changed=copy.deepcopy(engine);changed['options']['model']='other'
        with patch('semantic_scan.engine_binding',return_value=changed):self.cli(spec,expected=2)
        with patch('semantic_scan.engine_binding',return_value=engine):result=self.cli(spec)
        self.assertEqual(result['processed_frames'],1)
        self.assertEqual(path.read_bytes(),original)
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(entry['source_semantic_execution_history'],[old])
        self.assertEqual(entry['source_semantic_execution']['binding']['coordinate_order'],'yxyx')
        with patch('semantic_scan.engine_binding',return_value=engine):self.cli(spec,expected=2)

    def test_mainchain_executes_every_frame_and_reuses_without_model_or_decode(self):
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}), \
                patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=self.predictions
            result=self.cli(dict(action='scan-source',file_id='one'))
            self.assertEqual(model.return_value.__enter__.return_value.predict.call_count,6)
        self.assertEqual(result['processed_frames'],6)
        self.assertEqual(result['next_action'],'review_source_scan')
        saved=project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertNotIn('receipt',saved['source_observation']) # no automatic host signature
        data=semantic_scan.read_result(self.case,'one',saved)
        self.assertEqual(len(data['operations']['a']['records']),6)
        self.assertEqual(data['operations']['a']['records'][0]['instances'][0]['bbox'],[4.8,19.2,28.8,19.2])
        before=workflow.state_path(self.case).read_bytes()
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}), \
                patch('semantic_scan.LocalModel',side_effect=AssertionError('must not load model')), \
                patch('cv2.VideoCapture',side_effect=AssertionError('must not decode')):
            again=self.cli(dict(action='scan-source',file_id='one'))
            self.assertEqual(again,result)
        self.assertEqual(workflow.state_path(self.case).read_bytes(),before)

    def test_parser_preserves_multiple_instances_and_ambiguous_model_output(self):
        item=self.case['files'][0]
        p=json.loads(self.predictions(0));p['operations']['a']['instances'].append(dict(box=[600,300,950,600]))
        rows=semantic_scan.parse_prediction(json.dumps(p),item,0)
        self.assertEqual(len(rows['a']['instances']),2)
        p['operations']['a']['all_instances']=False
        rows=semantic_scan.parse_prediction(json.dumps(p),item,0)
        self.assertEqual(rows['a']['state'],'unresolved')
        self.assertEqual(len(rows['a']['predicted_instances']),2)
        self.assertTrue(all(r['state']=='unresolved' for r in semantic_scan.parse_prediction('broken',item,0).values()))
        p['operations']['a']['instances'][0]['box']=[700,0,100,600]
        self.assertEqual(semantic_scan.parse_prediction(json.dumps(p),item,0)['a']['state'],'unresolved')

    def test_resume_keeps_completed_frames_after_local_worker_failure(self):
        def fail(n):
            if n==2:raise RuntimeError('synthetic worker failure')
            return self.predictions(n)
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=fail
            self.cli(dict(action='scan-source',file_id='one'),expected=2)
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(semantic_scan.progress(self.case,'one',entry)['processed_frames'],2)
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=self.predictions
            result=self.cli(dict(action='scan-source',file_id='one',resume_reason='Synthetic worker repaired; resume same frozen model'))
            self.assertEqual([c.args[0] for c in model.return_value.__enter__.return_value.predict.call_args_list],[2,3,4,5])
        self.assertEqual(result['processed_frames'],6)

    def test_recovery_indexes_orphan_journal_without_model_or_rewriting_bytes(self):
        def fail(n):
            if n==2:raise RuntimeError('synthetic interruption')
            return self.predictions(n)
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=fail
            self.cli(dict(action='scan-source',file_id='one'),expected=2)
        with workflow.locked_state(self.case) as state:
            saved=state['files']['one']['source_semantic_execution']
            orphan=saved['frame_artifacts'].pop('1')
            original_error=saved['error']
        original=project.project_path(orphan['path']).read_bytes()
        with (patch('semantic_scan.engine_binding',side_effect=AssertionError('must not probe')),
                patch('semantic_scan.LocalModel',side_effect=AssertionError('must not infer')),
                patch('cv2.VideoCapture',side_effect=AssertionError('must not decode'))):
            result=self.cli(dict(action='recover-semantic-journal',file_id='one'))
            self.assertEqual(result['recovered_frames'],[1])
            self.assertEqual(self.cli(dict(action='recover-semantic-journal',file_id='one'))['recovered_frames'],[])
        saved=project.read_json(workflow.state_path(self.case))['files']['one']['source_semantic_execution']
        self.assertEqual(saved['frame_artifacts']['1'],orphan)
        self.assertEqual(saved['error'],original_error)
        self.assertEqual(project.project_path(orphan['path']).read_bytes(),original)
        project.project_path(orphan['path']).write_bytes(original+b' ')
        self.cli(dict(action='recover-semantic-journal',file_id='one'),expected=2)

    def test_runtime_maintenance_keeps_packet_provenance_and_refuses_model_change(self):
        import copy
        self.case['execution_policy']['mode']='method_trial'
        (self.root/'case.json').write_text(json.dumps(self.case))
        workflow.state_path(self.case).unlink();workflow.initialize(self.case,'case.json')
        engine=dict(options={'model':'fixed'},code={'semantic_scan.py':'old','semantic_worker.py':'old'})
        def fail(n):
            if n==2:raise RuntimeError('synthetic failure')
            return self.predictions(n)
        with patch('semantic_scan.engine_binding',return_value=engine),patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=fail
            self.cli(dict(action='scan-source',file_id='one'),expected=2)
        saved=project.read_json(workflow.state_path(self.case))['files']['one']['source_semantic_execution']
        ref=saved['frame_artifacts']['0'];original=project.project_path(ref['path']).read_bytes()
        new=copy.deepcopy(engine);new['code']['semantic_worker.py']='instrumented'
        diagnosis=self.root/'runtime-diagnosis.json'
        def request(updated):
            diagnosis.write_text(json.dumps(dict(source_sha256=self.case['files'][0]['source_sha256'],
                old_binding_sha256=workflow.hash_json(saved['binding']),new_engine_sha256=workflow.hash_json(updated),
                cause='synthetic telemetry maintenance',preserved_generation_contract={'native_image':True})))
            return dict(action='repair-semantic-runtime',file_id='one',diagnosis=semantic_scan.observation.artifact(diagnosis))
        changed=copy.deepcopy(new);changed['options']['model']='other'
        with patch('semantic_scan.engine_binding',return_value=changed):self.cli(request(changed),expected=2)
        with patch('semantic_scan.engine_binding',return_value=new):self.cli(request(new))
        self.assertEqual(project.project_path(ref['path']).read_bytes(),original)
        with patch('semantic_scan.engine_binding',return_value=new),patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=self.predictions
            result=self.cli(dict(action='scan-source',file_id='one',resume_reason='Synthetic telemetry update'))
            self.assertEqual([c.args[0] for c in model.return_value.__enter__.return_value.predict.call_args_list],[2,3,4,5])
        data=semantic_scan.read_result(self.case,'one',project.read_json(workflow.state_path(self.case))['files']['one'])
        self.assertEqual(data['runtime_maintenance'][0]['binding'],saved['binding'])
        self.assertEqual(result['processed_frames'],6)

    def test_host_consumes_saved_predictions_without_retyping_and_freezes_same_receipt(self):
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=self.predictions
            self.cli(dict(action='scan-source',file_id='one'))
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        request=dict(action='record-source-observation',file_id='one',from_semantic_scan=entry['source_semantic_execution']['result'],
            observer='synthetic-reviewer',reviewed_frames=list(range(6)))
        result=self.cli(request)
        self.assertEqual(result['next_action'],'freeze-route')
        self.cli(dict(action='freeze-route',file_id='one'))
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        self.assertEqual(entry['route_handoff']['observation'],entry['source_observation']['receipt'])
        receipt=project.read_json(project.project_path(entry['source_observation']['receipt']['path']))
        self.assertEqual(receipt['semantic_execution'],entry['source_semantic_execution']['result'])

    def test_maintenance_and_reuse_only_refuse_before_model_execution(self):
        (self.root/'config').mkdir(exist_ok=True)
        (self.root/'config/models.json').write_text(json.dumps(dict(execution_mode='maintenance')))
        for policy in ({'local_only':False},{'reuse_only':True}):
            self.case['execution_policy'].update(policy)
            (self.root/'case.json').write_text(json.dumps(self.case))
            with patch('semantic_scan.engine_binding',side_effect=AssertionError('model probe forbidden')):
                self.cli(dict(action='scan-source',file_id='one'),expected=2)

    def test_missing_local_weights_refuses_no_download(self):
        with self.assertRaisesRegex(ValueError,'local model'):
            semantic_scan.engine_binding(dict(model_path=str(self.root/'missing')))

    def test_model_uncertainty_keeps_original_predictions_without_signing_host_complete(self):
        def uncertain(n):
            p=json.loads(self.predictions(n))
            p['operations']['a'].update(all_instances=False,reason='Synthetic uncertain second target')
            p['operations']['b'].update(state='absent',instances=[],reason='Synthetic absent target')
            return json.dumps(p)
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=uncertain
            result=self.cli(dict(action='scan-source',file_id='one'))
        entry=project.read_json(workflow.state_path(self.case))['files']['one']
        self.cli(dict(action='record-source-observation',file_id='one',from_semantic_scan=result['result'],
            observer='synthetic-reviewer',reviewed_frames=list(range(6))))
        self.cli(dict(action='freeze-route',file_id='one'),expected=2)
        data=semantic_scan.read_result(self.case,'one',entry)
        self.assertEqual(data['operations']['a']['unresolved_frames_half_open'],[[0,6]])
        self.assertEqual(len(data['operations']['a']['records'][0]['predicted_instances']),1)
        self.assertEqual(data['operations']['b']['appearance_ranges_half_open'],[])
        self.assertEqual(data['operations']['a']['records'][1]['source_pts'],project.read_json(self.root/'timeline.json')['pts'][1])

    def test_interrupted_scan_refuses_missing_cached_frame_and_unexplained_retry(self):
        def fail(n):
            if n==2:raise RuntimeError('synthetic failure')
            return self.predictions(n)
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}),patch('semantic_scan.LocalModel') as model:
            model.return_value.__enter__.return_value.predict.side_effect=fail
            self.cli(dict(action='scan-source',file_id='one'),expected=2)
        with patch('semantic_scan.engine_binding',return_value={'fixture':True}), \
                patch('semantic_scan.LocalModel',side_effect=AssertionError('must not infer')):
            self.cli(dict(action='scan-source',file_id='one'),expected=2)
            entry=project.read_json(workflow.state_path(self.case))['files']['one']
            ref=entry['source_semantic_execution']['frame_artifacts']['0']
            project.project_path(ref['path']).write_text('{}')
            self.cli(dict(action='scan-source',file_id='one',resume_reason='Synthetic diagnosed worker failure'),expected=2)

    def test_new_policy_recommends_real_scan_and_worker_has_offline_structured_contract(self):
        from production_effort import new_policy
        from semantic_worker import response_schema
        self.assertEqual(new_policy()['source_semantic_execution'],'host_agent')
        self.case['execution_policy']['source_semantic_execution']='mlx_vlm'
        (self.root/'case.json').write_text(json.dumps(self.case))
        workflow.state_path(self.case).unlink() # disposable fixture only, no production state migration
        workflow.initialize(self.case,'case.json')
        result=self.cli(dict(action='prepare-source-observation',file_id='one'))
        self.assertEqual(result['next_action'],'scan-source')
        self.assertEqual(response_schema(['a','b'])['properties']['operations']['required'],['a','b'])
        with patch.dict('os.environ',{'RH_API_KEY':'never-forward'}):
            self.assertNotIn('RH_API_KEY',semantic_scan.worker_env())

    def test_worker_consumes_native_image_with_mature_loader_and_structured_generate(self):
        import semantic_worker
        from unittest.mock import Mock
        model=types.SimpleNamespace(config={})
        tokenizer=object();processor=types.SimpleNamespace(tokenizer=tokenizer)
        load=Mock(return_value=(model,processor))
        generate=Mock(return_value=types.SimpleNamespace(text=self.predictions(0)))
        template=Mock(return_value='image-prompt')
        structured=types.ModuleType('mlx_vlm.structured')
        structured.build_json_schema_logits_processor=Mock(return_value='grammar')
        incoming=io.StringIO(json.dumps(dict(model_path=str(self.root),operation_ids=['a','b']))+'\n'+
                            json.dumps(dict(frame=0,image='/synthetic/native.png',prompt='find both'))+'\n')
        outgoing=io.StringIO()
        with patch.object(semantic_worker,'memory_event'),patch.object(semantic_worker,'imports',return_value=(load,generate,template)), \
                patch.dict(sys.modules,{'mlx_vlm':types.ModuleType('mlx_vlm'),'mlx_vlm.structured':structured}), \
                patch.object(sys,'stdin',incoming),patch.object(sys,'argv',['semantic_worker.py']),redirect_stdout(outgoing):
            semantic_worker.main()
        load.assert_called_once_with(str(self.root),local_files_only=True,trust_remote_code=False,token=False)
        self.assertEqual(generate.call_args.kwargs['image'],['/synthetic/native.png'])
        self.assertEqual(generate.call_args.kwargs['logits_processors'],['grammar'])
        self.assertEqual(template.call_args.kwargs['num_images'],1)
        self.assertEqual(json.loads(outgoing.getvalue().splitlines()[1])['frame'],0)
        with self.assertRaisesRegex(RuntimeError,'network'):
            semantic_worker.deny_network()
