"""Binding and concurrency regressions; synthetic fixtures are not media acceptance."""
import copy
import json
import tempfile
import unittest
import threading
from pathlib import Path
from unittest.mock import patch

import project
import workflow


class WorkflowRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.root_patch = patch.object(project, 'ROOT', self.root)
        self.root_patch.start()
        for name, content in [('source.mp4', b'source'), ('out.mp4', b'candidate'), ('input.txt', b'remove logo')]:
            (self.root / name).write_bytes(content)
        self.case = {'batch_id': 'test', 'case_version': 'v1', 'user_input': 'input.txt', 'files': [
            {'id': 'one', 'source': 'source.mp4', 'source_sha256': project.digest(self.root/'source.mp4'),
             'duration_seconds': 10, 'fps_container_average': 30, 'operations': [
                 {'id': 'remove', 'kind': 'remove', 'target': 'logo', 'protect': ['hands']}]}]}
        self.candidate = {'id': 'v1', 'file_id': 'one', 'source_sha256': self.case['files'][0]['source_sha256'],
                          'output': 'out.mp4', 'output_sha256': project.digest(self.root/'out.mp4'),
                          'dependency': workflow.requirement_binding(self.case, self.case['files'][0]),
                          'reviews': [{'status': 'pass', 'output_sha256': 'old', 'kind': 'visual'}],
                          'parent_id': 'source', 'status': 'review_copy', 'accepted': False}
        self.state = {'revision': 1, 'batch_id': 'test', 'case_version': 'v1', 'files': {'one': {
            'current_candidate_id': 'v1', 'candidates': [self.candidate], 'issues': [], 'jobs': []}}}

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def test_exact_code_archive_preserves_frozen_candidate_without_qualifying_it(self):
        code=self.root/'scripts/execution_plan.py';code.parent.mkdir();code.write_bytes(b'old code')
        digest=project.digest(code);self.candidate['artifact_hashes']={'scripts/execution_plan.py':digest}
        code.write_bytes(b'new code')
        with self.assertRaises(ValueError):workflow.verify_historical_parent(self.case,'one',self.candidate)
        archive=self.root/'evidence/artifact-history'/(digest+'.py');archive.parent.mkdir(parents=True);archive.write_bytes(b'old code')
        workflow.verify_historical_parent(self.case,'one',self.candidate)
        workflow.verify_candidate(self.case,'one',self.candidate)
        self.assertFalse(workflow.quality_reviews_pass(self.candidate))
        (self.root/'mask.npz').write_bytes(b'original mask')
        self.candidate['artifact_hashes']['mask.npz']=project.digest(self.root/'mask.npz')
        (self.root/'mask.npz').write_bytes(b'changed mask')
        with self.assertRaisesRegex(ValueError,'dependency hash changed'):
            workflow.verify_candidate(self.case,'one',self.candidate)
        self.candidate['artifact_hashes'].pop('mask.npz')
        (self.root/'out.mp4').write_bytes(b'wrong bytes')
        with self.assertRaises(ValueError):workflow.verify_historical_parent(self.case,'one',self.candidate)

    def test_feedback_can_bind_exact_archived_parent_without_qualifying_it(self):
        code=self.root/'scripts/aigc_composite.py';code.parent.mkdir();code.write_bytes(b'old code')
        digest=project.digest(code);self.candidate['artifact_hashes']={'scripts/aigc_composite.py':digest}
        code.write_bytes(b'new code')
        body=dict(file_id='one',candidate_id='v1',candidate_sha256=self.candidate['output_sha256'],
                  source_sha256=self.candidate['source_sha256'],expected_revision=1,source_time=1,
                  text='Re-export under current compositor dependency',category='composition',actor='host_agent')
        with self.assertRaises(ValueError):workflow.bind_feedback(self.case,self.state,body)
        archive=self.root/'evidence/artifact-history'/(digest+'.py');archive.parent.mkdir(parents=True);archive.write_bytes(b'old code')
        issue=workflow.bind_feedback(self.case,self.state,body)
        self.assertEqual(issue['candidate_sha256'],self.candidate['output_sha256'])
        workflow.verify_candidate(self.case,'one',self.candidate)
        self.assertFalse(workflow.quality_reviews_pass(self.candidate))

    def test_report_after_first_export_has_no_missing_reason(self):
        self.state['case_binding'] = workflow.case_binding(self.case)
        self.state['files']['one']['raw_returns'] = []
        path = workflow.state_path(self.case)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.state))
        report = workflow.delivery_report(self.case)
        self.assertEqual(report['files'][0]['current_candidate_id'], 'v1')
        self.assertIsNone(report['files'][0]['missing_reason'])

    def test_cost_keeps_unknown_reserved_and_counts_repeated_charge_receipt_once(self):
        self.state['files']['one']['execution_units']=[
            dict(unit_id='paid',new_model_calls=1,reserved_cny=10,receipts=[
                {'response':{'actual_charge_cny':2.5}}, {'response':{'actual_charge_cny':2.5}},
                {'response':{'status':'raw_ready'}}]),
            dict(unit_id='unknown',new_model_calls=1,new_cost_cny=0,reserved_cny=7)]
        costs=workflow.cost_summary(self.state)
        self.assertEqual(costs['known_actual_receipts_cny'],2.5)
        self.assertIsNone(costs['new_cost_cny']);self.assertEqual(costs['reserved_total_cny'],17)
        self.assertEqual(costs['unsettled_provider_units'],['unknown'])

    def test_mask_app_nested_costs_are_reported_once_per_task_not_per_node_or_query(self):
        response = dict(output_receipt={'outputs': [dict(nodeId='68', consumeMoney='0.112', taskCostTime='67'),
                                                   dict(nodeId='70', consumeMoney='0.112', taskCostTime='67')]})
        self.state['files']['one']['execution_units'] = [
            dict(unit_id='paid', provider_task_id='task', reserved_cny=10, receipts=[
                dict(response=response), dict(response=response)]),
            dict(unit_id='failed', provider_task_id='failed-task', reserved_cny=10, receipts=[])]
        costs = workflow.cost_summary(self.state)
        self.assertEqual(costs['provider_reported_consumeMoney_sum'], '0.112')
        self.assertEqual(costs['provider_reported_compute_seconds'], '67')
        self.assertEqual(costs['provider_reported_missing_units'], ['failed'])
        self.assertIsNone(costs['new_cost_cny'])
        self.assertFalse(costs['account_bill_verified'])
        response['output_receipt']['outputs'][1]['consumeMoney'] = '0.2'
        costs = workflow.cost_summary(self.state)
        self.assertEqual(costs['provider_reported_consumeMoney_sum'], '0')
        self.assertIn('paid', costs['provider_reported_ambiguous_units'])

    def test_explicit_settlement_route_reconciles_usage_once_but_keeps_failed_unknown(self):
        response=dict(task_id='task',status='SUCCESS',usage=dict(consumeMoney='0.409',thirdPartyConsumeMoney=None))
        self.state['files']['one']['execution_units']=[
            dict(unit_id='settled',provider_task_id='task',reserved_cny=10,
                 receipts=[dict(action='settle',response=response),dict(action='settle',response=response)]),
            dict(unit_id='failed',provider_task_id='failed',reserved_cny=10,receipts=[])]
        costs=workflow.cost_summary(self.state)
        self.assertEqual(costs['known_actual_receipts_cny'],.409)
        self.assertEqual(costs['unsettled_provider_units'],['failed'])
        self.assertIsNone(costs['new_cost_cny'])

    def test_cost_deduplicates_task_aliases_but_not_distinct_tasks(self):
        response = dict(actual_charge_cny=2.5, output_receipt=dict(outputs=[
            dict(consumeMoney='0.112', taskCostTime='67')]))
        self.state['files']['one']['execution_units'] = [
            dict(unit_id=name, provider_task_id=task, new_model_calls=1,
                 reserved_cny=10, receipts=[dict(response=response)])
            for name, task in [('original', 'task-a'), ('alias', 'task-a'), ('other', 'task-b')]]
        before = copy.deepcopy(self.state)
        costs = workflow.cost_summary(self.state)
        self.assertEqual(costs['known_actual_receipts_cny'], 5)
        self.assertEqual(costs['new_model_calls'], 2)
        self.assertEqual(costs['reserved_total_cny'], 20)
        self.assertEqual(costs['provider_reported_consumeMoney_sum'], '0.224')
        self.assertEqual(costs['provider_reported_compute_seconds'], '134')
        self.assertEqual(self.state, before)

    def test_cost_conflicting_task_alias_receipts_remain_unknown(self):
        self.state['files']['one']['execution_units'] = [
            dict(unit_id=name, provider_task_id='same-task', reserved_cny=10,
                 receipts=[dict(response=dict(actual_charge_cny=amount))])
            for name, amount in [('original', 2.5), ('alias', 3)]]
        costs = workflow.cost_summary(self.state)
        self.assertIsNone(costs['new_cost_cny'])
        self.assertEqual(costs['known_actual_receipts_cny'], 0)
        self.assertEqual(costs['unsettled_provider_units'], ['original', 'alias'])

    def review_fixture(self, output_hash=None):
        self.state['case_binding'] = workflow.case_binding(self.case)
        self.candidate['reviews'] = [{'kind': kind, 'status': 'pass',
            'output_sha256': self.candidate['output_sha256']} for kind in ('technical', 'audio_preservation')]
        path = workflow.state_path(self.case)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.state))
        review = {'expected_revision': 1, 'file_id': 'one', 'candidate_id': 'v1',
            'source_sha256': self.candidate['source_sha256'],
            'output_sha256': output_hash or self.candidate['output_sha256'],
            'dependency': self.candidate['dependency'], 'checks': [
                {'kind': kind, 'status': 'pass', 'scope': 'test fixture only'}
                for kind in ('target_coverage', 'visual', 'playback', 'delivery')]}
        (self.root/'qa.json').write_text(json.dumps(review))
        return review

    def test_agent_review_rejects_another_output(self):
        self.review_fixture('wrong-output')
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            workflow.record_agent_review(self.case, 'qa.json')
        self.assertEqual(project.read_json(workflow.state_path(self.case))['revision'], 1)

    def test_stale_technical_review_cannot_qualify_new_output(self):
        self.review_fixture()
        path = workflow.state_path(self.case)
        state = project.read_json(path)
        state['files']['one']['candidates'][0]['reviews'][0]['output_sha256'] = 'old-output'
        path.write_text(json.dumps(state))
        self.assertNotEqual(workflow.record_agent_review(self.case, 'qa.json')['status'], 'qualified')

    def test_required_audio_review_cannot_be_ignored(self):
        self.review_fixture()
        path = workflow.state_path(self.case)
        state = project.read_json(path)
        state['files']['one']['candidates'][0]['reviews'].append({
            'kind': 'audio_listening', 'status': 'pending',
            'output_sha256': self.candidate['output_sha256']})
        path.write_text(json.dumps(state))
        self.assertNotEqual(workflow.record_agent_review(self.case, 'qa.json')['status'], 'qualified')

    def test_delivery_rechecks_stale_qualified_status(self):
        self.review_fixture()
        path = workflow.state_path(self.case)
        state = project.read_json(path)
        entry = state['files']['one']
        entry['raw_returns'] = []
        entry['candidates'][0]['status'] = 'qualified'
        entry['candidates'][0]['reviews'][0]['output_sha256'] = 'old-output'
        path.write_text(json.dumps(state))
        self.assertNotEqual(workflow.delivery_report(self.case)['files'][0]['delivery_status'], 'qualified')

    def test_changed_review_evidence_invalidates_candidate(self):
        self.review_fixture()
        result = workflow.record_agent_review(self.case, 'qa.json')
        self.assertEqual(result['status'], 'qualified')
        self.assertFalse(result['accepted'])
        candidate = project.read_json(workflow.state_path(self.case))['files']['one']['candidates'][0]
        (self.root/'qa.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'dependency hash changed'):
            workflow.verify_candidate(self.case, 'one', candidate)

    def test_review_closes_verified_ancestor_repairs_but_not_other_branch(self):
        review = self.review_fixture()
        state = project.read_json(workflow.state_path(self.case))
        state['files']['one']['candidates'] += [dict(self.candidate, id='v0', parent_id='source'),
                                               dict(self.candidate, id='other', parent_id='source')]
        state['files']['one']['candidates'][0]['parent_id'] = 'v0'
        state['files']['one']['issues'] = [{'id': 'fixed', 'status': 'pending_recheck',
            'result_candidate_id': 'v0'}, {'id': 'unrelated', 'status': 'pending_recheck',
            'result_candidate_id': 'other'}]
        workflow.state_path(self.case).write_text(json.dumps(state))
        review['resolved_issue_ids'] = ['unrelated']
        (self.root/'qa.json').write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError, 'unrelated repair'):
            workflow.record_agent_review(self.case, 'qa.json')
        review['resolved_issue_ids'] = ['fixed']
        (self.root/'qa.json').write_text(json.dumps(review))
        workflow.record_agent_review(self.case, 'qa.json')
        issues = project.read_json(workflow.state_path(self.case))['files']['one']['issues']
        self.assertEqual([i['status'] for i in issues], ['closed', 'pending_recheck'])

    def test_interrupted_first_export_does_not_require_a_repair_issue(self):
        self.state['files']['one']['jobs'] = [{'id': 'first-export', 'status': 'running', 'owner_pid': 99999999}]
        path = workflow.state_path(self.case)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.state))
        with patch.object(workflow.os, 'kill', side_effect=ProcessLookupError):
            workflow.reconcile_interrupted(self.case)
        self.assertEqual(project.read_json(path)['files']['one']['jobs'][0]['status'], 'interrupted')

    def test_new_batch_initializes_without_any_historical_sample(self):
        metadata = {'one': {'width': 64, 'height': 48, 'audio_present': False,
                           'duration_header_seconds': 10}}
        with patch.object(project, 'validate_case', return_value=metadata), \
             patch.object(project, 'source_inventory', return_value={'frame_count': 300, 'fps': 30}):
            workflow.initialize(self.case, 'case.json')
        state = project.read_json(workflow.state_path(self.case))
        self.assertEqual(set(state['files']), {'one'})
        self.assertEqual(state['files']['one']['raw_returns'], [])
        self.assertIsNone(state['files']['one']['current_candidate_id'])
        self.assertEqual(state['files']['one']['source_frame_count'], 300)
        self.assertFalse(workflow.delivery_report(self.case)['original_goal_complete'])

    def test_resume_rejects_same_batch_with_changed_source_or_requirements(self):
        self.state['case_binding'] = workflow.case_binding(self.case)
        path = workflow.state_path(self.case)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.state))
        with patch.object(project, 'validate_case', return_value={}):
            self.case['files'][0]['operations'][0]['target'] = 'different target'
            with self.assertRaisesRegex(ValueError, 'binding'):
                workflow.initialize(self.case, 'case.json')
        self.assertEqual(project.read_json(path)['revision'], 1)

    def test_intake_derives_media_identity_instead_of_copying_sample_metadata(self):
        request = copy.deepcopy(self.case)
        request['files'][0].pop('source_sha256')
        request['files'][0].pop('fps_container_average')
        request['files'][0].pop('duration_seconds')
        (self.root/'request.json').write_text(json.dumps(request))
        meta = {'width':64,'height':48,'audio_present':False,'duration_header_seconds':10}
        with patch.object(project, 'probe', return_value=meta), \
             patch.object(project, 'source_inventory', return_value={'frame_count': 300, 'fps': 30}), \
             patch('source_timeline.build', return_value={'pts': list(range(300)), 'scope': 'unit fixture'}):
            result = project.intake('cases/new/case.json', 'request.json')
        policy = project.read_json(self.root/'cases/new/case.json')['execution_policy']
        self.assertEqual(policy['version'], 3)
        self.assertEqual(policy['mode'], 'production')
        self.assertNotIn('deadline', policy)
        self.assertNotIn('feasibility_seconds', policy)
        self.assertEqual(policy['max_failed_previews_per_method'], 2)
        self.assertEqual(policy['editing_strategy'], 'provider_first_v1')
        self.assertNotIn('deformable_matte', policy)
        item = project.read_json(self.root/'cases/new/case.json')['files'][0]
        self.assertEqual(item['source_sha256'], project.digest(self.root/'source.mp4'))
        self.assertEqual(item['dimensions'], [64,48])
        self.assertEqual(item['operation_ids'], ['remove'])
        self.assertEqual(result['status'], 'intake_prepared_not_localized')
        self.assertFalse((self.root/'runs/test/workflow.json').exists())
        with self.assertRaisesRegex(ValueError, 'exists'):
            project.intake('cases/new/case.json', 'request.json')

    def test_wrong_source_and_changed_candidate_cannot_reuse_review(self):
        bad = copy.deepcopy(self.candidate)
        bad['source_sha256'] = 'other'
        with self.assertRaisesRegex(ValueError, 'source'):
            workflow.verify_candidate(self.case, 'one', bad)
        (self.root/'out.mp4').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash'):
            workflow.verify_candidate(self.case, 'one', self.candidate)

    def test_requirement_and_asset_revision_invalidate(self):
        self.case['files'][0]['operations'][0]['protect'].append('necklace')
        with self.assertRaisesRegex(ValueError, 'requirements'):
            workflow.verify_candidate(self.case, 'one', self.candidate)

    def test_actual_asset_change_invalidates_unchanged_case(self):
        (self.root/'logo.png').write_bytes(b'logo1')
        self.case['files'][0]['operations'][0]['replacement_asset']='logo.png'
        self.candidate['dependency']=workflow.requirement_binding(self.case,self.case['files'][0])
        (self.root/'logo.png').write_bytes(b'logo2')
        with self.assertRaisesRegex(ValueError,'asset'):
            workflow.verify_candidate(self.case,'one',self.candidate)

    def test_concurrent_feedback_keeps_one_and_reports_conflict(self):
        path=workflow.state_path(self.case);path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.state))
        body={'file_id':'one','candidate_id':'v1','candidate_sha256':self.candidate['output_sha256'],
              'source_sha256':self.candidate['source_sha256'],'source_time':1,'text':'保留手指','expected_revision':1}
        barrier=threading.Barrier(2);results=[]
        def save():
            barrier.wait()
            try:workflow.add_feedback(self.case,body);results.append('saved')
            except ValueError as exc:results.append(str(exc))
        workers=[threading.Thread(target=save) for _ in range(2)]
        for worker in workers:worker.start()
        for worker in workers:worker.join()
        self.assertEqual(results.count('saved'),1)
        self.assertEqual(len(project.read_json(path)['files']['one']['issues']),1)
        self.assertTrue(any('conflict' in result.lower() for result in results))

    def test_absent_candidate_feedback_is_bound_to_actual_source(self):
        self.state['files']['one']['candidates']=[]
        body={'file_id':'one','candidate_id':None,'candidate_sha256':self.candidate['source_sha256'],
              'source_sha256':self.candidate['source_sha256'],'source_time':1,'text':'门框被重画','expected_revision':1,'category':'model_internal'}
        issue=workflow.bind_feedback(self.case,self.state,body)
        self.assertEqual(issue['observation_basis'],'source_without_candidate')
        body['source_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'hash mismatch'):workflow.bind_feedback(self.case,self.state,body)

    def test_stale_review_is_pending(self):
        self.assertEqual(workflow.effective_reviews(self.candidate)[0]['status'], 'stale')

    def test_feedback_uses_displayed_frame_not_nearest_next_frame(self):
        body={'file_id':'one','candidate_id':'v1','candidate_sha256':self.candidate['output_sha256'],
              'source_sha256':self.candidate['source_sha256'],'source_time':1.029,
              'text':'当前帧问题','expected_revision':1}
        self.assertEqual(workflow.bind_feedback(self.case,self.state,body)['source_frame'],30)

    def test_feedback_requires_candidate_and_source_time(self):
        for body in [{}, {'file_id': 'one', 'candidate_id': 'missing', 'source_time': 1}]:
            with self.assertRaises(ValueError):
                workflow.bind_feedback(self.case, self.state, body)

    def test_browser_events_are_isolated_by_batch_instead_of_the_old_sample(self):
        import review_server
        for batch in ['first-batch', 'second-batch']:
            case = copy.deepcopy(self.case)
            case['batch_id'] = batch
            review_server.record_browser_event(case, {'file_id':'one', 'event':'source_ended'})
        for batch in ['first-batch', 'second-batch']:
            lines = (self.root/'runs'/batch/'browser-events.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])['batch_id'], batch)
            self.assertEqual(json.loads(lines[0])['case_version'], self.case['case_version'])
        self.assertFalse((self.root/'evidence/workflow-implementation-20261112/browser-events.jsonl').exists())

    def test_multi_operation_feedback_cannot_silently_bind_first_operation(self):
        self.case['files'][0]['operations'].append({'id':'other','kind':'remove','target':'microphone','protect':['hands']})
        self.candidate['dependency'] = workflow.requirement_binding(self.case, self.case['files'][0])
        body={'file_id':'one','candidate_id':'v1','candidate_sha256':self.candidate['output_sha256'],
              'source_sha256':self.candidate['source_sha256'],'source_time':1,
              'text':'麦克风还有残留','expected_revision':1}
        with self.assertRaisesRegex(ValueError,'operation'):
            workflow.bind_feedback(self.case,self.state,body)
        body['operation_id']='other'
        self.assertEqual(workflow.bind_feedback(self.case,self.state,body)['operation_id'],'other')

    def test_feedback_conflict_is_not_last_writer_wins(self):
        body = {'file_id': 'one', 'candidate_id': 'v1', 'candidate_sha256': self.candidate['output_sha256'],
                'source_sha256': self.candidate['source_sha256'], 'source_time': 1.2,
                'text': '这里还有残留', 'expected_revision': 1, 'category': 'protection'}
        issue = workflow.bind_feedback(self.case, self.state, body)
        self.assertEqual(issue['source_time_seconds'], 1.2)
        self.state['revision'] += 1
        with self.assertRaisesRegex(ValueError, 'conflict'):
            workflow.bind_feedback(self.case, self.state, body)

    def test_local_repair_never_routes_to_cloud(self):
        plan = workflow.dependency_plan(self.case, self.candidate, [1, 2], 'protection')
        self.assertFalse(plan['remote_submission_allowed'])
        self.assertNotIn('cloud edit', ' '.join(plan['recompute']))
        blocked = workflow.dependency_plan(self.case, self.candidate, [1, 2], 'model_internal')
        self.assertEqual(blocked['status'], 'needs_manual_or_valid_cache')

    def test_arbitrary_parent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'parent'):
            workflow.resolve_parent(self.case, 'one', 'nonexistent')


if __name__ == '__main__':
    unittest.main()
