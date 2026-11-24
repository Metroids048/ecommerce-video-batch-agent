import assert from 'node:assert/strict';
import {PrimePilot,digest} from './runninghub_pilot.mjs';

// Reproduce the actual price-discrepancy stop policy without making network calls.
export async function testSubmissionGate() {
  let calls=0;
  const client=new PrimePilot({key:'test-non-secret',directory:'unused',fetchImpl:async()=>{
    calls++; return {ok:true,json:async()=>({status:'read_only_fixture'})};
  }});
  for(const route of ['/openapi/v2/alibaba/wan-3.0/reference-to-video',
      '/openapi/v2/query','/openapi/v2/price-preview/alibaba/wan-3.0/reference-to-video',
      '/openapi/v2/media/upload/binary','/openapi/v2/free-model/generate']) {
    await assert.rejects(()=>client.request(route,{}),/All remote calls blocked/);
  }
  await assert.rejects(()=>client.submit('new-attempt',{}),/Paid submission blocked/);
  const {downloadResult}=await import('./runninghub_pilot.mjs');
  await assert.rejects(()=>downloadResult('https://example.invalid/raw','unused','127.0.0.1'),/All remote calls blocked/);
  assert.equal(calls,0,'No remote route may reach transport under the current ban');
  assert.equal(client.state.attempts.length,0);
  return {passed:true,network_calls:0};
}

// Reproduce the legacy runner's missing-ID path without network or disk writes.
export async function testUnknownReceipt() {
  let calls=0, saves=0;
  const client=new PrimePilot({key:'test-non-secret',directory:'unused'});
  client.assertSubmissionAllowed=async()=>{};
  client.save=async()=>{saves++;};
  client.request=async()=>{calls++;return {taskId:'',status:'',errorCode:'1007',errorMessage:'fixture invalid input'};};
  const body={fixture:true};
  for(const id of ['one','two','three']) client.state.quotes[id]={
    request_sha256:digest(JSON.stringify(body)),binding:{fixture:true},response:{estimatedPrice:1,currency:'CNY'}};
  await assert.rejects(()=>client.submit('one',body),/No task ID/);
  const attempt=client.state.attempts[0];
  assert.equal(attempt.state,'submission_unknown');
  assert.equal(attempt.task_id,null);
  assert.equal(attempt.create_response.errorCode,'1007');
  await assert.rejects(()=>client.submit('one',body),/already attempted/);
  assert.equal(calls,1);
  attempt.task_id='original';
  await client.query('one');
  assert.equal(attempt.state,'query_unknown');
  assert.equal(attempt.task_id,'original');
  assert.equal(saves,3);
  return {passed:true,network_calls:0};
}
