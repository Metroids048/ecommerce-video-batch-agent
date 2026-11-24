import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import vm from 'node:vm';
import {Buffer} from 'node:buffer';

// Execute the real bridge with only imports/network/I/O replaced by fixtures.
const source=(await fs.readFile(new URL('./runninghub_transport.mjs',import.meta.url),'utf8'))
  .replace(/^import .*;\n/gm,'');
async function run(action,route,reply,options={}) {
  let events=[];let output='',errors='',calls=0,payloads=[],credentialReads=0,policyChecked=false;
  const spec={journal_path:'fixture-journal',action,batch_id:'fixture',task_id:action==='submit'?null:'original',maximum_charge_cny:options.cap??21,
    api_origin:options.origin,usd_to_cny:options.fx,
    request:{route,reference_clip:'fixture.mp4',image_references:[],mask_video:route==='/task/openapi/ai-app/run'?'mask.mp4':undefined,parameters:route==='/task/openapi/create'?{workflowId:'1864499239568211970',instanceType:'default',workflow:{'1':{inputs:{video:'placeholder'}}}}:route==='/task/openapi/ai-app/run'?{webappId:'1963199782695178242',width:1280,height:720,instanceType:'default'}:{duration:'5'},
      prompt:'Erase only product lettering',source_time_seconds:[0,5]}};
  const context={Buffer,path,structuredClone,homedir:()=>'/fixture-user',randomUUID:()=> 'fixture',assertRemoteAllowed:async()=>{if(options.blocked)throw new Error('blocked');policyChecked=true;},
    fs:{appendFile:async(p,line)=>events.push(JSON.parse(line)),stat:async()=>({isFile:()=>true,mode:options.mode??0o600}),readFile:async p=>{if(p.endsWith('runninghub-api-key')){assert(policyChecked);credentialReads++;return 'test-key\n';}return Buffer.from('fixture');}},
    PrimePilot:class {async request(route,body){calls++;payloads.push({route,body});if(options.failAt && (options.failCreate?route===options.failAt:route.includes(options.failAt)))throw new Error('RunningHub HTTP 503 test-key https://example.invalid/?secret=yes');return route.includes('/media/upload/')?
      {code:0,data:{download_url:'https://example.invalid/input.mp4',file_name:'fixture.mp4'}}:
      route.includes('/price-preview/')?{estimatedPrice:1,currency:options.origin?'USD':'CNY'}:
      route==='/task/openapi/outputs'&&options.outputsReply?options.outputsReply:reply;}},
    process:{env:options.localCredential?{}:{RUNNINGHUB_API_KEY:'test-key'},stdin:(async function*(){yield JSON.stringify(spec);})(),
      stdout:{write:s=>output+=s},stderr:{write:s=>errors+=s}}};
  await vm.runInNewContext('(async()=>{'+source+'})()',context);
  if(options.blocked||options.mode===0o644){assert(errors);assert.equal(calls,0);assert.equal(credentialReads,0);return;}
  if(options.expectFailure)return {result:JSON.parse(output),calls,payloads,errors};
  assert.equal(errors,'',output);return {result:JSON.parse(output),calls,payloads,credentialReads,events};
}
for(const route of ['/openapi/v2/volc-subtitle-erase-pro/video',
  '/openapi/v2/kling-video-o3-std/video-edit',
  '/openapi/v2/alibaba/wan-3.0/reference-to-video']) {
  const {result,calls}=await run('submit',route,{taskId:'',status:'',errorCode:'1007',
    errorMessage:'fixture rejected test-key https://example.invalid/signed?token=fixture'});
  assert.equal(result.status,'submission_unknown');assert.equal(result.task_id,null);
  assert.equal(result.provider_error_code,'1007');assert.equal(calls,3);
  assert(!JSON.stringify(result).includes('test-key'));assert(!JSON.stringify(result).includes('token='));
  const good=await run('submit',route,{taskId:'new-task',status:'QUEUED',errorCode:''});
  assert(good.events.some(e=>e.stage==='create' && e.task_id==='new-task'));
  assert.equal(good.result.status,'QUEUED');assert.equal(good.result.task_id,'new-task');
  const actual=good.payloads.findLast(x=>x.route===route).body;
  assert.equal(good.result.prompt_transmitted,Object.hasOwn(actual,'prompt'));
  assert.equal(good.result.prompt_transmitted,!route.includes('volc-subtitle'));
}
const queried=await run('query','unused',{status:'',errorCode:'QUERY_ERROR',errorMessage:'fixture query failed'});
assert.equal(queried.result.status,'query_unknown');assert.equal(queried.result.task_id,'original');
assert.equal(queried.result.provider_error_code,'QUERY_ERROR');assert.equal(queried.calls,1);
for(const reply of [{code:0,data:{taskId:'mask-task',taskStatus:'QUEUED'}},{code:380,msg:'Missing workflow',data:null}]) {
 const tested=await run('submit','/task/openapi/ai-app/run',reply);assert.equal(tested.calls,3);assert.equal(tested.result.task_id,reply.data?.taskId||null);assert.equal(tested.result.status,reply.data?'QUEUED':'submission_unknown');
 assert.equal(tested.result.prompt_transmitted,false);
 assert(!JSON.stringify(tested.payloads.at(-1).body).includes('Erase only'));
}
const maskQuery=await run('query','/task/openapi/ai-app/run',{code:0,data:'RUNNING'});assert.equal(maskQuery.result.status,'RUNNING');
const usage=await run('query','/task/openapi/ai-app/run',{code:0,data:'SUCCESS'},
 {outputsReply:{code:0,data:[{nodeId:'68',fileType:'mp4',fileUrl:'https://example.invalid/signed?token=hidden',
 taskCostTime:'67',consumeMoney:'0.112',thirdPartyConsumeMoney:null,consumeCoins:'0',settleMode:'MONEY',chargedPoint:null}]}});
assert.equal(usage.result.output_receipt.outputs[0].settleMode,'MONEY');
assert.equal(usage.result.output_receipt.outputs[0].consumeMoney,'0.112');
assert(!JSON.stringify(usage.result).includes('signed'));
const stored=await run('query','unused',{status:'RUNNING'},{localCredential:true});assert.equal(stored.credentialReads,1);
await run('query','unused',{}, {localCredential:true,blocked:true});
await run('query','unused',{}, {localCredential:true,mode:0o644});
console.log('11 bridge receipt cases and 3 credential cases passed; no network calls');

const sam=await run('submit','/task/openapi/create',{code:0,data:{taskId:'sam-task',taskStatus:'RUNNING'}});
assert.equal(sam.result.task_id,'sam-task');
const sent=sam.payloads.at(-1).body;
assert.equal(sent.workflowId,'1864499239568211970');
assert.equal(JSON.parse(sent.workflow)['1'].inputs.video,'fixture.mp4');
assert(!JSON.stringify(sam.result).includes('test-key'));
const samStatus=await run('query','/task/openapi/create',{code:0,data:'RUNNING'});
assert.equal(samStatus.result.task_id,'original');
console.log('SAM2 full-workflow bridge and original-task query passed');

const budgetFailure=await run('submit','/openapi/v2/rhart-video-g-official/edit-video',{}, {cap:.5,expectFailure:true});
assert.equal(budgetFailure.result.failure_stage,'quote');
assert.equal(budgetFailure.result.create_attempted,false);
assert.equal(budgetFailure.result.status,'submission_not_created');
assert.equal(budgetFailure.calls,2);
const uploadFailure=await run('submit','/openapi/v2/rhart-video-g-official/edit-video',{}, {failAt:'/media/upload/',expectFailure:true});
assert.equal(uploadFailure.result.failure_stage,'upload');
assert.equal(uploadFailure.result.create_attempted,false);
assert(!JSON.stringify(uploadFailure.result).includes('test-key'));
assert(!JSON.stringify(uploadFailure.result).includes('secret=yes'));
const createFailure=await run('submit','/openapi/v2/kling-video-o3-std/video-edit',{}, {failAt:'/openapi/v2/kling-video-o3-std/video-edit',failCreate:true,expectFailure:true});
assert.equal(createFailure.result.create_attempted,true);
assert.equal(createFailure.result.status,'submission_unknown');
assert.equal(createFailure.result.failure_stage,'create');
const currentEditor=await run('submit','/openapi/v2/rhart-video-g-official/edit-video',{taskId:'editor-task',status:'RUNNING'}, {cap:1});
assert.equal(currentEditor.result.full_estimated_charge_cny,1);
assert.equal(currentEditor.result.task_id,'editor-task');
assert.equal(currentEditor.calls,3);
const preflight=await run('preflight','/openapi/v2/rhart-video-g-official/edit-video',{});
assert.equal(preflight.calls,2);
assert.equal(preflight.result.create_attempted,false);
const queryFailure=await run('query','unused',{}, {failAt:'/openapi/v2/query',expectFailure:true});
assert.equal(queryFailure.result.status,'query_unknown');
assert.equal(queryFailure.result.task_id,'original');
assert.equal(queryFailure.result.create_attempted,false);
console.log('Upload and quote failures remain distinguishable from unknown creates');
const rejected=await run('submit','/openapi/v2/rhart-video-g-official/edit-video',{errorCode:'40310',errorMessage:'CN model unavailable'});
assert.equal(rejected.result.status,'submission_rejected');assert.equal(rejected.result.provider_rejected,true);
const usd=await run('submit','/openapi/v2/rhart-video-g-official/edit-video',{taskId:'global-task',status:'RUNNING'},{origin:'https://www.runninghub.ai',fx:6.7152});
assert.equal(usd.result.full_estimated_charge_cny,6.7152);assert.equal(usd.result.quote_currency,'USD');
const settlement=await run('query','unused',{status:'SUCCESS',usage:{consumeMoney:'0.2',thirdPartyConsumeMoney:null}},{origin:'https://www.runninghub.ai',fx:6.7152});
assert.equal(settlement.result.actual_charge_cny,.2*6.7152);
console.log('Explicit CN rejection, global currency conversion and settlement fallback passed');
