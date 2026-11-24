// Thin protocol bridge. Uses the historical pilot's HTTP and direct-download
// implementation, never its fixed three-call/five-second budget methods.
import fs from 'node:fs/promises';
import {PrimePilot, assertRemoteAllowed, downloadResult} from './runninghub_pilot.mjs';
import {Buffer} from 'node:buffer';
import {randomUUID} from 'node:crypto';
import path from 'node:path';
import {homedir} from 'node:os';

let stage='policy', action, originalTaskId, createAttempted=false, runtimeKey='', journalPath, lastReceipt={};
const clean=value=>String(value).split(runtimeKey||'\0').join('[REDACTED]')
  .replace(/https?:\/\/\S+/g,'[URL]').slice(0,800);
const record=async event=>{
  if(journalPath)await fs.appendFile(journalPath,JSON.stringify({at:new Date().toISOString(),...event})+'\n',{mode:0o600});
};
try {
  let input='';for await (const chunk of process.stdin) input+=chunk;
  const spec=JSON.parse(input);
  action=spec.action; originalTaskId=spec.task_id;
  journalPath=spec.journal_path;
  await assertRemoteAllowed(spec.batch_id); // Must precede credential/env access.
  let key=process.env.RUNNINGHUB_API_KEY;
  if(!key) {
    const credentialPath=path.join(homedir(),'.config/aiw/credentials/runninghub-api-key');
    const metadata=await fs.stat(credentialPath);
    if(!metadata.isFile()||(metadata.mode&0o077))throw new Error('Credential file must be owner-only');
    key=(await fs.readFile(credentialPath,'utf8')).trim();
  }
  runtimeKey=key;
  const apiOrigin=spec.api_origin||'https://www.runninghub.cn';
  const conversion=apiOrigin==='https://www.runninghub.ai'?Number(spec.usd_to_cny):1;
  if(!Number.isFinite(conversion)||conversion<=0)throw new Error('Verified currency conversion required');
  const client=new PrimePilot({key,directory:'.',batchId:spec.batch_id,apiOrigin});
  const receipt=reply=>({
    provider_error_code:reply.errorCode??reply.code??null,
    provider_error_message:String(reply.errorMessage??reply.message??reply.msg??'')
      .replaceAll(key,'[REDACTED]').replace(/https?:\/\/\S+/g,'[URL]').slice(0,800),
    usage:reply.usage??null
  });
  const request=client.request.bind(client);
  client.request=async (route,...args)=>{
    stage=route.includes('/media/upload/')?'upload':route.includes('/price-preview/')?'quote':
      spec.action==='submit'?'create':spec.action;
    await record({stage,event:'started'});
    if(stage==='create')createAttempted=true;
    const reply=await request(route,...args);
    lastReceipt={...receipt(reply),task_id:reply.taskId||reply.data?.taskId||null,
      provider_code:reply.code??null,quote_cny:reply.estimatedPrice??null,currency:reply.currency??null,
      price_text:clean(reply.priceText??''),upload_url_present:!!reply.data?.download_url};
    await record({stage,event:'returned',...lastReceipt});
    return reply;
  };
  let result;
  if(spec.action==='settlement-query') {
    if(!/^\d+$/.test(String(spec.task_id)))throw new Error('Invalid task id');
    const reply=await client.request('/openapi/v2/query',{taskId:String(spec.task_id)});
    result={...receipt(reply),status:reply.status||'query_unknown',task_id:String(spec.task_id),
      task_usage_list:reply.taskUsageList?.map(x=>({task_id:String(x.taskId),status:x.taskStatus,usage:x.usage})),
      result_summary:reply.results?.map(x=>({keys:Object.keys(x),type:x.type,url_present:!!x.url})),response_keys:Object.keys(reply)};
  } else if(spec.action==='workflow-metadata') {
    if(!/^\d+$/.test(spec.workflow_id))throw new Error('Invalid workflow id');
    const reply=await client.request('/api/openapi/getJsonApiFormat',{apiKey:key,workflowId:spec.workflow_id});
    result={...receipt(reply),prompt:reply.data?.prompt?JSON.parse(reply.data.prompt):null};
  } else if(spec.action==='app-metadata') {
    if(!/^\d+$/.test(spec.webapp_id))throw new Error('Invalid app id');
    const reply=await client.request('/api/webapp/apiCallDemo?webappId='+spec.webapp_id,undefined,'GET');
    result={...receipt(reply),webapp_name:reply.data?.webappName,
      nodes:reply.data?.nodeInfoList}; // Omit curl example: it may echo the credential.
  } else if(['submit','preflight'].includes(spec.action)) {
    const upload=async file=>{
      const boundary='----Batch'+randomUUID();
      const bytes=await fs.readFile(file);const filename=path.basename(file);
      const mime=filename.endsWith('.png')?'image/png':'video/mp4';
      const form=Buffer.concat([Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="${filename}"\r\nContent-Type: ${mime}\r\n\r\n`),bytes,Buffer.from(`\r\n--${boundary}--\r\n`)]);
      const reply=await client.request('/openapi/v2/media/upload/binary',form,'POST',`multipart/form-data; boundary=${boundary}`);
      if(reply.code!==0||!reply.data?.download_url)throw new Error('Upload rejected');
      return reply.data; // Transient only; never emitted or saved.
    };
    if(spec.action==='preflight' && spec.request.route!=='/openapi/v2/rhart-video-g-official/edit-video')throw new Error('Preflight supports the current video editor only');
    const segmentation=spec.request.route==='/task/openapi/create';
    const maskApp=spec.request.route==='/task/openapi/ai-app/run';
    const textErase=spec.request.route==='/openapi/v2/volc-subtitle-erase-pro/video';
    const klingEdit=spec.request.route==='/openapi/v2/kling-video-o3-std/video-edit';
    const videoEdit=klingEdit||spec.request.route==='/openapi/v2/rhart-video-g-official/edit-video';
    const uploadedClip=await upload(spec.request.reference_clip);
    if(segmentation) {
      const p=spec.request.parameters;
      if(!uploadedClip.file_name&&!uploadedClip.fileName)throw new Error('Missing workflow video filename');
      if(spec.maximum_charge_cny<10)throw new Error('Missing bounded compute reservation');
      const graph=structuredClone(p.workflow);
      graph['1'].inputs.video=uploadedClip.file_name||uploadedClip.fileName;
      if(spec.request.seed_atlas) {
        const seed=await upload(spec.request.seed_atlas);
        if(!seed.file_name&&!seed.fileName)throw new Error('Missing dense seed filename');
        graph['2'].inputs.image=seed.file_name||seed.fileName;
      }
      if(spec.request.mask_video) {
        const uploadedMask=await upload(spec.request.mask_video);
        if(!uploadedMask.file_name&&!uploadedMask.fileName)throw new Error('Missing workflow mask filename');
        graph['2'].inputs.video=uploadedMask.file_name||uploadedMask.fileName;
      }
      const reply=await client.request(spec.request.route,{apiKey:key,workflowId:p.workflowId,
        workflow:JSON.stringify(graph),nodeInfoList:[],instanceType:p.instanceType});
      result={...receipt(reply),status:reply.data?.taskId?(reply.data.taskStatus||'submission_unknown'):'submission_unknown',
        task_id:reply.data?.taskId||null,prompt_transmitted:false,
        model_input_fields:spec.request.seed_atlas?['video','initial_mask','source_frame_selection','video_tracking']:spec.request.mask_video?['video','mask_video','local_pretrained_matting']:p.text_prompt?['video','text_prompt','video_tracking']:['video','positive_points','negative_points','video_tracking']};
    } else if(maskApp) {
      const uploadedMask=await upload(spec.request.mask_video);
      const fileName=data=>data.file_name||data.fileName;
      if(!fileName(uploadedClip)||!fileName(uploadedMask)||spec.maximum_charge_cny<10)throw new Error('Mask upload or compute reservation missing');
      const p=spec.request.parameters;
      const reply=await client.request(spec.request.route,{apiKey:key,webappId:p.webappId,instanceType:p.instanceType,
        nodeInfoList:[{nodeId:'23',fieldName:'video',fieldValue:fileName(uploadedClip)},
          {nodeId:'52',fieldName:'video',fieldValue:fileName(uploadedMask)},
          {nodeId:'69',fieldName:'value',fieldValue:String(p.width)},
          {nodeId:'70',fieldName:'value',fieldValue:String(p.height)}]});
      result={...receipt(reply),status:reply.data?.taskId?(reply.data.taskStatus||'submission_unknown'):'submission_unknown',task_id:reply.data?.taskId||null,
        compute_cny_per_hour:4,cancel_after_wall_seconds:900,
        prompt_transmitted:false,model_input_fields:['video','mask_video','width','height']};
    } else {
    const clipUrl=uploadedClip.download_url;
    const body=textErase?{...spec.request.parameters,videoUrl:clipUrl}:
      {...spec.request.parameters,prompt:spec.request.prompt,...(videoEdit?{videoUrl:clipUrl}:{videoUrls:[clipUrl]})};
    if(textErase)delete body.prompt;
    if(spec.request.image_references.length)body.imageUrls=await Promise.all(spec.request.image_references.map(async p=>(await upload(p)).download_url));
    const quote=await client.request(spec.request.route.replace('/openapi/v2/','/openapi/v2/price-preview/'),body);
    const price=Number(quote.estimatedPrice);
    // Official reference-video billing adds input seconds at the same tariff.
    // Auto duration needs a separate bounded cost policy before it can submit.
    const outputSeconds=Number(body.duration);
    const inputSeconds=spec.request.reference_duration_seconds??(spec.request.source_time_seconds[1]-spec.request.source_time_seconds[0]);
    // price-preview documents the model price for the identical call parameters.
    // RHX is an edit endpoint, not Wan reference-to-video: do not multiply its
    // returned call estimate by duration or charge input and output twice.
    const fullPrice=(klingEdit?2*Math.max(price,.81*Math.ceil(inputSeconds)):
      textErase?Math.max(price,4*inputSeconds/60):videoEdit?price:price*(1+Math.min(inputSeconds,15)/outputSeconds))*conversion;
    lastReceipt={...lastReceipt,full_estimated_charge_cny:fullPrice,reservation_basis:videoEdit&&!klingEdit?'official_identical_parameters_price_preview_estimate_not_settlement':'existing_route_policy'};
    await record({stage,event:'budget_checked',...lastReceipt,maximum_charge_cny:spec.maximum_charge_cny});
    if(quote.errorCode||quote.currency!==(apiOrigin==='https://www.runninghub.ai'?'USD':'CNY')||!Number.isFinite(fullPrice)||!(fullPrice>0)||(!textErase&&!videoEdit&&!(outputSeconds>=2))||fullPrice>spec.maximum_charge_cny)throw new Error('Full input/output cost outside reserved budget');
    if(spec.action==='preflight') {
      result={...lastReceipt,status:'preflight_ready',create_attempted:false,full_estimated_charge_cny:fullPrice};
    } else {
    const reply=await client.request(spec.request.route,body);
    const rejected=!reply.taskId&&String(reply.errorCode)==='40310';
    result={...receipt(reply),api_origin:apiOrigin,provider_rejected:rejected,status:reply.taskId?(reply.status||'submission_unknown'):rejected?'submission_rejected':'submission_unknown',task_id:reply.taskId||null,quote_currency:quote.currency,quote_amount:price,quote_cny:price*conversion,full_estimated_charge_cny:fullPrice,
      prompt_transmitted:Object.hasOwn(body,'prompt'),model_input_fields:Object.keys(body),create_attempted:true};
    }
    }
  } else if(['/task/openapi/create','/task/openapi/ai-app/run'].includes(spec.request.route)) {
    const reply=await client.request('/task/openapi/status',{apiKey:key,taskId:String(spec.task_id)});
    result={...receipt(reply),status:reply.code===0&&typeof reply.data==='string'?reply.data:'query_unknown',task_id:spec.task_id};
    if(result.status==='SUCCESS'||result.status==='FAILED') {
      const outputs=await client.request('/task/openapi/outputs',{apiKey:key,taskId:String(spec.task_id)});
      result.output_receipt={...receipt(outputs),outputs:Array.isArray(outputs.data)?outputs.data.map(x=>({nodeId:x.nodeId,fileType:x.fileType,taskCostTime:x.taskCostTime,consumeMoney:x.consumeMoney,
        thirdPartyConsumeMoney:x.thirdPartyConsumeMoney,consumeCoins:x.consumeCoins,settleMode:x.settleMode,chargedPoint:x.chargedPoint})):null,
        failed_reason:outputs.data?.failedReason?JSON.stringify(outputs.data.failedReason).replaceAll(key,'[REDACTED]').replace(/https?:\/\/\S+/g,'[URL]').slice(0,1600):null};
      if(spec.action==='download') {
        if(result.status!=='SUCCESS')throw new Error('Original mask task failed');
        const videos=outputs.data.filter(x=>x.fileType==='mp4'||/\.mp4(?:\?|$)/.test(x.fileUrl||''));
        if(videos.length!==1)throw new Error('Inspect ambiguous workflow outputs before download');
        const address=process.env.RUNNINGHUB_DIRECT_LOCAL_ADDRESS;
        if(!address)throw new Error('Verified direct Wi-Fi interface address is required');
        stage='download';
        const saved=await downloadResult(videos[0].fileUrl,spec.output,address,spec.batch_id);
        result={...result,status:'raw_ready',raw_path:spec.output,raw_sha256:saved.sha256};
      }
    }
  } else {
    const reply=await client.request('/openapi/v2/query',{taskId:String(spec.task_id)});
    const money=reply.usage?.thirdPartyConsumeMoney??reply.usage?.consumeMoney;
    result={...receipt(reply),api_origin:apiOrigin,status:reply.status||'query_unknown',task_id:spec.task_id,actual_charge_cny:money==null?null:Number(money)*conversion};
    if(spec.action==='download') {
      if(reply.status!=='SUCCESS')throw new Error('Original task is not ready');
      const url=reply.results?.[0]?.url;
      if(!url)throw new Error('Missing result URL; current contract needs verification');
      const address=process.env.RUNNINGHUB_DIRECT_LOCAL_ADDRESS;
      if(!address)throw new Error('Verified direct Wi-Fi interface address is required');
      stage='download';
      const receipt=await downloadResult(url,spec.output,address,spec.batch_id);
      result={...result,status:'raw_ready',raw_path:spec.output,raw_sha256:receipt.sha256};
    }
  }
  process.stdout.write(JSON.stringify(result));
} catch(error) {
  const submission=['submit','preflight'].includes(action);
  const failure={...lastReceipt,...(!submission&&originalTaskId?{task_id:String(originalTaskId)}:{}),
    status:submission?(createAttempted?'submission_unknown':'submission_not_created'):'query_unknown',
    failure_stage:stage,create_attempted:createAttempted,error:clean(error.message)};
  try {await record({stage,event:'failed',...failure});} catch {}
  process.stdout.write(JSON.stringify(failure));
  process.stderr.write('RunningHub '+stage+' failed; create_attempted='+createAttempted+'; inspect saved receipt.\n');
  process.exitCode=2;
}
