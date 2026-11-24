import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {Buffer} from 'node:buffer';
import https from 'node:https';
import {createWriteStream} from 'node:fs';
import {pipeline} from 'node:stream/promises';

const ROUTE = '/openapi/v2/alibaba/wan-3.0-prime/reference-to-video';
export const digest = value => createHash('sha256').update(value).digest('hex');
export function redact(value) {
  if (Array.isArray(value)) return value.map(redact);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k,v]) => [k,redact(v)]));
  if (typeof value === 'string' && /^https?:\/\//.test(value)) {
    const u = new URL(value); return u.origin + u.pathname + (u.search ? '?[REDACTED]' : '');
  }
  return value;
}

// Manual three-pilot runner. This is not the production natural-language Agent.
// Keep the supplied credential in this process; persist intent before any paid call.
export class PrimePilot {
  #key;
  constructor({key, directory, fetchImpl = fetch, batchId, apiOrigin='https://www.runninghub.cn'}) {
    if (!key) throw new Error('Missing runtime credential');
    if(!['https://www.runninghub.cn','https://www.runninghub.ai'].includes(apiOrigin))throw new Error('Unsupported RunningHub origin');
    this.apiOrigin=apiOrigin;
    this.#key = key; this.directory = directory; this.fetch = fetchImpl; this.batchId = batchId;
    this.state = {model: ROUTE, maximum_calls:3, ceiling_cny:30, uploads:[], quotes:{}, attempts:[]};
  }
  async init() {
    await fs.mkdir(this.directory,{recursive:true});
    try { this.state = JSON.parse(await fs.readFile(path.join(this.directory,'cloud-run.json'),'utf8')); }
    catch(e) { if(e.code !== 'ENOENT') throw e; }
    return this;
  }
  async save() {
    const target=path.join(this.directory,'cloud-run.json');
    await fs.writeFile(target+'.tmp',JSON.stringify(this.state,null,2)+'\n');
    await fs.rename(target+'.tmp',target);
  }
  async assertSubmissionAllowed() {
    const policy=await remotePolicy(this.batchId);
    if(policy.paid_submission_allowed!==true)
      throw new Error('Paid submission blocked: '+(policy.paid_submission_block_reason??'no active authorization'));
  }
  async request(route, body, method='POST', contentType) {
    await assertRemoteAllowed(this.batchId);
    if(route.startsWith('/openapi/v2/alibaba/')) await this.assertSubmissionAllowed();
    const policy=await remotePolicy(this.batchId);
    if(route.startsWith('/openapi/v2/media/upload/')) {
      if(policy.media_upload_allowed!==true) throw new Error('Upload blocked: no active authorization');
    } else if(route!=='/openapi/v2/query' && !route.startsWith('/openapi/v2/price-preview/') &&
        !(method==='GET' && /^\/api\/webapp\/apiCallDemo\?webappId=\d+$/.test(route)) &&
        !(method==='POST' && route==='/api/openapi/getJsonApiFormat')) {
      if(policy.remote_submission_allowed!==true) throw new Error('Remote submission blocked: no active authorization');
    }
    const headers={Authorization:`Bearer ${this.#key}`};
    if (body!==undefined) headers['Content-Type']=contentType??'application/json';
    // Official app metadata requires the query key as well as Bearer auth.
    // Keep the expanded URL transient; never return/log its curl example.
    const requestRoute=route.startsWith('/api/webapp/apiCallDemo?')?
      route+'&apiKey='+encodeURIComponent(this.#key):route;
    const response=await this.fetch(this.apiOrigin+requestRoute,{
      method,headers,body:body===undefined?undefined:contentType?body:JSON.stringify(body),
      signal:AbortSignal.timeout(120000)
    });
    if(!response.ok) throw new Error(`RunningHub HTTP ${response.status}`);
    return await response.json();
  }
  async upload(file, binding) {
    const bytes=await fs.readFile(file); const started=new Date().toISOString();
    const boundary='----PrimePilot'+randomUUID();
    const filename=path.basename(file); const mime=filename.endsWith('.png')?'image/png':'video/mp4';
    const form=Buffer.concat([Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="${filename}"\r\nContent-Type: ${mime}\r\n\r\n`),bytes,Buffer.from(`\r\n--${boundary}--\r\n`)]);
    const reply=await this.request('/openapi/v2/media/upload/binary',form,'POST',`multipart/form-data; boundary=${boundary}`);
    this.state.uploads.push({binding,sha256:digest(bytes),bytes:bytes.length,started,finished:new Date().toISOString(),response:redact(reply)});
    await this.save();
    if(reply.code!==0 || !reply.data?.download_url) throw new Error('Upload rejected; inspect receipt');
    return reply.data.download_url;
  }
  async quote(pilot, body, binding) {
    if(body.duration!=='5' || body.resolution!=='720P' || body.audio!==false || body.promptExtend!==false || body.videoUrls?.length!==1)
      throw new Error('Unexpected pilot request');
    const started=new Date().toISOString();
    const reply=await this.request(ROUTE.replace('/openapi/v2/','/openapi/v2/price-preview/'),body);
    this.state.quotes[pilot]={binding,body:redact(body),request_sha256:digest(JSON.stringify(body)),started,finished:new Date().toISOString(),response:reply};
    await this.save(); return reply;
  }
  async submit(pilot, body) {
    await this.assertSubmissionAllowed();
    if(this.state.attempts.some(a=>a.pilot===pilot)) throw new Error('Pilot already attempted; query the original task, never retry blindly');
    if(this.state.attempts.length>=3) throw new Error('Three-call limit reached');
    const quotes=Object.values(this.state.quotes);
    const quote=this.state.quotes[pilot];
    if(!quote || quote.request_sha256!==digest(JSON.stringify(body))) throw new Error('Exact request has not been quoted');
    if(quotes.length!==3 || quotes.some(q=>q.response.errorCode || q.response.currency!=='CNY' || !Number.isFinite(Number(q.response.estimatedPrice))) || quotes.reduce((s,q)=>s+Number(q.response.estimatedPrice),0)>30)
      throw new Error('Current three-pilot quotes exceed authorization or are incomplete');
    const attempt={pilot,binding:quote.binding,request_sha256:quote.request_sha256,submitted_at:new Date().toISOString(),state:'submission_unknown',task_id:null};
    this.state.attempts.push(attempt); await this.save();
    const reply=await this.request(ROUTE,body);
    attempt.create_response=redact(reply); attempt.create_returned_at=new Date().toISOString();
    attempt.task_id=reply.taskId||null; attempt.state=attempt.task_id?(reply.status||'submission_unknown'):'submission_unknown';
    await this.save();
    if(!attempt.task_id) throw new Error('No task ID; submission remains unknown, do not retry');
    return reply;
  }
  async query(pilot) {
    const attempt=this.state.attempts.find(a=>a.pilot===pilot);
    if(!attempt?.task_id) throw new Error('No original task ID to query');
    const reply=await this.request('/openapi/v2/query',{taskId:String(attempt.task_id)});
    attempt.state=reply.status||'query_unknown'; attempt.last_queried_at=new Date().toISOString(); attempt.last_response=redact(reply);
    attempt.observations??=[];
    attempt.observations.push({at:attempt.last_queried_at,status:reply.status,usage:reply.usage??null});
    await this.save(); return reply;
  }
}

// Explicit HTTPS connection bound to the observed Wi-Fi IPv4 address.
// No proxy lookup, protocol downgrade, automatic retry, or persisted signed URL.
export async function downloadResult(url, destination, localAddress, batchId) {
  await assertRemoteAllowed(batchId);
  const receiptPath=destination+'.receipt.json'; const partial=destination+'.part';
  try {
    const receipt=JSON.parse(await fs.readFile(receiptPath,'utf8'));
    const bytes=await fs.readFile(destination);
    if(bytes.length===receipt.bytes && digest(bytes)===receipt.sha256) return receipt;
    throw new Error('Existing complete result does not match receipt');
  } catch(e) { if(e.code!=='ENOENT') throw e; }
  await fs.mkdir(path.dirname(destination),{recursive:true});
  let offset=0, prior;
  try {offset=(await fs.stat(partial)).size;prior=JSON.parse(await fs.readFile(partial+'.json','utf8'));}
  catch(e){if(e.code!=='ENOENT')throw e;if(offset)throw new Error('Partial result lacks its receipt');}
  const started=new Date().toISOString();
  async function get(address, redirects=0) {
    const u=new URL(address);if(u.protocol!=='https:')throw new Error('Only HTTPS result downloads allowed');
    const headers={};if(offset){headers.Range=`bytes=${offset}-`;if(prior.etag)headers['If-Range']=prior.etag;}
    const response=await new Promise((resolve,reject)=>{
      const req=https.get(u,{localAddress,headers,timeout:60000},resolve);
      req.on('timeout',()=>req.destroy(new Error('Direct download timeout')));
      req.on('error',e=>reject(new Error('Direct download failed: '+(e.code??'network_error'))));
    });
    if([301,302,303,307,308].includes(response.statusCode)){
      response.resume();if(redirects>=4)throw new Error('Too many result redirects');
      return get(new URL(response.headers.location,u).href,redirects+1);
    }
    return response;
  }
  const response=await get(url);
  if(response.statusCode!==(offset?206:200)){response.resume();throw new Error('Unexpected direct-download HTTP status '+response.statusCode);}
  const length=Number(response.headers['content-length']);
  if(!Number.isSafeInteger(length)||length<=0){response.destroy();throw new Error('Missing result Content-Length');}
  if(offset && !String(response.headers['content-range']).startsWith(`bytes ${offset}-`)){response.destroy();throw new Error('Invalid resume range');}
  const total=offset+length;
  if(prior?.bytes && prior.bytes!==total){response.destroy();throw new Error('Resumed result changed size');}
  await fs.writeFile(partial+'.json',JSON.stringify({bytes:total,etag:response.headers.etag??null}));
  await pipeline(response,createWriteStream(partial,{flags:offset?'a':'w'}));
  const bytes=await fs.readFile(partial);if(bytes.length!==total)throw new Error('Incomplete direct download; retain partial');
  const receipt={started,finished:new Date().toISOString(),bytes:total,sha256:digest(bytes),etag:response.headers.etag??null,content_type:response.headers['content-type'],local_address:localAddress,proxy:false,official_sha256_available:false,url:redact(url)};
  await fs.rename(partial,destination);await fs.writeFile(receiptPath,JSON.stringify(receipt,null,2)+'\n');await fs.rm(partial+'.json');
  return receipt;
}

export async function remotePolicy(batchId) {
  if(batchId) {
    const retired=new URL('../runs/'+encodeURIComponent(batchId)+'/retired-after-cleanup.json',import.meta.url);
    // Check before reading policy or credentials, including direct pilot calls.
    try {await fs.stat(retired);throw new Error('Historical batch retired after user cleanup; do not resubmit old tasks');}
    catch(e) {if(e.code!=='ENOENT')throw e;}
  }
  const policy=JSON.parse(await fs.readFile(new URL('../config/models.json',import.meta.url),'utf8'));
  const grant=policy.scoped_remote_batches?.[batchId];
  return grant ? {...policy,...grant} : policy;
}

export async function assertRemoteAllowed(batchId) {
  const policy=await remotePolicy(batchId);
  if(policy.remote_calls_allowed!==true) throw new Error('All remote calls blocked by current policy (including query/quote/download)');
}
