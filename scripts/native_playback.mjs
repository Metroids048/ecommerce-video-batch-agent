import fs from 'node:fs/promises';
const [url, prefix, port] = process.argv.slice(2);
const tabs = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const ws = new WebSocket(tabs.find(t => t.type === 'page').webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r, {once:true}));
let id=0; const pending=new Map();
ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){pending.get(m.id)?.(m);pending.delete(m.id);}});
const call=(method,params={})=>new Promise((resolve,reject)=>{const i=++id;pending.set(i,m=>m.error?reject(new Error(JSON.stringify(m.error))):resolve(m.result));ws.send(JSON.stringify({id:i,method,params}));});
const evaluate=async expression=>(await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true})).result.value;
await call('Page.enable');
await call('Emulation.setDeviceMetricsOverride',{width:1000,height:1400,deviceScaleFactor:1,mobile:false});
await call('Page.navigate',{url});
for(let n=0;n<50;n++) {if(await evaluate('!!document.querySelector("video")?.readyState'))break;await new Promise(r=>setTimeout(r,100));}
const begin=await evaluate('(async()=>{const v=document.querySelector("video");v.pause();v.currentTime=0;v.playbackRate=1;v.muted=true;await v.play();return {url:v.currentSrc,duration:v.duration,rate:v.playbackRate,startedAt:new Date().toISOString()};})()');
const observations=[];
for(let n=0;n<Math.ceil(begin.duration/3)+4;n++){
  const state=await evaluate('(()=>{const v=document.querySelector("video");return {time:v.currentTime,duration:v.duration,ended:v.ended,paused:v.paused,rate:v.playbackRate,width:v.videoWidth,height:v.videoHeight,frames:(()=>{const q=v.getVideoPlaybackQuality();return {total:q.totalVideoFrames,dropped:q.droppedVideoFrames};})(),at:new Date().toISOString()};})()');
  const {data}=await call('Page.captureScreenshot',{format:'png'});
  const path=prefix+'-'+n+'.png';await fs.writeFile(path,Buffer.from(data,'base64'));observations.push({...state,screenshot:path});
  if(state.ended)break;await new Promise(r=>setTimeout(r,3000));
}
const result={begin,observations,ended:observations.at(-1).ended,scope:'Actual Chrome native video at 1x, muted; screenshots require separate agent inspection. No semantic audio claim.'};
await fs.writeFile(prefix+'.json',JSON.stringify(result,null,2)+'\n');
ws.close();console.log(JSON.stringify({ended:result.ended,samples:observations.length,begin,last:observations.at(-1)}));
if(!result.ended)process.exitCode=1;
