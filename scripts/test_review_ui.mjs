// Unit regression of the real submit handler; this does not replace browser QA.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const html=readFileSync(new URL('../web/review.html',import.meta.url),'utf8');
const handler=html.slice(html.indexOf("$('feedback').onsubmit="),html.indexOf("$('planRepair').onclick="));
assert.ok(handler.startsWith("$('feedback').onsubmit="));
for(const [time,count,duration,expected] of [
  [45.043855,1348,45.04,44.9],
  [41.133333,1234,41.13,41.1],
  [1.029,1348,45.04,1.029],
]){
  const elements={source:{currentTime:time},feedback:{},text:{value:'尾帧问题'},operationChoice:{value:'remove'},category:{value:'protection'},feedbackReceipt:{classList:{add(){}}},issueChoice:{}};
  let sent;
  vm.runInNewContext(handler,{
    $:id=>elements[id],fileId:'source',candidate:{id:'v1',output_sha256:'out'},
    entry:{source_frame_count:count,input:{fps_container_average:30,duration_seconds:duration,source_sha256:'source'}},
    state:{revision:1},region:null,
    request:async(path,body)=>{sent=body;return {id:'issue',operation_id:body.operation_id,candidate_id:'v1',source_time_seconds:body.source_time}},
    feedbackTime:()=>Math.min(time,(count-1)/30),event(){},load:async()=>{elements.operationChoice.value='another-operation'},seek(){},
  });
  await elements.feedback.onsubmit({preventDefault(){}});
  assert.equal(sent.operation_id,'remove','The selected operation must reach the real request');
  assert.equal(elements.operationChoice.value,'remove','A refresh after saving must restore the saved operation');
  assert.match(elements.feedbackReceipt.textContent,/修改项remove/);
  assert.ok(sent.source_time<duration,'Tail feedback must remain inside the declared source timeline');
  assert.ok(Math.abs(sent.source_time-expected)<1e-9,`Unexpected feedback time ${sent.source_time}`);
}
console.log('PASS: real feedback handler binds both native endings to their displayed last video frame, preserving an ordinary timestamp.');

// New cases may have no generated return. The actual UI handler must still initialize.
const rawHandler=html.slice(html.indexOf('function selectRaw()'),html.indexOf('async function selectCandidate()'));
const rawElements={raw:{value:''},rawVideo:{removeAttribute(){},load(){}},rawNote:{}};
let synchronized=false;
vm.runInNewContext(rawHandler+';selectRaw();',{$:id=>rawElements[id],entry:{raw_returns:[]},raw:undefined,sync(){synchronized=true}});
assert.ok(synchronized);
assert.match(rawElements.rawNote.textContent,/尚无模型回片/);
new vm.Script(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
console.log('PASS: operation binding, empty-return initialization and complete UI script syntax.');

const ptsHandlers=html.slice(html.indexOf('function sourceFrameAt('),html.indexOf('function drawRanges('));
const ptsContext={entry:{presentation_timeline:{pts:[0,100,300,500],durations:[100,200,200,100],time_base:'1/1000'}},$:()=>({currentTime:.6})};vm.createContext(ptsContext);vm.runInContext(ptsHandlers,ptsContext);assert.equal(vm.runInContext('sourceFrameAt(.25)',ptsContext),1);assert.equal(vm.runInContext('feedbackTime()',ptsContext),.5);console.log('PASS: indexed source PTS drives overlay and tail feedback.');

ptsContext.entry.presentation_timeline={pts:Array.from({length:1348},(_,n)=>512*n),durations:Array(1348).fill(512),time_base:'1/15360'};for(let n=0;n<1348;n++){ptsContext.t=n/30;assert.equal(vm.runInContext('sourceFrameAt(t)',ptsContext),n)}ptsContext.t=1/30-1e-8;assert.equal(vm.runInContext('sourceFrameAt(t)',ptsContext),0);console.log('PASS: displayed PTS float roundtrip preserves all 1348 frame boundaries.');
