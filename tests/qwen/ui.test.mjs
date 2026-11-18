import {test} from 'node:test';import assert from 'node:assert/strict';import {JSDOM} from 'jsdom';import {readFile} from 'node:fs/promises';
const html=await readFile(new URL('../../docs/qwen/index.html',import.meta.url),'utf8');let run=0;
async function setup(){const dom=new JSDOM(html,{url:'https://example.test/qwen/'});class FakeWorker{static all=[];constructor(){this.handlers={};this.sent=[];FakeWorker.all.push(this);}addEventListener(k,f){(this.handlers[k]??=[]).push(f);}postMessage(m){this.sent.push(m);}terminate(){this.terminated=true;}emit(data){for(const f of this.handlers.message||[])f({data});}}
let now=0,timerId=0;const timers=new Map();
dom.window.setTimeout=(callback,delay)=>{const id=++timerId;timers.set(id,{callback,at:now+delay});return id;};
dom.window.clearTimeout=id=>timers.delete(id);
function advance(ms){const target=now+ms;while(true){const next=[...timers].filter(([,v])=>v.at<=target).sort((a,b)=>a[1].at-b[1].at)[0];if(!next)break;now=next[1].at;timers.delete(next[0]);next[1].callback();}now=target;}
globalThis.document=dom.window.document;globalThis.Worker=FakeWorker;
await import(`../../client/qwen/app.mjs?test=${run++}`);const $=id=>dom.window.document.getElementById(id);return{dom,$,FakeWorker,advance,submit(q){$('msg').value=q;$('form').dispatchEvent(new dom.window.Event('submit',{cancelable:true}));}};}
test('model is opt-in, streams safely, and retains actual conversation',async()=>{const x=await setup();assert.equal(x.FakeWorker.all.length,0);assert.equal(x.$('send').disabled,true);x.$('load').click();const w=x.FakeWorker.all[0];w.emit({type:'ready',id:1});x.submit('孔子と孟子の違い');assert.equal(w.sent.at(-1).question,'孔子と孟子の違い');w.emit({type:'token',id:1,text:'stale'});assert.doesNotMatch(x.$('log').textContent,/stale/);w.emit({type:'token',id:2,text:'<img src=x>説明'});assert.equal(x.$('log').querySelectorAll('img').length,0);w.emit({type:'done',id:2,text:'孔子と孟子の説明'});x.submit('短くして');assert.deepEqual(w.sent.at(-1).history.map(m=>m.role),['user','assistant']);w.emit({type:'done',id:3,text:'短い説明'});x.$('clear').click();x.submit('こんにちは');assert.deepEqual(w.sent.at(-1).history,[]);});
test('loading can be cancelled and stale worker cannot re-enable sending',async()=>{const x=await setup();x.$('load').click();const old=x.FakeWorker.all[0];x.$('stop').click();assert.equal(old.terminated,true);old.emit({type:'ready',id:1});assert.equal(x.$('send').disabled,true);x.$('load').click();const current=x.FakeWorker.all[1];old.emit({type:'ready',id:1});assert.equal(x.$('send').disabled,true);current.emit({type:'ready',id:3});assert.equal(x.$('send').disabled,false);});
test('generation errors restore question and allow retry',async()=>{const x=await setup();x.$('load').click();const w=x.FakeWorker.all[0];w.emit({type:'ready',id:1});x.submit('長い質問');w.emit({type:'error',id:2,stage:'generate',message:'短くしてください',detail:'input too long'});assert.equal(x.$('msg').value,'長い質問');assert.equal(x.$('send').disabled,false);assert.match(x.$('log').textContent,/短くしてください/);});

test('generation stop terminates busy WASM and ignores late completion',async()=>{const x=await setup();x.$('load').click();const old=x.FakeWorker.all[0];old.emit({type:'ready',id:1});x.submit('孔子について');old.emit({type:'token',id:2,text:'孔子は'});x.$('stop').click();assert.equal(old.terminated,true);assert.equal(x.$('send').disabled,true);assert.equal(x.$('load').disabled,false);assert.equal(x.$('msg').value,'孔子について');old.emit({type:'done',id:2,text:'late reply'});assert.doesNotMatch(x.$('log').textContent,/late reply/);x.$('load').click();const current=x.FakeWorker.all[1];current.emit({type:'ready',id:4});x.submit('もう一度');assert.deepEqual(current.sent.at(-1).history,[]);});

test('100 percent download is not readiness and hung initialization releases the worker',async()=>{
  const x=await setup();x.$('load').click();const old=x.FakeWorker.all[0];
  old.emit({type:'progress',id:1,file:'model.onnx',loaded:100,total:100});
  assert.match(x.$('status').textContent,/取得は完了.*起動しています/);
  assert.equal(x.$('send').disabled,true);assert.equal(x.$('progress').hidden,true);
  x.advance(89_999);assert.equal(old.terminated,undefined);x.advance(1);
  assert.equal(old.terminated,true);assert.match(x.$('status').textContent,/90秒.*軽量版/);
  assert.equal(x.$('load').disabled,false);assert.equal(x.$('lightweight').getAttribute('href'),'../');
  old.emit({type:'ready',id:1});assert.equal(x.$('send').disabled,true);
  x.$('load').click();const current=x.FakeWorker.all[1];current.emit({type:'ready',id:3});
  x.advance(120_000);assert.equal(current.terminated,undefined);assert.equal(x.$('send').disabled,false);
});
test('cached model initialization has the same timeout without download progress',async()=>{
  const x=await setup();x.$('load').click();const w=x.FakeWorker.all[0];
  w.emit({type:'phase',id:1,phase:'initialize'});x.advance(90_000);
  assert.equal(w.terminated,true);assert.match(x.$('status').textContent,/起動完了を90秒/);
});
test('active download refreshes the stall deadline but a silent worker is released',async()=>{
  const x=await setup();x.$('load').click();const w=x.FakeWorker.all[0];x.advance(119_999);
  w.emit({type:'progress',id:1,file:'model.onnx',loaded:40,total:100});x.advance(119_999);
  assert.equal(w.terminated,undefined);x.advance(1);assert.equal(w.terminated,true);
  assert.match(x.$('status').textContent,/取得の進行.*2分/);
});
test('a failed load releases partial model memory and keeps the lightweight route available',async()=>{
  const x=await setup();x.$('load').click();const w=x.FakeWorker.all[0];
  w.emit({type:'error',id:1,stage:'load',message:'メモリ不足です。軽量版をご利用ください。',detail:'out of memory'});
  assert.equal(w.terminated,true);assert.equal(x.$('lightweight').hidden,false);assert.equal(x.$('load').disabled,false);
  x.advance(120_000);assert.match(x.$('status').textContent,/メモリ不足/);
});
