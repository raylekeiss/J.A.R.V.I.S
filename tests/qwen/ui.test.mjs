import {test} from 'node:test';import assert from 'node:assert/strict';import {JSDOM} from 'jsdom';import {readFile} from 'node:fs/promises';
import {MODEL_URLS} from '../../client/qwen/cache.mjs';
const html=await readFile(new URL('../../docs/qwen/index.html',import.meta.url),'utf8');let run=0;
async function setup({cached=false}={}){const dom=new JSDOM(html,{url:'https://example.test/qwen/'});class FakeWorker{static all=[];constructor(){this.handlers={};this.sent=[];FakeWorker.all.push(this);}addEventListener(k,f){(this.handlers[k]??=[]).push(f);}postMessage(m){this.sent.push(m);}terminate(){this.terminated=true;}emit(data){for(const f of this.handlers.message||[])f({data});}}
let now=0,timerId=0;const timers=new Map();
dom.window.setTimeout=(callback,delay)=>{const id=++timerId;timers.set(id,{callback,at:now+delay});return id;};
dom.window.clearTimeout=id=>timers.delete(id);
function advance(ms){const target=now+ms;while(true){const next=[...timers].filter(([,v])=>v.at<=target).sort((a,b)=>a[1].at-b[1].at)[0];if(!next)break;now=next[1].at;timers.delete(next[0]);next[1].callback();}now=target;}
globalThis.document=dom.window.document;globalThis.Worker=FakeWorker;
dom.window.caches={open:async()=>({keys:async()=>(cached?MODEL_URLS:[]).map(url=>({url}))})};
const app=await import(`../../client/qwen/app.mjs?test=${run++}`);await app.boot;const $=id=>dom.window.document.getElementById(id);return{dom,$,FakeWorker,advance,submit(q){$('msg').value=q;$('form').dispatchEvent(new dom.window.Event('submit',{cancelable:true}));}};}
test('model is opt-in, streams safely, and retains actual conversation',async()=>{const x=await setup();assert.equal(x.FakeWorker.all.length,0);assert.equal(x.$('send').disabled,true);x.$('load').click();const w=x.FakeWorker.all[0];w.emit({type:'ready',id:1});x.submit('孔子と孟子の違い');assert.equal(w.sent.at(-1).question,'孔子と孟子の違い');w.emit({type:'token',id:1,text:'stale'});assert.doesNotMatch(x.$('log').textContent,/stale/);w.emit({type:'token',id:2,text:'<img src=x>説明'});assert.equal(x.$('log').querySelectorAll('img').length,0);w.emit({type:'done',id:2,text:'孔子と孟子の説明'});x.submit('短くして');assert.deepEqual(w.sent.at(-1).history.map(m=>m.role),['user','assistant']);w.emit({type:'done',id:3,text:'短い説明'});x.$('clear').click();x.submit('こんにちは');assert.deepEqual(w.sent.at(-1).history,[]);});
test('loading can be cancelled and stale worker cannot re-enable sending',async()=>{const x=await setup();x.$('load').click();const old=x.FakeWorker.all[0];x.$('stop').click();assert.equal(old.terminated,true);old.emit({type:'ready',id:1});assert.equal(x.$('send').disabled,true);x.$('load').click();const current=x.FakeWorker.all[1];old.emit({type:'ready',id:1});assert.equal(x.$('send').disabled,true);current.emit({type:'ready',id:3});assert.equal(x.$('send').disabled,false);});
test('generation errors restore question and allow retry',async()=>{const x=await setup();x.$('load').click();const w=x.FakeWorker.all[0];w.emit({type:'ready',id:1});x.submit('長い質問');w.emit({type:'error',id:2,stage:'generate',message:'短くしてください',detail:'input too long'});assert.equal(x.$('msg').value,'長い質問');assert.equal(x.$('send').disabled,false);assert.match(x.$('log').textContent,/短くしてください/);});

test('ordinary stop retains the model and the next question uses the same worker',async()=>{
  const x=await setup();x.$('load').click();const w=x.FakeWorker.all[0];w.emit({type:'ready',id:1});
  x.submit('孔子について');x.$('stop').click();assert.equal(w.sent.at(-1).type,'stop');assert.equal(w.terminated,undefined);
  w.emit({type:'done',id:2,text:'孔子は',cancelled:true});x.advance(10_000);assert.equal(w.terminated,undefined);
  x.submit('もう一度');assert.equal(x.FakeWorker.all.length,1);assert.equal(w.sent.at(-1).type,'generate');assert.deepEqual(w.sent.at(-1).history,[]);
});
test('unresponsive generation still has an emergency stop and ignores stale completion',async()=>{
  const x=await setup();x.$('load').click();const old=x.FakeWorker.all[0];old.emit({type:'ready',id:1});x.submit('孔子について');x.$('stop').click();
  x.advance(9_999);assert.equal(old.terminated,undefined);x.advance(1);assert.equal(old.terminated,true);
  assert.equal(x.$('send').disabled,true);assert.equal(x.$('load').disabled,false);assert.equal(x.$('msg').value,'孔子について');
  old.emit({type:'done',id:2,text:'late reply'});assert.doesNotMatch(x.$('log').textContent,/late reply/);
});

test('100 percent download is not readiness and hung initialization releases the worker',async()=>{
  const x=await setup();x.$('load').click();const old=x.FakeWorker.all[0];
  old.emit({type:'progress',id:1,file:'model.onnx',loaded:100,total:100});
  assert.match(x.$('status').textContent,/起動中/);
  assert.equal(x.$('send').disabled,true);assert.equal(x.$('progress').hidden,true);
  x.advance(89_999);assert.equal(old.terminated,undefined);x.advance(1);
  assert.equal(old.terminated,true);assert.match(x.$('status').textContent,/起動できません.*軽量版/);
  assert.equal(x.$('load').disabled,false);assert.equal(x.dom.window.document.querySelector('header a').getAttribute('href'),'../');
  old.emit({type:'ready',id:1});assert.equal(x.$('send').disabled,true);
  x.$('load').click();const current=x.FakeWorker.all[1];current.emit({type:'ready',id:3});
  x.advance(120_000);assert.equal(current.terminated,undefined);assert.equal(x.$('send').disabled,false);
});
test('cached model initialization has the same timeout without download progress',async()=>{
  const x=await setup();x.$('load').click();const w=x.FakeWorker.all[0];
  w.emit({type:'phase',id:1,phase:'initialize'});x.advance(90_000);
  assert.equal(w.terminated,true);assert.match(x.$('status').textContent,/起動できません/);
});
test('active download refreshes the stall deadline but a silent worker is released',async()=>{
  const x=await setup();x.$('load').click();const w=x.FakeWorker.all[0];x.advance(119_999);
  w.emit({type:'progress',id:1,file:'model.onnx',loaded:40,total:100});x.advance(119_999);
  assert.equal(w.terminated,undefined);x.advance(1);assert.equal(w.terminated,true);
  assert.match(x.$('status').textContent,/取得が停止/);
});
test('a failed load releases partial model memory and keeps the lightweight route available',async()=>{
  const x=await setup();x.$('load').click();const w=x.FakeWorker.all[0];
  w.emit({type:'error',id:1,stage:'load',message:'メモリ不足です。軽量版をご利用ください。',detail:'out of memory'});
  assert.equal(w.terminated,true);assert.equal(x.dom.window.document.querySelector('header a').hidden,false);assert.equal(x.$('load').disabled,false);
  x.advance(120_000);assert.match(x.$('status').textContent,/メモリ不足/);
});

test('saved files auto-start in cache-only mode and storage reads are not called downloads',async()=>{
  const x=await setup({cached:true});assert.equal(x.FakeWorker.all.length,1);const w=x.FakeWorker.all[0];
  assert.equal(w.sent[0].type,'load');assert.equal(w.sent[0].cachedOnly,true);
  w.emit({type:'phase',id:1,phase:'restore'});w.emit({type:'progress',id:1,file:'model.onnx',loaded:50,total:100});
  assert.match(x.$('status').textContent,/保存済みモデルを読み込み中/);assert.doesNotMatch(x.$('status').textContent,/取得中/);
  w.emit({type:'ready',id:1});assert.equal(x.$('load').hidden,true);assert.equal(x.$('send').disabled,false);
});
test('storage failures remain visible without adding explanation panels',async()=>{
  const x=await setup();assert.equal(x.dom.window.document.querySelectorAll('details,aside,.setup,.footnote').length,0);
  x.$('load').click();const w=x.FakeWorker.all[0];w.emit({type:'cache-warning',id:1,detail:'quota exceeded'});w.emit({type:'ready',id:1});
  assert.match(x.$('status').textContent,/保存できません/);assert.equal(x.$('send').disabled,false);
});
