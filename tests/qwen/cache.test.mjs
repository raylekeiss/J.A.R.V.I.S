import {test} from 'node:test';
import assert from 'node:assert/strict';
import {CACHE_KEY,MODEL_URLS,MODEL_WEIGHT_URL,savedModelExists,modelCache,requestPersistence} from '../../client/qwen/cache.mjs';
import {yieldBetweenTokens} from '../../client/qwen/runtime.mjs';
function storage(entries=[]){
  const data=new Map(entries),names=[];
  return {names,data,async open(name){names.push(name);return {
    async keys(request){return [...data.keys()].filter(url=>!request||String(request)===url).map(url=>({url}));},
    async match(request){return data.get(String(request))?.clone();},
    async put(request,response){const bytes=await response.arrayBuffer();data.set(String(request),new Response(bytes,{headers:response.headers}));},
  };}};
}
test('only all files of the pinned revision qualify for automatic startup',async()=>{
  const s=storage(MODEL_URLS.map(url=>[url,new Response('saved')]));assert.equal(await savedModelExists(s),true);assert.equal(s.names[0],CACHE_KEY);
  s.data.delete(MODEL_WEIGHT_URL);assert.equal(await savedModelExists(s),false);
  s.data.set(MODEL_WEIGHT_URL.replace(/resolve\/[^/]+/,'resolve/another-revision'),new Response('other'));
  assert.equal(await savedModelExists(s),false);assert.equal(await savedModelExists(undefined),false);
});
test('model bytes survive a fresh cache instance and reads never need a network request',async()=>{
  const s=storage();let writes=0,reads=0;
  await modelCache(s,{onWrite:()=>writes++}).put(MODEL_WEIGHT_URL,new Response('model-bytes'));
  const response=await modelCache(s,{onRead:()=>reads++}).match(MODEL_WEIGHT_URL);
  assert.equal(await response.text(),'model-bytes');assert.equal(writes,1);assert.equal(reads,1);
});
test('cache failure is reported once and missing storage is not treated as saved',async()=>{
  let warnings=0;const broken={open:async()=>{throw new Error('quota exceeded');}};
  const c=modelCache(broken,{onFailure:()=>warnings++});assert.equal(await c.match(MODEL_WEIGHT_URL),undefined);
  await assert.rejects(c.put(MODEL_WEIGHT_URL,new Response('bytes')),/quota/);assert.equal(warnings,1);assert.equal(await savedModelExists(broken),false);
});
test('persistence denial or unsupported storage does not prevent starting the model',async()=>{
  let requests=0;assert.equal(await requestPersistence({persisted:async()=>true,persist:async()=>{requests++;}}),true);
  assert.equal(requests,0);assert.equal(await requestPersistence({persisted:async()=>false,persist:async()=>false}),false);
  assert.equal(await requestPersistence(undefined),false);
});
test('a queued stop runs between model forwards and the loaded model remains reusable',async()=>{
  let cancelled=false,calls=0;
  const model=yieldBetweenTokens({async forward(){if(++calls===1)setTimeout(()=>{cancelled=true;},0);return {call:calls};}});
  for(let i=0;i<100;i++){await model.forward();if(cancelled)break;}
  assert.equal(cancelled,true);assert.equal(calls,1);
  assert.deepEqual(await model.forward(),{call:2});
});
