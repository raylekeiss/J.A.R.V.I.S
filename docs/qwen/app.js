import {MODEL,MAX_USER_CHARACTERS,LOAD_STALL_TIMEOUT_MS,INITIALIZE_TIMEOUT_MS,STOP_TIMEOUT_MS} from './config.mjs?v=compact-cache-1';
import {savedModelExists,requestPersistence} from './cache.mjs?v=compact-cache-1';
const $=id=>document.getElementById(id);
const load=$('load'),send=$('send'),stop=$('stop'),clear=$('clear'),input=$('msg'),status=$('status'),progress=$('progress'),log=$('log');
const clock=document.defaultView;
let worker,ready=false,busy=false,sequence=0,pending=null,history=[];
let loadTimer,stopTimer,restoring=false,cacheWarning=false;
function clearLoadTimer(){clock.clearTimeout(loadTimer);loadTimer=undefined;}
function clearStopTimer(){clock.clearTimeout(stopTimer);stopTimer=undefined;}
function controls(){
  send.disabled=!ready||busy;input.disabled=!ready||busy;clear.disabled=busy;
  load.hidden=ready||busy;load.disabled=busy||ready;
  stop.hidden=!busy;stop.disabled=!busy;stop.textContent=ready?'停止':'中止';
}
function release(){
  clearLoadTimer();clearStopTimer();worker?.terminate();worker=null;sequence++;ready=false;busy=false;
  progress.hidden=true;load.textContent='再試行';controls();
}
function loadFailed(message){release();status.textContent=message;}
function armLoadTimer(initializing=false){
  clearLoadTimer();const active=worker,id=sequence;
  loadTimer=clock.setTimeout(()=>{
    if(worker!==active||sequence!==id||ready||!busy)return;
    loadFailed(initializing?'起動できませんでした。軽量版をお試しください。':'取得が停止しました。通信をご確認ください。');
  },initializing?INITIALIZE_TIMEOUT_MS:LOAD_STALL_TIMEOUT_MS);
}
function initializing(){progress.hidden=true;status.textContent='起動中…';armLoadTimer(true);}
function add(role,text){
  const row=document.createElement('div');row.className='row '+role;
  if(role==='ai'){const avatar=document.createElement('span');avatar.className='avatar';avatar.textContent='J';row.append(avatar);}
  const bubble=document.createElement('div');bubble.className='bubble';bubble.textContent=text;row.append(bubble);log.append(row);log.scrollTop=log.scrollHeight;return bubble;
}
function initWorker(){
  worker?.terminate();worker=new Worker(new URL('./worker.js?v=compact-cache-1',import.meta.url),{type:'module'});
  const active=worker;
  worker.addEventListener('message',({data})=>{
    if(worker!==active||data.id!==sequence)return;
    if(data.type==='cache-warning'){cacheWarning=true;console.warn('Model storage:',data.detail);return;}
    if(data.type==='phase'){
      if(data.phase==='initialize'){initializing();return;}
      if(data.phase==='restore')restoring=true;
      if(data.phase==='model')restoring=false;
      progress.hidden=false;progress.removeAttribute('value');
      status.textContent=data.phase==='restore'?'保存済みモデルを読み込み中…':data.phase==='save'?'保存中…':data.phase==='model'?'取得中…':'準備中…';
      armLoadTimer();
    }else if(data.type==='progress'){
      const percent=data.total?Math.min(100,Math.floor(data.loaded/data.total*100)):0;
      const modelFile=/\.onnx$/.test(data.file||'');
      if(modelFile&&percent===100){initializing();return;}
      progress.hidden=false;progress.value=percent;
      const label=modelFile?(restoring?'保存済みモデルを読み込み中':'取得中'):'準備中';
      status.textContent=percent?`${label}… ${percent}%`:`${label}…`;armLoadTimer();
    }else if(data.type==='ready'){
      clearLoadTimer();ready=true;busy=false;progress.hidden=true;
      status.textContent=cacheWarning?'準備完了（保存できませんでした）':'準備完了';controls();input.focus();
    }else if(data.type==='token'&&pending){
      pending.bubble.textContent=data.text||'考え中…';log.scrollTop=log.scrollHeight;
    }else if(data.type==='done'&&pending){
      clearStopTimer();
      pending.bubble.textContent=data.text||(data.cancelled?'停止しました。':'回答を生成できませんでした。');
      if(data.text&&!data.cancelled)history.push({role:'user',content:pending.question},{role:'assistant',content:data.text});
      history=history.slice(-6);pending=null;busy=false;status.textContent=data.cancelled?'停止しました':'準備完了';controls();input.focus();
    }else if(data.type==='error'){
      clearLoadTimer();clearStopTimer();console.error('Qwen runtime:',data.detail);
      if(data.stage==='load'){loadFailed(data.message);return;}
      if(pending){pending.bubble.textContent=data.message;input.value=pending.question;}
      pending=null;busy=false;status.textContent=data.message;controls();
    }
  });
  worker.addEventListener('error',()=>{
    if(worker!==active)return;
    if(pending){pending.bubble.textContent='処理が停止しました。';input.value=pending.question;pending=null;}
    loadFailed('処理が停止しました。再試行してください。');
  });
}
function start(cachedOnly=false){
  if(busy||ready)return;
  if(typeof Worker==='undefined'||typeof WebAssembly==='undefined'){status.textContent='このブラウザでは起動できません。';return;}
  sequence++;busy=true;restoring=cachedOnly;cacheWarning=false;controls();
  status.textContent=cachedOnly?'保存済みモデルを読み込み中…':'準備中…';progress.hidden=false;progress.removeAttribute('value');
  if(!cachedOnly)void requestPersistence(clock.navigator.storage);
  try{initWorker();armLoadTimer();worker.postMessage({type:'load',id:sequence,cachedOnly});}
  catch{loadFailed('起動できませんでした。');}
}
load.addEventListener('click',()=>start());
$('form').addEventListener('submit',event=>{
  event.preventDefault();const question=input.value.trim();if(!ready||busy||!question)return;
  if(question.length>MAX_USER_CHARACTERS){status.textContent='400文字以内で入力してください。';return;}
  sequence++;add('me',question);input.value='';pending={question,bubble:add('ai','考え中…')};busy=true;controls();status.textContent='生成中…';
  worker.postMessage({type:'generate',id:sequence,question,history});
});
stop.addEventListener('click',()=>{
  if(!busy)return;
  clearLoadTimer();
  if(!ready){release();status.textContent='中止しました';return;}
  stop.disabled=true;status.textContent='停止中…';worker.postMessage({type:'stop'});
  const active=worker,id=sequence;
  // A stalled native call may still need forced termination. Ordinary stops retain the model.
  stopTimer=clock.setTimeout(()=>{
    if(worker!==active||sequence!==id||!busy)return;
    if(pending){pending.bubble.textContent='停止しました。';input.value=pending.question;pending=null;}
    release();status.textContent='停止しました。再試行してください。';
  },STOP_TIMEOUT_MS);
});
clear.addEventListener('click',()=>{history=[];log.replaceChildren();greet();status.textContent=ready?'準備完了':'';});
input.addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('form').requestSubmit();}});
function greet(){add('ai','こんにちは、ジャーヴィスです。');}
load.textContent=`はじめる（約${Math.ceil(MODEL.downloadBytes/1e6)}MB）`;
controls();greet();
// Inspect actual stored files rather than relying on a flag left by an earlier page.
export const boot=savedModelExists(clock.caches).then(saved=>{if(saved&&sequence===0)start(true);});
