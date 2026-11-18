import {MODEL, MAX_USER_CHARACTERS} from './config.mjs';
const $ = id => document.getElementById(id);
const load = $('load'), send = $('send'), stop = $('stop'), clear = $('clear'), input = $('msg'), status = $('status'), progress = $('progress'), log = $('log');
let worker, ready=false, busy=false, sequence=0, pending=null;
let history=[];
function add(role, text) {
  const row=document.createElement('div'); row.className='row '+role;
  if (role==='ai') { const avatar=document.createElement('span');avatar.className='avatar';avatar.textContent='J';row.append(avatar); }
  const bubble=document.createElement('div');bubble.className='bubble';bubble.textContent=text;row.append(bubble);log.append(row);log.scrollTop=log.scrollHeight;return bubble;
}
function controls() {
  send.disabled=!ready || busy; input.disabled=!ready || busy; clear.disabled=busy;
  stop.hidden=!busy;stop.disabled=!busy;stop.textContent=ready?'生成を停止':'読み込みを中止'; load.disabled=busy || ready;
}
function initWorker() {
  if (worker) worker.terminate();
  worker=new Worker(new URL('./worker.js', import.meta.url),{type:'module'});
  const activeWorker=worker;
  worker.addEventListener('message', ({data}) => {
    if (worker!==activeWorker || data.id !== sequence) return;
    if (data.type==='progress') {
      const percent=data.total ? Math.min(100,Math.round(data.loaded/data.total*100)) : 0;
      progress.hidden=false;progress.value=percent;
      status.textContent=percent ? `モデルを準備しています… ${percent}%` : 'モデルを準備しています…';
    } else if (data.type==='ready') {
      ready=true;busy=false;progress.hidden=true;status.textContent='準備できました。端末内で回答を生成します。';load.textContent='読み込み済み';controls();input.focus();
    } else if (data.type==='token' && pending) {
      pending.bubble.textContent=data.text || '考えています…';log.scrollTop=log.scrollHeight;
    } else if (data.type==='done' && pending) {
      pending.bubble.textContent=data.text || '回答を生成できませんでした。質問を変えてお試しください。';
      if (data.text && !data.cancelled) history.push({role:'user',content:pending.question},{role:'assistant',content:data.text});
      history=history.slice(-6); pending=null;busy=false;status.textContent=data.cancelled ? '生成を停止しました。' : '端末内で生成しました。';controls();input.focus();
    } else if (data.type==='error') {
      if (pending) {pending.bubble.textContent=data.message;input.value=pending.question;}
      pending=null;busy=false;progress.hidden=true;status.textContent=data.message;
      if (data.stage==='load') { ready=false;load.textContent='もう一度読み込む'; }
      console.error('Qwen runtime:',data.detail);controls();
    }
  });
  worker.addEventListener('error', () => {
    if(worker!==activeWorker)return;
    if(pending){pending.bubble.textContent='処理が停止しました。モデルを読み込み直してください。';input.value=pending.question;}
    ready=false;busy=false;pending=null;progress.hidden=true;load.textContent='もう一度読み込む';status.textContent='モデルの処理が停止しました。端末の空きメモリや通信を確認して再度お試しください。';controls();
  });
}
load.addEventListener('click', () => {
  if (busy || ready) return;
  if (typeof Worker==='undefined' || typeof WebAssembly==='undefined') { status.textContent='このブラウザでは実行できません。「旧版を開く」をお試しください。';return; }
  sequence++;busy=true;controls();status.textContent='モデルを読み込んでいます。初回は時間がかかります。';progress.hidden=false;progress.removeAttribute('value');initWorker();worker.postMessage({type:'load',id:sequence});
});
$('form').addEventListener('submit', event => {
  event.preventDefault();const question=input.value.trim();if (!ready || busy || !question) return;
  if (question.length>MAX_USER_CHARACTERS) {status.textContent='質問は400文字以内で入力してください。';return;}
  sequence++;add('me',question);input.value='';pending={question,bubble:add('ai','考えています…')};busy=true;controls();status.textContent='回答を生成しています…';worker.postMessage({type:'generate',id:sequence,question,history});
});
stop.addEventListener('click',()=>{if (!ready) { worker.terminate();sequence++;busy=false;progress.hidden=true;load.textContent='もう一度読み込む';status.textContent='読み込みを中止しました。';controls();return; } stop.disabled=true;worker.postMessage({type:'stop'});status.textContent='生成を停止しています…';});
clear.addEventListener('click',()=>{history=[];log.replaceChildren();greet();status.textContent=ready?'新しい会話を始められます。':'最初にモデルを読み込んでください。';});
input.addEventListener('keydown', event => {if (event.key==='Enter' && !event.shiftKey && !event.isComposing) {event.preventDefault();$('form').requestSubmit();}});
function greet() {add('ai','こんにちは、ジャーヴィスです。世界史の質問や、説明の言い換えを試してみてください。');}
$('download-size').textContent=`初回約${Math.ceil(MODEL.downloadBytes/1e6)}MB＋実行ファイル`;
controls();greet();
