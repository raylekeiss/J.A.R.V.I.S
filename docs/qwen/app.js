import {MODEL, MAX_USER_CHARACTERS, LOAD_STALL_TIMEOUT_MS, INITIALIZE_TIMEOUT_MS} from './config.mjs?v=se2-startup-1';
const $ = id => document.getElementById(id);
const load = $('load'), send = $('send'), stop = $('stop'), clear = $('clear'), input = $('msg'), status = $('status'), progress = $('progress'), log = $('log');
let worker, ready=false, busy=false, sequence=0, pending=null;
let history=[];
let loadTimer;
const clock=document.defaultView;
function clearLoadTimer() { clock.clearTimeout(loadTimer);loadTimer=undefined; }
function loadFailed(message) {
  clearLoadTimer();worker?.terminate();worker=null;sequence++;ready=false;busy=false;
  progress.hidden=true;load.textContent='もう一度読み込む';status.textContent=message;controls();
}
function armLoadTimer(initializing=false) {
  clearLoadTimer();
  const activeWorker=worker, id=sequence;
  loadTimer=clock.setTimeout(()=>{
    if(worker!==activeWorker || sequence!==id || ready || !busy) return;
    loadFailed(initializing
      ? 'モデルの起動完了を90秒間確認できなかったため、読み込みを中止しました。この端末での実行を確認できていません。軽量版をご利用ください。'
      : '取得の進行を2分間確認できなかったため、読み込みを中止しました。通信状態を確認するか、軽量版をご利用ください。');
  },initializing?INITIALIZE_TIMEOUT_MS:LOAD_STALL_TIMEOUT_MS);
}
function initializing() {
  progress.hidden=true;
  status.textContent='モデルの取得は完了しました。起動しています。起動完了まで質問は送れません。';
  armLoadTimer(true);
}
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
  worker=new Worker(new URL('./worker.js?v=se2-startup-1', import.meta.url),{type:'module'});
  const activeWorker=worker;
  worker.addEventListener('message', ({data}) => {
    if (worker!==activeWorker || data.id !== sequence) return;
    if (data.type==='phase') {
      if(data.phase==='initialize') initializing();
      else {
        progress.hidden=false;progress.removeAttribute('value');
        status.textContent=data.phase==='tokenizer'?'会話に必要なデータを準備しています。':'モデルを取得しています。取得後に起動処理があります。';
        armLoadTimer();
      }
    } else if (data.type==='progress') {
      const percent=data.total ? Math.min(100,Math.floor(data.loaded/data.total*100)) : 0;
      const modelFile=/\.onnx$/.test(data.file || '');
      if(modelFile && percent===100) {initializing();return;}
      progress.hidden=false;progress.value=percent;
      const label=modelFile?'モデルを取得しています':'準備用データを取得しています';
      status.textContent=percent ? `${label}… ${percent}%（起動は未完了）` : `${label}…`;
      armLoadTimer();
    } else if (data.type==='ready') {
      clearLoadTimer();
      ready=true;busy=false;progress.hidden=true;status.textContent='準備できました。端末内で回答を生成します。';load.textContent='読み込み済み';controls();input.focus();
    } else if (data.type==='token' && pending) {
      pending.bubble.textContent=data.text || '考えています…';log.scrollTop=log.scrollHeight;
    } else if (data.type==='done' && pending) {
      pending.bubble.textContent=data.text || '回答を生成できませんでした。質問を変えてお試しください。';
      if (data.text && !data.cancelled) history.push({role:'user',content:pending.question},{role:'assistant',content:data.text});
      history=history.slice(-6); pending=null;busy=false;status.textContent=data.cancelled ? '生成を停止しました。' : '端末内で生成しました。';controls();input.focus();
    } else if (data.type==='error') {
      clearLoadTimer();
      if(data.stage==='load') {console.error('Qwen runtime:',data.detail);loadFailed(data.message);return;}
      if (pending) {pending.bubble.textContent=data.message;input.value=pending.question;}
      pending=null;busy=false;progress.hidden=true;status.textContent=data.message;
      console.error('Qwen runtime:',data.detail);controls();
    }
  });
  worker.addEventListener('error', () => {
    if(worker!==activeWorker)return;
    clearLoadTimer();worker.terminate();worker=null;sequence++;
    if(pending){pending.bubble.textContent='処理が停止しました。モデルを読み込み直してください。';input.value=pending.question;}
    ready=false;busy=false;pending=null;progress.hidden=true;load.textContent='もう一度読み込む';status.textContent='モデルの処理が停止しました。この端末での実行を確認できていません。軽量版をご利用ください。';controls();
  });
}
load.addEventListener('click', () => {
  if (busy || ready) return;
  if (typeof Worker==='undefined' || typeof WebAssembly==='undefined') { status.textContent='このブラウザでは実行できません。「軽量版を開く」をお試しください。';return; }
  sequence++;busy=true;controls();status.textContent='モデルを読み込んでいます。取得後に起動処理があります。';progress.hidden=false;progress.removeAttribute('value');
  try {initWorker();armLoadTimer();worker.postMessage({type:'load',id:sequence});}
  catch {loadFailed('モデルの処理を開始できませんでした。軽量版をご利用ください。');}
});
$('form').addEventListener('submit', event => {
  event.preventDefault();const question=input.value.trim();if (!ready || busy || !question) return;
  if (question.length>MAX_USER_CHARACTERS) {status.textContent='質問は400文字以内で入力してください。';return;}
  sequence++;add('me',question);input.value='';pending={question,bubble:add('ai','考えています…')};busy=true;controls();status.textContent='回答を生成しています…';worker.postMessage({type:'generate',id:sequence,question,history});
});
stop.addEventListener('click',()=>{
  clearLoadTimer();
  // Synchronous WASM may starve worker messages, so terminate instead of waiting for a stop message.
  const wasReady=ready;
  worker.terminate();worker=null;sequence++;ready=false;busy=false;
  if(pending){
    if(pending.bubble.textContent==='考えています…') pending.bubble.textContent='生成を停止しました。';
    else pending.bubble.textContent+='\n（生成を停止しました）';
    input.value=pending.question;pending=null;
  }
  progress.hidden=true;load.textContent='もう一度読み込む';
  status.textContent=wasReady?'生成を停止しました。再開するにはモデルを読み込み直してください。':'読み込みを中止しました。';
  controls();
});
clear.addEventListener('click',()=>{history=[];log.replaceChildren();greet();status.textContent=ready?'新しい会話を始められます。':'最初にモデルを読み込んでください。';});
input.addEventListener('keydown', event => {if (event.key==='Enter' && !event.shiftKey && !event.isComposing) {event.preventDefault();$('form').requestSubmit();}});
function greet() {add('ai','こんにちは、ジャーヴィスです。世界史の質問や、説明の言い換えを試してみてください。');}
$('download-size').textContent=`初回約${Math.ceil(MODEL.downloadBytes/1e6)}MB＋実行ファイル`;
controls();greet();
