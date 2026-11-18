import {AutoTokenizer, AutoModelForCausalLM, TextStreamer, InterruptableStoppingCriteria, env} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/+esm';
import {MODEL, MAX_INPUT_TOKENS, MAX_NEW_TOKENS, MAX_USER_CHARACTERS, visibleAnswer, messagesFor, errorMessage} from './config.mjs';
env.allowLocalModels = false;
env.useBrowserCache = true;
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;
env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/';
let tokenizer, model, busy = false;
const stopping = new InterruptableStoppingCriteria();
function post(type, id, extra={}) { self.postMessage({type, id, ...extra}); }
async function load(id) {
  if (busy) return;
  busy = true;
  try {
    const progress_callback = (event) => {
      if (event.status === 'progress') post('progress', id, {file:event.file, loaded:event.loaded || 0, total:event.total || 0});
    };
    tokenizer = await AutoTokenizer.from_pretrained(MODEL.id, {revision:MODEL.revision, progress_callback});
    model = await AutoModelForCausalLM.from_pretrained(MODEL.id, {
      revision:MODEL.revision, subfolder:MODEL.subfolder, dtype:MODEL.dtype,
      use_external_data_format:false, device:'wasm', progress_callback,
    });
    post('ready', id);
  } catch (error) {
    post('error', id, {stage:'load', message:errorMessage(error), detail:String(error?.message || error)});
  } finally { busy = false; }
}
async function generate({id, question, history}) {
  if (busy || !model) return;
  busy = true; stopping.reset();
  try {
    if (typeof question !== 'string' || !question.trim() || question.length > MAX_USER_CHARACTERS) throw new Error('質問は400文字以内で入力してください。');
    const messages = messagesFor(history || [], question.trim());
    let inputs;
    while (true) {
      inputs = tokenizer.apply_chat_template(messages, {add_generation_prompt:true, return_dict:true, enable_thinking:false});
      if (inputs.input_ids.dims[1] <= MAX_INPUT_TOKENS) break;
      if (messages.length > 2) messages.splice(1, 2);
      else throw new Error('質問が長いため処理できませんでした。少し短くしてください。');
    }
    let raw = '';
    const streamer = new TextStreamer(tokenizer, {
      skip_prompt:true, skip_special_tokens:false,
      callback_function(text) { raw += text; post('token', id, {text:visibleAnswer(raw)}); },
    });
    const output = await model.generate({...inputs, max_new_tokens:MAX_NEW_TOKENS, do_sample:true,
      temperature:0.6, top_p:0.95, repetition_penalty:1.08, streamer, stopping_criteria:stopping});
    const text = visibleAnswer(tokenizer.decode(output.tolist()[0].slice(inputs.input_ids.dims[1]), {skip_special_tokens:false})).trim();
    post('done', id, {text, cancelled:stopping.interrupted});
  } catch (error) {
    const detail = String(error?.message || error);
    post('error', id, {stage:'generate', message:/質問/.test(detail) ? detail : errorMessage(error), detail});
  } finally { busy = false; }
}
self.addEventListener('message', ({data}) => {
  if (data.type === 'stop') { stopping.interrupt(); return; }
  if (data.type === 'load') void load(data.id);
  if (data.type === 'generate') void generate(data);
});
