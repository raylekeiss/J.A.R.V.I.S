export const MODEL = Object.freeze({
  id: 'onnx-community/Qwen3-0.6B-ONNX',
  revision: '79346d370dfae34801d26993fda9ae2372900fd0',
  subfolder: 'onnxruntime/cpu_and_mobile/cpu-int4-kld-block-128',
  // This graph is already INT4-quantized; fp32 selects model.onnx and float32 KV cache.
  dtype: 'fp32',
  use_external_data_format: false,
  downloadBytes: 533717673,
});
export const SYSTEM_PROMPT = 'You are JARVIS, a helpful world-history tutor. Always respond in Japanese, not Chinese or English. 日本語で、質問に直接答えてください。自然で丁寧な口調で、まず結論を述べ、必要な説明を簡潔に続けます。歴史上の人物・年代・国名を推測で作らず、不確かな場合は不確かだと伝えてください。';
export const MAX_INPUT_TOKENS = 640;
export const MAX_NEW_TOKENS = 160;
export const MAX_USER_CHARACTERS = 400;
export const LOAD_STALL_TIMEOUT_MS = 120_000;
export const INITIALIZE_TIMEOUT_MS = 90_000;
// This graph is already optimized. Re-optimizing it raises peak startup memory.
export const SESSION_OPTIONS = Object.freeze({graphOptimizationLevel:'disabled', enableCpuMemArena:false, enableMemPattern:false});
export function visibleAnswer(raw) {
  return raw.replace(/<think>[\s\S]*?<\/think>/g, '')
    .replace(/<think>[\s\S]*$/g, '').replace(/<\|[^>]*\|>/g, '')
    .replace(/<[^>]*$/g, '').trimStart();
}
export function messagesFor(history, question) {
  const recent = history.slice(-6).map(({role, content}) => ({role, content}));
  return [{role:'system', content:SYSTEM_PROMPT}, ...recent, {role:'user', content:question}];
}
export function errorMessage(error) {
  const detail = String(error?.message || error);
  if (/memory|allocation|out of bounds/i.test(detail)) return '端末のメモリが足りず処理を続けられませんでした。約3.2MBの軽量版をご利用ください。';
  if (/fetch|network|download|HTTP|file.*not found/i.test(detail)) return 'モデルを取得できませんでした。通信状態を確認して、もう一度お試しください。';
  return 'この端末ではモデルを実行できませんでした。「軽量版を開く」から自作モデルを利用できます。';
}
