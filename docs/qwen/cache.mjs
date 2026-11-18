import {MODEL} from './config.mjs?v=compact-cache-2';

// Keep the existing cache name and immutable model URLs so updates reuse saved weights.
export const CACHE_KEY='transformers-cache';
const base=`https://huggingface.co/${MODEL.id}/resolve/${MODEL.revision}/`;
export const MODEL_WEIGHT_URL=base+MODEL.subfolder+'/model.onnx';
export const COMPRESSED_WEIGHT_URL=MODEL_WEIGHT_URL+'?jarvis-cache=gzip-v1';
export const MODEL_URLS=Object.freeze([
  'config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
  MODEL.subfolder+'/model.onnx',
].map(path=>base+path));

export async function savedModelExists(storage) {
  try {
    if(!storage?.open)return false;
    const cache=await storage.open(CACHE_KEY);
    const keys=new Set((await cache.keys()).map(request=>request.url));
    return MODEL_URLS.every(url=>keys.has(url)||(url===MODEL_WEIGHT_URL&&typeof DecompressionStream==='function'&&keys.has(COMPRESSED_WEIGHT_URL)));
  } catch {return false;}
}

export function modelCache(storage,{onRead=()=>{},onWrite=()=>{},onFailure=()=>{}}={}) {
  let cachePromise,reported=false;
  const open=()=>cachePromise??=storage?.open
    ? storage.open(CACHE_KEY) : Promise.reject(new Error('Model storage unavailable'));
  const report=error=>{if(!reported){reported=true;onFailure(error);}};
  return {
    async match(request) {
      try {
        const cache=await open();
        let response=await cache.match(request);
        if(!response&&String(request)===MODEL_WEIGHT_URL&&typeof DecompressionStream==='function'){
          const compressed=await cache.match(COMPRESSED_WEIGHT_URL);
          if(compressed){
            const headers=new Headers(compressed.headers);
            const length=headers.get('x-jarvis-original-length');
            headers.delete('x-jarvis-original-length');headers.delete('content-length');
            if(length)headers.set('content-length',length);
            response=new Response(compressed.body.pipeThrough(new DecompressionStream('gzip')),{headers});
          }
        }
        if(response&&String(request)===MODEL_WEIGHT_URL)onRead();
        return response;
      } catch(error) {report(error);return undefined;}
    },
    async put(request,response) {
      try {
        const cache=await open();
        let key=request;
        if(String(request)===MODEL_WEIGHT_URL){
          onWrite();
          if(typeof CompressionStream==='function'&&typeof DecompressionStream==='function'){
            key=COMPRESSED_WEIGHT_URL;
            const headers=new Headers(response.headers);
            const length=headers.get('content-length');
            headers.delete('content-length');headers.delete('content-encoding');
            if(length)headers.set('x-jarvis-original-length',length);
            response=new Response(response.body.pipeThrough(new CompressionStream('gzip')),{headers});
          }
        }
        await cache.put(key,response);
        if(!(await cache.keys(key)).length)throw new Error('Model storage write was not retained');
      } catch(error) {report(error);throw error;}
    },
  };
}

export async function requestPersistence(storage) {
  try {
    if(await storage?.persisted?.())return true;
    return Boolean(await storage?.persist?.());
  } catch {return false;}
}
