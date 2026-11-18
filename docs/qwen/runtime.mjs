// A WASM token loop can starve worker messages. Yield after each forward pass so
// a stop request can reach the stopping criteria while keeping the model loaded.
export function yieldBetweenTokens(model,pause=()=>new Promise(resolve=>setTimeout(resolve,0))) {
  const forward=model.forward.bind(model);
  model.forward=async (...args)=>{
    const output=await forward(...args);
    await pause();
    return output;
  };
  return model;
}
