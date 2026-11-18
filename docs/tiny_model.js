// tiny_model.js ― tiny_model.py と同じ計算をブラウザで行う（生成だけ。学習はしない）
"use strict";

class TinyModelJS {
  static async load(base = ".") {
    const meta = await (await fetch(`${base}/model.json`, { cache: "no-store" })).json();
    const buf = await (await fetch(`${base}/model.bin?v=${meta.version}`)).arrayBuffer();
    return new TinyModelJS(meta, new Float32Array(buf));
  }

  constructor(meta, all) {
    this.meta = meta;
    this.chars = meta.chars;
    this.stoi = new Map(this.chars.map((c, i) => [c, i]));
    this.V = this.chars.length; this.D = meta.D; this.L = meta.L; this.ctx = meta.ctx;
    this.p = {};
    for (const t of meta.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      this.p[t.name] = { data: all.subarray(t.offset, t.offset + n), shape: t.shape };
    }
  }

  encode(text) {
    const ids = [];
    for (const c of text) if (this.stoi.has(c)) ids.push(this.stoi.get(c));
    return ids;
  }

  // x: (T, n) 行列 × W: (n, m) → (T, m)
  matmul(x, T, n, W, m, bias) {
    const out = new Float32Array(T * m);
    for (let t = 0; t < T; t++) {
      const o = t * m;
      if (bias) out.set(bias, o);
      for (let k = 0; k < n; k++) {
        const v = x[t * n + k];
        if (v === 0) continue;
        const wr = k * m;
        for (let j = 0; j < m; j++) out[o + j] += v * W[wr + j];
      }
    }
    return out;
  }

  // 最後の文字の「次の文字の点数」を返す
  lastLogits(ids) {
    const { D, L, V, p } = this, T = ids.length;
    let h = new Float32Array(T * D);
    for (let t = 0; t < T; t++)
      for (let d = 0; d < D; d++)
        h[t * D + d] = p.Wte.data[ids[t] * D + d] + p.Wpe.data[t * D + d];
    const scale = 1 / Math.sqrt(D);
    for (let l = 0; l < L; l++) {
      const q = this.matmul(h, T, D, p[`Wq${l}`].data, D);
      const k = this.matmul(h, T, D, p[`Wk${l}`].data, D);
      const v = this.matmul(h, T, D, p[`Wv${l}`].data, D);
      const a = new Float32Array(T * D);
      const s = new Float64Array(T);
      for (let i = 0; i < T; i++) {          // 自己注意（未来の文字は見ない）
        let mx = -Infinity;
        for (let j = 0; j <= i; j++) {
          let dot = 0;
          for (let d = 0; d < D; d++) dot += q[i * D + d] * k[j * D + d];
          s[j] = dot * scale; if (s[j] > mx) mx = s[j];
        }
        let sum = 0;
        for (let j = 0; j <= i; j++) { s[j] = Math.exp(s[j] - mx); sum += s[j]; }
        for (let j = 0; j <= i; j++) {
          const w = s[j] / sum;
          for (let d = 0; d < D; d++) a[i * D + d] += w * v[j * D + d];
        }
      }
      const o = this.matmul(a, T, D, p[`Wo${l}`].data, D);
      const h1 = new Float32Array(T * D);
      for (let i = 0; i < h1.length; i++) h1[i] = h[i] + o[i];
      const m1 = this.matmul(h1, T, D, p[`W1${l}`].data, 4 * D, p[`b1${l}`].data);
      for (let i = 0; i < m1.length; i++) if (m1[i] < 0) m1[i] = 0;
      const m2 = this.matmul(m1, T, 4 * D, p[`W2${l}`].data, D, p[`b2${l}`].data);
      for (let i = 0; i < h1.length; i++) h1[i] += m2[i];
      h = h1;
    }
    const last = h.subarray((T - 1) * D, T * D);
    return this.matmul(last, 1, D, p.Wout.data, V, p.bout.data);
  }

  reply(userText, temperature = 0.6, maxLen = 80) {
    const prompt = this.encode(`U:${userText.trim()}\nA:`);
    const out = [];
    const END = this.stoi.get("\n");
    for (let n = 0; n < maxLen; n++) {
      const ids = prompt.concat(out).slice(-this.ctx);
      const logits = this.lastLogits(ids);
      const t = Math.max(temperature, 1e-3);
      let mx = -Infinity;
      for (const x of logits) if (x / t > mx) mx = x / t;
      const pr = new Float64Array(this.V); let sum = 0;
      for (let i = 0; i < this.V; i++) { pr[i] = Math.exp(logits[i] / t - mx); sum += pr[i]; }
      let r = Math.random() * sum, next = this.V - 1;
      for (let i = 0; i < this.V; i++) { r -= pr[i]; if (r <= 0) { next = i; break; } }
      if (next === END) break;
      out.push(next);
    }
    const text = out.map(i => this.chars[i]).join("");
    return text || "……（うまく言葉が出てこない）";
  }
}

if (typeof module !== "undefined") module.exports = { TinyModelJS };
