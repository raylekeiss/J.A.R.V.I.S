"""
tiny_model.py ― NumPyだけで動く、文字単位の小さなTransformer言語モデル

仕組み:
  会話を「U:質問\nA:回答\n」という1本の文字列にして、
  「次の1文字は何か」を当てる練習をくり返す（＝学習）。
  答えるときは「U:質問\nA:」まで与え、改行が出るまで1文字ずつ予測して足していく（＝生成）。

  重みの更新（誤差逆伝播）もすべてこのファイルに手書きしてある。
"""
import json
import numpy as np

BOS_USER = "U:"
BOS_AI = "A:"
END = "\n"


# ------------------------------------------------------------
# データ
# ------------------------------------------------------------
def load_dialogues(path):
    pairs = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pairs.append((d["user"].replace("\n", " "), d["assistant"].replace("\n", " ")))
    return pairs


def format_pair(user, assistant):
    return f"{BOS_USER}{user}{END}{BOS_AI}{assistant}{END}"


# ------------------------------------------------------------
# 小さな部品
# ------------------------------------------------------------
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.99, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = np.clip(grads[k], -1.0, 1.0)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ------------------------------------------------------------
# モデル本体
# ------------------------------------------------------------
class TinyTransformer:
    def __init__(self, chars, d_model=96, n_layers=2, ctx=96, seed=0):
        rng = np.random.default_rng(seed)
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.V, self.D, self.L, self.ctx = len(self.chars), d_model, n_layers, ctx
        s = 0.08
        D, V = self.D, self.V
        p = {
            "Wte": rng.normal(0, s, (V, D)),       # 文字 → ベクトル
            "Wpe": rng.normal(0, s, (ctx, D)),     # 位置 → ベクトル
            "Wout": rng.normal(0, s, (D, V)),      # ベクトル → 次の文字の点数
            "bout": np.zeros(V),
        }
        for l in range(n_layers):
            p[f"Wq{l}"] = rng.normal(0, s, (D, D))   # 自己注意：何を探すか
            p[f"Wk{l}"] = rng.normal(0, s, (D, D))   # 自己注意：何を持っているか
            p[f"Wv{l}"] = rng.normal(0, s, (D, D))   # 自己注意：渡す中身
            p[f"Wo{l}"] = rng.normal(0, s, (D, D))
            p[f"W1{l}"] = rng.normal(0, s, (D, 4 * D))  # 文字ごとの小さな計算（MLP）
            p[f"b1{l}"] = np.zeros(4 * D)
            p[f"W2{l}"] = rng.normal(0, s, (4 * D, D))
            p[f"b2{l}"] = np.zeros(D)
        self.p = p

    # ---------- 文字 ⇔ 番号 ----------
    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    # ---------- 順伝播 ----------
    def forward(self, x):
        """x: (B,T) の文字番号。logits と、逆伝播用のメモを返す"""
        p = self.p
        B, T = x.shape
        h = p["Wte"][x] + p["Wpe"][:T]
        mask = np.triu(np.full((T, T), -1e9), 1)   # 未来の文字は見ない
        caches = []
        for l in range(self.L):
            q, k, v = h @ p[f"Wq{l}"], h @ p[f"Wk{l}"], h @ p[f"Wv{l}"]
            S = q @ k.transpose(0, 2, 1) / np.sqrt(self.D) + mask
            P = softmax(S)                               # どの文字をどれだけ参照するか
            a = P @ v
            h1 = h + a @ p[f"Wo{l}"]
            m1 = h1 @ p[f"W1{l}"] + p[f"b1{l}"]
            r = np.maximum(m1, 0)
            h2 = h1 + r @ p[f"W2{l}"] + p[f"b2{l}"]
            caches.append((h, q, k, v, P, a, h1, m1, r))
            h = h2
        logits = h @ p["Wout"] + p["bout"]
        return logits, (x, caches, h)

    # ---------- 逆伝播（重みをどちらに動かせば当たるようになるか） ----------
    def loss_and_grads(self, x, y, weight):
        p = self.p
        logits, (x, caches, hL) = self.forward(x)
        B, T, V = logits.shape
        probs = softmax(logits)
        n = weight.sum()
        idx = np.arange(B)[:, None], np.arange(T)[None, :], y
        loss = -(np.log(probs[idx] + 1e-12) * weight).sum() / n

        g = {k: np.zeros_like(v) for k, v in p.items()}
        dlog = probs.copy()
        dlog[idx] -= 1
        dlog *= (weight / n)[..., None]
        g["Wout"] = hL.reshape(-1, self.D).T @ dlog.reshape(-1, V)
        g["bout"] = dlog.sum((0, 1))
        dh = dlog @ p["Wout"].T

        for l in reversed(range(self.L)):
            h, q, k, v, P, a, h1, m1, r = caches[l]
            # MLP
            dh1 = dh.copy()
            g[f"W2{l}"] = r.reshape(-1, r.shape[-1]).T @ dh.reshape(-1, self.D)
            g[f"b2{l}"] = dh.sum((0, 1))
            dm1 = (dh @ p[f"W2{l}"].T) * (m1 > 0)
            g[f"W1{l}"] = h1.reshape(-1, self.D).T @ dm1.reshape(-1, dm1.shape[-1])
            g[f"b1{l}"] = dm1.sum((0, 1))
            dh1 += dm1 @ p[f"W1{l}"].T
            # 自己注意
            dhin = dh1.copy()
            g[f"Wo{l}"] = a.reshape(-1, self.D).T @ dh1.reshape(-1, self.D)
            da = dh1 @ p[f"Wo{l}"].T
            dP = da @ v.transpose(0, 2, 1)
            dv = P.transpose(0, 2, 1) @ da
            dS = P * (dP - (dP * P).sum(-1, keepdims=True)) / np.sqrt(self.D)
            dq = dS @ k
            dk = dS.transpose(0, 2, 1) @ q
            hf = h.reshape(-1, self.D)
            g[f"Wq{l}"] = hf.T @ dq.reshape(-1, self.D)
            g[f"Wk{l}"] = hf.T @ dk.reshape(-1, self.D)
            g[f"Wv{l}"] = hf.T @ dv.reshape(-1, self.D)
            dhin += dq @ p[f"Wq{l}"].T + dk @ p[f"Wk{l}"].T + dv @ p[f"Wv{l}"].T
            dh = dhin

        np.add.at(g["Wte"], x, dh)
        g["Wpe"][:T] = dh.sum(0)
        return loss, g

    # ---------- 学習 ----------
    def train(self, pairs, steps=3000, batch=24, lr=3e-3, log=print, seed=0):
        rng = np.random.default_rng(seed)
        seqs = []
        for u, a in pairs:
            text = format_pair(u, a)
            ids = self.encode(text)[: self.ctx + 1]
            # 回答部分（A: の後ろ）だけを「当てるべき文字」として重視する
            ans_start = len(self.encode(f"{BOS_USER}{u}{END}{BOS_AI}"))
            seqs.append((ids, ans_start))
        opt = Adam(self.p, lr=lr)
        for step in range(1, steps + 1):
            chosen = [seqs[i] for i in rng.integers(0, len(seqs), batch)]
            T = max(len(s) for s, _ in chosen) - 1
            x = np.zeros((batch, T), dtype=np.int64)
            y = np.zeros((batch, T), dtype=np.int64)
            w = np.zeros((batch, T))
            for b, (ids, st) in enumerate(chosen):
                n = len(ids) - 1
                x[b, :n], y[b, :n] = ids[:-1], ids[1:]
                w[b, :n] = 0.2                       # 質問部分も少しだけ学ぶ
                w[b, max(st - 1, 0):n] = 1.0         # 回答部分はしっかり学ぶ
            loss, g = self.loss_and_grads(x, y, w)
            # 学習率を後半に向けて少しずつ下げる（最後の仕上げを細かくするため）
            opt.lr = lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / steps)))
            opt.step(self.p, g)
            if step % 200 == 0 or step == 1:
                log(f"step {step:5d}  loss {loss:.3f}")

    # ---------- 生成 ----------
    def reply(self, user_text, temperature=0.6, max_len=80, seed=None):
        rng = np.random.default_rng(seed)
        prompt = f"{BOS_USER}{user_text.strip()}{END}{BOS_AI}"
        ids = self.encode(prompt)
        out = []
        for _ in range(max_len):
            ctx_ids = (ids + out)[-self.ctx:]
            logits, _ = self.forward(np.array([ctx_ids]))
            last = logits[0, -1] / max(temperature, 1e-3)
            nxt = int(rng.choice(self.V, p=softmax(last)))
            if self.chars[nxt] == END:
                break
            out.append(nxt)
        return self.decode(out) or "……（うまく言葉が出てこない）"

    # ---------- 追加学習用：知らない文字を一覧に足す ----------
    def extend_vocab(self, texts, seed=1):
        new = sorted(set("".join(texts)) - set(self.chars))
        if not new:
            return []
        rng = np.random.default_rng(seed)
        n = len(new)
        self.chars += new
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.V = len(self.chars)
        p = self.p
        p["Wte"] = np.vstack([p["Wte"], rng.normal(0, 0.08, (n, self.D))])
        p["Wout"] = np.hstack([p["Wout"], rng.normal(0, 0.08, (self.D, n))])
        p["bout"] = np.concatenate([p["bout"], np.zeros(n)])
        return new

    # ---------- ブラウザ用に書き出す（GitHub Pages のトーク画面が読む） ----------
    def export_web(self, outdir):
        import os
        import time
        os.makedirs(outdir, exist_ok=True)
        tensors, offset, blobs = [], 0, []
        for name, arr in self.p.items():
            a = np.ascontiguousarray(arr, dtype="<f4")
            tensors.append({"name": name, "shape": list(a.shape), "offset": offset})
            offset += a.size
            blobs.append(a.tobytes())
        version = time.strftime("%Y%m%d%H%M%S")
        with open(os.path.join(outdir, "model.bin"), "wb") as f:
            f.write(b"".join(blobs))
        meta = {"version": version, "chars": self.chars, "D": self.D, "L": self.L,
                "ctx": self.ctx, "tensors": tensors}
        with open(os.path.join(outdir, "model.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    # ---------- 保存・読み込み ----------
    def save(self, path):
        meta = json.dumps({"chars": self.chars, "D": self.D, "L": self.L, "ctx": self.ctx})
        np.savez_compressed(path, __meta__=np.array(meta), **self.p)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["__meta__"]))
        m = cls(meta["chars"], meta["D"], meta["L"], meta["ctx"])
        m.p = {k: z[k] for k in m.p}
        return m


def build_from_pairs(pairs, **kw):
    chars = sorted(set("".join(format_pair(u, a) for u, a in pairs)))
    return TinyTransformer(chars, **kw)
