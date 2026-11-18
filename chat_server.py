"""
chat_server.py ― 自作モデルを「チャットAIのトーク画面」で使うためのサーバー

使い方:
  python chat_server.py                 # model.npz があれば読み込む。なければ dialogues.jsonl で学習して保存
  python chat_server.py --retrain       # データを変えたあと、学習し直す
  python chat_server.py --steps 5000    # 学習回数を増やす

起動したらブラウザで http://localhost:8000 を開く。
（Python標準の http.server だけで動く。追加で必要なのは NumPy のみ）
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tiny_model import TinyTransformer, build_from_pairs, load_dialogues

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = None
TEMPERATURE = 0.6


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "web", "index.html"), encoding="utf-8") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/chat":
            return self._send(404, "not found", "text/plain; charset=utf-8")
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        message = str(req.get("message", ""))[:200]
        temp = float(req.get("temperature", TEMPERATURE))
        reply = MODEL.reply(message, temperature=temp)
        self._send(200, json.dumps({"reply": reply}, ensure_ascii=False),
                   "application/json; charset=utf-8")

    def log_message(self, fmt, *args):  # アクセスログは出さない
        pass


DEFAULT_DIM = 128
BIG_BATCH = 300        # 一度にこれ以上の会話が追加されたら、最初から学習し直す


def auto_steps(n_pairs):
    """会話の数に合わせて学習回数を決める"""
    return int(min(max(n_pairs * 2.6, 3000), 16000))


def read_pairs(path):
    """jsonlを読む。壊れた行は飛ばして、どこが壊れていたかを返す"""
    pairs, errors = [], []
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8-sig") as f:
        for no, line in enumerate(f, 1):
            line = line.strip()
            pos = 0
            while pos < len(line):          # 1行に2組くっついていても読めるように
                try:
                    d, end = dec.raw_decode(line, pos)
                    u, a = str(d["user"]).strip(), str(d["assistant"]).strip()
                    if u and a:
                        pairs.append((u.replace("\n", " "), a.replace("\n", " ")))
                    pos = end
                    while pos < len(line) and line[pos] in " ,\t":
                        pos += 1
                except Exception:
                    errors.append(f"{os.path.basename(path)} の {no}行目")
                    break
    return pairs, errors


def write_pairs(path, pairs):
    with open(path, "w", encoding="utf-8") as f:
        for u, a in pairs:
            f.write(json.dumps({"user": u, "assistant": a}, ensure_ascii=False) + "\n")


def merge(old, new):
    """同じ質問があれば新しい答えで上書きして、1つにまとめる"""
    d = {}
    for u, a in old + new:
        d[u] = a
    return list(d.items())


def save_all(model, args):
    """model.npz と、GitHub Pages 用のブラウザ版モデル（docs フォルダ）を保存する"""
    model.save(args.model)
    model.export_web(os.path.join(os.path.dirname(os.path.abspath(args.model)), "docs"))


def add_training(args):
    import time
    target = args.add
    files = ([os.path.join(target, f) for f in sorted(os.listdir(target)) if f.endswith(".jsonl")]
             if os.path.isdir(target) else [target])
    if not files:
        print(f"{target} に .jsonl ファイルがありません。")
        sys.exit(1)

    new_pairs, errors = [], []
    for f in files:
        p, e = read_pairs(f)
        print(f"  {os.path.basename(f)}: {len(p)}組")
        new_pairs += p
        errors += e
    if errors:
        print(f"※ 読めなかった行が{len(errors)}か所ありました（飛ばして続けます）: " + "、".join(errors[:10]))
    if not new_pairs:
        print("追加できる会話がありませんでした。")
        sys.exit(1)

    old_pairs, _ = read_pairs(args.data) if os.path.exists(args.data) else ([], [])
    all_pairs = merge(old_pairs, new_pairs)
    write_pairs(args.data, all_pairs)
    print(f"{os.path.basename(args.data)} は {len(all_pairs)}組になりました（新しく{len(new_pairs)}組）")

    if not os.path.exists(args.model) or len(new_pairs) >= BIG_BATCH or args.retrain:
        steps = args.steps or auto_steps(len(all_pairs))
        print(f"量が多いので、全部の会話で最初から学習し直します（{steps}回）")
        model = build_from_pairs(all_pairs, d_model=args.dim, n_layers=args.layers)
        model.train(all_pairs, steps=steps)
    else:
        steps = args.steps or 1500
        model = TinyTransformer.load(args.model)
        added = model.extend_vocab(f"U:{u}\nA:{a}\n" for u, a in new_pairs)
        if added:
            print(f"初めて出てくる文字が{len(added)}個あったので、文字の一覧に追加します: {''.join(added)}")
        print(f"これまでの会話に新しい{len(new_pairs)}組を多めに混ぜて追加学習します（{steps}回）")
        model.train(all_pairs + new_pairs * 3, steps=steps, lr=2e-3)
    save_all(model, args)

    # 使い終わったファイルは added フォルダへ（同じ会話を二重に足さないため）
    base = os.path.dirname(os.path.abspath(args.data))
    done_dir = os.path.join(base, "added")
    os.makedirs(done_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for f in files:
        os.replace(f, os.path.join(done_dir, f"{stamp}_{os.path.basename(f)}"))
    print(f"保存しました: {args.model}（使ったファイルは added フォルダに移しました）")
    return model


def main():
    global MODEL, TEMPERATURE
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "dialogues.jsonl"))
    ap.add_argument("--model", default=os.path.join(HERE, "model.npz"))
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--add", help="追加学習する会話ファイル、またはフォルダ（例: add.jsonl / inbox）")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--train-only", action="store_true", help="学習して保存したら終了する（GitHub Actions用）")
    args = ap.parse_args()
    TEMPERATURE = args.temperature

    if args.add:
        MODEL = add_training(args)
    elif os.path.exists(args.model) and not args.retrain:
        print(f"モデルを読み込み中: {args.model}")
        MODEL = TinyTransformer.load(args.model)
        if not os.path.exists(os.path.join(HERE, "docs", "model.json")):
            MODEL.export_web(os.path.join(HERE, "docs"))
    else:
        pairs, errors = read_pairs(args.data)
        if errors:
            print(f"※ 読めなかった行を飛ばしました: " + "、".join(errors[:10]))
        steps = args.steps or auto_steps(len(pairs))
        print(f"{len(pairs)}組の会話で最初から学習します（{steps}回）")
        MODEL = build_from_pairs(pairs, d_model=args.dim, n_layers=args.layers)
        MODEL.train(pairs, steps=steps)
        save_all(MODEL, args)
        print(f"保存しました: {args.model}")

    if args.train_only:
        return
    url = f"http://localhost:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\nトーク画面: {url}  （この画面を閉じると止まります）")
    if not args.no_browser:
        import webbrowser
        webbrowser.open(url)          # ブラウザで自動的に開く
    server.serve_forever()


if __name__ == "__main__":
    main()
