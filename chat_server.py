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


def main():
    global MODEL, TEMPERATURE
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "dialogues.jsonl"))
    ap.add_argument("--model", default=os.path.join(HERE, "model.npz"))
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--add", help="追加学習する会話ファイル（例: add.jsonl）")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--train-only", action="store_true", help="学習して保存したら終了する（GitHub Actions用）")
    args = ap.parse_args()
    TEMPERATURE = args.temperature
    if args.steps is None:
        args.steps = 1500 if args.add else 3000

    if args.add:
        # 追加学習：前の重みを引き継ぎ、新しい会話を多めに混ぜて短く学習する
        old_pairs = load_dialogues(args.data)
        new_pairs = load_dialogues(args.add)
        MODEL = TinyTransformer.load(args.model)
        added = MODEL.extend_vocab(f"U:{u}\nA:{a}\n" for u, a in new_pairs)
        if added:
            print(f"初めて出てくる文字が{len(added)}個あったので、文字の一覧に追加します。")
            print("（下はその一覧です。文字化けではありません）")
            print("  " + "".join(added))
        print(f"これまでの{len(old_pairs)}組 ＋ 新しい{len(new_pairs)}組で追加学習（{args.steps}回）")
        MODEL.train(old_pairs + new_pairs * 3, steps=args.steps, lr=2e-3)
        MODEL.save(args.model)
        # 新しい会話を dialogues.jsonl の後ろに書き足しておく（次回のやり直し学習用）
        with open(args.add, encoding="utf-8") as src, open(args.data, "a", encoding="utf-8") as dst:
            lines = [l.strip() for l in src if l.strip()]
            dst.write("\n" + "\n".join(lines) + "\n")
        # 同じ会話を二重に足さないよう、使い終わったファイルは名前を変えておく
        import time
        done_dir = os.path.join(os.path.dirname(os.path.abspath(args.add)), "added")
        os.makedirs(done_dir, exist_ok=True)
        done = os.path.join(done_dir, time.strftime("added_%Y%m%d_%H%M%S.jsonl"))
        os.replace(args.add, done)
        print(f"保存しました: {args.model}（追加分は {os.path.basename(args.data)} に追記し、"
              f"{os.path.basename(args.add)} は {os.path.basename(done)} に名前を変えました）")
    elif os.path.exists(args.model) and not args.retrain:
        print(f"モデルを読み込み中: {args.model}")
        MODEL = TinyTransformer.load(args.model)
    else:
        pairs = load_dialogues(args.data)
        print(f"{len(pairs)}組の会話で学習します（{args.steps}回）")
        MODEL = build_from_pairs(pairs, d_model=args.dim, n_layers=args.layers)
        MODEL.train(pairs, steps=args.steps)
        MODEL.save(args.model)
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
