# ちいさな世界史AI（自作Transformer＋トーク画面）

## 起動のしかた
`start_chat.py` をダブルクリック → 番号を選ぶ → ブラウザが自動で開く
  1: そのまま起動（初回だけ学習が走る）
  2: add.jsonl を追加学習してから起動
  3: dialogues.jsonl で最初から学習し直してから起動
黒い画面はトーク中ずっと開いたままにする（閉じるとAIが止まる）。

## ファイル
- `start_chat.py` … 起動用
- `chat_server.py` … 学習とトーク画面のサーバー
- `tiny_model.py` … モデル本体（NumPyだけで書いたTransformer）
- `web/index.html` … トーク画面（直接開いても動かない。必ず start_chat.py から）
- `dialogues.jsonl` … 学習データ。全部の会話がここにたまる
- `add.jsonl` … 追加したい会話だけを書くファイル（自分で作る）。
  追加学習が終わると `added/added_日付.jsonl` に移される
- `model.npz` … 学習済みの重み（学習後にできる）

## 会話データの書き方（1行1組、UTF-8で保存）
{"user":"科挙とは","assistant":"隋の文帝が始めた筆記試験による役人の選抜制度です。"}

## GitHubで自動学習する場合
1. 非公開（Private）リポジトリを作り、このフォルダの中身をアップロードする
   （dialogues.jsonl と model.npz は自分の最新のものを入れる）
2. `.github/workflows/train.yml` は「Add file → Create new file」で、
   名前欄に `.github/workflows/train.yml` と打ち、中身を貼り付けて作る
3. 以後は add.jsonl をアップロードするだけで、Actions が追加学習して model.npz を更新する
4. 手元では GitHub から model.npz と dialogues.jsonl をダウンロードして差し替え、start_chat.py で起動
