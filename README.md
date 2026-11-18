# ちいさな世界史AI

NumPyだけで実装した文字単位のTransformerと、世界史・日常会話のチャット画面です。OpenAI APIキーは使用しません。

## ローカルで使う

`pip install -r requirements.txt` の後、`start_chat.py` を起動します。

1. 保存済みモデルで会話する。
2. inboxまたはadd.jsonlを使って候補を追加学習し、評価に合格した場合だけ更新する。
3. 全データで候補を作り直し、評価に合格した場合だけ更新する。

初回モデルがない場合だけ通常の初期学習を行います。黒いサーバー画面を閉じるとローカルの会話も止まります。ブラウザ版はdocs/index.htmlとモデルをGitHub Pagesで配信すると、ブラウザ内で推論できます。

## 自動学習

inboxへ290行単位のJSONLを追加するとGitHub Actionsが候補を学習・評価します。件数が300組を超えただけで初期化することはありません。合格時だけ学習済みデータとモデルを保存し、Pagesへ公開します。不合格時のモデル・データは変更せず、結果をActionsの成果物に保存します。

```json
{"user":"科挙とは","assistant":"隋の文帝が始めた筆記試験による役人の選抜制度です。"}
```

新規ファイルはUTF-8、各290行、userとassistantの2項目を使います。質問25字以内、回答1文40字以内、です終止で、問い返しは入れません。訂正はdata/correctionsへ置き、同じ質問の回答を置き換えます。

使い方・評価基準・教材作成と定期タスクの指示は [AUTOMATION.md](AUTOMATION.md) を参照してください。GitHub Actionsの週次実行は待機データの処理であり、教材生成にはChatGPT側の定期タスクが別途必要です。

## 検証

```
python -m unittest discover -s tests -v
python -m scripts.evaluate
```

evaluation/cases.jsonlの固定評価と新規例の確認問題で比較します。未知の知識への一般化や、歴史的正確性を全面的に保証するテストではありません。
