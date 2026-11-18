"""
start_chat.py ― ダブルクリックで起動する用

  1. そのまま起動（学習済みモデルを読み込む。初回だけ学習する）
  2. add.jsonl を追加学習してから起動
  3. dialogues.jsonl で最初から学習し直してから起動
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def pause(msg="Enterキーで閉じます"):
    try:
        input(msg)
    except EOFError:
        pass


def main():
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("NumPyが入っていないので、インストールします…")
        subprocess.call([sys.executable, "-m", "pip", "install", "numpy"])

    print("==== ちいさな世界史AI ====")
    print("  1: そのまま起動")
    print("  2: add.jsonl を追加学習してから起動")
    print("  3: dialogues.jsonl で最初から学習し直してから起動")
    choice = input("番号を入力してEnter（何も入れずにEnterなら1）: ").strip() or "1"

    cmd = [sys.executable, os.path.join(HERE, "chat_server.py")]
    if choice == "2":
        if not os.path.exists(os.path.join(HERE, "add.jsonl")):
            print("add.jsonl が見つかりません。同じフォルダに置いてください。")
            return pause()
        if not os.path.exists(os.path.join(HERE, "model.npz")):
            print("まだ学習済みモデルがありません。先に 1 か 3 で学習してください。")
            return pause()
        cmd += ["--add", "add.jsonl"]
    elif choice == "3":
        cmd += ["--retrain"]
    elif choice != "1":
        print("1〜3で選んでください。")
        return pause()

    code = subprocess.call(cmd)
    if code not in (0, None):
        print("\nエラーで止まりました。上のメッセージを確認してください。")
    pause()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
