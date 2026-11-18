from pathlib import Path
import tempfile
import unittest

from scripts.quality import write_rows
from scripts.train_candidate import conversation_rehearsal


class ConversationReplayTests(unittest.TestCase):
    def test_archived_membership_never_reintroduces_old_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root / 'added/daily_conversation_1.jsonl', [{'user': 'おはよう', 'assistant': '訂正前の回答です。'}])
            current = [{'user': 'おはよう', 'assistant': '現在の回答です。'}]
            self.assertEqual(conversation_rehearsal(root, current), current)

    def test_uses_both_source_names_without_selecting_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = [{'user': 'おはよう', 'assistant': 'おはようございます。'},
                       {'user': 'ありがとう', 'assistant': 'お役に立てて何よりです。'},
                       {'user': '科挙とは', 'assistant': '試験です。'}]
            write_rows(root / 'added/daily_conversation_1.jsonl', current[:1])
            write_rows(root / 'added/01_日常会話.jsonl', current[1:2])
            write_rows(root / 'added/history.jsonl', current[2:])
            self.assertEqual(conversation_rehearsal(root, current), current[:2])

    def test_missing_archives_do_not_add_any_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(conversation_rehearsal(directory, []), [])


if __name__ == '__main__':
    unittest.main()
