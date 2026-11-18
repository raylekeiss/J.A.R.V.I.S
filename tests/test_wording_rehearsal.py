import unittest

from scripts.train_candidate import wording_rehearsal
from scripts.quality import normalized


class WordingRehearsalTests(unittest.TestCase):
    def test_uses_training_answers_and_excludes_every_evaluation_question(self):
        base = [{'user': '科挙とは', 'assistant': '役人を選ぶ試験です。'}]
        cases = [{'user': '科挙を短く教えて', 'assistant': '評価だけの秘密の回答です。'}]
        result = wording_rehearsal(base, cases)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(r['assistant'] == base[0]['assistant'] for r in result))
        self.assertNotIn(normalized(cases[0]['user']), {normalized(r['user']) for r in result})

    def test_deduplicates_normalized_questions_and_obeys_context_limit(self):
        base = [{'user': '科挙とは', 'assistant': '試験です。'},
                {'user': '科挙の説明をして', 'assistant': '試験です。'},
                {'user': '長' * 23 + 'とは', 'assistant': '説明です。'}]
        result = wording_rehearsal(base, [])
        self.assertEqual(len(result), len({normalized(r['user']) for r in result}))
        self.assertNotIn('科挙の説明をして', {r['user'] for r in result})
        self.assertTrue(all(len(r['user']) <= 25 for r in result))
        self.assertEqual(result, wording_rehearsal(base, []))

    def test_does_not_rephrase_conversation_or_take_facts_from_cases(self):
        base = [{'user': 'おはよう', 'assistant': 'おはようございます。'}]
        cases = [{'user': '未知の用語とは', 'assistant': '評価専用の説明です。'}]
        self.assertEqual(wording_rehearsal(base, cases), [])

    def test_corrected_canonical_answer_is_never_reintroduced_by_rehearsal(self):
        base = [{'user': '科挙とは', 'assistant': '古い説明です。'}]
        incoming = [{'user': '科挙とは', 'assistant': '訂正した説明です。'}]
        self.assertEqual(wording_rehearsal(base, [], incoming), [])


if __name__ == '__main__':
    unittest.main()
