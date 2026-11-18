import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.quality import decide, greedy, merge, normalized, read_rows, validate, write_json, write_rows
from scripts.train_candidate import run
from tiny_model import build_from_pairs


class DataTests(unittest.TestCase):
    def test_reject_malformed_record_without_partial_read(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.jsonl'
            p.write_text('{"user":"質問","assistant":"回答です。"}\n{"user":3,"assistant":"回答です。"}\n')
            with self.assertRaises(ValueError):
                read_rows(p)

    def test_reject_overflow_followup_and_holdout(self):
        for row in [{'user': '長' * 26, 'assistant': '説明です。'},
                    {'user': '質問', 'assistant': '元気ですか？'}]:
            with self.assertRaises(ValueError):
                validate([row])
        with self.assertRaises(ValueError):
            validate([{'user': '科挙とは', 'assistant': '試験です。'}], heldout={normalized('科挙とは')})

    def test_latest_correction_replaces_spelling_variant(self):
        old = [{'user': 'マンサ・ムーサとは', 'assistant': '古い回答です。'}]
        new = [{'user': 'マンサムーサとは', 'assistant': '新しい回答です。'}]
        self.assertEqual(merge(old, new), new)
        with self.assertRaises(ValueError):
            validate(old + new)

    def test_eval_is_deterministic_and_unknown_is_not_silently_dropped(self):
        pairs = [('こんにちは', 'ご挨拶です。')]
        model = build_from_pairs(pairs, d_model=8, n_layers=1)
        self.assertEqual(greedy(model, 'こんにちは'), greedy(model, 'こんにちは'))
        self.assertEqual(greedy(model, '未登録文字'), '')

    def test_rejected_candidate_leaves_all_published_files_and_inbox_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for directory in ['docs', 'inbox', 'added', 'evaluation', 'data/corrections']:
                (root / directory).mkdir(parents=True)
            base = [{'user': '挨拶', 'assistant': 'ご挨拶です。'}]
            write_rows(root / 'dialogues.jsonl', base)
            model = build_from_pairs([('挨拶', 'ご挨拶です。')], d_model=8, n_layers=1)
            model.save(root / 'model.npz')
            model.export_web(root / 'docs')
            write_rows(root / 'inbox/new.jsonl', [{'user': f'項目{i}', 'assistant': '追加の説明です。'} for i in range(290)])
            write_json(root / 'unused.json', {})
            (root / 'evaluation/cases.jsonl').write_text(json.dumps({**base[0], 'group': 'history_retention'}, ensure_ascii=False) + '\n')
            paths = ['model.npz', 'dialogues.jsonl', 'docs/model.json', 'docs/model.bin', 'inbox/new.jsonl']
            before = {n: (root / n).read_bytes() for n in paths}
            with patch('scripts.train_candidate.decide', return_value={'accepted': False, 'reasons': ['test rejection']}):
                self.assertFalse(run(root, steps=1, batch=1))
            self.assertEqual(before, {n: (root / n).read_bytes() for n in paths})
            self.assertEqual(list((root / 'added').iterdir()), [])
            self.assertFalse(json.loads((root / 'reports/latest.json').read_text())['accepted'])

    def test_gate_rejects_lost_answers_even_if_aggregate_improves(self):
        detail = {'user': '質問', 'exact': True}
        before = {'details': [detail], 'groups': {'history': {'exact': .5, 'digits_correct': 1, 'format_ok': 1}}}
        after = {'details': [{**detail, 'exact': False}], 'groups': {'history': {'exact': .6, 'digits_correct': 1, 'format_ok': 1}}}
        self.assertFalse(decide(before, after)['accepted'])

    def test_gate_accepts_real_improvement_with_retention(self):
        before = {'details': [{'user': '質問', 'exact': False}], 'groups': {'history': {'exact': 0, 'digits_correct': 1, 'format_ok': 1}}}
        after = {'details': [{'user': '質問', 'exact': True}], 'groups': {'history': {'exact': 1, 'digits_correct': 1, 'format_ok': 1}}}
        self.assertTrue(decide(before, after)['accepted'])

    def test_repository_data_and_holdout_separation(self):
        root = Path(__file__).resolve().parents[1]
        cases = [json.loads(l) for l in (root / 'evaluation/cases.jsonl').read_text().splitlines()]
        heldout = {normalized(c['user']) for c in cases if c['group'] == 'unseen_wording'}
        validate(read_rows(root / 'dialogues.jsonl'), heldout=heldout, strict=False)
        for p in (root / 'inbox').glob('*.jsonl'):
            rs = validate(read_rows(p), heldout=heldout)
            self.assertEqual(len(rs), 290)
        for p in (root / 'data/corrections').glob('*.jsonl'):
            validate(read_rows(p), heldout=heldout)


if __name__ == '__main__':
    unittest.main()

class BatchPreparationTests(unittest.TestCase):
    def test_source_quotes_are_required_and_duplicate_run_cannot_overwrite(self):
        from scripts.prepare_batch import prepare
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'data/sources').mkdir(parents=True)
            (root / 'inbox').mkdir()
            (root / 'evaluation').mkdir()
            write_json(root / 'data/sources/used_topics.json', [])
            write_rows(root / 'dialogues.jsonl', [])
            (root / 'evaluation/cases.jsonl').write_text('')
            source = root / 'source.txt'
            source.write_text('原資料の内容を確認するための引用です。')
            facts = root / 'facts.jsonl'
            facts.write_text(''.join(json.dumps({'topic': f'新項目{i}', 'answer': '原資料にある説明です。', 'source_excerpt': source.read_text()}, ensure_ascii=False) + '\n' for i in range(29)))
            paths = prepare(root, facts, source, 'batch_1')
            self.assertEqual(len(read_rows(paths[0])), 290)
            with self.assertRaises(ValueError):
                prepare(root, facts, source, 'batch_1')
            source.write_text('原資料には引用した記述が存在しません。')
            with self.assertRaises(ValueError):
                prepare(root, facts, source, 'batch_2')

    def test_promotion_rolls_back_on_midway_io_error(self):
        from scripts.train_candidate import promote
        import os
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            temp = root / 'candidate'
            for p in [root / 'docs', root / 'reports', root / 'inbox', temp / 'docs']:
                p.mkdir(parents=True)
            names = ['model.npz', 'dialogues.jsonl', 'docs/model.json', 'docs/model.bin']
            for name in names:
                (root / name).write_bytes(b'old')
                (temp / name).write_bytes(b'new')
            original = os.replace
            calls = []
            def broken(src, dst):
                calls.append(src)
                if len(calls) == 2:
                    raise OSError('simulated disk failure')
                return original(src, dst)
            with patch('scripts.train_candidate.os.replace', side_effect=broken):
                with self.assertRaises(OSError):
                    promote(root, temp, [])
            self.assertTrue(all((root / name).read_bytes() == b'old' for name in names))

    def test_promotion_publishes_complete_candidate_and_archives_input(self):
        from scripts.train_candidate import promote
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            temp = root / 'candidate'
            for p in [root / 'docs', root / 'reports', root / 'inbox', temp / 'docs']:
                p.mkdir(parents=True)
            for name in ['model.npz', 'dialogues.jsonl', 'docs/model.json', 'docs/model.bin']:
                (root / name).write_bytes(b'old')
                (temp / name).write_bytes(b'new')
            input_path = root / 'inbox/new.jsonl'
            input_path.write_text('input')
            promote(root, temp, [input_path])
            self.assertEqual((root / 'model.npz').read_bytes(), b'new')
            self.assertEqual((root / 'reports/previous_model.npz').read_bytes(), b'old')
            self.assertFalse(input_path.exists())
            self.assertEqual(next((root / 'added').iterdir()).read_text(), 'input')

class PagesTests(unittest.TestCase):
    def test_existing_unrelated_publishing_source_is_not_changed(self):
        from scripts.publish_pages import inspect
        with patch('scripts.publish_pages.api', return_value={'build_type': 'legacy', 'source': {'branch': 'other', 'path': '/'}, 'html_url': 'https://example.test/'}):
            with self.assertRaises(ValueError):
                inspect('owner/repo')

    def test_legacy_publish_requests_build_and_verifies_exact_commit(self):
        from scripts.publish_pages import publish_legacy
        site = {'build_type': 'legacy', 'source': {'branch': 'main', 'path': '/docs'}, 'html_url': 'https://example.test/'}
        with patch('scripts.publish_pages.api', side_effect=[site, {'status': 'queued'}, {'status': 'built', 'commit': 'new-sha'}]) as mocked:
            self.assertEqual(publish_legacy('owner/repo', 'new-sha', pause=lambda _: None), site['html_url'])
            self.assertEqual(mocked.call_args_list[1].args, ('repos/owner/repo/pages/builds', 'POST'))

    def test_unrelated_build_is_not_reported_as_our_success(self):
        from scripts.publish_pages import publish_legacy
        site = {'build_type': 'legacy', 'source': {'branch': 'main', 'path': '/docs'}, 'html_url': 'https://example.test/'}
        with patch('scripts.publish_pages.api', side_effect=[site, {}, {'status': 'built', 'commit': 'other-sha'}]):
            with self.assertRaises(TimeoutError):
                publish_legacy('owner/repo', 'new-sha', tries=1, pause=lambda _: None)
