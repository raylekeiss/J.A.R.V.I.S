"""Turn source-backed, agent-written facts into 290-row batches without an API.

This checks quotation locations and schema, not the historical truth of an answer.
"""
import argparse
import json
from pathlib import Path

from scripts.quality import digest, normalized, read_rows, validate, write_json, write_rows

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ['{}とは', '{}って何', '{}について教えて', '{}を説明して', '{}の意味を教えて', '{}を簡単に説明して', '{}を短くまとめて', '{}の基本を教えて', '{}の要点を教えて', '世界史でいう{}って何', '{}の説明をお願い', '{}を一言でいうと']


def prepare(root, facts_path, source_path, prefix):
    root, facts_path, source_path = map(Path, (root, facts_path, source_path))
    if not prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in prefix):
        raise ValueError('prefix must contain only letters, digits, underscore or dash')
    facts = [json.loads(l) for l in facts_path.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    if not facts or len(facts) % 29 or len(facts) > 290:
        raise ValueError('Supply 29 facts per file, up to 290 facts / 10 files')
    source = source_path.read_text(encoding='utf-8-sig')
    used_topics = set(json.loads((root / 'data/sources/used_topics.json').read_text()))
    used_questions = set()
    for p in [root / 'dialogues.jsonl', *sorted((root / 'inbox').glob('*.jsonl'))]:
        used_questions.update(normalized(r['user']) for r in read_rows(p))
    for c in [json.loads(l) for l in (root / 'evaluation/cases.jsonl').read_text().splitlines() if l.strip()]:
        used_questions.add(normalized(c['user']))
    for p in (root / 'data/provenance').glob('*.json'):
        used_topics.update(normalized(f['topic']) for f in json.loads(p.read_text())['facts'])
    records, provenance = [], []
    for fact in facts:
        topic, answer = fact['topic'], fact['answer']
        quote = fact['source_excerpt']
        if not all(isinstance(v, str) and v.strip() for v in (topic, answer, quote)):
            raise ValueError('topic, answer and source_excerpt must be nonempty strings')
        if normalized(topic) in used_topics:
            raise ValueError(f'already used topic: {topic}')
        if len(quote) < 8 or quote not in source:
            raise ValueError(f'exact source excerpt missing: {topic}')
        used_topics.add(normalized(topic))
        questions = []
        for pattern in PATTERNS:
            q = pattern.format(topic)
            if len(q) <= 25 and normalized(q) not in used_questions:
                questions.append(q)
                used_questions.add(normalized(q))
            if len(questions) == 10:
                break
        if len(questions) != 10:
            raise ValueError(f'cannot produce ten new short questions: {topic}')
        records.extend({'user': q, 'assistant': answer} for q in questions)
        provenance.append({**fact, 'source_line': source[:source.index(quote)].count('\n') + 1,
                           'evidence_status': 'quoted_source_requires_semantic_review'})
    validate(records)
    destinations = [root / 'inbox' / f'{prefix}_{i+1:02}_290.jsonl' for i in range(len(facts) // 29)]
    manifest = root / 'data/provenance' / f'{prefix}.json'
    if manifest.exists() or any(p.exists() for p in destinations):
        raise ValueError('batch already exists; do not overwrite it')
    for i, dest in enumerate(destinations):
        write_rows(dest, records[i*290:(i+1)*290])
    write_json(manifest, {'source_name': source_path.name, 'source_sha256': digest(source_path),
                         'facts': provenance, 'files': [str(p.relative_to(root)) for p in destinations]})
    return destinations


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--facts', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--prefix', required=True)
    a = p.parse_args()
    for path in prepare(ROOT, a.facts, a.source, a.prefix):
        print(path)
