"""Strict JSONL validation and reproducible evaluation; no external model/API."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def normalized(text):
    return re.sub(r'[\s、。・＝=!?！？「」『』]', '', unicodedata.normalize('NFKC', text))


def read_rows(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if set(row) != {'user', 'assistant'} or any(not isinstance(v, str) or not v.strip() for v in row.values()):
                raise ValueError('user/assistant must be nonempty strings')
            if any('\n' in v or '\r' in v for v in row.values()):
                raise ValueError('embedded newline')
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f'{path}:{number}: {exc}') from exc
        rows.append(row)
    return rows


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def validate(rows, ctx=96, heldout=(), strict=True):
    seen = set()
    for row in rows:
        u, a = row['user'], row['assistant']
        q = normalized(u)
        if q in seen:
            raise ValueError(f'duplicate question: {u}')
        if q in heldout:
            raise ValueError(f'held-out question in training: {u}')
        seen.add(q)
        if len(u) > 25 or len(a) > 40 or len(f'U:{u}\nA:{a}\n') > ctx + 1:
            raise ValueError(f'length/context overflow: {u}')
        if strict and (re.search(r'[?？A-Za-z０-９()（）]', a) or not a.endswith('です。') or a.count('。') != 1):
            raise ValueError(f'answer format: {u}')
    return rows


def merge(old, incoming):
    """Normalize question spellings; latest correction wins, preserving latest text."""
    result = {normalized(r['user']): r for r in old}
    for row in incoming:
        result[normalized(row['user'])] = row
    return list(result.values())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def greedy(model, user, max_len=41):
    import numpy as np
    chars = list(f'U:{user}\nA:')
    if any(c not in model.stoi for c in chars):
        return ''  # Never silently drop unknown characters during evaluation.
    ids = model.encode(''.join(chars))
    out = []
    for _ in range(max_len):
        logits, _ = model.forward(np.array([(ids + out)[-model.ctx:]]))
        nxt = int(np.argmax(logits[0, -1]))
        if model.chars[nxt] == '\n':
            break
        out.append(nxt)
    return model.decode(out)


def evaluate(model, cases):
    from difflib import SequenceMatcher
    details = []
    for c in cases:
        actual = greedy(model, c['user'])
        expected = c['assistant']
        details.append({**c, 'actual': actual, 'exact': actual == expected,
                        'digits_correct': re.findall(r'\d+', actual) == re.findall(r'\d+', expected),
                        'format_ok': bool(actual) and len(actual) <= 40 and actual.endswith('。') and not re.search('[?？]', actual) and (c['group'] != 'new_training_probe' or (actual.endswith('です。') and actual.count('。') == 1)),
                        'similarity': SequenceMatcher(None, actual, expected).ratio()})
    groups = {}
    for group in sorted({c['group'] for c in cases}):
        ds = [d for d in details if d['group'] == group]
        groups[group] = {'count': len(ds), **{k: sum(d[k] for d in ds) / len(ds) for k in ('exact', 'digits_correct', 'format_ok', 'similarity')}}
    return {'groups': groups, 'details': details}


def decide(before, after):
    reasons = []
    old = {d['user']: d for d in before['details']}
    for d in after['details']:
        prev = old[d['user']]
        if prev['exact'] and not d['exact']:
            reasons.append(f'previous correct answer lost: {d["user"]}')
    for group, scores in after['groups'].items():
        for metric in ('exact', 'digits_correct', 'format_ok'):
            if scores[metric] + 1e-12 < before['groups'][group][metric]:
                reasons.append(f'{group}: {metric} regressed')
        if scores['format_ok'] < .9:
            reasons.append(f'{group}: format_ok below 90%')
    # New material must actually be learned; memorization probes are labeled as such.
    if 'new_training_probe' in after['groups'] and after['groups']['new_training_probe']['exact'] < .5:
        reasons.append('new training probe exact match below 50%')
    improved = any(scores['exact'] > before['groups'][g]['exact'] + 1e-12 for g, scores in after['groups'].items())
    if not improved:
        reasons.append('no exact-match improvement')
    return {'accepted': not reasons, 'reasons': reasons}
