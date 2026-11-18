"""Train in an isolated directory; publish only after deterministic quality checks."""
import argparse
import copy
import json
import os
from pathlib import Path
import numpy as np
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_model import TinyTransformer, build_from_pairs
from scripts.quality import digest, evaluate, decide, merge, normalized, read_rows, validate, write_json, write_rows


def promote(root, temp, files):
    names = ['model.npz', 'dialogues.jsonl', 'docs/model.json', 'docs/model.bin']
    if (temp / 'data/applied_corrections.json').exists():
        names.append('data/applied_corrections.json')
    stamp = time.strftime('%Y%m%d_%H%M%S')
    destinations = [(p, root / 'added' / f'{stamp}_{p.name}') for p in files]
    if len({dst for _, dst in destinations}) != len(destinations) or any(dst.exists() for _, dst in destinations):
        raise ValueError('archive collision; nothing published')
    (root / 'added').mkdir(exist_ok=True)
    originals = {root / name: (root / name).read_bytes() if (root / name).exists() else None for name in names}
    shutil.copy2(root / 'model.npz', root / 'reports/previous_model.npz')
    moved = []
    try:
        for name in names:
            os.replace(temp / name, root / name)
        for src, dst in destinations:
            os.replace(src, dst)
            moved.append((src, dst))
    except BaseException:
        for path, content in originals.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        for src, dst in moved:
            dst.rename(src)
        raise


def run(root, steps=1500, mode='incremental', batch=24, targets=None, lr=None):
    root = Path(root)
    lr = lr if lr is not None else (1e-4 if mode == 'incremental' else 3e-3)
    if not 1 <= steps <= 16000 or not 1 <= batch <= 24 or not 1e-6 <= lr <= 3e-3:
        raise ValueError('invalid training steps, batch or learning rate')
    report_dir = root / 'reports'
    report_dir.mkdir(exist_ok=True)
    # Separate immutable evaluation references from data being merged.
    cases = [json.loads(l) for l in (root / 'evaluation/cases.jsonl').read_text().splitlines() if l.strip()]
    heldout = {normalized(c['user']) for c in cases if c['group'] == 'unseen_wording'}
    base = read_rows(root / 'dialogues.jsonl')
    validate(base, heldout=heldout, strict=False)
    files = sorted((root / 'inbox').glob('*.jsonl'))
    if (root / 'add.jsonl').exists():
        files.insert(0, root / 'add.jsonl')
    if targets is not None:
        files = []
        for target in targets:
            target = Path(target).resolve()
            target.relative_to(root.resolve())  # Keep committed inputs inside this repository.
            files.extend(sorted(target.glob('*.jsonl')) if target.is_dir() else [target])
        files = sorted(set(files))
    all_corrections = sorted((root / 'data/corrections').glob('*.jsonl'))
    ledger_path = root / 'data/applied_corrections.json'
    applied = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    # Compatibility with the first accepted run, before this ledger existed.
    previous_report = report_dir / 'latest.json'
    if not applied and previous_report.exists():
        previous = json.loads(previous_report.read_text())
        if previous.get('accepted'):
            applied.update({p: sha for p, sha in previous.get('inputs', {}).items() if p.startswith('data/corrections/')})
    corrections = [p for p in all_corrections if applied.get(str(p.relative_to(root))) != digest(p)]
    new = []
    existing = {normalized(r['user']): r for r in base}
    for p in corrections + files:
        rs = validate(read_rows(p), heldout=heldout)
        if p in files and len(rs) != 290:
            raise ValueError(f'{p}: expected 290 rows')
        for r in rs:
            prior = existing.get(normalized(r['user']))
            if prior != r:
                new.append(r)
            existing[normalized(r['user'])] = r
    new = merge([], new)
    if not new and mode != 'retrain':
        write_json(report_dir / 'latest.json', {'status': 'no_new_data', 'accepted': False})
        (report_dir / 'latest.md').write_text('新しい学習データはありません。モデルは更新していません。\n')
        return False
    combined = merge(base, new)
    validate(combined, heldout=heldout, strict=False)
    model_path = root / 'model.npz'
    if not model_path.exists():
        raise ValueError('A baseline model is required before automatic publication')
    baseline = TinyTransformer.load(model_path)
    # Restrict probes to distinct answers, sampled independently of evaluation cases.
    probes = []
    answers = set()
    for r in new:
        if r['assistant'] not in answers:
            probes.append({**r, 'group': 'new_training_probe'})
            answers.add(r['assistant'])
    probes = sorted(probes, key=lambda r: r['user'])
    if len(probes) > 20:
        probes = [probes[i * (len(probes) - 1) // 19] for i in range(20)]
    eval_cases = cases + probes
    started = time.monotonic()
    before = evaluate(baseline, eval_cases)
    # Baseline is never passed to the optimizer.
    candidate = TinyTransformer.load(model_path) if mode == 'incremental' else build_from_pairs([(r['user'], r['assistant']) for r in combined], d_model=baseline.D, n_layers=baseline.L, ctx=baseline.ctx)
    candidate.extend_vocab(f'U:{r["user"]}\nA:{r["assistant"]}\n' for r in combined)
    teacher = copy.deepcopy(candidate) if mode == 'incremental' else None
    pairs = [(r['user'], r['assistant']) for r in combined]
    replay = pairs + [(r['user'], r['assistant']) for r in new] * 3
    candidate.train(replay, steps=steps, batch=batch, lr=lr, teacher=teacher,
                    new_users={r['user'] for r in new})
    if not all(np.isfinite(p).all() for p in candidate.p.values()):
        raise ValueError('Non-finite candidate weights; published model remains untouched')
    after = evaluate(candidate, eval_cases)
    decision = decide(before, after)
    report = {**decision, 'status': 'evaluated', 'mode': mode, 'steps': steps,
              'learning_rate': lr, 'elapsed_seconds': round(time.monotonic() - started, 2),
              'retention_distillation': teacher is not None,
              'new_pairs': len(new), 'merged_pairs': len(combined),
              'baseline_sha256': digest(model_path),
              'evaluation_sha256': digest(root / 'evaluation/cases.jsonl'),
              'inputs': {str(p.relative_to(root)): digest(p) for p in corrections + files},
              'before': before, 'after': after}
    # Materialize the complete candidate before touching any published files.
    with tempfile.TemporaryDirectory(dir=root, prefix='.candidate-') as directory:
        temp = Path(directory)
        candidate.save(temp / 'model.npz')
        candidate.export_web(temp / 'docs')
        write_rows(temp / 'dialogues.jsonl', combined)
        applied.update({str(p.relative_to(root)): digest(p) for p in corrections})
        write_json(temp / 'data/applied_corrections.json', applied)
        report['candidate_sha256'] = digest(temp / 'model.npz')
        if decision['accepted']:
            promote(root, temp, files)
    write_json(report_dir / 'latest.json', report)
    write_json(report_dir / 'input_topics.json', sorted({r['user'] for r in combined}))
    summary = ['# 学習・評価レポート', '', f'採用: {decision["accepted"]}', f'追加: {len(new)}組 / 学習: {steps}ステップ', '', '| 評価区分 | 学習前の完全一致 | 学習後の完全一致 |', '|---|---:|---:|']
    for g, scores in after['groups'].items():
        summary.append(f'| {g} | {before["groups"][g]["exact"]:.1%} | {scores["exact"]:.1%} |')
    summary += ['', '新規問題は学習内容の確認であり、未知問題の正答率ではありません。', 'unseen_wording は質問文未学習、同じ用語の知識は学習済みの場合があります。']
    summary += ['', *decision['reasons']]
    (report_dir / 'latest.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print(json.dumps({**decision, 'new_pairs': len(new), 'elapsed_seconds': report['elapsed_seconds']}, ensure_ascii=False))
    return decision['accepted']


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--mode', choices=['incremental', 'retrain'], default='incremental')
    p.add_argument('--batch', type=int, default=24)
    p.add_argument('--lr', type=float)
    a = p.parse_args()
    if not 1 <= a.steps <= 16000 or not 1 <= a.batch <= 24:
        p.error('steps must be 1..16000, batch 1..24')
    try:
        run(a.root, a.steps, a.mode, a.batch, lr=a.lr)
    except Exception as exc:
        write_json(a.root / 'reports/latest.json', {'status': 'error', 'accepted': False, 'error': str(exc)})
        (a.root / 'reports/latest.md').write_text('学習は失敗しました。現モデルは採用していません。\n' + str(exc) + '\n')
        raise
