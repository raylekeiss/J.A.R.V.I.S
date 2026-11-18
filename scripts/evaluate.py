import argparse
import json
from pathlib import Path
from tiny_model import TinyTransformer
from scripts.quality import evaluate, write_json

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('model.npz'))
    p.add_argument('--cases', type=Path, default=Path('evaluation/cases.jsonl'))
    p.add_argument('--output', type=Path, default=Path('reports/baseline.json'))
    a = p.parse_args()
    cases = [json.loads(l) for l in a.cases.read_text().splitlines() if l.strip()]
    report = evaluate(TinyTransformer.load(a.model), cases)
    write_json(a.output, report)
    print(json.dumps(report['groups'], ensure_ascii=False, indent=2))
