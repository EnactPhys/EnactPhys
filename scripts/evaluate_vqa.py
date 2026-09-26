"""Score videos from an explicit CSV manifest with FAST-VQA or FasterVQA."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--fast-repo', type=Path, required=True)
    parser.add_argument('--scorer', choices=('FAST-VQA', 'FasterVQA'), default='FasterVQA')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error('--out-dir must be a new directory')
    with args.manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or not {'model', 'task_id', 'video'}.issubset(rows[0]):
        parser.error('Manifest requires model, task_id and video columns')
    if len({(r['model'], r['task_id']) for r in rows}) != len(rows):
        parser.error('Duplicate model/task identity')
    for row in rows:
        video = Path(row['video'])
        if not video.is_absolute():
            video = args.manifest.resolve().parent / video
        if not video.is_file():
            raise FileNotFoundError(video)
        row['video'] = str(video)
    config = args.fast_repo / ('options/fast/f3dvqa-b.yml' if args.scorer == 'FasterVQA' else 'options/fast/fast-b.yml')
    if not config.is_file():
        raise FileNotFoundError(config)
    if args.dry_run:
        print(json.dumps(dict(videos=len(rows), scorer=args.scorer, config=str(config), gpu_started=False)))
        return
    sys.path.insert(0, str(ROOT))
    import torch
    from evaluation.vqa import FastScorer, task_seed
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    scorer = FastScorer(args.scorer, args.fast_repo)
    args.out_dir.mkdir(parents=True)
    groups, errors = {}, []
    with (args.out_dir / 'per_video.jsonl').open('w') as f:
        for row in rows:
            try:
                value, raw = scorer(row['video'], task_seed(row['task_id']))
                if not math.isfinite(value) or not math.isfinite(raw):
                    raise ValueError('Nonfinite evaluator output')
                result = dict(row, status='VALID', score=value, raw_score=raw, sampling_seed=task_seed(row['task_id']))
                groups.setdefault(row['model'], []).append(value)
            except Exception as exc:
                result = dict(row, status='ERROR', error_type=type(exc).__name__)
                errors.append(result)
            f.write(json.dumps(result) + '\n')
            f.flush()
    summary = dict(status='PARTIAL' if errors else 'COMPLETE', scorer=args.scorer,
                   expected=len(rows), errors=len(errors),
                   per_model={model: dict(n=len(values), mean=statistics.mean(values),
                                          percent=100*statistics.mean(values)) for model, values in groups.items()})
    (args.out_dir / 'RESULT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
