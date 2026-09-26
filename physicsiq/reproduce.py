#!/usr/bin/env python3
"""Aggregate the current Solid Mechanics records across three seeds."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from recorded_metrics import physicsiq_scores, SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--from-metrics', action='store_true',
                        help='Also recompute all baseline scores from their supplied metric CSVs.')
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error('Use a new output directory')
    models = physicsiq_scores()
    if args.from_metrics:
        sys.path.insert(0, str(ROOT / 'physicsiq/vendor'))
        from physiq.calculate_iq_score_stable import IQTable
        for model, result in models.items():
            if model == 'h2':
                continue
            per_seed = []
            for seed in SEEDS:
                path = ROOT / 'physicsiq/data/baselines' / model / f'metrics_seed{seed}.csv'
                score = 100 * float(IQTable.from_csv(str(path)).get_output_dict()['final_score_view'])
                reference = next(r['verified_view'] for r in result['per_seed'] if int(r['seed']) == seed)
                if not math.isclose(score, reference, abs_tol=1e-8, rel_tol=0):
                    raise ValueError(f'Metric CSV mismatch: {model}, {seed}: {score} != {reference}')
                per_seed.append({'seed':seed, 'verified_view':score})
            result.update(per_seed=per_seed, verified_view=statistics.mean(r['verified_view'] for r in per_seed))
    score = models['h2']['verified_view']
    output = {'status':'PASS', 'scope':'saved_video_score_aggregation', 'views':114, 'videos':342,
              'seeds':list(SEEDS), 'verified_view':score, 'displayed_score':f'{score:.2f}',
              'baseline_metric_csvs_recomputed':args.from_metrics, 'models':models}
    args.out_dir.mkdir(parents=True)
    (args.out_dir / 'RESULT.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
