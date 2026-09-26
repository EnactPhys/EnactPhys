"""Recompute architecture-ablation aggregates from recorded measurements."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('Sim PC', 'Sim OC', 'Real PC', 'Real OC')


def compute(row):
    c = row['sim_pc_components']
    pairs = row['real_oc_E_pairs']
    if len(pairs) != 24 or len({p['task_id'] for p in pairs}) != 24:
        raise ValueError('Expected 24 unique Real elasticity comparisons')
    scores = []
    for pair in pairs:
        target = float(pair['target_ADE'] or 0)
        leak = float(pair['max_non_target_ADE'] or 0)
        valid = (float(pair['signed_height_delta_px'] or 'nan') > 5
                 and target > 0.1692766811711215
                 and pair['old_status'] == 'measured'
                 and int(pair['valid_frames']) >= 2)
        score = target / (target + leak) if valid else 0
        if not math.isclose(score, float(pair['score']), abs_tol=1e-10):
            raise ValueError(f'Measurement mismatch: {pair["task_id"]}')
        scores.append(score)
    if row['real_pc_n'] != 708:
        raise ValueError('Expected 708 Real PC comparisons')
    if 'real_pc_by_seed' in row:
        parts = row['real_pc_by_seed']
        if sum(p['other_success'] + p['s03_success'] for p in parts) != row['real_pc_success']:
            raise ValueError('Real PC successes disagree with per-seed records')
        if sum(p['other_n'] + p['s03_n'] for p in parts) != row['real_pc_n']:
            raise ValueError('Real PC comparison count disagrees with per-seed records')
    return (
        100 * (513*c['magnitude'] + 513*c['direction'] + 101*c['invariance']) / 1127,
        100 * statistics.mean(row['sim_oc'].values()),
        100 * row['real_pc_success'] / row['real_pc_n'],
        (row['real_oc_F'] + row['real_oc_G'] + 100*statistics.mean(scores)) / 3,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error('--out-dir must be a new directory')
    rows = json.loads((ROOT / 'results/ablation_inputs.json').read_text())
    if len(rows) != 7 or len({r['model'] for r in rows}) != 7:
        raise ValueError('Expected seven unique ablation rows')
    tex = (ROOT / 'results/architecture_ablation.tex').read_text()
    tex = re.sub(r'\\(?:textbf|underline)\{([^{}]*)\}', r'\1', tex)
    checks = []
    for row in rows:
        match = re.search(re.escape(row['display_name']) + r'\s*&\s*(.*?)\\\\', tex, re.S)
        if not match:
            raise ValueError(f'Missing table row: {row["display_name"]}')
        cells = match.group(1).split('&')
        if len(cells) != 6:
            raise ValueError('Expected two structure columns and four metric columns')
        reference = [float(c.strip()) for c in cells[2:]]
        for metric, value, expected in zip(METRICS, compute(row), reference):
            checks.append(dict(model=row['model'], metric=metric, computed=value,
                               displayed=f'{value:.2f}', paper=f'{expected:.2f}',
                               matches=f'{value:.2f}' == f'{expected:.2f}'))
    failed = [c for c in checks if not c['matches']]
    summary = dict(status='FAIL' if failed else 'PASS', cells=len(checks),
                   matched=len(checks)-len(failed), mismatches=failed,
                   scope='Recorded measurement aggregation; no new generation or pixel scoring.')
    args.out_dir.mkdir(parents=True)
    with (args.out_dir / 'ablation.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    (args.out_dir / 'RESULT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
