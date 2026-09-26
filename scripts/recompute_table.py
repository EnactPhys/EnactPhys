"""Recompute table aggregates from released measurements and judgments."""
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from recorded_metrics import physicsiq_scores, morpheus_scores

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['Sim VQA', 'Sim PP', 'Sim PC', 'Sim OC', 'Real VQA', 'Real PP', 'Real PC', 'Real OC', 'Physics-IQ', 'MORPHEUS D', 'MORPHEUS I']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    inputs = json.loads((ROOT / 'results/main_table_inputs.json').read_text())
    tex = (ROOT / 'results/main_benchmark_table.tex').read_text()
    tex = re.sub(r'\\(?:textbf|underline)\{([^{}]*)\}', r'\1', tex)
    checks = []
    iq = physicsiq_scores()
    morpheus = morpheus_scores()
    for row in inputs:
        pp = [json.loads(line) for line in (ROOT / f'results/sim_pp/{row["model"]}.jsonl').read_text().splitlines()]
        if len(pp) != 215 or len({x['task_id'] for x in pp}) != 215:
            raise ValueError('Expected 215 unique simulation judgments')
        if any(x['status'] not in ['VALID', 'VALID_MANUAL'] or x['label'] not in ['PASS', 'FAIL'] for x in pp):
            raise ValueError('Invalid released judgment')
        c = row['sim_pc_components']
        es = []
        for pair in row['real_oc_E_pairs']:
            target = float(pair['target_ADE'] or 0)
            leak = float(pair['max_non_target_ADE'] or 0)
            valid = (float(pair['signed_height_delta_px'] or 'nan') > 5 and target > 0.1692766811711215
                     and pair['old_status'] == 'measured' and int(pair['valid_frames']) >= 2)
            score = target / (target + leak) if valid else 0
            if not math.isclose(score, float(pair['score']), abs_tol=1e-10):
                raise ValueError(f'Measurement mismatch: {pair["task_id"]}')
            es.append(score)
        if len(es) != 24:
            raise ValueError('Expected 24 Real elasticity pairs')
        values = [row['sim_vqa'], 100 * sum(x['label'] == 'PASS' for x in pp) / len(pp),
                  100 * (513*c['magnitude'] + 513*c['direction'] + 101*c['invariance']) / 1127,
                  100 * statistics.mean(row['sim_oc'].values()), row['real_vqa'],
                  100 * row['real_pp_pass'] / row['real_pp_n'],
                  100 * row['real_pc_success'] / row['real_pc_n'],
                  (row['real_oc_F'] + row['real_oc_G'] + 100*statistics.mean(es)) / 3,
                  iq[row['model']]['verified_view'], morpheus[row['model']]['D'], morpheus[row['model']]['I']]
        match = re.search(r'&\s*' + re.escape(row['display_name']) + r'\s*&\s*(.*?)\\\\', tex, re.S)
        if not match:
            raise ValueError('Missing manuscript row')
        cells = match.group(1).split('&')
        pairs = [list(map(float, re.findall(r'\d+\.\d+', cell))) for cell in cells[:4]]
        if len(cells) != 7 or any(len(pair) != 2 for pair in pairs):
            raise ValueError('Expected four Sim/Real pairs and three benchmark columns')
        expected = [pair[0] for pair in pairs] + [pair[1] for pair in pairs] + [float(x.strip()) for x in cells[4:]]
        if len(expected) != len(values):
            raise ValueError('Metric count differs')
        for metric, actual, reference in zip(METRICS, values, expected):
            digits = 4 if metric.startswith('MORPHEUS') else 2
            checks.append(dict(model=row['model'], metric=metric, computed=actual, displayed=f'{actual:.{digits}f}',
                               paper=f'{reference:.{digits}f}', matches=f'{actual:.{digits}f}' == f'{reference:.{digits}f}'))
    args.out_dir.mkdir(parents=True)
    with (args.out_dir / 'main_table.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    failed = [x for x in checks if not x['matches']]
    summary = dict(status='FAIL' if failed else 'PASS', cells=len(checks), matched=len(checks)-len(failed),
                   scope='Reaggregation of saved measurements and judgments; no new generation or pixel scoring.', mismatches=failed)
    (args.out_dir / 'RESULT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
