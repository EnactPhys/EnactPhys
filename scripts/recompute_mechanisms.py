"""Recompute mechanism-figure statistics from released numerical measurements."""
import argparse
import csv
import json
import math
import statistics as s
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results/mechanisms'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    if a.out_dir.exists():
        raise FileExistsError(a.out_dir)
    with (ROOT / 'state_probe.csv').open() as f:
        probe = [r for r in csv.DictReader(f) if r['split'] == 'test']
    ys = [float(r['measured']) for r in probe]
    predictions = [float(r['readout']) for r in probe]
    mean = s.mean(ys)
    r2 = 1 - sum((v-y)**2 for v, y in zip(predictions, ys)) / sum((y-mean)**2 for y in ys)
    mae = s.mean(abs(v-y) for v, y in zip(predictions, ys))
    if len(probe) != 5238 or not math.isclose(r2, 0.8106705230862501, abs_tol=1e-10) or not math.isclose(mae, 1.4764765674895026, abs_tol=1e-10):
        raise ValueError('State-probe statistics differ')
    edits = json.loads((ROOT / 'state_edit.json').read_text())
    reductions = {}
    for key in ['state', 'trajectory']:
        inputs = list(edits[key]['per_input'].values())
        reduction = 1 - s.mean(x['edited_mse'] for x in inputs) / s.mean(x['baseline_mse'] for x in inputs)
        if len(inputs) != 42 or not math.isclose(reduction, edits[key]['error_reduction'], abs_tol=1e-10):
            raise ValueError('State-edit statistics differ')
        reductions[key] = dict(inputs=len(inputs), error_reduction=reduction)
    with (ROOT / 'message_intervention.csv').open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 408:
        raise ValueError('Expected 408 message-intervention measurements')
    curves = []
    for family, direction in [('drop', 1), ('collision', 1), ('ballwall', 1), ('ramp', -1)]:
        for variant in ['original', 'removed']:
            selected = [r for r in rows if r['family'] == family and r['variant'] == variant]
            xs = sorted({float(r['axis_value']) for r in selected})
            seeds = sorted({int(r['seed']) for r in selected})
            values = {(int(r['seed']), float(r['axis_value'])): float(r['outcome_px']) for r in selected}
            if len(xs) != 17 or len(seeds) != 3 or len(values) != 51:
                raise ValueError('Incomplete message-intervention sweep')
            displayed_x = [x for x in xs if family != 'ballwall' or x <= 0.60]
            medians = [s.median(direction * (values[seed, x] - values[seed, xs[0]]) for seed in seeds) for x in displayed_x]
            curves.append(dict(family=family, variant=variant, x=displayed_x, median=medians))
    result = dict(status='PASS', scope='Recomputation from recorded numerical measurements; no new training, generation or tracking.',
                  state_probe=dict(test_points=len(probe), r2=r2, mae=mae), state_edit=reductions,
                  message_intervention=dict(measurements=len(rows), curves=curves))
    a.out_dir.mkdir(parents=True)
    (a.out_dir / 'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'message_intervention'}, indent=2))


if __name__ == '__main__':
    main()
