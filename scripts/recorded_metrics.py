"""Aggregate the released Physics-IQ and MORPHEUS measurement records."""
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (43278311, 56382197, 68491523)
MORPHEUS_MODELS = {'wan':'wan_base', 'cog':'cogvideo', 'h3':'h3', 'phantom':'phantom',
                  'textonly':'textonly', 'controlnet':'controlnet', 'force':'force_prompting',
                  'official_phyco':'official_phyco', 'ours':'h2'}


def physicsiq_scores():
    data = ROOT / 'physicsiq/data'
    points = json.loads((data / 'per_video_scores.json').read_text())
    tasks = list(csv.DictReader((data / 'generation_manifest.csv').open()))
    keys = {(r['benchmark_id'], int(r['seed'])) for r in points}
    ids = {r['benchmark_id'] for r in points}
    if len(points) != 342 or len(ids) != 114 or keys != {(bid,s) for bid in ids for s in SEEDS}:
        raise ValueError('Expected 114 views and all three seeds')
    task_map = {(r['benchmark_id'], int(r['seed'])): r for r in tasks}
    if len(tasks) != 342 or set(task_map) != keys:
        raise ValueError('Generation manifest coverage differs from score records')
    for point in points:
        task = task_map[(point['benchmark_id'], int(point['seed']))]
        if task['task_id'] != point['task_id'] or not math.isfinite(float(point['score'])):
            raise ValueError('Task identity or score is invalid')
    route_counts = {name:len({r['benchmark_id'] for r in tasks if r['model_route']==name})
                    for name in ('H2_ADAPTER','WAN_BASE')}
    if route_counts != {'H2_ADAPTER':80,'WAN_BASE':34}:
        raise ValueError('Unexpected route coverage')
    by_model = json.loads((data / 'baseline_scores.json').read_text())
    by_model['h2'] = {'per_seed':[{'seed':s,'verified_view':statistics.mean(
        float(p['score']) for p in points if int(p['seed'])==s)} for s in SEEDS]}
    for model, result in by_model.items():
        if len(result['per_seed']) != 3 or {int(p['seed']) for p in result['per_seed']} != set(SEEDS):
            raise ValueError(f'Incomplete seed coverage: {model}')
        result['verified_view'] = statistics.mean(float(p['verified_view']) for p in result['per_seed'])
    expected = json.loads((data / 'expected_result.json').read_text())
    if not math.isclose(by_model['h2']['verified_view'], expected['verified_view'], abs_tol=1e-9, rel_tol=0):
        raise ValueError('Physics-IQ reference comparison failed')
    return by_model


def morpheus_scores():
    rows = list(csv.DictReader((ROOT / 'results/morpheus/per_video.csv').open()))
    models = set(MORPHEUS_MODELS)
    cases = {r['case'] for r in rows}
    seeds = {937,5318,1888}
    keys = {(r['model'],r['case'],int(r['seed'])) for r in rows}
    if len(rows) != 432 or len(cases) != 16 or keys != {(m,c,s) for m in models for c in cases for s in seeds}:
        raise ValueError('Expected nine models, sixteen cases and three seeds')
    for r in rows:
        reject = r['confirmed_reject'].lower() == 'true'
        for metric in ('D','I'):
            value, raw = float(r[metric]), float(r[metric+'_raw'])
            if not math.isfinite(value) or not math.isfinite(raw):
                raise ValueError('Non-finite MORPHEUS measurement')
            if not math.isclose(value, 0.0 if reject else raw, abs_tol=1e-12, rel_tol=0):
                raise ValueError('Screening decision and recorded score disagree')
    output = {}
    for model, target in MORPHEUS_MODELS.items():
        group = [r for r in rows if r['model'] == model]
        output[target] = {'n':len(group), **{k:statistics.mean(float(r[k]) for r in group)
                         for k in ('D','I','D_raw','I_raw','D_uncertain_zero','I_uncertain_zero')}}
    return output
