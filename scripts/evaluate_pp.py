"""Evaluate PP using the fixed 16-frame prompt and SAM2 trajectory rules."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.pp_rules import classify, parse_answer, trajectory_audit


def call_api(row, args, prompt):
    encoded = base64.b64encode((args.out_dir / 'grids' / (row['id'] + '.jpg')).read_bytes()).decode()
    payload = dict(model=args.model, temperature=0, messages=[dict(role='user', content=[
        dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + encoded)),
        dict(type='text', text=prompt)])])
    url = os.environ['EVAL_API_BASE_URL'].rstrip('/') + '/chat/completions'
    for attempt in range(3):
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), method='POST',
                  headers={'Authorization': 'Bearer ' + os.environ['EVAL_API_KEY'], 'Content-Type': 'application/json'})
        response = None
        try:
            with urllib.request.urlopen(request, timeout=300) as handle:
                response = json.load(handle)
            answer = parse_answer(response)
            label = str(answer.get('label', '')).strip().upper()
            if label not in {'PASS', 'FAIL', 'UNCERTAIN'}:
                raise ValueError('Invalid VLM label')
            record = dict(ok=True, pred_label=label, response=response, requested_model=args.model)
        except urllib.error.HTTPError as exc:
            record = dict(ok=False, error_type='HTTPError', http_status=exc.code)
        except Exception as exc:
            record = dict(ok=False, error_type=type(exc).__name__)
            if response is not None:
                record['response'] = response
        (args.out_dir / 'responses' / f'{row["id"]}.attempt{attempt+1}.json').write_text(json.dumps(record, indent=2) + '\n')
        if record['ok']:
            break
        if attempt < 2:
            time.sleep(30*(attempt+1))
    (args.out_dir / 'responses' / f'{row["id"]}.json').write_text(json.dumps(record, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='JSONL: id, model, task_id, video_path, trajectory_path.')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--model', default='qwen3-vl-235b-a22b-instruct')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--responses-dir', type=Path, help='Aggregate existing response JSONs without API calls or frame extraction.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error('--out-dir must be a new directory')
    if args.workers < 1:
        parser.error('--workers must be positive')
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    required = {'id', 'model', 'task_id', 'video_path', 'trajectory_path'}
    if not rows or any(not required.issubset(row) for row in rows):
        parser.error('Manifest lacks required fields')
    if len({r['id'] for r in rows}) != len(rows) or len({(r['model'], r['task_id']) for r in rows}) != len(rows):
        parser.error('Duplicate ID or model/task identity')
    for row in rows:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', row['id']):
            parser.error('IDs must be safe filenames')
        for field in ('video_path', 'trajectory_path'):
            path = Path(row[field])
            if not path.is_absolute():
                path = args.manifest.resolve().parent / path
            row[field] = str(path)
            if (field == 'trajectory_path' or not args.responses_dir) and not path.is_file():
                raise FileNotFoundError(path)
    if args.dry_run:
        print(json.dumps(dict(videos=len(rows), model=args.model, mode='recorded_responses' if args.responses_dir else 'video_and_api', api_calls_started=False)))
        return
    if not args.responses_dir and not all(os.environ.get(k) for k in ('EVAL_API_BASE_URL', 'EVAL_API_KEY')):
        parser.error('Set EVAL_API_BASE_URL (including /v1) and EVAL_API_KEY')
    args.out_dir.mkdir(parents=True)
    manifest = args.out_dir / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    if not args.responses_dir:
        subprocess.run([sys.executable, str(ROOT / 'evaluation/pp_grid.py'), '--manifest', str(manifest),
                        '--out-dir', str(args.out_dir / 'grids')], check=True)
        (args.out_dir / 'responses').mkdir()
        prompt = (ROOT / 'docs/pp_prompt.txt').read_text().rstrip('\n')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda row: call_api(row, args, prompt), rows))
    response_root = args.responses_dir or args.out_dir / 'responses'
    results = []
    for row in rows:
        try:
            raw = json.loads((response_root / f'{row["id"]}.json').read_text())
            if not raw.get('ok'):
                raise ValueError('No valid evaluator response')
            answer = parse_answer(raw['response'])
            track = trajectory_audit(row['trajectory_path'])
            result = dict(row, status='VALID', label=classify(answer, track),
                          vlm_label=answer['label'], issue_type=answer['issue_type'],
                          evidence=answer.get('evidence', ''), track=track,
                          decision_source='vlm_and_trajectory_rules')
        except Exception as exc:
            result = dict(row, status='ERROR', error_type=type(exc).__name__)
        results.append(result)
    summaries = {}
    for model in sorted({r['model'] for r in rows}):
        subset = [r for r in results if r['model'] == model]
        valid = [r for r in subset if r['status'] == 'VALID']
        passed = sum(r['label'] == 'PASS' for r in valid)
        errors = len(subset)-len(valid)
        summaries[model] = dict(n=len(subset), valid=len(valid), errors=errors, passed=passed,
                               score=passed/len(valid) if valid else None,
                               full_denominator_interval=[passed/len(subset), (passed+errors)/len(subset)],
                               status='PARTIAL' if errors else 'COMPLETE')
    summary = dict(status='COMPLETE' if all(s['status'] == 'COMPLETE' for s in summaries.values()) else 'PARTIAL', per_model=summaries)
    (args.out_dir / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
    (args.out_dir / 'RESULT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if summary['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
