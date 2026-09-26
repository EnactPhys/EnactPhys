"""Run the frozen Physics-IQ configurations using explicit local asset roots."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route', choices=['adapter', 'base'], required=True)
    p.add_argument('--assets-root', type=Path, required=True)
    p.add_argument('--base-model', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--benchmark-id', help='Optional exact benchmark view ID for a single-view example.')
    p.add_argument('--seed', type=int, help='Optional seed from the frozen manifest.')
    p.add_argument('--dry-run', action='store_true', help='Validate task selection and print its configuration without loading models.')
    args = p.parse_args()
    if args.route == 'adapter' and args.checkpoint is None:
        p.error('--checkpoint is required for adapter generation')
    filename = 'adapter_on_240.csv' if args.route == 'adapter' else 'adapter_off_102.csv'
    with (ROOT / 'physicsiq/data' / filename).open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if (not args.benchmark_id or r['benchmark_id'] == args.benchmark_id)
                and (args.seed is None or int(r['seed']) == args.seed)]
    if not rows:
        p.error('No tasks match the requested route, view, and seed')
    for row in rows:
        for key in ['video', 'source_video', 'condition_path']:
            if row.get(key):
                row[key] = str((args.assets_root / row[key]).resolve())
                if not args.dry_run and not Path(row[key]).is_file():
                    raise FileNotFoundError(row[key])
    if args.dry_run:
        print(json.dumps({'mode': 'configuration_only', 'route': args.route, 'tasks': len(rows),
                          'sampling': [{k: r[k] for k in ['task_id', 'seed', 'num_inference_steps', 'cfg_scale', 'fps']}
                                       for r in rows]}, indent=2))
        return
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / 'generation_manifest.csv'
    with manifest.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    entry = 'generate_adapter.py' if args.route == 'adapter' else 'generate_base.py'
    cmd = [sys.executable, str(ROOT / 'scripts' / entry), '--manifest', str(manifest),
           '--model-root', str(args.base_model.resolve()), '--output-root', str(output / 'videos'),
           '--shard-index', '0', '--shard-count', '1']
    if args.route == 'adapter':
        cmd += ['--checkpoint', str(args.checkpoint.resolve()), '--architecture', 'h2_mass', '--mass-encoder', 'mlp']
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'runtime/inference'))
    subprocess.run(cmd, env=env, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
