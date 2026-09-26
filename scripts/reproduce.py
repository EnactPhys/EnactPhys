"""Run a selected recorded-measurement aggregation workflow."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "ablation-recorded": ("scripts/recompute_ablation.py", "Architecture-ablation measurements, seven variants"),
    "main-table-recorded": ("scripts/recompute_table.py", "Current main-table measurements and judgments"),
    "physdelta-recorded": ("scripts/recompute_table.py", "Current main-table measurements and judgments (compatibility alias)"),
    "physicsiq-recorded": ("physicsiq/reproduce.py", "Current Solid Mechanics per-video scores"),
    "mechanisms-recorded": ("scripts/recompute_mechanisms.py", "Recorded object-state measurements"),
    "morpheus-recorded": ("scripts/recompute_morpheus.py", "MORPHEUS per-video measurements and review decisions"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List available workflows.")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--out-dir", type=Path, help="New directory for computed outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    args = parser.parse_args()
    if args.list:
        print(json.dumps({name: description for name, (_, description) in TASKS.items()}, indent=2))
        return
    if args.task is None or args.out_dir is None:
        parser.error("--task and --out-dir are required unless --list is specified")
    output = args.out_dir.resolve()
    if output.exists():
        parser.error("--out-dir must be a new directory")
    script, _ = TASKS[args.task]
    command = [sys.executable, str(ROOT / script), "--out-dir", str(output)]
    if args.dry_run:
        print(json.dumps({"task": args.task, "scope": "recorded_measurement_aggregation",
                          "command": command}, indent=2))
        return
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
