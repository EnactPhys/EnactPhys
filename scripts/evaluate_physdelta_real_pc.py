"""Score matched Real parameter-control pairs from newly extracted trajectories."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.real_parameter_control import response


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--tracking-dir", type=Path, required=True)
    p.add_argument("--model-name", default="enactphys")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    if Path(args.model_name).name != args.model_name:
        p.error("Invalid model name")
    with (args.dataset / "real/parameter_control_pairs.csv").open(newline="") as f:
        pairs = [r for r in csv.DictReader(f) if args.seed is None or int(r["seed"]) == args.seed]
    if not pairs:
        p.error("No matching pairs")
    expected = 236 if args.seed is not None else 708
    if len(pairs) != expected:
        raise ValueError("Expected %d fixed comparisons, got %d" % (expected, len(pairs)))
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cache, details = {}, []
    for pair in pairs:
        values = []
        for key in ["low_task", "high_task"]:
            task = pair[key]
            if Path(task).name != task or Path(pair["group_id"]).name != pair["group_id"]:
                raise ValueError("Invalid task or group ID")
            identity = (task, pair["scene"], int(pair["object_index"]))
            if identity not in cache:
                path = args.tracking_dir / "trajectories" / args.model_name / pair["group_id"] / (task + ".json")
                cache[identity] = response(path, pair["scene"], int(pair["object_index"]))
            values.append(cache[identity])
        observed = all(v is not None for v in values)
        delta = float(pair["direction"]) * (values[1] - values[0]) if observed else None
        success = int(observed and delta > float(pair["threshold_pixels"]))
        details.append(dict(pair, low_response=values[0], high_response=values[1],
                            signed_delta=delta, observed=observed, success=success))
    groups = defaultdict(list)
    for row in details:
        groups[row["seed"]].append(row)
    def summarize(rows):
        return dict(comparisons=len(rows), successes=sum(x["success"] for x in rows),
                    missing_readouts=sum(not x["observed"] for x in rows),
                    score=100 * sum(x["success"] for x in rows) / len(rows))
    result = dict(model=args.model_name, **summarize(details),
                  by_seed={k:summarize(v) for k,v in sorted(groups.items())})
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader(); writer.writerows(details)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
