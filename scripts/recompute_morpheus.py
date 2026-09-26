"""Aggregate the fixed MORPHEUS per-video measurements and review decisions."""
import argparse
import json
from pathlib import Path
from recorded_metrics import morpheus_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error('Use a new output directory')
    results = morpheus_scores()
    args.out_dir.mkdir(parents=True)
    output = {'status':'PASS', 'scope':'saved_measurements_and_screening_decisions', 'models':results}
    (args.out_dir / 'RESULT.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
