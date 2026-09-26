"""Join fixed anonymous screening decisions with MORPHEUS measurements."""
import argparse
import csv
import json
import math
from pathlib import Path


def truth(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() not in ('true', 'false'):
        raise ValueError(f'Expected a boolean, received {value!r}')
    return str(value).lower() == 'true'


def finalize(rows, decisions):
    if not rows or len({r['blind_id'] for r in rows}) != len(rows):
        raise ValueError('Expected nonempty, unique anonymous IDs')
    if set(decisions) != {r['blind_id'] for r in rows}:
        raise ValueError('Decisions must cover exactly the input anonymous IDs')
    if len({(r['model'], r['task_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate model/task identity')
    result = []
    for source in rows:
        row = dict(source)
        decision = decisions[row['blind_id']]
        if decision.get('verdict') not in ('pass', 'reject', 'uncertain'):
            raise ValueError(f'Invalid verdict: {row["blind_id"]}')
        if not all(decision.get(k) for k in ('reason', 'evidence', 'frames_reviewed')):
            raise ValueError(f'Missing review evidence: {row["blind_id"]}')
        low = truth(row['low_motion_candidate'])
        if low and decision['frames_reviewed'] != '0-48':
            raise ValueError(f'Low-motion candidates require all 49 frames: {row["blind_id"]}')
        if truth(decision.get('low_motion_confirmed', False)):
            if not low or decision['verdict'] != 'reject':
                raise ValueError('Confirmed low motion must be a rejected low-motion candidate')
        row.update(visual_verdict=decision['verdict'], visual_reason=decision['reason'],
                   visual_evidence=decision['evidence'], frames_reviewed=decision['frames_reviewed'],
                   confirmed_reject=truth(row['trajectory_reject']) or decision['verdict'] == 'reject')
        row['uncertain_retained'] = decision['verdict'] == 'uncertain' and not row['confirmed_reject']
        for metric in ('D', 'I'):
            raw = float(row[metric + '_raw'])
            if not math.isfinite(raw):
                raise ValueError(f'Nonfinite {metric}: {row["blind_id"]}')
            row[metric + '_raw'] = raw
            row[metric] = 0.0 if row['confirmed_reject'] else raw
            row[metric + '_uncertain_zero'] = 0.0 if row['confirmed_reject'] or row['uncertain_retained'] else raw
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Scored CSV joined to the private anonymous-ID map.')
    parser.add_argument('--decisions', type=Path, required=True, help='Frozen JSON decisions keyed by anonymous ID.')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error('--out-dir must be a new directory')
    with args.input.open(newline='') as f:
        rows = list(csv.DictReader(f))
    result = finalize(rows, json.loads(args.decisions.read_text()))
    args.out_dir.mkdir(parents=True)
    with (args.out_dir / 'per_video.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    summary = dict(status='PASS', inputs=len(rows), outputs=len(result),
                   confirmed_reject=sum(r['confirmed_reject'] for r in result),
                   uncertain_retained=sum(r['uncertain_retained'] for r in result))
    (args.out_dir / 'RESULT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
