"""Summarize durable episode records without treating infrastructure errors as failures."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time


def summarize(root):
    plan = json.loads((root/'plan.json').read_text())
    rows = []
    for job in plan['jobs']:
        folder = root/'jobs'/job['track']/job['task']
        records = [json.loads(p.read_text()) for p in folder.glob('episode-*.json')]
        errors = len(list(folder.glob('error-*.json')))
        ended = (folder/'job-result.json').exists()
        count = len(records)
        successes = sum(bool(r['success']) for r in records)
        rows.append({'track': job['track'], 'task': job['task'], 'expected': job['count'],
                     'completed': count, 'successes': successes,
                     'success_rate': successes/count if count else None,
                     'infrastructure_errors': errors, 'missing': job['count']-count,
                     'job_ended': ended,
                     'mean_episode_seconds': statistics.mean(r['elapsed_seconds'] for r in records) if count else None})
    tracks = []
    for name in sorted({r['track'] for r in rows}):
        group = [r for r in rows if r['track'] == name]
        rates = [r['success_rate'] for r in group if r['success_rate'] is not None]
        tracks.append({'track': name, 'tasks': len(group), 'tasks_with_results': len(rates),
                       'completed': sum(r['completed'] for r in group),
                       'expected': sum(r['expected'] for r in group),
                       'macro_success_rate': statistics.mean(rates) if rates else None})
    result = {'time': time.time(), 'run': str(root), 'complete': (root/'complete.json').exists(),
              'expected': sum(r['expected'] for r in rows), 'completed': sum(r['completed'] for r in rows),
              'successes': sum(r['successes'] for r in rows),
              'infrastructure_errors': sum(r['infrastructure_errors'] for r in rows),
              'jobs': rows, 'tracks': tracks}
    (root/'summary.json').write_text(json.dumps(result, indent=2))
    with (root/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run), indent=2))
