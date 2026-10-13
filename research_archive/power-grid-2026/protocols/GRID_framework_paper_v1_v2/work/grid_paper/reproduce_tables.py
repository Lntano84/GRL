"""Reproduce the paper's summary table using only public small JSON receipts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists(), 'Never replace a prior reproduction.'
    root = a.source_root.resolve()
    design = json.loads((root / 'outputs/grid62/design.json').read_text(encoding='utf-8'))
    finished = json.loads((root / 'outputs/grid62/finished.json').read_text(encoding='utf-8'))
    review = json.loads((root / 'outputs/grid62/review.json').read_text(encoding='utf-8'))
    assert finished['passed_engineering'] and review['passed_saved_data']
    ix = {(r['repeat'], r['week'], r['method']): r for r in finished['summaries']}
    avg = lambda values: math.fsum(values) / len(values) if values else None
    result = []
    for method in [m['method'] for m in design['methods']]:
        times, queries, costs, completed, quality = [], [], [], 0, 0
        for week in design['weeks']:
            rs = [ix[repeat, week, method] for repeat in range(design['repeats'])]
            fs = [ix[repeat, week, 'FULL'] for repeat in range(design['repeats'])]
            equal = all(r['steps'] == f['steps'] and r['complete'] == f['complete'] for r, f in zip(rs, fs))
            done = all(r['complete'] for r in rs)
            completed += done
            survive = all(r['complete'] if f['complete'] else r['steps'] >= f['steps'] for r, f in zip(rs, fs))
            delta = (avg([r['cost'] for r in rs]) - avg([f['cost'] for f in fs])) / abs(avg([f['cost'] for f in fs])) if done and all(f['complete'] for f in fs) else None
            quality += survive and (delta is None or delta <= design['quality_threshold'])
            if delta is not None:
                costs.append(delta)
            if equal:
                times.append(1 - avg([r['controller_s'] for r in rs]) / avg([f['controller_s'] for f in fs]))
                denominator = avg([f['n1_forecasts'] for f in fs])
                if denominator:
                    queries.append(1 - avg([r['n1_forecasts'] for r in rs]) / denominator)
        row = {'method': method, 'completed_families': completed, 'quality_families': quality,
            'macro_control_reduction_full': avg(times), 'macro_query_reduction_full': avg(queries),
            'worst_complete_cost_delta_full': max(costs) if costs else None}
        reference = next(x for x in review['summaries'] if x['method'] == method)
        for field, value in row.items():
            if isinstance(value, float):
                assert abs(value - reference[field]) <= 1e-12, (method, field)
            else:
                assert value == reference[field], (method, field)
        result.append(row)
    a.out.mkdir(parents=True)
    with (a.out / 'main_table.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    sources = []
    for name in ('design.json', 'finished.json', 'review.json'):
        path = root / 'outputs/grid62' / name
        sources.append({'path': 'outputs/grid62/' + name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (a.out / 'reproduction.json').write_text(json.dumps({'passed': True, 'rows': result, 'sources': sources,
        'scope': 'Independent aggregate re-computation from saved summary receipts; no raw-vector or AC re-solve.'}, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'methods': len(result)}))


if __name__ == '__main__':
    main()
