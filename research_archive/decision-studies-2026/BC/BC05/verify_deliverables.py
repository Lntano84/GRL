"""Independent table-level checks for the saved BC05 deliverables (no replay)."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = BASE/'outputs'
ARMS = ['BASE-CONTROL', 'BASE-CF', 'HIGH-CONTROL', 'HIGH-CF']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    manifest = json.loads((WORK/'frozen_manifest.json').read_text(encoding='utf-8'))
    assert sha(WORK/'frozen_targets.json') == manifest['targets_sha256']
    for name, digest in manifest['source_sha256'].items():
        assert sha(Path(name)) == digest
    summary_rows = read_csv(OUT/'BC05_four_arm_summary.csv')
    window_rows = read_csv(OUT/'BC05_all_windows.csv')
    target_rows = read_csv(OUT/'BC05_target_events.csv')
    intervention_costs = read_csv(OUT/'BC05_intervention_request_costs.csv')
    peak_costs = read_csv(OUT/'BC05_peak_request_costs.csv')
    assert [r['arm'] for r in summary_rows] == ARMS
    assert len(window_rows) == 4032 and len(target_rows) == 5956
    summaries = {r['arm']: r for r in summary_rows}
    audit = json.loads((OUT/'BC05_audit.json').read_text(encoding='utf-8'))
    assert audit['protocol_passed'] and audit['run_status']['status'] == 'completed'
    assert audit['run_status']['wall_seconds'] <= audit['run_status']['budget_seconds'] == 3600
    assert audit['run_status']['completed'] == ARMS
    for arm in ARMS:
        rows = [r for r in window_rows if r['arm'] == arm]
        assert len(rows) == 1008
        eval_rows = [r for r in rows if r['phase'] == 'evaluation']
        assert len(eval_rows) == 864
        peak = max(eval_rows, key=lambda r: float(r['total_dt_pct']))
        actual_peak = float(summaries[arm]['evaluation_peak_dt_pct'])
        assert abs(float(peak['total_dt_pct'])-actual_peak) <= 1e-10
        assert int(peak['window_index']) == int(summaries[arm]['peak_window'])
        dur = math.fsum(float(r['window_duration_s']) for r in eval_rows)
        service = math.fsum((float(r['demand_service_time_s'])+
                             float(r['prefetch_extra_service_time_s'])+
                             float(r['put_service_time_s'])) for r in eval_rows)
        avg = service/(.036*dur)*100
        assert abs(avg-float(summaries[arm]['evaluation_average_dt_pct_time_weighted'])) <= 1e-10
        writes = sum(int(r['write_chunks']) for r in eval_rows)
        pfreads = sum(int(r['prefetch_read_chunks']) for r in eval_rows)
        assert writes == int(summaries[arm]['evaluation_write_chunks'])
        assert pfreads == int(summaries[arm]['evaluation_prefetch_read_chunks'])
        events = [r for r in target_rows if r['arm'] == arm]
        assert len(events) == 1489
        assert all(r['evicted_before_corresponding_peak_request'] in ('True', 'False') for r in events)
        for cpath in (intervention_costs, peak_costs):
            arm_costs = [r for r in cpath if r['arm'] == arm]
            assert all(abs(float(r['demand_service_time_s'])+
                           float(r['extra_prefetch_service_time_s'])-
                           float(r['total_get_service_time_s'])) <= 1e-12 for r in arm_costs)
    base_peak = float(summaries['BASE-CONTROL']['evaluation_peak_dt_pct'])
    high_cf_peak = float(summaries['HIGH-CF']['evaluation_peak_dt_pct'])
    gain = (base_peak-high_cf_peak)/base_peak
    assert abs(gain-float(summaries['HIGH-CF']['peak_gain_vs_BASE'])) <= 1e-12
    assert gain >= .05 and summaries['HIGH-CF']['resource_eligible'] == 'True'
    assert summaries['HIGH-CF']['peak_window'] == '578'
    assert int(summaries['HIGH-CF']['evaluation_write_chunks']) <= 148731
    assert int(summaries['HIGH-CF']['max_cache_entries_observed']) <= 3002
    assert int(summaries['BASE-CF']['evaluation_write_chunks']) > 148731
    counts = {arm: dict(Counter(r['status'] for r in target_rows if r['arm'] == arm)) for arm in ARMS}
    assert counts['BASE-CONTROL'] == {'parent_reject_unchanged': 1489}
    assert counts['HIGH-CONTROL'] == {'parent_reject_unchanged': 1489}
    assert counts['BASE-CF'] == {'overridden': 1489}
    assert counts['HIGH-CF'] == {'overridden': 1489}
    result = {'passed': True, 'checks': ['frozen_inputs_and_target_hashes', 'all_four_arm_csv_row_counts',
             'peak_and_duration_weighted_average_recomputed_from_windows', 'write_and_prefetch_totals_recomputed',
             'request_service_cost_sums', 'counterfactual_gate_and_target_statuses'],
              'high_cf_peak_gain_recomputed': gain, 'high_cf_writes': int(summaries['HIGH-CF']['evaluation_write_chunks']),
              'base_cf_writes': int(summaries['BASE-CF']['evaluation_write_chunks']),
              'high_cf_peak_dt': high_cf_peak, 'peak_window': 578}
    (WORK/'deliverable_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
