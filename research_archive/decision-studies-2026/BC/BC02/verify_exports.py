"""Verify exported windows against frozen statistics; never re-read the trace."""
import csv
import hashlib
import json
from pathlib import Path

import compress_json
import numpy as np

base = Path(__file__).resolve().parents[2]
repo = base / 'work/bc01/Baleen-FAST24'
audit = json.loads((base / 'work/bc02/audit.json').read_text(encoding='utf-8'))
rows = list(csv.DictReader((base / 'outputs/BC02_all_windows.csv').open(encoding='utf-8-sig', newline='')))
if len(rows) != 1008 or [int(r['window_index']) for r in rows] != list(range(1008)):
    raise AssertionError('missing/duplicated/reordered exported windows')

def col(key):
    a = np.asarray([float(r[key]) for r in rows])
    if not np.isfinite(a).all():
        raise AssertionError(f'nonfinite export: {key}')
    return a

time_end = col('trace_end_s')
time_start = col('trace_start_s')
duration = col('window_duration_s')
np.testing.assert_array_equal(time_start, np.r_[0, time_end[:-1]])
np.testing.assert_array_equal(duration, time_end - time_start)
np.testing.assert_array_equal([r['phase'] for r in rows], ['train_and_warmup'] * 144 + ['evaluation'] * 864)
np.testing.assert_array_equal(col('lower_bound_dt_pct'), col('put_dt_pct') + col('first_get_dt_pct'))

for relpath, original_sha in audit['inputs_sha256'].items():
    path = repo / relpath
    if hashlib.sha256(path.read_bytes()).hexdigest() != original_sha:
        raise AssertionError('frozen raw stats changed')
    b = compress_json.load(str(path))['batches']
    np.testing.assert_array_equal(time_end, b['time_elapsed_phy'])
    np.testing.assert_array_equal(col('scan_put_service_seconds_sample'), np.diff(b['service_time_writes_stats'], prepend=0))
    np.testing.assert_array_equal(col('scan_get_nocache_service_seconds_sample'), np.diff(b['service_time_nocache_stats'], prepend=0))
    if 'baleen' in relpath:
        # Equivalent scale expressed independently: sampled seconds / sampling
        # fraction / 36 disks / physical duration, converted to percent.
        src_dt = (np.diff(b['service_time_used_stats'], prepend=0) + np.diff(b['service_time_writes_stats'], prepend=0)) / .001 / 36 / duration * 100
        np.testing.assert_allclose(col('baleen_actual_dt_pct'), src_dt, rtol=0, atol=1e-12)

old_rows = list(csv.DictReader((base / 'outputs/BC01_10min_trajectory.csv').open(encoding='utf-8-sig', newline='')))
old_baleen = [float(r['dt_total_util_pct']) for r in old_rows if r['method'] == 'Baleen']
old_error = float(np.max(np.abs(col('baleen_actual_dt_pct') - old_baleen)))
if old_error > 5.1e-10:
    raise AssertionError('Baleen reference changed from BC01 beyond CSV rounding')
if np.any(col('lower_bound_dt_pct') > col('baleen_actual_dt_pct') + 1e-8):
    raise AssertionError('export violates the per-window lower-bound condition')
seen = col('seen_get_blocks_cumulative')
if np.any(np.diff(seen) < 0) or int(seen[-1]) != audit['unique_get_blocks']:
    raise AssertionError('seen-state discontinuity')
if sum(int(r['first_get_requests']) for r in rows) != audit['unique_get_blocks']:
    raise AssertionError('first GETs not counted exactly once per block')

actual = col('baleen_actual_dt_pct')[144:]
lower = col('lower_bound_dt_pct')[144:]
peak_actual = float(actual.max())
peak_lower = float(lower.max())
gain = (peak_actual - peak_lower) / peak_actual
np.testing.assert_array_equal([peak_actual, peak_lower, gain], [audit['p_baleen_pct'], audit['p_lb_pct'], audit['upper_gain_fraction']])
if int(lower.argmax()) + 144 != audit['lb_peak_window']:
    raise AssertionError('lower-bound peak window mismatch')
print(json.dumps({'export_checks_passed': True, 'windows': len(rows), 'evaluation_windows': len(lower),
                  'bc01_csv_rounding_max_error_pp': old_error, 'p_baleen_pct': peak_actual,
                  'p_lb_pct': peak_lower, 'upper_gain_pct': gain * 100,
                  'lb_peak_window': int(lower.argmax()) + 144}, indent=2))
