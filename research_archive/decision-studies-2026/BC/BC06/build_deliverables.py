"""Offline sparse BC05 linkage and descriptive summaries after BC06 freeze."""
import csv
import gzip
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
BASE = WORK.parents[1]
ARM = WORK / 'HIGH-CONTROL'
OUT = BASE / 'outputs'
MANIFEST_PATH = WORK / 'snapshot_manifest.json'
LINK_PATH = OUT / 'BC06_BC05_label_links.csv'
LABEL_AUDIT_PATH = WORK / 'target_link_audit.json'

DESC_FEATURES = [
    'prior_block_get_count', 'seconds_since_block_last_get',
    'prior_chunk_get_count', 'seconds_since_chunk_last_get',
    'prior_actual_rejection_count', 'seconds_since_latest_actual_rejection',
    'current_request_chunk_count', 'cache_occupancy_fraction_at_request_start',
    'evaluation_write_budget_fraction_before_request',
    'last_completed_total_get_dt_pct', 'last_completed_demand_read_dt_pct',
    'last_completed_extra_prefetch_dt_pct', 'last_completed_put_dt_pct',
    'last_completed_writes', 'last_completed_evictions',
    'last_six_mean_total_get_dt_pct', 'last_six_max_total_get_dt_pct',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def canon(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def bool_csv(value):
    return str(value).strip().lower() == 'true'


def stream_jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise AssertionError(f'invalid snapshot JSON line {line_no} in {path}') from e


class GroupSummary:
    def __init__(self, name, n_features):
        self.name = name
        self.n_features = n_features
        self.n = 0
        self.missing = [0] * n_features
        self.value_n = [0] * n_features
        self.sums = [0.0] * n_features
        self.mins = [math.inf] * n_features
        self.maxs = [-math.inf] * n_features
        self.by_request = {}
        self.by_block = {}
        self.request_events = Counter()
        self.block_events = Counter()

    def _update_unit(self, table, key, values, event_counts):
        row = table.get(key)
        if row is None:
            row = [0.0] * self.n_features + [0] * self.n_features
            table[key] = row
        event_counts[key] += 1
        n = self.n_features
        for j, value in enumerate(values):
            if value is None:
                continue
            value = float(value)
            row[j] += value
            row[n + j] += 1

    def add(self, request_id, block_id, values):
        self.n += 1
        for j, value in enumerate(values):
            if value is None:
                self.missing[j] += 1
            else:
                value = float(value)
                self.value_n[j] += 1
                self.sums[j] += value
                self.mins[j] = min(self.mins[j], value)
                self.maxs[j] = max(self.maxs[j], value)
        self._update_unit(self.by_request, request_id, values, self.request_events)
        self._update_unit(self.by_block, block_id, values, self.block_events)

    def event_rows(self):
        out = []
        for j, feature in enumerate(DESC_FEATURES):
            n = self.value_n[j]
            out.append({
                'population': self.name, 'aggregation_unit': 'candidate_chunk_event',
                'feature': feature, 'candidate_events': self.n,
                'unique_units': self.n, 'feature_nonmissing_units': n,
                'feature_missing_events': self.missing[j],
                'mean_of_unit_values': self.sums[j] / n if n else None,
                'median_of_unit_values': None, 'p25_of_unit_values': None,
                'p75_of_unit_values': None,
                'min_of_unit_values': self.mins[j] if n else None,
                'max_of_unit_values': self.maxs[j] if n else None,
                'independence_note': 'candidate/chunk events are clustered; descriptive only',
            })
        return out

    def _unit_rows(self, table, event_counts, unit_name):
        means = [[] for _ in DESC_FEATURES]
        missing_units = [0] * self.n_features
        for key, row in table.items():
            for j in range(self.n_features):
                count = row[self.n_features + j]
                if count:
                    means[j].append(row[j] / count)
                else:
                    missing_units[j] += 1
        candidate_events = sum(event_counts.values())
        unique_units = len(table)
        out = []
        for j, feature in enumerate(DESC_FEATURES):
            values = means[j]
            if values:
                arr = np.asarray(values, dtype=float)
                p25, median, p75 = np.percentile(arr, [25, 50, 75]).tolist()
                mean = float(arr.mean())
                mn, mx = float(arr.min()), float(arr.max())
            else:
                p25 = median = p75 = mean = mn = mx = None
            out.append({
                'population': self.name, 'aggregation_unit': unit_name,
                'feature': feature, 'candidate_events': candidate_events,
                'unique_units': unique_units, 'feature_nonmissing_units': len(values),
                'feature_missing_events': self.missing[j],
                'mean_of_unit_values': mean, 'median_of_unit_values': median,
                'p25_of_unit_values': p25, 'p75_of_unit_values': p75,
                'min_of_unit_values': mn, 'max_of_unit_values': mx,
                'independence_note': 'unit means summarize within request/block clusters; descriptive only',
            })
        return out

    def all_rows(self):
        return self.event_rows() + self._unit_rows(self.by_request, self.request_events, 'request') + \
            self._unit_rows(self.by_block, self.block_events, 'block')


def context_features(context):
    last = context['last_completed_window']
    six = context['last_six_completed_windows']
    totals = [float(x['total_get_dt_pct']) for x in six]
    vals = {
        'current_request_chunk_count': context['current_request_chunk_count'],
        'cache_occupancy_fraction_at_request_start': context['cache_occupancy_fraction_at_request_start'],
        'evaluation_write_budget_fraction_before_request': context['evaluation_write_budget_fraction_before_request'],
        'last_completed_total_get_dt_pct': None if last is None else last['total_get_dt_pct'],
        'last_completed_demand_read_dt_pct': None if last is None else last['demand_read_dt_pct'],
        'last_completed_extra_prefetch_dt_pct': None if last is None else last['extra_prefetch_dt_pct'],
        'last_completed_put_dt_pct': None if last is None else last['put_dt_pct'],
        'last_completed_writes': None if last is None else last['writes'],
        'last_completed_evictions': None if last is None else last['evictions'],
        'last_six_mean_total_get_dt_pct': None if not totals else float(np.mean(totals)),
        'last_six_max_total_get_dt_pct': None if not totals else float(np.max(totals)),
    }
    return vals


def history_features(row):
    h = row['legal_key_history_at_decision_request_start']
    return {**h}


def main():
    # Verify byte identity before consulting any future outcome file.
    frozen = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    if not frozen.get('snapshot_frozen_before_BC05_target_link'):
        raise AssertionError('BC06 input snapshot is not sealed')
    for name, info in frozen['files'].items():
        path = ARM / name
        if sha(path) != info['sha256']:
            raise AssertionError(('sealed snapshot changed before linkage', name))

    # Only after the immutable snapshot check do we open the BC05 label sidecars.
    target_path = OUT / 'BC05_frozen_targets.json'
    targets = json.loads(target_path.read_text(encoding='utf-8'))
    if len(targets) != 1489:
        raise AssertionError(('BC05 frozen list size', len(targets)))
    key_to_target = {}
    for t in targets:
        ident = (int(t['reject_request_sequence']), canon(t['full_block_key']), int(t['chunk']))
        if ident in key_to_target:
            raise AssertionError(('duplicate target event identity', ident))
        key_to_target[ident] = t

    event_csv = OUT / 'BC05_target_events.csv'
    with event_csv.open(encoding='utf-8-sig', newline='') as f:
        all_outcomes = list(csv.DictReader(f))
    high_control = {int(r['event_id']): r for r in all_outcomes if r['arm'] == 'HIGH-CONTROL'}
    high_cf = {int(r['event_id']): r for r in all_outcomes if r['arm'] == 'HIGH-CF'}
    if len(high_control) != 1489 or len(high_cf) != 1489:
        raise AssertionError(('BC05 per-arm target outcomes incomplete', len(high_control), len(high_cf)))
    if any(r['status'] != 'parent_reject_unchanged' for r in high_control.values()):
        raise AssertionError('HIGH-CONTROL target source outcomes differ from BC05 audit')

    contexts = {}
    for row in stream_jsonl(ARM / 'get_request_contexts.jsonl.gz'):
        contexts[int(row['request_sequence'])] = context_features(row)
    if len(contexts) != 127305:
        raise AssertionError(('context map count', len(contexts)))

    group_names = [
        'all_actual_rejections_full_trace', 'BC05_frozen_members',
        'BC05_HIGH_CF_write_success', 'BC05_HIGH_CF_hit_at_corresponding_peak',
        'BC05_HIGH_CF_evicted_before_corresponding_peak',
    ]
    groups = {name: GroupSummary(name, len(DESC_FEATURES)) for name in group_names}
    linked = {}
    candidate_counts = Counter()
    target_decision_counts = Counter()
    target_request_ids = set()
    target_block_ids = set()
    reject_request_ids = set()
    reject_block_ids = set()
    block_to_target_hits = set()
    for row in stream_jsonl(ARM / 'candidate_decisions.jsonl.gz'):
        candidate_counts['actual_candidates'] += 1
        accepted = bool(row['author_decision_accept'])
        candidate_counts['accepted' if accepted else 'rejected'] += 1
        seq = int(row['actual_decision_request_sequence'])
        block_token = canon(row['full_block_key'])
        block_chunk_key = (block_token, int(row['chunk_id']))
        identity = (seq, block_token, int(row['chunk_id']))
        target = key_to_target.get(identity)
        outcome = None
        if target is not None:
            event_id = int(target['event_id'])
            if event_id in linked:
                raise AssertionError(('target event matched more than one candidate', event_id))
            hc = high_control[event_id]
            cf = high_cf[event_id]
            outcome = {
                'candidate_id': row['candidate_id'],
                'bc05_event_id': event_id,
                'reject_request_sequence': seq,
                'full_block_key_json': canon(row['full_block_key']),
                'chunk': int(row['chunk_id']),
                'high_control_candidate_found': True,
                'high_control_original_decision_accept': accepted,
                'high_control_model_score': row['author_model_score'],
                'high_control_threshold': row['threshold'],
                'high_control_decision_matches_score': accepted == (float(row['author_model_score']) > float(row['threshold'])),
                'bc05_high_cf_intervention_changed_reject_to_accept': bool_csv(cf['intervention_changed_reject_to_accept']),
                'bc05_high_cf_write_success': bool_csv(cf['write_success']),
                'bc05_high_cf_evicted_after_write': bool_csv(cf['evicted_after_write']),
                'bc05_high_cf_evicted_before_corresponding_peak': bool_csv(cf['evicted_before_corresponding_peak_request']),
                'bc05_high_cf_corresponding_peak_chunk_missing': bool_csv(cf['corresponding_peak_chunk_missing']),
                'bc05_high_cf_corresponding_peak_hit': not bool_csv(cf['corresponding_peak_chunk_missing']),
                'bc05_outcome_interpretation': 'privileged future-selected outcome; not a ground-truth accept/reject label or net utility',
            }
            if hc['parent_actual_decision'].strip().lower() != 'false':
                raise AssertionError(('HIGH-CONTROL did not reject manifest target in BC05', event_id))
            linked[event_id] = outcome
            target_decision_counts['accept' if accepted else 'reject'] += 1
            target_request_ids.add(seq)
            target_block_ids.add(block_token)
            if outcome['bc05_high_cf_corresponding_peak_hit']:
                block_to_target_hits.add(block_token)

        feature_values = history_features(row)
        feature_values.update(contexts[seq])
        values = [feature_values.get(f) for f in DESC_FEATURES]
        if not accepted:
            groups['all_actual_rejections_full_trace'].add(seq, block_token, values)
            reject_request_ids.add(seq)
            reject_block_ids.add(block_token)
        if target is not None:
            groups['BC05_frozen_members'].add(seq, block_token, values)
            if outcome['bc05_high_cf_write_success']:
                groups['BC05_HIGH_CF_write_success'].add(seq, block_token, values)
            if outcome['bc05_high_cf_corresponding_peak_hit']:
                groups['BC05_HIGH_CF_hit_at_corresponding_peak'].add(seq, block_token, values)
            if outcome['bc05_high_cf_evicted_before_corresponding_peak']:
                groups['BC05_HIGH_CF_evicted_before_corresponding_peak'].add(seq, block_token, values)

    missing_ids = sorted(set(high_control) - set(linked))
    if missing_ids:
        raise AssertionError(('BC05 target events not linked to exact HIGH-CONTROL candidate', len(missing_ids), missing_ids[:10]))
    if len(linked) != 1489:
        raise AssertionError(('target linkage count', len(linked)))

    label_fields = [
        'candidate_id', 'bc05_event_id', 'reject_request_sequence', 'full_block_key_json', 'chunk',
        'high_control_candidate_found', 'high_control_original_decision_accept',
        'high_control_model_score', 'high_control_threshold', 'high_control_decision_matches_score',
        'bc05_high_cf_intervention_changed_reject_to_accept', 'bc05_high_cf_write_success',
        'bc05_high_cf_evicted_after_write', 'bc05_high_cf_evicted_before_corresponding_peak',
        'bc05_high_cf_corresponding_peak_chunk_missing', 'bc05_high_cf_corresponding_peak_hit',
        'bc05_outcome_interpretation',
    ]
    dump_csv(LINK_PATH, [linked[i] for i in sorted(linked)], label_fields)

    coverage = []
    def cov(name, n, requests, blocks, decision, detail):
        coverage.append({'population': name, 'candidate_or_event_count': n,
                         'unique_requests': requests, 'unique_blocks': blocks,
                         'decision_or_outcome': decision, 'interpretation': detail})
    cov('all actual HIGH-CONTROL candidates', candidate_counts['actual_candidates'], 72899, 13688,
        f"accept={candidate_counts['accepted']}; reject={candidate_counts['rejected']}",
        'actual QueueCache admission decisions only; check_only excluded')
    cov('all actual HIGH-CONTROL rejections', candidate_counts['rejected'], len(reject_request_ids), len(reject_block_ids),
        'author model returned reject', 'observed action, not a label that rejection was wrong')
    cov('BC05 frozen manifest events', len(targets), len({int(t['reject_request_sequence']) for t in targets}),
        len({canon(t['full_block_key']) for t in targets}), 'membership provenance only',
        'sparse sidecar lists members; other candidates are unlabeled, not negative')
    cov('BC05 targets matched in HIGH-CONTROL', len(linked), len(target_request_ids), len(target_block_ids),
        f"accept={target_decision_counts['accept']}; reject={target_decision_counts['reject']}",
        'exact join on request sequence, complete block key, and chunk')
    cf_written = [r for r in linked.values() if r['bc05_high_cf_write_success']]
    cf_hit = [r for r in linked.values() if r['bc05_high_cf_corresponding_peak_hit']]
    cf_evicted = [r for r in linked.values() if r['bc05_high_cf_evicted_before_corresponding_peak']]
    cov('BC05 HIGH-CF target write success', len(cf_written), len({r['reject_request_sequence'] for r in cf_written}),
        len({r['full_block_key_json'] for r in cf_written}), 'write_success=true',
        'post-intervention outcome; does not establish positive net utility')
    cov('BC05 HIGH-CF target hit at corresponding peak request', len(cf_hit), len({r['reject_request_sequence'] for r in cf_hit}),
        len({r['full_block_key_json'] for r in cf_hit}), 'corresponding_peak_chunk_missing=false',
        'peak hit outcome, still clustered and not an independent sample')
    cov('BC05 HIGH-CF target evicted before corresponding peak', len(cf_evicted), len({r['reject_request_sequence'] for r in cf_evicted}),
        len({r['full_block_key_json'] for r in cf_evicted}), 'evicted_before_corresponding_peak=true',
        'does not imply the forced write had zero global effect')
    cov('check_only model calls', 49204, None, None, 'separate file',
        'qualification predictions; not actual admission candidates or rejection history')
    dump_csv(OUT / 'BC06_coverage.csv', coverage,
             ['population', 'candidate_or_event_count', 'unique_requests', 'unique_blocks', 'decision_or_outcome', 'interpretation'])

    # Field dictionary: list every top-level field in each persisted table.
    sample_candidate = next(stream_jsonl(ARM / 'candidate_decisions.jsonl.gz'))
    sample_context = next(stream_jsonl(ARM / 'get_request_contexts.jsonl.gz'))
    sample_check = next(stream_jsonl(ARM / 'check_only_calls.jsonl.gz'))
    descriptions = {
        'candidate_id': ('monotonic instrumentation ID assigned when QueueCache.insert creates an actual candidate', 'candidate creation'),
        'full_block_key': ('complete native block identity serialized with tuple/type structure preserved', 'audit key only'),
        'chunk_id': ('chunk identifier within the complete block key', 'audit key only'),
        'chunk_key': ('full native (block, chunk) key', 'audit key only'),
        'candidate_created_request_sequence': ('trace request sequence at QueueCache.insert', 'audit/time join'),
        'candidate_created_physical_time_s': ('author physical timestamp at candidate insertion', 'candidate creation'),
        'candidate_created_trace_elapsed_s': ('physical time relative to trace start', 'candidate creation'),
        'candidate_creation_window_index': ('completed author statistics window count before this GET', 'candidate creation'),
        'candidate_source': ('demand miss from sim_cache.run_get, prefetch only with explicit prefetch metadata, otherwise unknown', 'candidate provenance'),
        'candidate_source_basis': ('code path or explicit metadata that established source', 'candidate provenance'),
        'candidate_source_metadata_field_present': ('whether metadata explicitly carried a prefetch flag', 'candidate provenance'),
        'candidate_metadata_fields_present': ('metadata key names passed to QueueCache.insert; values such as Episode are not exported', 'candidate provenance'),
        'candidate_metadata_size': ('author metadata size used by insertion path', 'candidate creation'),
        'candidate_metadata_acc_chunk_range': ('author request chunk range attached to candidate', 'candidate creation'),
        'candidate_metadata_promotion': ('author promotion flag when supplied', 'candidate provenance'),
        'candidate_metadata_at_episode_start': ('author episode-start metadata when supplied', 'candidate provenance'),
        'candidate_model_input_vector': ('feature vector stored by QueueCache.insert before scoring', 'author input; verified equal to scorer input here'),
        'candidate_request_context_sequence': ('GET context table sequence for candidate creation', 'audit join'),
        'candidate_key_history_at_request_start': ('new GET-count, gap, and actual rejection history frozen before candidate request', 'new legal features'),
        'candidate_current_request_chunk_count': ('current GET width; recorded separately from historic features', 'current request feature'),
        'decision_batch_id': ('instrumentation ID for one actual NewMLAP.batchAccept call', 'audit/execution context'),
        'decision_batch_position': ('candidate order within author batch', 'audit/execution context'),
        'decision_batch_size': ('number of actual candidates scored in this author batch', 'audit/execution context'),
        'actual_decision_request_sequence': ('trace request sequence at author scoring call', 'audit/time join'),
        'actual_decision_physical_time_s': ('physical timestamp passed to NewMLAP.batchAccept', 'decision time'),
        'actual_decision_trace_elapsed_s': ('decision physical time relative to trace start', 'decision time'),
        'actual_decision_window_index': ('author completed-window count at decision request start', 'decision time'),
        'candidate_to_decision_delay_s': ('physical delay from insert to batch scoring', 'execution timing'),
        'author_model_input_vector': ('exact ordered numeric vector passed to the author booster', 'existing author input'),
        'author_model_score': ('unmodified author booster prediction for this candidate', 'author model output'),
        'threshold': ('actual threshold used at this decision, including HIGH-CONTROL evaluation schedule', 'author decision parameter'),
        'author_decision_accept': ('author output decision; equals score > threshold', 'observed action, not an outcome label'),
        'author_decision_rule': ('author comparator used by LearnedAP.batchAccept', 'decision rule'),
        'legal_key_history_at_decision_request_start': ('new per-key GET count/gap and actual rejection count/gap frozen before this GET', 'new legal features'),
        'decision_request_context_sequence': ('GET context table sequence at the actual decision', 'audit join'),
        'decision_current_request_chunk_count': ('current GET width at the actual decision', 'current request feature'),
        'decision_kind': ('check_only qualification call; stored outside actual candidates', 'check-only provenance'),
        'request_sequence': ('trace request sequence after author pre-request checkpoint handling', 'audit key'),
        'physical_time_s': ('physical timestamp of the GET or check-only call', 'decision time'),
        'trace_elapsed_s': ('physical time relative to trace origin', 'decision time'),
        'window_index_at_request_start': ('number of fully ended author windows before this GET', 'request context'),
        'operation': ('author operation, always GET in this context table', 'current request'),
        'current_request_chunk_count': ('requested chunks in this GET', 'current request feature'),
        'current_request_chunk_start': ('first requested chunk index', 'current request feature'),
        'current_request_chunk_end_exclusive': ('exclusive end of requested chunk range', 'current request feature'),
        'cache_occupancy_chunks_at_request_start': ('cache entries at GET entry before processing this request', 'new legal feature'),
        'cache_capacity_chunks': ('frozen configured capacity, 3,002 entries', 'configuration'),
        'cache_occupancy_fraction_at_request_start': ('occupancy divided by configured capacity', 'new legal feature'),
        'evaluation_cumulative_writes_before_request': ('successful writes since evaluation start, before this GET; the cap is not enforced online', 'new legal feature'),
        'evaluation_write_cap_chunks': ('frozen post-hoc BC05 qualification cap, 148,731 writes', 'known budget reference'),
        'evaluation_write_budget_fraction_before_request': ('observed evaluation writes so far divided by that cap', 'new legal feature'),
        'last_completed_window': ('last ended 10-minute author window with total/demand/prefetch/PUT DT and write/eviction counts', 'new legal history'),
        'last_six_completed_windows': ('up to six ended windows with same load and event measures; active window excluded', 'new legal history'),
        'completed_window_count': ('number of fully ended windows before this GET', 'request context'),
        'full_block_key': ('complete native block identity for this request', 'audit key only'),
        'requested_chunks': ('all chunks in the current GET request', 'current request'),
        'author_current_request_metadata': ('op, namespace, user, pipeline, byte offset/end/size, repeat count from the current trace request', 'current request; overlaps author meta inputs'),
        'check_only': ('check-only rows never enter the actual decision table', 'check-only provenance'),
        'high_control_candidate_found': ('exact join located an actual candidate for a BC05 event', 'future sidecar provenance'),
        'bc05_outcome_interpretation': ('explicit warning that intervention outcomes are privileged future results, not ground truth', 'future sidecar provenance'),
    }
    field_rows = []
    for artifact, sample in [('candidate_decisions.jsonl.gz', sample_candidate),
                             ('get_request_contexts.jsonl.gz', sample_context),
                             ('check_only_calls.jsonl.gz', sample_check)]:
        for field in sample:
            desc, use = descriptions.get(field, (
                f'{field} recorded from the original simulator/admission call for trace audit', 'audit context'))
            model_used = 'yes: see exact ordered feature names in BC06_feature_schema.json' if field in {
                'author_model_input_vector', 'candidate_model_input_vector'} else 'no'
            field_rows.append({'artifact': artifact, 'field': field, 'source_or_update': use,
                               'author_model_input_used': model_used, 'description': desc})
    for field in label_fields:
        desc, use = descriptions.get(field, (
            f'joined after snapshot freeze from BC05 event records: {field}', 'future sidecar only'))
        field_rows.append({'artifact': 'BC06_BC05_label_links.csv (sparse sidecar)', 'field': field,
                           'source_or_update': use, 'author_model_input_used': 'no', 'description': desc})
    dump_csv(OUT / 'BC06_field_dictionary.csv', field_rows,
             ['artifact', 'field', 'source_or_update', 'author_model_input_used', 'description'])

    overlap = [
        {'feature': 'meta op / namespace / user / byte offset / end / size', 'author_input': 'exact existing inputs', 'relation': 'already present', 'interpretation': 'Stored in feat_metadata|op, feat_metadata|ns, feat_metadata|user, feat_metadata_size|start/end/size; not a new signal.'},
        {'feature': 'prior_block_get_count', 'author_input': 'feat_dynamic_b|0..5', 'relation': 'partial overlap', 'interpretation': 'Exact all-time count; the author input has six rolling hourly occurrence buckets, so recent count information overlaps but the all-time total is not a duplicate.'},
        {'feature': 'prior_chunk_get_count', 'author_input': 'feat_dynamic_c_combined|0..5', 'relation': 'partial overlap', 'interpretation': 'Exact all-time count for this chunk; the author input sums six recent bucket histories over every chunk in the current request.'},
        {'feature': 'seconds_since_block_last_get / seconds_since_chunk_last_get', 'author_input': 'none in captured 18-vector', 'relation': 'new temporal form', 'interpretation': 'DynamicFeatures has a last-access helper, but sim_features.collect_features does not include it in meta+block+chunk.'},
        {'feature': 'prior_actual_rejection_count / seconds_since_latest_actual_rejection', 'author_input': 'none', 'relation': 'new decision history', 'interpretation': 'Counts only completed actual admission rejections; check_only and current-request decisions are excluded.'},
        {'feature': 'cache occupancy at request start', 'author_input': 'none', 'relation': 'new competition context', 'interpretation': 'Observed cache entries before current GET processing.'},
        {'feature': 'last completed window writes / evictions', 'author_input': 'none', 'relation': 'new competition context', 'interpretation': 'Measured successful writes and evictions from the immediately preceding completed window.'},
        {'feature': 'evaluation cumulative writes / budget fraction', 'author_input': 'none', 'relation': 'new resource context', 'interpretation': 'Observed writes before the request against the frozen post-hoc cap; no online limiter was added.'},
        {'feature': 'last six completed windows total / demand / prefetch / PUT DT', 'author_input': 'none', 'relation': 'new load history', 'interpretation': 'Derived from completed author windows only; no current or future window value is used.'},
        {'feature': 'current request chunk count', 'author_input': 'feat_metadata_size|start/end/size', 'relation': 'partly overlapping', 'interpretation': 'Explicit chunk count is current-request metadata and is related to existing byte offset/size fields.'},
        {'feature': 'block key / chunk key / request sequence', 'author_input': 'not treated as predictor', 'relation': 'audit only', 'interpretation': 'Retained for exact joins and replay lookup; not proposed as a feature.'},
        {'feature': 'score / threshold / accept decision', 'author_input': 'author model output/action', 'relation': 'not a predictor feature', 'interpretation': 'Logged verbatim to audit behavior; not used as a future label.'},
    ]
    dump_csv(OUT / 'BC06_feature_overlap.csv', overlap,
             ['feature', 'author_input', 'relation', 'interpretation'])

    desc_rows = []
    for group in groups.values():
        desc_rows.extend(group.all_rows())
    dump_csv(OUT / 'BC06_descriptive.csv', desc_rows,
             ['population', 'aggregation_unit', 'feature', 'candidate_events', 'unique_units',
              'feature_nonmissing_units', 'feature_missing_events', 'mean_of_unit_values',
              'median_of_unit_values', 'p25_of_unit_values', 'p75_of_unit_values',
              'min_of_unit_values', 'max_of_unit_values', 'independence_note'])

    target_counts = Counter()
    for r in linked.values():
        target_counts['manifest_events'] += 1
        target_counts['high_cf_write_success'] += int(r['bc05_high_cf_write_success'])
        target_counts['high_cf_peak_hit'] += int(r['bc05_high_cf_corresponding_peak_hit'])
        target_counts['high_cf_evicted_before_peak'] += int(r['bc05_high_cf_evicted_before_corresponding_peak'])
    target_hit_blocks = {r['full_block_key_json'] for r in linked.values() if r['bc05_high_cf_corresponding_peak_hit']}
    link_audit = {
        'bc06_snapshot_manifest_sha256': sha(MANIFEST_PATH),
        'bc05_frozen_targets_sha256': sha(target_path),
        'bc05_target_events_sha256': sha(event_csv),
        'exact_key': '(reject request sequence, complete block key, chunk)',
        'target_events_expected': len(targets),
        'target_events_linked': len(linked),
        'target_requests': len(target_request_ids),
        'target_blocks': len(target_block_ids),
        'HIGH_CONTROL_decisions_on_targets': dict(target_decision_counts),
        'HIGH_CF_post_intervention_outcomes': dict(target_counts),
        'HIGH_CF_target_hit_unique_blocks': len(target_hit_blocks),
        'nonmembers_labeled_negative': False,
        'sidecar_path': str(LINK_PATH.resolve()),
        'sidecar_sha256': sha(LINK_PATH),
        'snapshot_hashes_unchanged_after_label_link': {
            name: sha(ARM / name) == info['sha256'] for name, info in frozen['files'].items()
        },
        'interpretation': 'membership and forced-branch outcomes are descriptive privileged information; not accept/reject truth and not per-item net utility',
    }
    if not all(link_audit['snapshot_hashes_unchanged_after_label_link'].values()):
        raise AssertionError('label join modified a frozen snapshot artifact')
    LABEL_AUDIT_PATH.write_text(json.dumps(link_audit, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copyfile(LABEL_AUDIT_PATH, OUT / 'BC06_BC05_label_link_audit.json')
    make_report(validation=json.loads((WORK / 'snapshot_validation.json').read_text(encoding='utf-8')),
                frozen=frozen, audit=link_audit, target_counts=target_counts,
                reject_requests=len(reject_request_ids), reject_blocks=len(reject_block_ids),
                candidate_counts=candidate_counts, target_decision_counts=target_decision_counts,
                target_requests=len(target_request_ids), target_blocks=len(target_block_ids),
                target_hit_blocks=len(target_hit_blocks), groups=groups)
    print(json.dumps({'linked_targets': len(linked), 'target_requests': len(target_request_ids),
                      'target_blocks': len(target_block_ids), 'target_outcomes': dict(target_counts),
                      'snapshot_hashes_unchanged': all(link_audit['snapshot_hashes_unchanged_after_label_link'].values()),
                      'outputs': [str(p.resolve()) for p in [LINK_PATH, OUT/'BC06_coverage.csv', OUT/'BC06_field_dictionary.csv', OUT/'BC06_feature_overlap.csv', OUT/'BC06_descriptive.csv', OUT/'BC06_report.md']]}, ensure_ascii=False, indent=2))


def make_report(validation, frozen, audit, target_counts, reject_requests, reject_blocks,
                candidate_counts, target_decision_counts, target_requests, target_blocks,
                target_hit_blocks, groups):
    out = OUT / 'BC06_report.md'
    all_rej = groups['all_actual_rejections_full_trace']
    target = groups['BC05_frozen_members']
    text = f'''# BC06：合法决策快照与 BC05 事后关联

## 结论范围

BC06 完成了一次完整的 HIGH-CONTROL Region1 回放，得到 {validation['actual_candidate_rows']:,} 个实际准入候选决策和 {validation['get_request_context_count']:,} 个 GET 请求上下文。快照保存作者模型的准确 18 维输入、分数、实际阈值及接受／拒绝结果，并追加了请求开始前冻结的历史字段。全回放成本、读写、预取及淘汰序列与 BC03 HIGH-CONTROL **逐项完全一致**；窗口特征与原始统计在 {validation['completed_windows_in_request_contexts']:,} 个对决策可见的已结束窗口上对账误差为零。

BC05 清单中的 {audit['target_events_linked']:,}/{audit['target_events_expected']:,} 个目标事件都以请求序号、完整块键、chunk 精确关联到 HIGH-CONTROL 的实际拒绝。这个关联提供了可审计的分母与事后结果，**不构成训练集**：未入清单候选没有负标签，HIGH-CF 命中或淘汰也不是单个候选的净系统收益。

## 覆盖与执行核对

| 项目 | 数值 |
|---|---:|
| 完整轨迹请求数 | {validation['trace_request_count']:,} |
| GET 上下文数 | {validation['get_request_context_count']:,} |
| 实际候选 | {validation['actual_candidate_rows']:,} |
| 接受 / 拒绝 | {validation['actual_accepted_rows']:,} / {validation['actual_rejected_rows']:,} |
| 需求候选 / 预取候选 | {validation['candidate_source_counts']['demand']:,} / {validation['candidate_source_counts']['prefetch']:,} |
| check-only 项（独立文件） | {validation['check_only_rows']:,} |
| 唯一实际决策请求 / 块 | {validation['unique_decision_requests']:,} / {validation['unique_candidate_blocks']:,} |
| BC05 目标事件 / 目标请求 / 目标块 | {audit['target_events_linked']:,} / {target_requests} / {target_blocks} |
| HIGH-CF 强制写入成功 | {target_counts['high_cf_write_success']:,}/{target_counts['manifest_events']:,} |
| HIGH-CF 对应峰值请求命中 | {target_counts['high_cf_peak_hit']:,}，分布于 {target_hit_blocks} 个块 |
| HIGH-CF 到对应峰值前已淘汰 | {target_counts['high_cf_evicted_before_peak']:,} |
| 评价段写入 | {validation['evaluation_write_count']:,}，原始窗口核算一致 |

候选记录都在同一请求内进入模型批次，队列延迟均为零；仍分别保留了候选产生与实际决定时间及上下文，以便此结论不依赖推断。需求和预取来源分别通过 `run_get` 的候选插入路径与预取元数据识别，来源缺失时保留 `unknown` 类别；本次 HIGH-CONTROL 没有 unknown 候选。

## 作者输入与新增合法历史

作者模型输入严格为以下 18 列，详见 `BC06_feature_schema.json`：

`feat_metadata|op`, `feat_metadata|ns`, `feat_metadata|user`, `feat_metadata_size|start/end/size`, `feat_dynamic_b|0..5`, `feat_dynamic_c_combined|0..5`。

`block_get_count` 和 chunk 历史与作者六个滚动小时桶部分重叠；chunk 输入还是当前请求所有 chunk 历史的聚合，不是当前 chunk 的独立累计值。访问间隔、此前真实拒绝次数／间隔、缓存占用、上个完整窗口写入与淘汰、评价写入预算使用率，以及最近六个已结束窗口的 GET／需求／预取／PUT DT 分量，都没有进入这 18 维作者模型输入。当前请求字节大小已在作者 meta 特征中；chunk 数作为显式上下文另存。

字段来源、冻结时点、作者模型是否使用以及 BC05 稀疏标签侧车的字段定义见 `BC06_field_dictionary.csv` 与 `BC06_feature_overlap.csv`。

## 描述性对照

`BC06_descriptive.csv` 同时给出候选事件、按请求聚合和按块聚合的特征摘要。全轨迹共有 {all_rej.n:,} 个作者实际拒绝候选，分布在 {reject_requests:,} 个请求和 {reject_blocks:,} 个块。BC05 成员有 {target.n:,} 项，分布在 {target_requests} 个拒绝请求、{target_blocks} 个块；逐chunk不是独立样本。请求与块表先在组内求均值，再给出这些单位均值的四分位数。

结果标签仅通过 `BC06_BC05_label_links.csv` 稀疏关联到目标成员。该表记录 HIGH-CONTROL 原决定，以及 HIGH-CF 写入、淘汰和对应峰值请求是否命中的后验观察。它们是使用未来目标清单得到的特权反事实结果；不得把它们转写成“应接受／应拒绝”的逐项真值，或者将节省成本相加当作全局净收益。

## 权限隔离与执行说明

候选快照在读取 BC05 目标或结果之前已封存 SHA-256 清单。标签关联后，候选、请求上下文、check-only 文件、配置和原始运行审计的哈希全部保持不变。完整回放输出中的全局 `ml_predictions` 还累计了预取模型推理；它比只针对 NewMLAP 捕获的准入输入多 {validation['additional_shared_counter_predictions_outside_admission_rows']:,} 项。这个统计口径差异触发了运行器末尾一个过宽的总数断言，但仿真已经处理全部请求、写完原始结果及快照；独立后验核验通过。我们没有将这部分预取模型推理误记为准入候选，也没有重跑求解器。

BC05 标签文件只进入此后生成的稀疏 sidecar。`BC06_snapshot_manifest.json` 保留封存哈希，`BC06_snapshot_validation.json` 保留原始统计逐窗复现结果。
'''
    out.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
