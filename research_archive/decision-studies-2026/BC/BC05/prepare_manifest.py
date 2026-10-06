"""Freeze BC05 intervention events and arm configs before any replay."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parents[2]
WORK = BASE / 'work/bc05'
OUT = BASE / 'outputs'
CHUNKS = OUT / 'BC04_peak_578_chunks.csv'
REQUESTS = OUT / 'BC04_peak_578_requests.csv'
AUDIT = BASE / 'work/bc04/audit.json'
EVAL_START = 86401.23277902603
PEAK_START = 346810.0965330601
PEAK = 578
EXPECTED = {'before_peak': 1489, 'peak': 59, 'warmup': 64}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def token(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


def main():
    assert not any(p.name != 'prepare_manifest.py' for p in WORK.iterdir()), 'BC05 outputs already exist; refusing to overwrite'
    WORK.mkdir(parents=True, exist_ok=True)
    source_audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    assert source_audit['protocol_passed'] and source_audit['run_status']['status'] == 'completed'
    rows = list(csv.DictReader(CHUNKS.open(encoding='utf-8-sig', newline='')))
    peak_rejected = [r for r in rows if r['request_category'] == 'REJECTED-BEFORE']
    assert len(peak_rejected) == 1612 and len({r['request_sequence'] for r in peak_rejected}) == 77
    assert len({r['full_block_key_json'] for r in peak_rejected}) == 34

    partitions = {'before_peak': [], 'peak': [], 'warmup': []}
    unique = {}
    for r in peak_rejected:
        seq = int(r['last_rejection_request'])
        rejected_s = float(r['last_rejection_s'])
        block = json.loads(r['full_block_key_json'])
        chunk = int(r['chunk'])
        identity = (seq, token(block), chunk)
        event = {
            'event_id': None,
            'reject_request_sequence': seq,
            'full_block_key': block,
            'chunk': chunk,
            'last_rejection_trace_elapsed_s': rejected_s,
            'peak_request_sequence': int(r['request_sequence']),
            'source_window': int(r['window_index']),
            'source_category': r['request_category'],
            'source_rejection_count': int(r['rejection_count']),
            'source_candidate_count': int(r['candidate_count']),
            'source_pending_before_peak_request': r['pending_before_request'].lower() == 'true',
        }
        if rejected_s < EVAL_START:
            bucket = 'warmup'
        elif rejected_s < PEAK_START:
            bucket = 'before_peak'
        else:
            bucket = 'peak'
        partitions[bucket].append(identity)
        unique[identity] = event

    observed = {name: len(set(events)) for name, events in partitions.items()}
    assert observed == EXPECTED, observed
    targets = []
    for identity in sorted(set(partitions['before_peak'])):
        event = unique[identity]
        event['event_id'] = len(targets) + 1
        targets.append(event)
    assert len(targets) == 1489
    assert all(EVAL_START <= e['last_rejection_trace_elapsed_s'] < PEAK_START for e in targets)
    assert len({(e['reject_request_sequence'], token(e['full_block_key']), e['chunk']) for e in targets}) == 1489

    # Independently preserve the specifically audited request 82821 example.
    example = [r for r in peak_rejected if int(r['request_sequence']) == 82821]
    assert len(example) == 64
    assert {int(r['last_rejection_request']) for r in example} == {81555}
    assert {int(r['rejection_count']) for r in example} == {30}

    target_path = WORK / 'frozen_targets.json'
    target_path.write_text(json.dumps(targets, ensure_ascii=False, indent=2), encoding='utf-8')
    target_hash = sha(target_path)
    parent_cfgs = {
        'BASE': BASE / 'work/bc03/STATIC-BASE/config.json',
        'HIGH': BASE / 'work/bc03/STATIC-HIGH/config.json',
    }
    base_cfg = json.loads(parent_cfgs['BASE'].read_text(encoding='utf-8'))
    high_cfg = json.loads(parent_cfgs['HIGH'].read_text(encoding='utf-8'))
    base_compare, high_compare = dict(base_cfg), dict(high_cfg)
    base_compare.pop('output_dir', None)
    high_compare.pop('output_dir', None)
    assert base_compare == high_compare, 'BASE/HIGH config differs beyond the BC03 runtime threshold schedule'

    arms = ['BASE-CONTROL', 'BASE-CF', 'HIGH-CONTROL', 'HIGH-CF']
    for arm in arms:
        cfg = dict(base_cfg if arm.startswith('BASE') else high_cfg)
        cfg['output_dir'] = str((WORK / arm / 'raw').resolve()).replace('\\', '/')
        arm_dir = WORK / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')

    source_paths = [CHUNKS, REQUESTS, AUDIT,
                    parent_cfgs['BASE'], parent_cfgs['HIGH'],
                    BASE / 'work/bc01/Baleen-FAST24/data/tectonic/201910/Region1/full_0_0.1.trace',
                    BASE / 'work/bc01/Baleen-FAST24/tmp/example/201910_Region1_0_0.1/ea_5892.86_wr_35.599_admit_threshold_binary.model',
                    BASE / 'work/bc01/Baleen-FAST24/tmp/example/201910_Region1_0_0.1/ea_5892.86_wr_35.599_prefetch_offset_end.model',
                    BASE / 'work/bc01/Baleen-FAST24/tmp/example/201910_Region1_0_0.1/ea_5892.86_wr_35.599_prefetch_offset_start.model',
                    BASE / 'work/bc01/Baleen-FAST24/tmp/example/201910_Region1_0_0.1/ea_5892.86_wr_35.599_prefetch_pred_net_pf_st_binary.model',
                    BASE / 'work/bc01/Baleen-FAST24/tmp/example/201910_Region1_0_0.1/ea_5892.86_wr_35.599_prefetch_size.model']
    manifest = {
        'protocol': 'BC05 frozen privileged counterfactual; no model training',
        'targets_count': len(targets), 'targets_sha256': target_hash,
        'source_partition_unique_events': observed,
        'selection': {'source_window': PEAK, 'category': 'REJECTED-BEFORE',
                      'time_rule': 'evaluation_start <= latest_actual_rejection < peak_578_start',
                      'dedupe': '(rejection request sequence, complete block key, chunk id)'},
        'constants': {'evaluation_start_trace_s': EVAL_START, 'peak_578_start_trace_s': PEAK_START,
                      'evaluation_write_cap_chunks': 148731, 'cache_capacity_chunks': 3002,
                      'full_trace_requests': 147794, 'get_requests': 127305},
        'example_82821_check': {'missing_chunks': len(example), 'latest_rejection_sequence': 81555,
                                'prior_rejections_each': 30},
        'source_sha256': {str(p.resolve()): sha(p) for p in source_paths},
        'parent_configs_sha256': {name: sha(path) for name, path in parent_cfgs.items()},
        'arms': arms,
    }
    (WORK / 'frozen_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'BC05_frozen_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'BC05_frozen_targets.json').write_text(target_path.read_text(encoding='utf-8'), encoding='utf-8')
    print(json.dumps({'targets': len(targets), 'partition': observed, 'sha256': target_hash,
                      'earliest_target_seq': min(e['reject_request_sequence'] for e in targets),
                      'latest_target_seq': max(e['reject_request_sequence'] for e in targets),
                      'request_82821': {'missing_chunks': len(example), 'latest_rejection': 81555}}, indent=2))


if __name__ == '__main__':
    main()
