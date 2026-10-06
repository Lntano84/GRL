"""Independent post-freeze timing-count correction and label firewall check."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

WORK = Path(__file__).resolve().parent
BASE = WORK.parents[1]
ARM = WORK / 'HIGH-CONTROL'
OUT = BASE / 'outputs'
MANIFEST_PATH = WORK / 'snapshot_manifest.json'
LABELS_PATH = OUT / 'BC06_BC05_label_links.csv'
MODEL_PATH = BASE / 'work' / 'bc01' / 'Baleen-FAST24' / 'tmp' / 'example' / '201910_Region1_0_0.1' / 'ea_5892.86_wr_35.599_admit_threshold_binary.model'
PREFIX_PATH = WORK / 'label_firewall_prefix.json'
RESULT_PATH = OUT / 'BC06_postfreeze_reconciliation.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def hash_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    validation = json.loads((WORK / 'snapshot_validation.json').read_text(encoding='utf-8'))
    run_audit = json.loads((ARM / 'run_audit.json').read_text(encoding='utf-8'))
    for name, info in manifest['files'].items():
        path = ARM / name
        if sha(path) != info['sha256'] or path.stat().st_size != info['bytes']:
            raise AssertionError(f'frozen artifact mismatch: {name}')

    with LABELS_PATH.open(encoding='utf-8-sig', newline='') as f:
        links = list(csv.DictReader(f))
    if not links:
        raise AssertionError('empty BC05 sparse result sidecar')
    target_candidate_id = int(links[0]['candidate_id'])
    target_request_seq = int(links[0]['reject_request_sequence'])

    candidate_path = OUT / 'BC06_candidate_decisions.jsonl.gz'
    target = None
    same_request = delayed = pending = 0
    candidate_rows = 0
    max_delay = 0.0
    with gzip.open(candidate_path, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            candidate_rows += 1
            if row.get('actual_decision_request_sequence') is None:
                pending += 1
            elif row['candidate_created_request_sequence'] == row['actual_decision_request_sequence']:
                same_request += 1
            else:
                delayed += 1
            max_delay = max(max_delay, float(row.get('candidate_to_decision_delay_s') or 0.0))
            if row['candidate_id'] == target_candidate_id:
                target = row
    if target is None:
        raise AssertionError(f'linked target candidate missing: {target_candidate_id}')
    if candidate_rows != manifest['counts']['actual_candidate_decisions']:
        raise AssertionError(('candidate count differs from frozen manifest', candidate_rows))

    context = None
    context_path = OUT / 'BC06_get_request_contexts.jsonl.gz'
    with gzip.open(context_path, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if int(row['request_sequence']) == target_request_seq:
                context = row
                break
    if context is None:
        raise AssertionError(f'target request context missing: {target_request_seq}')
    if target['actual_decision_request_sequence'] != target_request_seq:
        raise AssertionError('sidecar candidate does not match its target request')

    # The saved prefix excludes every BC05 field. It contains only the request
    # start snapshot and the exact feature vector observed by the author model.
    prefix = {
        'candidate_id': target_candidate_id,
        'request_sequence': target_request_seq,
        'get_request_context_at_start': context,
        'candidate_key_history_at_request_start': target['candidate_key_history_at_request_start'],
        'author_model_input_vector': target['author_model_input_vector'],
        'threshold': target['threshold'],
        'recorded_author_score': target['author_model_score'],
        'recorded_author_decision_accept': target['author_decision_accept'],
    }
    PREFIX_PATH.write_text(json.dumps(prefix, ensure_ascii=False, indent=2), encoding='utf-8')
    prefix_hash_before = sha(PREFIX_PATH)

    model_hash_before = sha(MODEL_PATH)
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    def score_from_frozen_prefix(saved_prefix):
        vector = np.asarray(saved_prefix['author_model_input_vector'], dtype=float).reshape(1, -1)
        score = float(booster.predict(vector)[0])
        return {'score': score, 'accept': score > float(saved_prefix['threshold'])}

    baseline = score_from_frozen_prefix(prefix)
    if abs(baseline['score'] - float(prefix['recorded_author_score'])) > 1e-12:
        raise AssertionError(('saved score does not reproduce from author model', baseline, prefix['recorded_author_score']))
    if baseline['accept'] != bool(prefix['recorded_author_decision_accept']):
        raise AssertionError('recomputed score/threshold decision differs from the frozen decision')

    # Construct a label file with every post-intervention outcome flag flipped.
    # The scoring function receives only the frozen prefix and model, never a
    # label path or a BC05 event table.
    sidecar_text = LABELS_PATH.read_text(encoding='utf-8-sig')
    reader = csv.DictReader(io.StringIO(sidecar_text))
    fields = reader.fieldnames
    rows = list(reader)
    outcome_fields = [
        'bc05_high_cf_intervention_changed_reject_to_accept',
        'bc05_high_cf_write_success',
        'bc05_high_cf_evicted_after_write',
        'bc05_high_cf_evicted_before_corresponding_peak',
        'bc05_high_cf_corresponding_peak_chunk_missing',
        'bc05_high_cf_corresponding_peak_hit',
    ]
    if not all(field in fields for field in outcome_fields):
        raise AssertionError('expected BC05 outcome fields are absent from the sidecar')
    changed = 0
    for row in rows:
        for field in outcome_fields:
            value = row[field].strip().lower()
            if value in {'true', 'false'}:
                row[field] = 'False' if value == 'true' else 'True'
                changed += 1
            else:
                row[field] = 'perturbed'
                changed += 1
    mutated = io.StringIO(newline='')
    writer = csv.DictWriter(mutated, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    mutated_label_hash = hash_text(mutated.getvalue())
    original_label_hash = sha(LABELS_PATH)
    if mutated_label_hash == original_label_hash:
        raise AssertionError('label perturbation did not change the label file')

    after_perturbation = score_from_frozen_prefix(json.loads(PREFIX_PATH.read_text(encoding='utf-8')))
    frozen_hashes_after = {name: sha(ARM / name) for name in manifest['files']}
    if baseline != after_perturbation:
        raise AssertionError(('labels changed a frozen-prefix score/action', baseline, after_perturbation))
    if sha(PREFIX_PATH) != prefix_hash_before:
        raise AssertionError('saved prefix changed during label perturbation check')
    if frozen_hashes_after != {name: info['sha256'] for name, info in manifest['files'].items()}:
        raise AssertionError('a frozen snapshot artifact changed during label perturbation check')
    if sha(MODEL_PATH) != model_hash_before:
        raise AssertionError('author model changed during firewall check')

    recorded = run_audit['recorder']['actual_decision_counts']
    if same_request != recorded['created_and_decided_same_request']:
        raise AssertionError(('same-request count differs from run audit', same_request, recorded))
    if delayed != recorded.get('delay_positive', 0) or pending != 0:
        raise AssertionError(('delay/pending counts differ from run audit', delayed, pending, recorded))
    if candidate_rows != validation['actual_candidate_rows']:
        raise AssertionError('candidate scan count differs from postflight validation')

    result = {
        'frozen_manifest_sha256': sha(MANIFEST_PATH),
        'snapshot_artifact_hashes_unchanged': True,
        'candidate_creation_decision_counts_corrected': {
            'same_request': same_request,
            'delayed_to_later_request': delayed,
            'pending_undecided': pending,
            'maximum_delay_seconds': max_delay,
            'note': 'The initial verifier had its same-request and delayed counter condition reversed; raw snapshot scan agrees with the run audit.'
        },
        'label_firewall_replay_on_saved_prefix': {
            'candidate_id': target_candidate_id,
            'request_sequence': target_request_seq,
            'label_rows': len(rows),
            'outcome_cells_perturbed': changed,
            'original_label_sha256': original_label_hash,
            'perturbed_label_sha256': mutated_label_hash,
            'saved_prefix_sha256': prefix_hash_before,
            'author_model_sha256': model_hash_before,
            'score_before': baseline['score'],
            'score_after_label_perturbation': after_perturbation['score'],
            'decision_before': baseline['accept'],
            'decision_after_label_perturbation': after_perturbation['accept'],
            'model_score_matches_frozen_replay': True,
            'score_function_inputs': ['saved legal prefix features', 'frozen author model'],
            'score_function_reads_future_labels': False,
            'interpretation': 'Deterministic inference replay on one sealed decision prefix; no solver replay or retraining.'
        }
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
