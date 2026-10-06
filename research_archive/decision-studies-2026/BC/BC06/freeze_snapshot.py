"""Seal BC06 score/input snapshots before any BC05 label join."""
import hashlib
import json
import shutil
from pathlib import Path

WORK = Path(__file__).resolve().parent
BASE = WORK.parents[1]
ARM = WORK / 'HIGH-CONTROL'
OUT = BASE / 'outputs'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    validation = json.loads((WORK / 'snapshot_validation.json').read_text(encoding='utf-8'))
    if not validation.get('snapshot_validation_passed'):
        raise AssertionError('cannot freeze an unvalidated snapshot')
    names = [
        'candidate_decisions.jsonl.gz', 'get_request_contexts.jsonl.gz',
        'check_only_calls.jsonl.gz', 'feature_schema.json', 'config.json',
        'run_audit.json',
    ]
    files = {name: {'sha256': sha(ARM / name), 'bytes': (ARM / name).stat().st_size}
             for name in names}
    manifest = {
        'protocol': 'BC06 HIGH-CONTROL legal decision snapshot; no retraining or policy change',
        'snapshot_frozen_before_BC05_target_link': True,
        'validation': str((WORK / 'snapshot_validation.json').resolve()),
        'validation_sha256': sha(WORK / 'snapshot_validation.json'),
        'files': files,
        'counts': {
            'trace_requests': validation['trace_request_count'],
            'get_contexts': validation['get_request_context_count'],
            'actual_candidate_decisions': validation['actual_candidate_rows'],
            'actual_accepts': validation['actual_accepted_rows'],
            'actual_rejects': validation['actual_rejected_rows'],
            'check_only_items': validation['check_only_rows'],
            'demand_candidates': validation['candidate_source_counts']['demand'],
            'prefetch_candidates': validation['candidate_source_counts']['prefetch'],
        },
        'label_firewall': 'snapshot generation/validation contains no BC05 target or outcome input; any later join is a separate sidecar',
    }
    path = WORK / 'snapshot_manifest.json'
    if path.exists():
        raise RuntimeError('snapshot manifest already exists; refusing to rewrite frozen manifest')
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    copies = {
        'candidate_decisions.jsonl.gz': OUT / 'BC06_candidate_decisions.jsonl.gz',
        'get_request_contexts.jsonl.gz': OUT / 'BC06_get_request_contexts.jsonl.gz',
        'check_only_calls.jsonl.gz': OUT / 'BC06_check_only_calls.jsonl.gz',
        'feature_schema.json': OUT / 'BC06_feature_schema.json',
    }
    for src_name, dst in copies.items():
        if dst.exists():
            raise RuntimeError(f'output already exists; refusing to overwrite: {dst}')
        shutil.copyfile(ARM / src_name, dst)
    for src, dst in [(WORK / 'snapshot_validation.json', OUT / 'BC06_snapshot_validation.json'),
                     (path, OUT / 'BC06_snapshot_manifest.json')]:
        if dst.exists():
            raise RuntimeError(f'output already exists; refusing to overwrite: {dst}')
        shutil.copyfile(src, dst)
    print(json.dumps({'frozen': True, 'manifest_sha256': sha(path), 'files': files,
                      'outputs': {k: str(v.resolve()) for k, v in copies.items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
