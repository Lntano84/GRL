"""BC05 source, frozen-manifest, and intervention-interface checks; no replay."""
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
REPO = BASE / 'work/bc01/Baleen-FAST24'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from replay import Recorder, THETA0, THETAH, EVAL_START, START, WORK
from BCacheSim.cachesim import admission_policies as aps, eviction_policies as evp


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeAP:
    def __init__(self, pred):
        self.pred = pred
        self.threshold = None
        self.counted = []

    def _predict(self, batch, ts):
        return self.pred

    def count_decisions(self, decisions):
        self.counted.append(dict(decisions))


def configure_single(r, t, seq=777):
    r.by_seq = {seq: [t]}
    r.by_key = {(seq, t['key_token']): t}
    r.current_seq = seq
    s = t['state']
    for k, v in {'decision_seen': False, 'check_only_seen': False, 'parent_decision': None,
                 'intervention_changed': False}.items():
        s[k] = v


def main():
    manifest = json.loads((WORK / 'frozen_manifest.json').read_text(encoding='utf-8'))
    targets_path = WORK / 'frozen_targets.json'
    targets = json.loads(targets_path.read_text(encoding='utf-8'))
    checks = {}
    assert len(targets) == 1489 and sha(targets_path) == manifest['targets_sha256']
    checks['frozen_target_count_and_sha256'] = True
    for path, digest in manifest['source_sha256'].items():
        assert sha(Path(path)) == digest, f'input changed after freezing: {path}'
    checks['frozen_input_hashes_unchanged'] = True

    for name, parent_path in [('BASE', BASE/'work/bc03/STATIC-BASE/config.json'),
                              ('HIGH', BASE/'work/bc03/STATIC-HIGH/config.json')]:
        parent = json.loads(parent_path.read_text(encoding='utf-8'))
        for arm in [f'{name}-CONTROL', f'{name}-CF']:
            cfg = json.loads((WORK/arm/'config.json').read_text(encoding='utf-8'))
            expected = dict(parent)
            expected['output_dir'] = str((WORK/arm/'raw').resolve()).replace('\\', '/')
            assert cfg == expected
    checks['configs_only_change_output_directory'] = True
    assert aps.NewMLAP.batchAccept is aps.LearnedAP.batchAccept
    assert callable(evp.QueueCache.admit)
    checks['actual_batch_accept_and_write_hooks_resolve'] = True

    # False decisions are flipped only for matching actual (non-check-only) decisions.
    r = Recorder('BASE-CF')
    target = r.targets[0]
    key = target['native_key']
    configure_single(r, target)
    ap = FakeAP([0.2])
    out = r.score_decisions(ap, {key: object()}, SimpleNamespace(physical=START+EVAL_START+1), False)
    assert out == {key: True} and ap.counted == [{key: True}]
    assert target['state']['parent_decision'] is False and target['state']['intervention_changed']
    checks['actual_reject_overridden_and_counted_as_accept'] = True

    r = Recorder('BASE-CF')
    target = r.targets[0]
    key = target['native_key']
    configure_single(r, target)
    ap = FakeAP([0.2])
    out = r.score_decisions(ap, {key: object()}, SimpleNamespace(physical=START+EVAL_START+1), True)
    assert out == {key: False} and ap.counted == []
    assert target['state']['intervention_changed'] is False
    checks['check_only_decision_unchanged_and_not_counted'] = True

    r = Recorder('BASE-CF')
    target = r.targets[0]
    key = target['native_key']
    configure_single(r, target)
    ap = FakeAP([0.95])
    out = r.score_decisions(ap, {key: object()}, SimpleNamespace(physical=START+EVAL_START+1), False)
    assert out == {key: True} and ap.counted == [{key: True}]
    assert target['state']['parent_decision'] is True and target['state']['intervention_changed'] is False
    checks['parent_accept_unchanged'] = True

    r = Recorder('BASE-CF')
    target = r.targets[0]
    configure_single(r, target)
    ap = FakeAP([])
    out = r.score_decisions(ap, {}, SimpleNamespace(physical=START+EVAL_START+1), False)
    assert out == {} and ap.counted == [{}] and not target['state']['decision_seen']
    checks['noncandidate_has_no_forced_decision'] = True

    assert Recorder('BASE-CONTROL').threshold(EVAL_START+1) == THETA0
    assert Recorder('HIGH-CF').threshold(EVAL_START-1) == THETA0
    assert Recorder('HIGH-CF').threshold(EVAL_START+1) == THETAH
    checks['parent_threshold_schedules_frozen'] = True

    result = {'passed': True, 'checks': checks, 'targets': len(targets),
              'targets_sha256': manifest['targets_sha256'], 'no_simulator_replay': True}
    (WORK/'preflight.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
