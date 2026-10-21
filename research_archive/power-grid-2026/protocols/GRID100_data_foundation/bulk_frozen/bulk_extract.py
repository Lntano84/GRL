"""Validate saved teacher records and split visible inputs from gain targets."""
import argparse
import collections
import gzip
import shutil
import numpy as np
from labels import collect, key
from bulk_common import ROOT, OUT, RUNTIME, items, read, sha, save, test_sealed, verify


def digest(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def check_label(call, subs, row):
    truth = {r['pool_id']: r for r in call['outcomes']}
    n, x, y = row['visible_count'], row['input'], row['label']
    incumbent = max((truth[j] for j in call['order'][:n] if truth[j]['strict_admissible']), key=key)
    station = subs[incumbent['pool_id']]
    representatives = {int(rep) for original, rep in call['aliases'].items() if subs[int(original)] == station}
    extra = [j for j in call['order'][n:] if j in representatives]
    assert extra == x['remaining_same_station_ids'] and len(extra) == y['known_completion_queries']
    after = max([incumbent] + [truth[j] for j in extra if truth[j]['strict_admissible']], key=key)
    assert float(np.float32(after['reward'])) - float(np.float32(incumbent['reward'])) == y['local_reward_gain']
    assert after['pool_id'] == y['after_id'] and incumbent['pool_id'] == y['before_id']


def extract(index):
    verify(read(OUT / 'bulk_code_freeze.json'))
    test_sealed()
    item = items()[index]
    assert index > 0
    folder = RUNTIME / 'runs' / f"r0__{item['scenario']}__FULL_TEACHER"
    result_path = RUNTIME / f'worker_result_{index:02d}.json'
    result = read(result_path)
    assert result == read(folder / 'summary.json')
    assert result['week'] == item['scenario'] and result['partition'] == item['partition']
    with gzip.open(folder / 'steps.jsonl.gz', 'rt', encoding='utf8') as f:
        steps = [__import__('json').loads(line) for line in f]
    assert len(steps) == result['steps'] <= 2017
    assert abs(sum(x['raw_cost'] for x in steps) - result['cost']) < 1e-6
    with np.load(folder / 'vectors.npz') as vectors:
        assert len(vectors['action']) == len(vectors['observation']) == len(steps)
        for row, action, obs in zip(steps, vectors['action'], vectors['observation']):
            assert digest(action) == row['action_hash'] and digest(obs) == row['after_hash']
    for before, after in zip(steps, steps[1:]):
        assert before['after_hash'] == after['before_hash']
    for row in steps:
        if row['ledger']:
            assert abs(row['ledger']['recomputed_raw_cost'] - row['raw_cost']) <= row['ledger_tolerance']
    subs = read(ROOT / 'outputs/grid28/offline.json')['public_subids']
    dataset = OUT / 'datasets' / item['scenario']
    assert not dataset.exists(), 'Preserve partial/completed extraction; never overwrite'
    dataset.mkdir(parents=True)
    manifest, excluded = [], []
    counts = collections.Counter()
    for path in sorted(folder.glob('call[0-9][0-9][0-9].json')):
        call = read(path)
        assert call['policy'] == 'FULL'
        assert call['public_queries'] == len(call['order']) == len(call['outcomes'])
        truth = {r['pool_id']: r for r in call['outcomes']}
        assert len(truth) == len(call['order']) and set(truth) == set(call['order'])
        for row in call['outcomes']:
            admissible = bool(0 < row['rho'] < call['rho_limit'] and not row['done']
                              and not row['illegal'] and not row['ambiguous'] and not row['exceptions'])
            assert admissible == row['strict_admissible']
            counts['feedback_legality_checks'] += 1
        best = max((r for r in call['outcomes'] if r['strict_admissible']), key=key, default=None)
        assert (None if best is None else best['pool_id']) == call['chosen_pool_id']
        state = path.with_name(path.stem + '_state.npz')
        with np.load(state) as vectors:
            assert digest(vectors['observation']) == call['state_hash'] == steps[call['step'] - 1]['before_hash']
        dest = dataset / 'states' / state.name
        dest.parent.mkdir(exist_ok=True); shutil.copy2(state, dest)
        rows, rejected = collect(call, subs, path.relative_to(ROOT).as_posix())
        excluded.extend(dict(call=path.stem, **x) for x in rejected)
        for row in rows:
            sample_id, n = row['sample_id'], row['visible_count']
            x = dict(row['input'], observation_file='states/' + state.name,
                     public_action_library='work/grid02/ljn_nn/assets/action_N1_unsafe.npz')
            y = row['label']
            assert not set(x) & (set(y) - {'known_completion_queries'})
            assert [r['pool_id'] for r in x['visible']] == call['order'][:n]
            # Recompute local gain directly from paid feedback, separately from collect().
            check_label(call, subs, row)
            save(dataset / 'inputs' / (sample_id + '.json'), x)
            save(dataset / 'targets' / (sample_id + '.json'), y)
            manifest.append(dict(sample_id=sample_id, partition=item['partition'], calendar_group=item['date'],
                                 scenario=item['scenario'], physical_state_group=item['scenario'] + '|' + path.stem,
                                 source_call=path.relative_to(ROOT).as_posix(), source_sha256=sha(path),
                                 state_hash=call['state_hash'], step=call['step'], visible_count=n,
                                 input_file='inputs/' + sample_id + '.json', target_file='targets/' + sample_id + '.json'))
            counts['positive_gain'] += y['local_reward_gain'] > 0
            counts['zero_gain'] += y['local_reward_gain'] == 0
            counts['independent_gain_checks'] += 1
        counts['full_teacher_calls'] += 1
        counts['paid_candidate_queries'] += len(call['order'])
    assert counts['full_teacher_calls'] == result['n1_calls']
    assert counts['paid_candidate_queries'] == result['n1_forecasts']
    save(dataset / 'manifest.json', dict(samples=manifest, excluded_prefixes=excluded, partition=item['partition'],
         calendar_groups=1, teacher_complete=result['complete'], teacher_steps=result['steps'],
         not_independent_samples='Prefixes share FULL physical states and date groups; not FRAME closed-loop labels.'))
    for row in manifest:
        x, y = read(dataset / row['input_file']), read(dataset / row['target_file'])
        assert y['sample_id'] == row['sample_id'] and x['known_completion_queries'] == y['known_completion_queries']
    files = {p.relative_to(ROOT).as_posix(): sha(p) for p in dataset.rglob('*') if p.is_file()}
    files.update({p.relative_to(ROOT).as_posix(): sha(p) for p in folder.rglob('*') if p.is_file()})
    files[result_path.relative_to(ROOT).as_posix()] = sha(result_path)
    receipt = dict(passed=True, scenario=item['scenario'], partition=item['partition'], calendar_group=item['date'],
                   teacher_steps=result['steps'], teacher_complete=result['complete'], teacher_cost=result['cost'],
                   samples=len(manifest), excluded_prefixes=len(excluded), counts=dict(counts),
                   dataset_path=dataset.relative_to(ROOT).as_posix(), files=files,
                   final_test_materialized=False, model_fits=0, reused_pilot=False,
                   scope='Immediate completion gain on FULL teacher paths; no long-term value or deployment claim.')
    save(OUT / 'dataset_receipts' / (item['scenario'] + '.json'), receipt)
    print(__import__('json').dumps({k: v for k, v in receipt.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--index', type=int, required=True)
    extract(parser.parse_args().index)
