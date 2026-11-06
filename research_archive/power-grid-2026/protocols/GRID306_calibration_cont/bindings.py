"""Bind the approved GRID305 rows to the exact archived public prefixes."""
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path

R = Path('D:/Codex/GRL研究')
I = R / 'work/GRL-results-integration'
OUT = R / 'outputs/grid306_calibration_cont'
SOURCE = Path(__file__).resolve().parent
RECOVERY = R / 'work/grid301_remaining_collection/recovery03'
LEGACY = R / 'work/grid297_paired_collection_preparation'
OLD37 = R / 'outputs/grid297_paired_collection_preparation'
OLD38 = R / 'outputs/grid301_remaining_collection'
GRID305 = R / 'outputs/grid305_saved_calibration_candidates/proposed_scope.json'
GRID305_BLOB = 'e566db3f92f0da7a63e2ab04dcb17520218f9a88'
CAPS = dict(episode_s=1200, wall_s=3600, physical_steps=34289, public_forecasts=250000)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


state = module('grid306_original_state', RECOVERY / 'state.py')
guard = module('grid306_original_guard', RECOVERY.parent / 'guard.py')
read, save, sha, git = guard.read, guard.save, guard.sha, guard.git


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def folder(item):
    return f"r{item['repeat']}__{item['week']}__{item['method']}"


def checked_bytes(rel, reference):
    data = (R / rel).read_bytes()
    if len(data) != reference['bytes'] or digest_bytes(data) != reference['sha256']:
        raise ValueError(f'Frozen baseline bytes differ: {rel}')
    return data


def ledger(item):
    rel = item['baseline_folder'] + '/steps.jsonl.gz'
    with gzip.GzipFile(fileobj=io.BytesIO(checked_bytes(rel, item['baseline_input_refs'][rel]))) as stream:
        rows = [{key: row[key] for key in state.KEYS} for row in map(json.loads, stream)]
    if (len(rows) != 2017 or not rows[-1]['complete']
        or [row['step'] for row in rows] != list(range(1, 2018))
        or any(a['after_hash'] != b['before_hash'] for a,b in zip(rows, rows[1:]))):
        raise ValueError(f'Incomplete or discontinuous baseline: {rel}')
    return rows


def validate_event(marker, expected):
    if (marker['ordinary261_continue'] != expected['ordinary261_continue']
        or any(marker['common'].get(k) != v for k,v in expected['common'].items())):
        raise ValueError('Selected frozen public marker differs')


def build(year):
    builder = module('grid306_prior_builder', RECOVERY / 'source_builder.py')
    original = builder.build

    def assemble():
        source = original()
        source = builder.old.replace_once(source, "OUT=ROOT/'outputs/grid301_remaining_collection'",
                                          "OUT=ROOT/'outputs/grid306_calibration_cont'")
        return builder.old.replace_once(source, "D=load(OUT/'runtime_design.json')",
                                         f"D=load(OUT/'runtime_design_{year}.json')")
    builder.build = assemble
    return builder


def prepare():
    assert not (SOURCE / 'execution_order.json').exists()
    scope = json.loads(git('cat-file', 'blob', GRID305_BLOB))
    assert scope == read(GRID305)
    assert len(scope['fixed_representatives']) == 17
    designs = {2037: read(OLD37 / 'design.json'), 2038: read(OLD38 / 'runtime_design.json')}
    roles = {y: {v['pool']:v for v in read(old / 'pool_roles.json')}
             for y,old in [(2037, OLD37), (2038, OLD38)]}
    records = {y: {folder(i):i for i in d['execution_order']} for y,d in designs.items()}
    execution = []
    for index, row in enumerate(scope['fixed_representatives']):
        year, pool, date, step = (row[k] for k in ('year','pool','date','step'))
        role = roles[year][pool]
        original = records[year][Path(row['baseline_folder']).name]
        assert original['partition'] == 'own_training_cal4' and date in role['cal_dates']
        assert date not in role['outer_dates_excluded'] and original['KEEP'] == role['KEEP']
        assert (original['pool'],original['date'],original['week']) == (pool,date,row['week'])
        assert row['new_arm'] == 'CONT_NEXT' and row['coverage_status'] == 'new_state'
        assert row['baseline_folder'] == (OLD37 if year == 2037 else OLD38).relative_to(R).as_posix() + '/runs/' + folder(original)
        rows = ledger(row)
        assert rows[step-1]['before_hash'] == row['state_hash']
        prefix = rows[:step-1]
        prefix_cost = 0.
        for entry in prefix:
            prefix_cost += entry['raw_cost']
        prefix_marker = dict(prefix_signature=state.digest(prefix),prefix_cost=prefix_cost)
        record = read(R / row['baseline_folder'] / 'intervention.json')
        if year == 2037:
            if record['event']:
                assert record['event']['forced_continue'] == record['event']['ordinary261']['continue_']
            rel = row['source_call']
            assert Path(rel).parent.as_posix() == row['baseline_folder']
            assert int(Path(rel).stem[4:]) == row['call_ordinal']
            call = json.loads(checked_bytes(rel,row['baseline_input_refs'][rel]))
            decision = call['learned_decisions'][row['decision_index']]
            assert (call['model_pool'],call['step'],call['state_hash']) == (pool,step,row['state_hash'])
            assert decision['optional_extension'] is True and decision['action'] == 0
            assert decision['owner_pool'] == pool and decision['batch'] == row['batch']
            assert decision['next_ids'] == row['next_ids'] and 1 <= len(row['next_ids']) <= 64
            assert decision['native_q'][1] <= decision['native_q'][0]
            assert state.digest(decision['raw_features']) == row['raw41_hash']
            assert state.digest(decision['NN_input']) == row['public_NN_input_hash']
            target = {'ordinary261_continue':False,'common':{
                **prefix_marker, **{k:row[k] for k in ('step','state_hash','batch','next_ids',
                    'raw41_hash','public_NN_input_hash')}, 'call_index':row['call_ordinal'],
                'native_q':decision['native_q'], 'order_hash':state.digest(call['order']),
                'aliases_hash':state.digest(call['aliases']), 'rho_limit':call['rho_limit']}}
        else:
            target = row['target']
            common = target['common']
            assert original['index'] == row['baseline_index'] and original['target'] is None
            assert record['candidates'].count(target) == 1 and target['ordinary261_continue'] is False
            assert all(common[k] == row[k] for k in ('step','state_hash','batch','public_NN_input_hash'))
            assert all(common[k] == v for k,v in prefix_marker.items())
        execution.append({**original, 'repeat':index, 'index':index, 'pair_index':index,
                          'within_pair':0, 'method':f'CONT_NEXT__{pool}', 'arm':'CONT_NEXT',
                          'target':target, 'year':year, 'baseline_folder':row['baseline_folder'],
                          'baseline_input_refs':row['baseline_input_refs']})
    assert len({(x['date'],x['target']['common']['step'],x['target']['common']['state_hash'])
                for x in execution}) == 17
    assert sha(OLD37/'model_references.json') == sha(OLD38/'model_references.json')
    dependencies = [OLD37/'data_freeze.json', OLD37/'source_freeze.json', OLD38/'recovery03_source_freeze.json',
                    OLD38/'model_references.json', OLD38/'code_freeze.json', LEGACY/'intervention.py',
                    RECOVERY/'adapter.py', RECOVERY/'state.py', RECOVERY.parent/'guard.py',
                    OLD38/'baseline_ready.json', OLD38/'baseline_publication.json',
                    R/'outputs/grid300_new_training_dates/generated_freeze.json']
    for year in designs:
        compile(build(year).build(), str(R/'work/grid55/run.py'), 'exec')
        compile(build(year).instrument_worker((R/'work/grid55/worker.py').read_text(encoding='utf8')),
                str(R/'work/grid55/worker.py'), 'exec')
    assert not OUT.exists()
    OUT.mkdir()
    for year, d in designs.items():
        (OUT/f'runtime_design_{year}.json').write_text(json.dumps({
            'environment':d['environment'],'timing':d['timing'],'caps':CAPS,
            'execution_order':execution}, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    for name in ('model_references.json','code_freeze.json'):
        (OUT/name).write_bytes((OLD38/name).read_bytes())
    save(SOURCE/'execution_order.json', execution)
    files = [*SOURCE.glob('*.py'), SOURCE/'execution_order.json', *OUT.glob('*.json'), *dependencies]
    save(SOURCE/'approval.json', {
        'user_reply':'批准','user_message_reference':'Direct reply in this chat to GRID305 proposed 17 CONT replays',
        'grid305_scope_git_blob':GRID305_BLOB,'caps':CAPS,'runs':17,'no_retry':True,
        'source_sha256':{p.relative_to(R).as_posix():sha(p) for p in files}})
    print('17 original baseline ledgers, 16 saved calls, and 1 full saved marker verified. Zero live imports.')


if __name__ == '__main__':
    prepare()
