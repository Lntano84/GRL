"""Qualify and separately seal the fixed 40-scenario continuation, once."""
import copy
import shutil
from bulk_common import ROOT, W, OUT, RUNTIME, ARCHIVE, COLLAB, read, sha, save, verify, items, test_sealed, utc
from bulk_extract import check_label
from labels import collect, input_at


def main():
    assert not (OUT / 'bulk_code_freeze.json').exists(), 'Do not reseal an existing run'
    verify(read(OUT / 'freeze.json')['sha256'])
    verify(read(OUT / 'collector_code_freeze.json'))
    assert read(OUT / 'pilot_collection_finished.json')['passed']
    assert read(OUT / 'collector_qualification.json')['passed']
    planned = items(); test_sealed()
    for path in W.glob('*.py'):
        compile(path.read_text(encoding='utf8'), str(path), 'exec')
    pilot = planned[0]
    dataset = OUT / 'dataset_pilot'
    raw = OUT / 'teacher_runtime/runs' / f"r0__{pilot['scenario']}__FULL_TEACHER"
    subs = read(ROOT / 'outputs/grid28/offline.json')['public_subids']
    saved_samples = {r['sample_id']: r for r in read(dataset / 'manifest.json')['samples']}
    checks = 0
    for path in sorted(raw.glob('call[0-9][0-9][0-9].json')):
        call = read(path)
        rows, _ = collect(call, subs, path.relative_to(ROOT).as_posix())
        for row in rows:
            check_label(call, subs, row)
            saved = saved_samples[row['sample_id']]
            assert read(dataset / saved['target_file']) == row['label']
            x = read(dataset / saved['input_file'])
            assert {k: v for k, v in x.items() if k not in ('observation_file', 'public_action_library')} == __import__('json').loads(__import__('json').dumps(row['input']))
            visible = copy.deepcopy(call['outcomes'][:row['visible_count']])
            public = dict(order=call['order'], aliases=call['aliases'], rho_limit=call['rho_limit'], visible=visible)
            # Feature function only receives prefix feedback: future values have no argument route.
            assert input_at(public, subs, row['visible_count']) == row['input']
            checks += 1
    assert checks == len(saved_samples) == 96
    roles = {}
    for item in read(OUT / 'design.json')['scenario_plan']:
        roles.setdefault(item['date'], set()).add(item['partition'])
    assert len(roles) == 24 and all(len(v) == 1 for v in roles.values())
    assert len({x['date'] for x in planned if x['partition'] == 'train'}) == 16
    assert len({x['date'] for x in planned if x['partition'] == 'validation'}) == 4
    save(OUT / 'bulk_preflight.json', dict(passed=True, syntax_checked=True, pilot_gain_checks=checks,
         pilot_targets_reproduced=checks, prefix_input_checks=checks, date_roles_disjoint=True,
         generator_entry='generate.py --mode trainval', pilot_reused=True,
         new_physical_steps=0, model_fits=0, final_test_materialized=False, utc=utc()))
    plan = dict(stage='GRID100_TRAIN_VALIDATION_COLLECTION', utc=utc(), scenarios=planned,
                counts=dict(train=32, validation=8, final_test_unmaterialized=8), reused_pilot=pilot['scenario'],
                generation_wall_seconds=86400, total_wall_seconds=86400, worker_wall_seconds=1320,
                generation_item_watchdog_seconds=1920,
                execution='Serial generation of remaining39; serial FULL teacher replay of remaining39; saved-data extraction and archive. No model fitting.',
                source_protocol='design.json and split_manifest.json unchanged; supplement enables the predeclared train/validation continuation.',
                failure='Preserve errors and partial outputs, stop without replacing scenarios or retrying completed trajectories.',
                final_test_gate='No test generation, inputs, teacher labels or final evaluation.',
                training_contract='training_input_contract.json: raw vectors contain calendar fields; explicit feature projection required later.')
    save(OUT / 'bulk_plan.json', plan)
    execution = [dict(rule='STATIC_ID', budget=128, checkpoint=None, cache_dc=True, cache_safe=True,
                      policy='FULL', order_mode='STATIC_ID', repeat=0, stage_phase='data_collection',
                      week=x['scenario'], method='FULL_TEACHER', partition=x['partition']) for x in planned]
    save(RUNTIME / 'design.json', dict(execution_order=execution,
         timing=read(OUT / 'teacher_runtime/design.json')['timing'],
         caps=dict(episode_s=1200, wall_s=1320, physical_steps=2017 * 40, public_forecasts=200000 * 40)))
    receipt = read(OUT / 'pilot_collection_finished.json')
    receipt.update(calendar_group=pilot['date'], dataset_path=dataset.relative_to(ROOT).as_posix(), reused_pilot=True,
                   files={p.relative_to(ROOT).as_posix(): sha(p) for folder in [dataset, raw] for p in folder.rglob('*') if p.is_file()})
    result = OUT / 'teacher_runtime/worker_result_00.json'
    receipt['files'][result.relative_to(ROOT).as_posix()] = sha(result)
    save(OUT / 'dataset_receipts' / (pilot['scenario'] + '.json'), receipt)
    codes = read(OUT / 'collector_code_freeze.json')
    paths = list(W.glob('*.py')) + [OUT / 'bulk_plan.json', OUT / 'bulk_preflight.json', RUNTIME / 'design.json',
             OUT / 'training_input_contract.json', OUT / 'pilot_collection_finished.json']
    codes.update({p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    save(OUT / 'bulk_code_freeze.json', codes)
    save(RUNTIME / 'code_freeze.json', codes)
    frozen = ARCHIVE / 'bulk_frozen'; frozen.mkdir(parents=True, exist_ok=False)
    for path in list(W.glob('*.py')) + [OUT / 'bulk_plan.json', OUT / 'bulk_preflight.json',
                                      OUT / 'bulk_code_freeze.json', OUT / 'training_input_contract.json']:
        shutil.copy2(path, frozen / path.name)
    shutil.copy2(RUNTIME / 'design.json', frozen / 'bulk_runtime_design.json')
    save(frozen / 'archive_manifest.json', {p.name: sha(p) for p in frozen.iterdir() if p.is_file()})
    COLLAB.mkdir(parents=True, exist_ok=True)
    for name in ['bulk_plan.json', 'bulk_preflight.json', 'bulk_code_freeze.json']:
        shutil.copy2(OUT / name, COLLAB / name)
    print('Bulk sealed:', len(codes), 'hashes; pilot reused;40 train/validation scenarios; test sealed.')


if __name__ == '__main__':
    main()
