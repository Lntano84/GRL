"""Repair only the saved legacy alias identity and bind unstarted GRID306 indices."""
import json
from pathlib import Path
import importlib.util

SOURCE = Path(__file__).resolve().parent
PRIOR = SOURCE.parent / 'GRID306_calibration_cont'
spec = importlib.util.spec_from_file_location('grid306_frozen_bindings', PRIOR / 'bindings.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
R, I, LEGACY = prior.R, prior.I, prior.LEGACY
GRID305, GRID305_BLOB = prior.GRID305, prior.GRID305_BLOB
module, read, save, sha, git = prior.module, prior.read, prior.save, prior.sha, prior.git
guard, state, ledger, validate_event = prior.guard, prior.state, prior.ledger, prior.validate_event
OLD = prior.OUT
OUT = R / 'outputs/grid307_unstarted_calibration_cont'
INDICES = list(range(1, 17))
CAPS = dict(episode_s=1200, wall_s=3526, physical_steps=32272, public_forecasts=245336)


def alias_hash(call):
    aliases = call['aliases']
    integer_keys = {int(k):v for k,v in aliases.items()}
    assert len(integer_keys) == len(aliases)
    assert all(k == str(int(k)) and type(v) is int for k,v in aliases.items())
    return state.digest(integer_keys)


def build(year):
    builder = prior.build(year)
    old_build = builder.build

    def assemble():
        return builder.old.replace_once(
            old_build(), "OUT=ROOT/'outputs/grid306_calibration_cont'",
            "OUT=ROOT/'outputs/grid307_unstarted_calibration_cont'")
    builder.build = assemble
    return builder


def assert_prior_unstarted(item):
    index = item['index']
    assert index in INDICES
    assert not (OLD / 'runs' / prior.folder(item)).exists()
    assert not (OLD / f'worker_result_{index:02d}.json').exists()
    assert not list((OLD / 'claims').glob(f'replay_{index:03d}.*'))


def prepare():
    assert not OUT.exists()
    assert read(OLD/'pipeline_state.json') == {'phase':'stopped','new_accepted':0}
    close = read(OLD/'closeout.json')
    assert close['unstarted_indices'] == INDICES
    assert close['physical_steps_new'] == 1505 and close['forecasts_new'] == 4664
    assert CAPS['wall_s'] <= 3600-close['wall_s']
    assert CAPS['physical_steps'] == len(INDICES)*2017 <= 34289-close['physical_steps_new']
    assert CAPS['public_forecasts'] == 250000-close['forecasts_new']
    old_source = read(PRIOR/'approval.json')['source_sha256']
    assert all(sha(R/rel) == value for rel,value in old_source.items())
    previous = read(PRIOR/'execution_order.json')
    execution = {}
    fixture_checks = []
    bridge = module('grid307_pure_bridge',SOURCE/'bridge.py')
    for item in previous:
        index = item['index']
        # Keep the spent row as provenance in the old run only.
        if index == 0:
            continue
        assert_prior_unstarted(item)
        rows = ledger(item)
        common = item['target']['common']
        prefix = rows[:common['step']-1]
        assert state.digest(prefix) == common['prefix_signature']
        assert rows[common['step']-1]['before_hash'] == common['state_hash']
        if item['year'] == 2037:
            fixed = json.loads(git('cat-file','blob',GRID305_BLOB))['fixed_representatives'][index]
            rel = fixed['source_call']
            call = json.loads(prior.checked_bytes(rel,item['baseline_input_refs'][rel]))
            decision = call['learned_decisions'][fixed['decision_index']]
            assert type(decision['action']) is int and decision['action'] == 0 and decision['optional_extension'] is True
            assert (call['step'],call['state_hash'],decision['batch'],decision['next_ids']) == (
                common['step'],common['state_hash'],common['batch'],common['next_ids'])
            assert state.digest(decision['raw_features']) == common['raw41_hash']
            assert state.digest(decision['NN_input']) == common['public_NN_input_hash']
            assert state.digest(call['aliases']) == common['aliases_hash']
            corrected = alias_hash(call)
            revised = {**item, 'target':{**item['target'],'common':{**common,'aliases_hash':corrected}}}
            archived = read(R/item['baseline_folder']/'intervention.json')['event']
            event_match = None
            if archived:
                event = archived['common']
                original_call = read(R/item['baseline_folder']/f"call{event['call_index']:03d}.json")
                event_match = alias_hash(original_call) == event['aliases_hash']
                assert event_match
            # The archived JSON provides every input used by this pure gate fixture.
            fixture = bridge.LegacyState(revised,prefix)
            fixture.prefix = prefix
            fixture.prefix_cost = 0.
            for row in prefix:
                fixture.prefix_cost += row['raw_cost']
            fixture.call = {k:common[k] for k in ('step','state_hash','call_index')}
            fixture.call.update(order_hash=state.digest(call['order']), aliases_hash=corrected,
                                rho_limit=call['rho_limit'])
            fixture.offer = {'next_ids':decision['next_ids'],'batch':decision['batch']}
            native = {k:decision[k] for k in ('NN_input','native_q')}
            ordinary = {'continue_':False,'log':{}}
            assert fixture.force(decision['raw_features'],native,ordinary)['continue_'] is True
            assert fixture.event['delivered'] is None
            negative = bridge.LegacyState(revised,prefix)
            negative.prefix,negative.prefix_cost = prefix,fixture.prefix_cost
            negative.call = {**fixture.call,'aliases_hash':common['aliases_hash']}
            negative.offer = fixture.offer
            try:
                negative.force(decision['raw_features'],native,ordinary)
            except ValueError:
                assert negative.event is None and len(negative.candidates) == 1
            else:
                raise AssertionError('Wrong saved-key hash was not rejected')
            fixture_checks.append({'index':index,'old_hash':common['aliases_hash'],
                'restored_hash':corrected,'saved_live_marker_matches':event_match,
                'pure_gate_accepted_restored_and_rejected_old':True})
        else:
            assert item['year'] == 2038
            record = read(R/item['baseline_folder']/'intervention.json')
            assert record['candidates'].count(item['target']) == 1
            revised = item
            fixture_checks.append({'index':index,'complete_marker_unchanged':True})
        execution[str(index)] = revised
    assert list(map(int,execution)) == INDICES
    for year in (2037,2038):
        builder = build(year)
        compile(builder.build(),str(R/'work/grid55/run.py'),'exec')
        compile(builder.instrument_worker((R/'work/grid55/worker.py').read_text(encoding='utf8')),
                str(R/'work/grid55/worker.py'),'exec')

    OUT.mkdir()
    for year in (2037,2038):
        d = read(OLD/f'runtime_design_{year}.json')
        d['caps'] = CAPS
        d['execution_order'] = [None]+list(execution.values())
        save(OUT/f'runtime_design_{year}.json',d)
    for name in ('code_freeze.json','model_references.json'):
        (OUT/name).write_bytes((OLD/name).read_bytes())
    save(SOURCE/'execution_order.json',list(execution.values()))
    save(SOURCE/'offline_review.json',{'passed':True,'prior_indices':INDICES,'fixtures':fixture_checks,
        'excluded_paid_partial_index':0,'new_model_loads':0,'new_predictions':0,'new_physical_steps':0})
    artifacts = [*SOURCE.glob('*.py'), SOURCE/'execution_order.json', SOURCE/'offline_review.json',
                 *OUT.glob('*.json'), OLD/'closeout.json', OLD/'pipeline_state.json']
    save(SOURCE/'approval.json',{'user_reply':'那继续补呗',
        'user_message_reference':'Direct instruction following the GRID306 outcome and missing-label discussion',
        'grid305_scope_git_blob':GRID305_BLOB,'caps':CAPS,'runs':16,'no_retry':True,
        'source_sha256':{**old_source, **{p.relative_to(R).as_posix():sha(p) for p in artifacts},
            **{p.relative_to(R).as_posix():sha(p) for p in PRIOR.glob('*.py')}}})
    print('GRID307: verified 15 repaired legacy markers, 1 unchanged full marker, and no paid duplicates.')


if __name__ == '__main__':
    # Permit the pure bridge to import this instance during standalone preparation.
    import sys
    sys.modules['bindings'] = sys.modules[__name__]
    prepare()
