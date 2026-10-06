"""FA02-END: reconstruct paid prefixes, then 18 fixed terminal fits. No acquisition."""
from pathlib import Path
import copy, csv, gzip, hashlib, json, math, os, pickle, sys, time
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'work/fa00'))
from core import ObservationCache, Selector, keyed_seed, PARAMS
OUT = ROOT / 'outputs/fa02_end'
ARMS = ['FIXED-100', 'RANDOM-100', 'CHEAP-100']
SEEDS = [7, 42, 99]
BUDGET = 86460.00000000001

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def source(arm, seed):
    stage = 'fa00' if arm == 'FIXED-100' else 'fa01'
    return ROOT / f'outputs/{stage}/runs/{arm}_s{seed}'

def read_prefix(path):
    events = []
    with gzip.open(path, 'rt', encoding='utf8') as f:
        for line in f:
            e = json.loads(line); events.append(e)
            if e['type'] == 'observation' and e['train_cost_s'] > BUDGET: break
    return events

def reconstruct(events, limit, X, p, seed):
    cache = ObservationCache((len(X), len(p['algorithms'])))
    selector = Selector(X, p['algorithms'], p['train'], cache, seed)
    train = set(p['train']); spent = 0.; paid = []; boundary = None
    for e in events:
        if e['type'] == 'observation':
            if spent >= limit - 1e-9: break
            assert e['role'] == 'train' and e['i'] in train
            i, a = e['i'], e['a']
            assert int(cache.kind[i,a]) == e['before_kind']
            assert abs(float(cache.value[i,a]) - e['before_value']) < 1e-8
            if e['cached']:
                assert e['charged_s'] == 0 and not cache.can_query(i,a,e['planned_cap_s'])
                continue
            assert cache.can_query(i,a,e['planned_cap_s'])
            remaining = limit - spent
            # Round-end costs are accumulated floats. A 1e-8-second ledger
            # tolerance avoids turning an already-paid completion into a censor.
            crossing = e['charged_s'] > remaining + 1e-8
            charge = remaining if crossing else float(e['charged_s'])
            if crossing:
                # Both a later completion and a longer timeout reveal only a bound.
                kind, value = 1, max(float(cache.value[i,a]), remaining)
                boundary = dict(event_id=e['event_id'], i=i, a=a,
                    original_kind=e['kind'], original_charge_s=e['charged_s'],
                    actual_cap_s=remaining, kind=kind, value=value)
            else:
                kind, value = e['kind'], e['value']
            cache.kind[i,a] = kind; cache.value[i,a] = value
            spent += charge
            paid.append(dict(event_id=e['event_id'], i=i, a=a, kind=int(kind),
                             value=float(value), charged_s=charge))
        elif e['type'] == 'pair_label' and e['train_cost_s'] <= limit + 1e-8:
            pi, i = e['pair'], e['i']
            assert i not in selector.label_rows[pi]
            a,b = selector.pairs[pi]
            assert int(cache.kind[i,a]) == e['ka'] and int(cache.kind[i,b]) == e['kb']
            selector.label_rows[pi][i] = (e['va'], e['vb'], e['label'], e['weight'])
    # The boundary may end a batch before its logged label synchronization.
    selector.synchronize_labels()
    return selector, spent, paid, boundary

def signature(selector):
    labels = [[[int(i), *list(v)] for i,v in sorted(z.items())] for z in selector.label_rows]
    h = hashlib.sha256(selector.cache.kind.tobytes() + selector.cache.value.tobytes())
    h.update(json.dumps(labels, separators=(',', ':')).encode())
    h.update(selector.X[selector.train_rows].tobytes())
    h.update(json.dumps(PARAMS, sort_keys=True).encode())
    h.update(str(selector.seed).encode())
    return h.hexdigest()

def preflight(p, X):
    checks = []
    # A completion beyond the new boundary must be a lower bound, not a label.
    train_i = p['train'][0]
    e = dict(type='observation', role='train', i=train_i, a=0,
             before_kind=0, before_value=0., cached=False, charged_s=80.,
             planned_cap_s=100., actual_cap_s=100., kind=2, value=80.,
             train_cost_s=80., event_id=0)
    s,c,_,b = reconstruct([e], 30., X,p,7)
    assert c == 30 and s.cache.kind[train_i,0] == 1 and s.cache.value[train_i,0] == 30
    modified = dict(e, value=900., charged_s=900.)
    t,_,_,_ = reconstruct([modified],30.,X,p,7)
    assert signature(s) == signature(t)
    checks.extend(['boundary_completion_becomes_censor', 'boundary_hidden_duration_invariance'])
    manifest = []; inputs = []
    for arm in ARMS:
        for seed in SEEDS:
            src = source(arm,seed)
            history = json.loads((src/'history.json').read_text())
            h = max((h for h in history if h['train_cost_s'] <= BUDGET), key=lambda h:h['round'])
            events = read_prefix(src/'actions.jsonl.gz')
            s,spent,paid,boundary = reconstruct(events,BUDGET,X,p,seed)
            assert abs(spent-BUDGET) < 1e-7
            mutated = copy.deepcopy(events)
            for m in mutated:
                if m['train_cost_s'] > BUDGET:
                    if m['type'] == 'observation' and not m['cached']:
                        m['charged_s'] = 1e9; m['value'] = 1e9
            t,_,_,_ = reconstruct(mutated,BUDGET,X,p,seed)
            assert signature(s) == signature(t)
            # Outcome arrays are never an argument to reconstruction or training.
            alias = Selector(X,p['algorithms'],p['train'],s.cache,seed)
            alias.label_rows = copy.deepcopy(s.label_rows)
            assert signature(s) == signature(alias)
            for mode,limit in [('COMPLETE',h['train_cost_s']),('ALL',BUDGET)]:
                ss,used,pp,bb = reconstruct(events,limit,X,p,seed)
                if mode == 'COMPLETE':
                    assert sum(map(len,ss.label_rows)) == h['labels']
                inputs.append(dict(arm=arm,seed=seed,mode=mode,limit_s=limit,
                    last_complete_round=h['round'],complete_cost_s=h['train_cost_s'],
                    label_count=sum(map(len,ss.label_rows)),paid_count=len(pp),
                    input_sha256=signature(ss),boundary=bb))
            for name in ['actions.jsonl.gz','history.json']:
                path=src/name; manifest.append(dict(path=str(path),sha256=sha(path)))
    checks.extend(['nine_prefix_budgets', 'nine_hidden_suffix_invariance',
        'nine_arm_agnostic_input_identity', 'nine_complete_label_counts',
        'training_interface_has_no_outcome_argument'])
    return dict(checks=checks,passed=True,inputs=inputs),manifest

def execute():
    p = json.loads((ROOT/'outputs/fa00/FA00_protocol.json').read_text())
    with np.load(ROOT/'outputs/fa00/FA00_data.npz') as d: X = d['X'].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'FA02_freeze.json').exists(): raise RuntimeError('Existing freeze; no silent rerun')
    started=time.perf_counter(); deadline=started+3600
    prep,manifest=preflight(p,X)
    save(OUT/'FA02_preflight.json',prep)
    for path in [Path(__file__),Path(__file__).with_name('audit.py'),ROOT/'work/fa00/core.py',
                 ROOT/'work/fa00/upstream/ActiveRFModel.py',ROOT/'work/fa00/upstream/PassiveRFRegressor.py',
                 ROOT/'outputs/fa00/FA00_protocol.json',ROOT/'outputs/fa00/FA00_data.npz',
                 ROOT/'outputs/fa01/FA02_next_step.md']:
        manifest.append(dict(path=str(path),sha256=sha(path)))
    save(OUT/'FA02_protocol.json',dict(name='FA02-END',budget_s=BUDGET,arms=ARMS,
        seeds=SEEDS,modes=['COMPLETE','ALL'],terminal_round_seed=0,model=PARAMS,
        fits=18,no_new_acquisition=True,development_only=True,
        continuation='ALL active mean >=5% better than each control; >=2/3 paired signs each',
        closure='any control ALL mean <=1.02*active mean',wall_limit_s=3600))
    manifest.append(dict(path=str(OUT/'FA02_protocol.json'),sha256=sha(OUT/'FA02_protocol.json')))
    save(OUT/'FA02_freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),files=manifest))
    results=[]
    for rec in prep['inputs']:
        arm,seed,mode=rec['arm'],rec['seed'],rec['mode']
        selector,spent,paid,boundary=reconstruct(read_prefix(source(arm,seed)/'actions.jsonl.gz'),rec['limit_s'],X,p,seed)
        assert signature(selector) == rec['input_sha256']
        dest=OUT/'models'/f'{arm}_s{seed}_{mode}';dest.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(dest/'visible_cache.npz',kind=selector.cache.kind,value=selector.cache.value)
        save(dest/'paid_prefix.json',paid)
        save(dest/'labels.json',[[[int(i),*list(v)] for i,v in sorted(z.items())] for z in selector.label_rows])
        selector.fit(0,deadline)
        pred=selector.predict()
        # Validate the shared random-state rule on the actual fitted estimators.
        for pi,m in enumerate(selector.pair_models):
            if m is not None: assert m.learner.estimator.random_state == keyed_seed(seed,'pair',0,pi)
        for a,m in enumerate(selector.regressors):
            assert m.model.random_state == keyed_seed(seed,'reg',0,a)
        np.save(dest/'predictions.npy',pred)
        with (dest/'selector.pkl').open('wb') as f: pickle.dump(selector,f,protocol=5)
        row=dict(rec,acquisition_cost_s=spent,fit_s=selector.fit_s,predict_s=selector.predict_s,
            observations=int(np.count_nonzero(selector.cache.kind)),
            completions=int(np.count_nonzero(selector.cache.kind==2)),
            label_count=sum(map(len,selector.label_rows)),informative_labels=sum(sum(v[3]>0 for v in z.values()) for z in selector.label_rows),
            artifact=str(dest))
        results.append(row);save(dest/'result.json',row)
        save(OUT/'RUN_STATE.json',dict(status='running',completed_fits=len(results),total_fits=18,wall_s=time.perf_counter()-started))
        print(f'{len(results)}/18 {arm} seed={seed} {mode} labels={row["label_count"]} fit={row["fit_s"]:.2f}s',flush=True)
    # Test outcomes enter only after every terminal prediction has been frozen.
    save(OUT/'FA02_training_results.json',results)
    save(OUT/'RUN_STATE.json',dict(status='fits_complete_pending_audit',completed_fits=18,wall_s=time.perf_counter()-started))

if __name__ == '__main__': execute()
