"""FA05-Q: whitelist-only training record audit. No ML or raw runtime table."""
import ast
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from information import describe, proxy_possible, update_visible

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/fa05_q'
SEEDS = [7, 42, 99]
BUDGET = 86460.0
T = 600.0
PAIRS = list(itertools.combinations(range(11), 2))
PROTOCOL = ROOT/'outputs/fa00/FA00_protocol.json'
CORE = ROOT/'work/fa00/core.py'
UPSTREAM = ROOT/'work/fa00/upstream/Algorithm_Selection_Pareto.py'
RF_SOURCE = ROOT/'work/fa00/upstream/ActiveRFModel.py'


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT)).replace('\\', '/')


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def configs():
    runs = []
    for seed in SEEDS:
        for arm, parent in [('FIXED-100', 'fa02_end'),
                            ('RANDOM-ROW', 'fa03b'),
                            ('COARSE-PC-100', 'fa04')]:
            folder = (ROOT/f'outputs/{parent}/models/FIXED-100_s{seed}_ALL'
                      if parent == 'fa02_end' else
                      ROOT/f'outputs/{parent}/runs/{arm}_s{seed}')
            log = (ROOT/f'outputs/fa00/runs/FIXED-100_s{seed}/actions.jsonl.gz'
                   if parent == 'fa02_end' else folder/'actions.jsonl.gz')
            runs.append(dict(arm=arm, seed=seed, folder=folder, log=log,
                             prefix=folder/'paid_prefix.json' if parent == 'fa02_end' else None))
    return runs


RUNS = configs()
ALLOWED = {PROTOCOL.resolve(), CORE.resolve(), UPSTREAM.resolve(), RF_SOURCE.resolve()}
for run in RUNS:
    for p in [run['folder']/'labels.json', run['folder']/'visible_cache.npz', run['log'], run['prefix']]:
        if p is not None:
            ALLOWED.add(p.resolve())
READS = Counter()


def read_bytes(p):
    p = Path(p).resolve()
    if p not in ALLOWED:
        raise PermissionError(f'Not a whitelisted training/source input: {p}')
    READS[rel(p)] += 1
    return p.read_bytes()


def load_json(p):
    return json.loads(read_bytes(p))


def historical_snapshot():
    # Byte hashes only: these files are never deserialized as outcomes or models.
    files = [p for p in (ROOT/'outputs').rglob('*')
             if p.is_file() and OUT not in p.parents]
    return {rel(p): digest(p) for p in sorted(files)}


def iter_events(p, stop_before=None):
    p = Path(p).resolve()
    if p not in ALLOWED:
        raise PermissionError(p)
    READS[rel(p)] += 1
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        for line in f:
            # Avoid deserializing the original boundary observation, whose
            # feedback exceeded FA02's shortened remaining-budget cutoff.
            match = re.search(r'"event_id"\s*:\s*(\d+)', line)
            if stop_before is not None and match and int(match[1]) >= stop_before:
                break
            e = json.loads(line)
            assert e.get('role', 'train') != 'validation'
            assert not any(k in e for k in ['par10', 'test_par10', 'test_predictions'])
            yield e


def close(a, b, eps=1e-9):
    return abs(float(a)-float(b)) <= eps*max(1.0, abs(float(a)), abs(float(b)))


def category(ka, kb):
    if ka == kb == 2:
        return 'complete_complete'
    if 3 in (ka, kb):
        return 'known_native_timeout'
    if set((ka, kb)) == {1, 2}:
        return 'complete_censor'
    return 'other'


def code_evidence():
    s = read_bytes(UPSTREAM).decode('utf-8')
    tree = ast.parse(s)
    defs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    hard = defs['hard_voting_mechanism']
    hard_timeout_loads = [n.lineno for n in ast.walk(hard)
                          if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                          and n.id == 'timeout_predictor_dict']
    def calls(name):
        return [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == name]
    assert not hard_timeout_loads
    assert not calls('find_runtime_for_each_row')
    assert calls('hard_voting_mechanism') and calls('active_learning_algorithm_regressor')
    lines = s.splitlines()
    evidence = {'source': rel(UPSTREAM), 'sha256': digest(UPSTREAM),
                'hard_voting': {'definition_line': hard.lineno,
                                'call_lines': calls('hard_voting_mechanism'),
                                'timeout_dict_loads_in_body': hard_timeout_loads},
                'find_runtime_for_each_row': {'definition_line': defs['find_runtime_for_each_row'].lineno,
                                              'call_lines': calls('find_runtime_for_each_row')},
                'cost_regressor': {'definition_line': defs['active_learning_algorithm_regressor'].lineno,
                                   'call_lines': calls('active_learning_algorithm_regressor'),
                                   'penalty_type_lines': [i+1 for i,l in enumerate(lines) if 'config["penalty_type"]' in l]},
                'free_label_propagation': {'definition_line': defs['find_cost_free_instances'].lineno,
                                          'call_lines': calls('find_cost_free_instances')},
                'pair_update_source': rel(RF_SOURCE),
                'pair_weight_lines': [i+1 for i,l in enumerate(read_bytes(RF_SOURCE).decode().splitlines())
                                     if 'sample_weight' in l and 'timeout_limit' in l],
                'controlled_core_source': rel(CORE), 'controlled_core_sha256': digest(CORE),
                'controlled_core_relevant_lines': [i+1 for i,l in enumerate(read_bytes(CORE).decode().splitlines())
                                                   if any(x in l for x in ['10*va', '10*vb', 'self.cache.value[rows, a]', 'if i in self.label_rows'])],
                'interpretation': 'Static source call sites verified; upstream was not executed in FA05-Q.'}
    return evidence


def process(run, train, n_rows, detail_writer):
    arm, seed, folder = run['arm'], run['seed'], run['folder']
    labels = load_json(folder/'labels.json')
    assert len(labels) == len(PAIRS)
    with np.load(io.BytesIO(read_bytes(folder/'visible_cache.npz')), allow_pickle=False) as z:
        final_kind, final_value = z['kind'].copy(), z['value'].copy()
    assert final_kind.shape == final_value.shape == (n_rows, 11)
    outside = sorted(set(range(n_rows))-train)
    assert not np.any(final_kind[outside]) and not np.any(final_value[outside])
    state = {}
    first = {}
    event_labels = {}
    paid = []
    charge = 0.0
    cache_hits = 0
    prefix = load_json(run['prefix']) if run['prefix'] else None
    cutoff_event = int(prefix[-1]['event_id']) if prefix else None
    for e in iter_events(run['log'], stop_before=cutoff_event):
        typ = e.get('type')
        if typ == 'observation':
            i, a = int(e['i']), int(e['a'])
            assert i in train and 0 <= a < 11
            assert e['actual_cap_s'] <= 100.0+1e-9
            key = (i,a)
            before = state.get(key, (0,0.0))
            assert int(e['before_kind']) == before[0] and close(e['before_value'], before[1])
            assert not e['cached'] or e['charged_s'] == 0
            after = update_visible(before, e['kind'], e['value'])
            state[key] = after
            charge += float(e['charged_s'])
            assert charge <= BUDGET+1e-6
            if e['cached']:
                cache_hits += 1
            else:
                paid.append({k:e[k] for k in ['event_id','i','a','kind','value','charged_s']})
            first.setdefault(key, after)
        elif typ == 'pair_label':
            key = (int(e['pair']), int(e['i']))
            assert key not in event_labels
            a,b = PAIRS[key[0]]
            assert e['a'] == a and e['b'] == b and key[1] in train
            sa,sb = state[(key[1],a)],state[(key[1],b)]
            assert e['ka'] == sa[0] and e['kb'] == sb[0]
            assert close(e['va'],sa[1]) and close(e['vb'],sb[1])
            event_labels[key] = (e,sa,sb)
    if prefix:
        assert len(prefix) == len(paid)+1
        for x,y in zip(prefix[:-1],paid):
            assert x.keys() == y.keys()
            assert all(close(x[k],y[k]) for k in x)
        e = prefix[-1]
        assert int(e['i']) in train
        key = (int(e['i']),int(e['a']))
        after = update_visible(state.get(key,(0,0.0)),e['kind'],e['value'])
        state[key] = after
        first.setdefault(key,after)
        charge += float(e['charged_s'])
        paid.append(e)
    assert close(charge,BUDGET,1e-11)
    for i in train:
        for a in range(11):
            k,v = state.get((i,a),(0,0.0))
            assert int(final_kind[i,a]) == k and close(final_value[i,a],v)
    model_rows = []
    all_detail = []
    terminal_added = original_labels = changed = refined = 0
    max_weight_error = 0.0
    for pi,((a,b), table) in enumerate(zip(PAIRS,labels)):
        counter = Counter()
        weights = Counter()
        classes = Counter()
        widths = []
        stored_ids = set()
        for i,va,vb,label,w in table:
            i,label = int(i),int(label)
            assert i in train and i not in stored_ids
            stored_ids.add(i)
            key = (pi,i)
            if key in event_labels:
                e,sa,sb = event_labels[key]
                assert all(close(x,y) for x,y in zip([va,vb,label,w],[e['va'],e['vb'],e['label'],e['weight']]))
                source, event_id = 'first_pair_label_event',int(e['event_id'])
                original_labels += 1
            else:
                assert prefix is not None, 'Only FA02-END has separately generated terminal labels'
                sa,sb = state[(i,a)],state[(i,b)]
                assert close(va,sa[1]) and close(vb,sb[1])
                source, event_id = 'terminal_paid_cache',None
                terminal_added += 1
            ka,kb = sa[0],sb[0]
            assert ka and kb and 2 in (ka,kb)
            weight_check = abs((va if ka == 2 else 10*va)-(vb if kb == 2 else 10*vb))
            assert close(w,weight_check)
            max_weight_error = max(max_weight_error,abs(w-weight_check))
            init = describe(ka,sa[1],kb,sb[1],T)
            assert init['preference'] != 'unidentified'
            assert (label == 0 and init['preference'] in ['a_better','a_no_worse','tie']) or (label == 1 and init['preference'] in ['b_better','b_no_worse','tie'])
            end = describe(int(final_kind[i,a]),final_value[i,a],int(final_kind[i,b]),final_value[i,b],T)
            assert end['preference'] == init['preference']
            is_changed = sa != state[(i,a)] or sb != state[(i,b)]
            is_refined = (end['regret_lower'] > init['regret_lower']
                          or end['regret_upper'] < init['regret_upper'])
            changed += int(is_changed); refined += int(is_refined)
            cat = category(ka,kb)
            counter[cat] += 1; weights[cat] += w; classes[label] += int(w>0)
            counter['informative_labels'] += int(w>0)
            counter['zero_weight_labels'] += int(w==0)
            counter['regret_exact'] += int(init['regret_exact'])
            counter['regret_unidentified'] += int(not init['regret_exact'])
            counter['proxy_outside_feasible_regret'] += int(not proxy_possible(init,w))
            counter['terminal_added'] += int(source=='terminal_paid_cache')
            counter['later_changed'] += int(is_changed)
            counter['later_refined'] += int(is_refined)
            widths.append(init['regret_upper']-init['regret_lower'])
            d = dict(arm=arm,seed=seed,pair=pi,a=a,b=b,i=i,label=label,proxy_weight=w,
                     first_source=source,event_id=event_id,ka_first=ka,kb_first=kb,
                     va_recorded=va,vb_recorded=vb,va_first_visible=sa[1],vb_first_visible=sb[1],
                     ka_final=int(final_kind[i,a]),kb_final=int(final_kind[i,b]),
                     va_final_visible=float(final_value[i,a]),vb_final_visible=float(final_value[i,b]),
                     category=cat,preference=init['preference'],regret_exact=init['regret_exact'],
                     regret_lower=init['regret_lower'],regret_upper=init['regret_upper'],
                     native_regret_components=json.dumps(init['regret_segments'],separators=(',',':')),
                     final_regret_lower=end['regret_lower'],final_regret_upper=end['regret_upper'],
                     later_feedback_changed=is_changed,later_range_refined=is_refined,
                     proxy_is_possible_native_regret=proxy_possible(init,w))
            detail_writer.writerow(d); all_detail.append(d)
        assert len(stored_ids) == len(table)
        pending = Counter()
        for i in train:
            if i in stored_ids:
                continue
            ka,kb = int(final_kind[i,a]),int(final_kind[i,b])
            if not ka or not kb:
                pending['unlabelled_at_least_one_unobserved'] += 1
            elif ka == kb == 1:
                pending['unlabelled_both_censored'] += 1
            elif 3 in (ka,kb):
                pending['unlabelled_native_timeout'] += 1
            else:
                info = describe(ka,final_value[i,a],kb,final_value[i,b],T)
                assert info['preference'] == 'unidentified', 'A legal label was lost'
                pending['unlabelled_both_paid_preference_unresolved'] += 1
        total_weight = math.fsum(weights.values())
        row = dict(arm=arm,seed=seed,pair=pi,a=a,b=b,labels=len(table),total_raw_weight=total_weight,
                   complete_complete=counter['complete_complete'],complete_censor=counter['complete_censor'],
                   known_native_timeout=counter['known_native_timeout'],other=counter['other'],
                   censor_label_fraction=counter['complete_censor']/len(table) if table else None,
                   censor_raw_weight_fraction=weights['complete_censor']/total_weight if total_weight else None,
                   complete_complete_weight=weights['complete_complete'],complete_censor_weight=weights['complete_censor'],
                   zero_weight_labels=counter['zero_weight_labels'],informative_labels=counter['informative_labels'],
                   label0_informative=classes[0],label1_informative=classes[1],
                   empty_model=not bool(counter['informative_labels']),
                   single_class_model=sum(classes[c]>0 for c in [0,1])==1,
                   regret_exact=counter['regret_exact'],regret_unidentified=counter['regret_unidentified'],
                   proxy_outside_feasible_regret=counter['proxy_outside_feasible_regret'],
                   terminal_added=counter['terminal_added'],later_changed=counter['later_changed'],later_refined=counter['later_refined'],
                   regret_range_width_median=float(np.median(widths)) if widths else None,
                   training_pair_population=len(train),**{k:pending[k] for k in [
                       'unlabelled_at_least_one_unobserved','unlabelled_both_censored',
                       'unlabelled_native_timeout','unlabelled_both_paid_preference_unresolved']})
        assert row['labels'] + sum(pending.values()) == len(train)
        model_rows.append(row)
    assert set(event_labels) == {(pi,int(r[0])) for pi,t in enumerate(labels) for r in t if (pi,int(r[0])) in event_labels}
    return model_rows,dict(arm=arm,seed=seed,paid_executions=len(paid),cache_hits=cache_hits,charged_s=charge,
                          original_first_labels=original_labels,terminal_added=terminal_added,
                          changed_after_label=changed,refined_after_label=refined,
                          max_weight_reconstruction_error=max_weight_error,
                          cache_verified_slots=len(train)*11,
                          nontraining_cache_nonzero=0),all_detail


def write_csv(p, rows):
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    inputs={rel(p):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(ALLOWED)}
    save(OUT/'FA05_input_manifest.json',dict(stage='FA05-Q',budget_s=BUDGET,native_cutoff_s=T,
         allowlisted_semantic_inputs=inputs,
         exclusion='No raw runtime matrix, validation/test outcomes, predictions, or model objects.',
         historical_hash_mode='Raw bytes only for immutability; no outcome/model deserialization.',
         code_sha256={rel(p):digest(p) for p in Path(__file__).parent.glob('*.py')}))
    before=historical_snapshot(); save(OUT/'FA05_historical_before.json',before)
    protocol=load_json(PROTOCOL)
    assert protocol['native_cutoff_s']==T and protocol['timeout_score']==6000
    assert len(protocol['algorithms'])==11
    train=set(map(int,protocol['train'])); n_rows=len(protocol['ids'])
    fields=['arm','seed','pair','a','b','i','label','proxy_weight','first_source','event_id',
            'ka_first','kb_first','va_recorded','vb_recorded','va_first_visible','vb_first_visible',
            'ka_final','kb_final','va_final_visible','vb_final_visible','category','preference',
            'regret_exact','regret_lower','regret_upper','native_regret_components',
            'final_regret_lower','final_regret_upper','later_feedback_changed','later_range_refined',
            'proxy_is_possible_native_regret']
    models=[]; runs=[]
    with gzip.open(OUT/'FA05_label_information.csv.gz','wt',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for run in RUNS:
            rows,checks,_=process(run,train,n_rows,writer)
            models.extend(rows); runs.append(checks)
            print(run['arm'],run['seed'],'labels',sum(r['labels'] for r in rows),'terminal_added',checks['terminal_added'],flush=True)
    assert len(models)==495
    write_csv(OUT/'FA05_pair_information.csv',models)
    write_csv(OUT/'FA05_run_checks.csv',runs)
    save(OUT/'FA05_code_path_evidence.json',code_evidence())
    summary=[]
    for run in runs:
        ms=[m for m in models if m['arm']==run['arm'] and m['seed']==run['seed']]
        def median(k): return float(np.median([m[k] for m in ms]))
        def total(k): return sum(m[k] for m in ms)
        summary.append(dict(arm=run['arm'],seed=run['seed'],models=len(ms),labels=total('labels'),
          informative_labels=total('informative_labels'),complete_censor=total('complete_censor'),
          censor_label_fraction_model_median=median('censor_label_fraction'),
          censor_weight_fraction_model_median=median('censor_raw_weight_fraction'),
          censor_weight_fraction_model_min=min(m['censor_raw_weight_fraction'] for m in ms),
          censor_weight_fraction_model_max=max(m['censor_raw_weight_fraction'] for m in ms),
          regret_exact=total('regret_exact'),regret_unidentified=total('regret_unidentified'),
          proxy_outside_feasible_regret=total('proxy_outside_feasible_regret'),
          terminal_added=run['terminal_added'],later_changed=run['changed_after_label'],
          later_refined=run['refined_after_label'],single_class_models=total('single_class_model'),
          empty_models=total('empty_model'),unlabelled_both_censored=total('unlabelled_both_censored')))
    write_csv(OUT/'FA05_method_seed_summary.csv',summary)
    after=historical_snapshot()
    assert before==after, 'A historical artifact changed'
    assert all(digest(ROOT/k)==v['sha256'] for k,v in inputs.items())
    result=dict(stage='FA05-Q',status='training_information_audit_passed',runs=9,models=495,
                labels=sum(s['labels'] for s in summary),pair_population=9*55*len(train),
                native_regret_unidentified=sum(s['regret_unidentified'] for s in summary),
                proxy_outside_feasible_regret=sum(s['proxy_outside_feasible_regret'] for s in summary),
                terminal_added=sum(s['terminal_added'] for s in summary),
                later_changed=sum(s['later_changed'] for s in summary),
                later_refined=sum(s['later_refined'] for s in summary),
                historical_files_hashed=len(before),historical_changed=0,
                no_new_acquisition=True,no_fit=True,no_prediction_evaluation=True,
                semantic_read_log=dict(READS),wall_s=time.perf_counter()-started,
                summary=summary,
                interpretation='Information qualification only; not learner influence, performance loss, learnability, or novelty proof.')
    save(OUT/'FA05_training_audit.json',result)
    print(json.dumps({k:result[k] for k in ['status','labels','native_regret_unidentified','terminal_added','wall_s']}),flush=True)


if __name__=='__main__':
    main()
