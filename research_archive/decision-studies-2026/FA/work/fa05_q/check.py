"""Separate FA05-Q checker: case formulas, original records, no audit helper import."""
import csv
import gzip
import hashlib
import itertools
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/fa05_q'
PAIRS = list(itertools.combinations(range(11), 2))

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def near(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9), (a,b)

def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def main():
    begin=time.perf_counter()
    protocol=read(ROOT/'outputs/fa00/FA00_protocol.json')
    train=set(protocol['train'])
    details=defaultdict(dict)
    with gzip.open(OUT/'FA05_label_information.csv.gz','rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            key=(r['arm'],int(r['seed']))
            pos=(int(r['pair']),int(r['i']))
            assert pos not in details[key]
            details[key][pos]=r
    totals={'labels':0,'unidentified':0,'proxy_outside':0,'terminal_added':0,'later_changed':0}
    aggregate={}
    paid_total=cache_total=0
    run_table={(r['arm'],int(r['seed'])):r for r in rows(OUT/'FA05_run_checks.csv')}
    for (arm,seed),table in details.items():
        folder=(ROOT/f'outputs/fa02_end/models/FIXED-100_s{seed}_ALL' if arm=='FIXED-100'
                else ROOT/f"outputs/{'fa03b' if arm=='RANDOM-ROW' else 'fa04'}/runs/{arm}_s{seed}")
        path=(ROOT/f'outputs/fa00/runs/FIXED-100_s{seed}/actions.jsonl.gz' if arm=='FIXED-100'
              else folder/'actions.jsonl.gz')
        prefix=read(folder/'paid_prefix.json') if arm=='FIXED-100' else None
        boundary=int(prefix[-1]['event_id']) if prefix else None
        visible={}; first={}; fees=[]; paid=cache=0
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                match=re.search(r'"event_id"\s*:\s*(\d+)',line)
                if boundary is not None and match and int(match[1])>=boundary:
                    break  # original boundary feedback must not enter this audit
                e=json.loads(line)
                if e['type']=='observation':
                    p=(int(e['i']),int(e['a']))
                    assert p[0] in train and e.get('role','train')=='train'
                    old=visible.get(p,(0,0.0))
                    assert old[0]==int(e['before_kind']); near(old[1],e['before_value'])
                    kind=int(e['kind']); value=float(e['value'])
                    if kind==1 and old[0]==1: value=max(old[1],value)
                    if old[0] in (2,3):
                        assert kind==old[0]; near(old[1],value)
                    visible[p]=(kind,value)
                    fees.append(float(e['charged_s']))
                    if e['cached']:
                        near(e['charged_s'],0);cache+=1
                    else:paid+=1
                elif e['type']=='pair_label':
                    p=(int(e['pair']),int(e['i']))
                    assert p not in first
                    a,b=PAIRS[p[0]]
                    sa,sb=visible[(p[1],a)],visible[(p[1],b)]
                    assert (e['ka'],e['kb'])==(sa[0],sb[0])
                    near(e['va'],sa[1]);near(e['vb'],sb[1])
                    first[p]=(e,sa,sb)
        if prefix:
            assert len(prefix)==paid+1
            e=prefix[-1]; p=(int(e['i']),int(e['a']))
            visible[p]=(int(e['kind']),float(e['value']))
            fees.append(float(e['charged_s']));paid+=1
        near(math.fsum(fees),86460)
        rt=run_table[(arm,seed)]
        near(rt['paid_executions'],paid);near(rt['cache_hits'],cache)
        paid_total+=paid;cache_total+=cache
        with np.load(folder/'visible_cache.npz',allow_pickle=False) as z:
            kinds=z['kind'];values=z['value']
            for i in range(len(protocol['ids'])):
                for a in range(11):
                    k,v=visible.get((i,a),(0,0))
                    assert kinds[i,a]==k;near(values[i,a],v)
        labels=read(folder/'labels.json')
        assert sum(map(len,labels))==len(table)
        for pi,stored in enumerate(labels):
            a,b=PAIRS[pi]
            acc=defaultdict(float); widths=[]
            for i,va,vb,y,w in stored:
                i=int(i);r=table[(pi,i)];assert i in train
                near(r['proxy_weight'],w);assert int(r['label'])==y
                if (pi,i) in first:
                    e,sa,sb=first[(pi,i)]
                    assert r['first_source']=='first_pair_label_event'
                    assert int(r['event_id'])==e['event_id']
                    near(e['weight'],w);near(e['va'],va);near(e['vb'],vb)
                else:
                    assert prefix is not None
                    assert r['first_source']=='terminal_paid_cache'
                    sa,sb=visible[(i,a)],visible[(i,b)]
                    acc['terminal_added']+=1
                ka,x=sa;kb,z=sb
                assert (ka,kb) in ((2,2),(1,2),(2,1))
                assert (int(r['ka_first']),int(r['kb_first']))==(ka,kb)
                near(r['va_first_visible'],x);near(r['vb_first_visible'],z)
                near(w,abs((va if ka==2 else 10*va)-(vb if kb==2 else 10*vb)))
                acc['labels']+=1;acc['weight']+=w
                if ka==kb==2:
                    lo=hi=abs(x-z);possible=math.isclose(w,lo,abs_tol=1e-9,rel_tol=1e-9)
                    acc['complete_complete']+=1;acc['complete_weight']+=w
                    assert r['regret_exact']=='True'
                else:
                    v,l=(x,z) if ka==2 else (z,x)
                    assert l>=v and l<600
                    lo=l-v;hi=6000-v
                    possible=(w>lo and w<=600-v) or math.isclose(w,hi,abs_tol=1e-9,rel_tol=1e-9)
                    acc['complete_censor']+=1;acc['censor_weight']+=w
                    acc['unidentified']+=1
                    assert r['regret_exact']=='False'
                    assert y==(0 if ka==2 else 1)
                assert possible==(r['proxy_is_possible_native_regret']=='True')
                acc['proxy_outside']+=int(not possible)
                near(r['regret_lower'],lo);near(r['regret_upper'],hi)
                widths.append(hi-lo)
                is_changed=(sa!=visible[(i,a)] or sb!=visible[(i,b)])
                assert is_changed==(r['later_feedback_changed']=='True')
                acc['later_changed']+=int(is_changed)
            labelled={int(r[0]) for r in stored}
            unknown=both_censored=unresolved=0
            for i in train-labelled:
                ka=visible.get((i,a),(0,0))[0];kb=visible.get((i,b),(0,0))[0]
                if 0 in (ka,kb):unknown+=1
                elif ka==kb==1:both_censored+=1
                elif (ka,kb) in ((1,2),(2,1)):
                    x=visible[(i,a)][1];z=visible[(i,b)][1]
                    v,l=(x,z) if ka==2 else (z,x)
                    assert l<v
                    unresolved+=1
                else:raise AssertionError(('unexpected pending category',ka,kb))
            assert len(stored)+unknown+both_censored+unresolved==len(train)
            acc['unlabelled_at_least_one_unobserved']=unknown
            acc['unlabelled_both_censored']=both_censored
            acc['unlabelled_both_paid_preference_unresolved']=unresolved
            acc['width_median']=statistics.median(widths)
            aggregate[(arm,seed,pi)]=acc
            for k in totals:totals[k]+=int(acc[k])
    models=rows(OUT/'FA05_pair_information.csv')
    assert len(models)==len(aggregate)==495
    for r in models:
        acc=aggregate[(r['arm'],int(r['seed']),int(r['pair']))]
        for k in ['labels','complete_complete','complete_censor','terminal_added',
                  'unlabelled_at_least_one_unobserved','unlabelled_both_censored',
                  'unlabelled_both_paid_preference_unresolved']:
            near(r[k],acc[k])
        near(r['regret_unidentified'],acc['unidentified'])
        near(r['total_raw_weight'],acc['weight'])
        near(r['complete_complete_weight'],acc['complete_weight'])
        near(r['complete_censor_weight'],acc['censor_weight'])
        near(r['proxy_outside_feasible_regret'],acc['proxy_outside'])
        near(r['regret_range_width_median'],acc['width_median'])
    for r in rows(OUT/'FA05_method_seed_summary.csv'):
        vals=[aggregate[(r['arm'],int(r['seed']),p)] for p in range(55)]
        near(r['censor_label_fraction_model_median'],statistics.median(v['complete_censor']/v['labels'] for v in vals))
        near(r['censor_weight_fraction_model_median'],statistics.median(v['censor_weight']/v['weight'] for v in vals))
    expected=read(OUT/'FA05_training_audit.json')
    for our,their in [('labels','labels'),('unidentified','native_regret_unidentified'),
                      ('proxy_outside','proxy_outside_feasible_regret'),('terminal_added','terminal_added'),
                      ('later_changed','later_changed')]:near(totals[our],expected[their])
    hist=read(OUT/'FA05_historical_before.json')
    assert all(digest(ROOT/p)==h for p,h in hist.items())
    manifest=read(OUT/'FA05_input_manifest.json')
    assert all(digest(ROOT/p)==v['sha256'] for p,v in manifest['allowlisted_semantic_inputs'].items())
    source=read(ROOT/'work/fa00/source_manifest.json')
    verified=[]
    for r in source['files']:
        if r['path'] in ['Algorithm_Selection_Pareto.py','ActiveRFModel.py']:
            assert digest(ROOT/'work/fa00/upstream'/r['path'])==r['sha256']
            verified.append(r['path'])
    result=dict(status='passed',models=495,runs=9,totals=totals,
                paid_executions=paid_total,cache_hits=cache_total,
                historical_files_unchanged=len(hist),source_hash_verified=verified,
                scope='Separate case formulas plus original training records; no run.py/information.py import, no model fit, no score evaluation.',
                wall_s=time.perf_counter()-begin)
    (OUT/'FA05_independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
