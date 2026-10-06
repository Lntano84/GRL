"""FA06 legal-feedback training interface. No test outcomes are parsed here."""
import csv
import hashlib
import itertools
import json
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
import arff
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/fa06'
DATA=ROOT/'work/fa00/upstream/DATASETS/ASP-POTASSCO'
sys.path.insert(0,str(ROOT/'work/fa00'))
from core import ObservationCache,Ledger,ReplayEnvironment,Exhausted,keyed_seed
PAIRS=list(itertools.combinations(range(11),2))
ARMS=['ORIGINAL','UNIT','NATIVE-UPPER']

def row_batch(cache,train_rows,seed,round_no,size=11,cap=100):
    pool=np.array(sorted(int(i) for i in train_rows if any(
        cache.can_query(int(i),a,cap) for a in range(cache.kind.shape[1]))),dtype=int)
    if not len(pool):return pool,[],[]
    rng=np.random.default_rng(keyed_seed(seed,'row_acquire',round_no))
    selected=pool[rng.permutation(len(pool))[:min(size,len(pool))]].copy()
    np.random.default_rng(keyed_seed(seed,'row_execute',round_no)).shuffle(selected)
    work=[]
    for i in selected:
        algorithms=np.arange(cache.kind.shape[1])
        np.random.default_rng(keyed_seed(seed,'row_alg',round_no,int(i))).shuffle(algorithms)
        work.extend([[int(i),int(a)] for a in algorithms if cache.can_query(int(i),int(a),cap)])
    return pool,selected.tolist(),work

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)

def dump(p,x):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp')
    with temp.open('wb') as f:pickle.dump(x,f,protocol=5);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)

def design():
    d=json.loads((ROOT/'outputs/fa06_design/FA06_design_freeze.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in d['input_hashes'].items())
    assert d['acquisition_budget_s']==86460 and d['cap_s']==100
    return d

def source_rows(name):
    with (DATA/f'{name}.arff').open(encoding='utf-8') as f:
        for line in f:
            if line.strip().lower()=='@data':break
        for r in csv.reader(f):
            if r and not r[0].lstrip().startswith('%'):yield r

def train_truth(d):
    """Parse numeric outcomes only for frozen training rows; rest are dummy."""
    mapping={name:i for i,name in enumerate(d['ids'])}
    algorithms={name:a for a,name in enumerate(d['algorithms'])}
    train=set(d['train']);seen=set()
    runtime=np.full((len(mapping),11),600.,dtype=float)
    ok=np.zeros(runtime.shape,dtype=bool)
    for r in source_rows('algorithm_runs'):
        if int(r[1])!=1:continue
        i=mapping[r[0]]
        if i not in train:continue  # no numeric conversion/status use outside train
        a=algorithms[r[2]];assert (i,a) not in seen;seen.add((i,a))
        value=float(r[3]);status=r[4]
        assert status in ['ok','timeout'] and 0<value<=600
        assert status=='ok' or value==600
        runtime[i,a]=value;ok[i,a]=status=='ok'
    assert len(seen)==len(train)*11
    return runtime,ok

def raw_features(d):
    with (DATA/'feature_values.arff').open(encoding='utf-8') as f:tab=arff.load(f)
    values={r[0]:r[2:] for r in tab['data'] if r[1]==1}
    assert set(values)==set(d['ids'])
    x=np.array([[float(v) if v is not None else np.nan for v in values[i]] for i in d['ids']])
    assert not np.isinf(x).any()
    return x

def features(d):
    raw=raw_features(d);train=np.array(d['train']);keep=~np.all(np.isnan(raw[train]),axis=0)
    imp=SimpleImputer(strategy='median').fit(raw[train][:,keep])
    imputed=imp.transform(raw[:,keep])
    scale=StandardScaler().fit(imputed[train]);x=scale.transform(imputed)
    assert np.isfinite(x).all()
    return x,dict(raw_features=raw,keep=keep,medians=imp.statistics_,mean=scale.mean_,scale=scale.scale_)

def make_labels(cache,train):
    tables=[]
    for a,b in PAIRS:
        table=[]
        for i in sorted(train):
            ka,kb=int(cache.kind[i,a]),int(cache.kind[i,b])
            if not ka or not kb or 2 not in (ka,kb):continue
            x,z=float(cache.value[i,a]),float(cache.value[i,b])
            if ka!=2 and x<z-1e-10 or kb!=2 and z<x-1e-10:continue
            va,vb=x,z
            if ka!=2 and va==vb:va=float(np.nextafter(va,np.inf))
            if kb!=2 and vb==va:vb=float(np.nextafter(vb,np.inf))
            y=0 if va<=vb else 1
            original=abs((va if ka==2 else 10*va)-(vb if kb==2 else 10*vb))
            if original<=0:continue
            assert (ka==kb==2) or (y==0 and ka==2) or (y==1 and kb==2)
            upper=abs(va-vb) if ka==kb==2 else 6000-(x if ka==2 else z)
            table.append(dict(i=int(i),label=y,ka=ka,kb=kb,va=va,vb=vb,
                              va_visible=x,vb_visible=z,ORIGINAL=original,UNIT=1.,
                              **{'NATIVE-UPPER':upper}))
        tables.append(table)
    return tables

def support_hash(tables,x):
    support=[[[r['i'],r['label'],r['ka'],r['kb'],r['va_visible'],r['vb_visible']] for r in t] for t in tables]
    h=hashlib.sha256(json.dumps(support,separators=(',',':')).encode())
    rows=sorted({r['i'] for t in tables for r in t});h.update(x[rows].tobytes())
    return h.hexdigest()

def hard_vote(pair_votes):
    return np.array([Counter(row).most_common(1)[0][0] for row in np.asarray(pair_votes).T],dtype=np.int16)

def historical():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'outputs').rglob('*'))
            if p.is_file() and OUT not in p.parents}

def collect(d,runtime,ok,seed,path=None,deadline=None):
    """Real runner, from empty state or replayed paid prefix; no model access."""
    import time
    start=time.perf_counter();train=set(d['train'])
    cache=ObservationCache(runtime.shape);ledger=Ledger(d['acquisition_budget_s'])
    phase='init';work=[];cursor=0;round_no=0;events=[];prior_wall=0.
    dest=Path(path) if path else None;log=None
    if dest:
        dest.mkdir(parents=True,exist_ok=True);p=dest/'actions.jsonl'
        if p.exists():
            valid=0
            with p.open('rb') as f:
                for line in f:
                    if not line.endswith(b'\n'):break
                    e=json.loads(line);events.append(e);valid+=len(line)
            with p.open('r+b') as f:f.truncate(valid)
            for e in events:
                ledger.event_id=e['event_id']+1;round_no=e['round'];ledger.round=round_no
                prior_wall=e['active_wall_s']
                if e['type']=='batch':work=e['work'];cursor=0;phase='execute'
                elif e['type']=='observation':
                    i,a=e['i'],e['a'];assert i in train
                    assert cache.kind[i,a]==e['before_kind'];assert cache.value[i,a]==e['before_value']
                    cache.kind[i,a]=e['kind'];cache.value[i,a]=e['value'];cursor=e['work_index']+1
                    if e['cached']:ledger.cache_hits+=1
                    else:
                        ledger.executions+=1
                        if e['kind']==2:ledger.completions+=1
                        else:ledger.cancellations+=1
                elif e['type']=='batch_complete':phase='plan'
                elif e['type']=='collection_end':phase='done'
                ledger.train=e['train_cost_s'];ledger.validation=e['validation_cost_s']
        log=p.open('ab')
    context={}
    def emit(e):
        if e['type']=='observation':e['work_index']=context['cursor']
        e['active_wall_s']=prior_wall+time.perf_counter()-start
        if log:
            log.write((json.dumps(e,separators=(',',':'))+'\n').encode());log.flush();os.fsync(log.fileno())
        events.append(e)
    ledger.writer=emit;env=ReplayEnvironment(runtime,ok,ledger)
    try:
        while phase!='done':
            if deadline is not None and time.perf_counter()>=deadline:raise TimeoutError('FA06 wall limit; paid prefix retained')
            if ledger.remaining<=1e-9:
                ledger.event(dict(type='collection_end',reason='budget'));phase='done';break
            if phase=='init':
                work=[[int(i),a] for i in d['initial_rows'][str(seed)] for a in range(11)]
                cursor=0;ledger.event(dict(type='batch',initial=True,work=work));phase='execute'
            elif phase=='plan':
                round_no+=1;ledger.round=round_no
                pool,selected,work=row_batch(cache,d['train'],seed,round_no,11,100)
                if not len(pool):
                    ledger.event(dict(type='collection_end',reason='pool_exhausted'));phase='done';break
                cursor=0;ledger.event(dict(type='batch',initial=False,selected_rows=selected,work=work));phase='execute'
            elif phase=='execute':
                while cursor<len(work) and ledger.remaining>1e-9:
                    if deadline is not None and time.perf_counter()>=deadline:raise TimeoutError('FA06 wall limit; paid prefix retained')
                    i,a=work[cursor];assert i in train
                    context['cursor']=cursor;env.observe(cache,i,a,100,'train');cursor+=1
                if ledger.remaining<=1e-9:continue
                ledger.event(dict(type='batch_complete'));phase='plan'
            else:raise ValueError(phase)
            if dest:
                save(dest/'progress.json',dict(phase=phase,seed=seed,spent_s=ledger.train,
                    completed_executions=ledger.executions,active_wall_s=prior_wall+time.perf_counter()-start))
    finally:
        if log:log.close()
    labels=make_labels(cache,d['train'])
    outside=sorted(set(range(len(d['ids'])))-train)
    assert not np.any(cache.kind[outside]) and ledger.validation==0
    result=dict(seed=seed,charged_s=ledger.train,executions=ledger.executions,
        cache_hits=ledger.cache_hits,completions=ledger.completions,cancellations=ledger.cancellations,
        labels=sum(map(len,labels)),active_wall_s=prior_wall+time.perf_counter()-start,
        acquisition_signature=hashlib.sha256(cache.kind.tobytes()+cache.value.tobytes()+json.dumps(labels,sort_keys=True).encode()).hexdigest())
    if dest:
        np.savez_compressed(dest/'visible_cache.npz',kind=cache.kind,value=cache.value)
        save(dest/'labels.json',labels);save(dest/'result.json',result)
    return cache,labels,events,result
