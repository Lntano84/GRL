"""Check the manuscript algorithm against frozen code using saved feedback only."""
from pathlib import Path
import gzip,json,importlib.util,struct,hashlib,random

ROOT=Path('D:/Codex/GRL研究')
OUT=ROOT/'outputs/frame_paper_sections_20261017/04_方法'
spec=importlib.util.spec_from_file_location('methods_frozen_grid70',ROOT/'work/grid70/policies.py')
production=importlib.util.module_from_spec(spec);spec.loader.exec_module(production)

def manuscript(order,feedback,aliases,subids):
    # Direct transcription of Algorithm 1. No production policy function reused.
    seen=set();cache={};chosen=None;trace=[];events=[]
    def score(i):
        return struct.unpack('f',struct.pack('f',feedback[i]['reward']))[0],-i
    def query(batch):
        nonlocal chosen
        before=None if chosen is None else score(chosen)
        for i in batch:
            if i not in cache:cache[i]=feedback[i];seen.add(i);trace.append(i)
            if cache[i]['strict_admissible'] and (chosen is None or score(i)>score(chosen)):chosen=i
        after=None if chosen is None else score(chosen)
        improved=after is not None and (before is None or after>before)
        events.append((len(seen),chosen,improved))
        return improved
    n=min(128,len(order));query(order[:n]);prefix=chosen
    initial=prefix is not None and feedback[prefix]['rho']<.9
    stale=0;j=n
    while j<len(order):
        batch=order[j:j+64];improved=query(batch);j+=len(batch)
        stale=0 if improved else stale+1
        if not initial and chosen is not None and feedback[chosen]['rho']<.9:break
        if chosen is not None and len(seen)>=256 and stale>=2 and len(seen)<len(order):break
    if chosen is not None and len(seen)<len(order) and (not initial or chosen!=prefix):
        station=subids[chosen]
        targets={rep for original,rep in aliases.items() if subids[original]==station}
        query([i for i in order if i in targets and i not in seen])
    return chosen,trace,events

def compare(order,feedback,aliases,subids,label):
    trace=[]
    def observe(i,phase):
        assert i not in trace,(label,'duplicate actual query',i)
        assert i in feedback,(label,'unavailable feedback',i)
        trace.append(i)
        return {**feedback[i],'phase':phase}
    expected=production.run(order,observe,'FRAME_GUARD',subids,aliases)
    chosen,got,events=manuscript(order,feedback,aliases,subids)
    assert got==trace,(label,'query order')
    assert chosen==expected['chosen_pool_id'],(label,'action')
    assert events==[(e['queries'],e['chosen'],e['improved']) for e in expected['events']],(label,'batch events')
    assert len(got)==len(set(got))<=len(order)
    return len(got)

source=ROOT/'outputs/grid85_mechanism/mechanism_feedback.json.gz'
pack=json.load(gzip.open(source,'rt',encoding='utf-8'))
fields=pack['fields'];count_full=0;count_partial=0;queries=0
for label,r in pack['records'].items():
    feedback={row[0]:dict(zip(fields,row)) for row in r['outcomes']}
    aliases={int(k):v for k,v in r['aliases'].items()}
    queries+=compare(r['order'],feedback,aliases,pack['subids'],label)
    if set(r['order'])<=set(feedback):count_full+=1
    else:count_partial+=1

rng=random.Random(20261017)
synthetic=0
for n in (0,1,63,64,127,128,129,191,192,255,256,257,320,512,909):
    for mode in range(8):
        order=list(range(n))
        if mode%2:order.reverse()
        feedback={i:dict(pool_id=i,rho=(.8 if (i+mode)%4 else .95),reward=(1.+rng.randrange(5)*1e-9 if mode==6 else rng.random()),strict_admissible=(False if mode==0 else (i+mode)%3!=0)) for i in range(n)}
        if mode==1:
            for row in feedback.values():row.update(rho=.8,reward=1.,strict_admissible=True)
        if mode==2:
            for row in feedback.values():row.update(rho=.95,reward=1.,strict_admissible=True)
        aliases={i:i for i in range(n)}
        subs=[i%11 for i in range(n)]
        # Extra aliases preserve multi-station membership after deduplication.
        for i in range(n):aliases[n+i]=i;subs.append((i+3)%11)
        compare(order,feedback,aliases,subs,f'synthetic-{n}-{mode}')
        synthetic+=1

result=dict(passed=True,saved_complete_feedback_records=count_full,saved_partial_feedback_records=count_partial,
            reproduced_query_items=queries,synthetic_boundary_cases=synthetic,
            checks=['query identity and order','selected action','per-batch events','unique queries','maximum M queries'],
            scope='documentation algorithm consistency; saved feedback only; no simulation, training, or new policy comparison',
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            production_sha256=hashlib.sha256((ROOT/'work/grid70/policies.py').read_bytes()).hexdigest())
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'algorithm_consistency.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
