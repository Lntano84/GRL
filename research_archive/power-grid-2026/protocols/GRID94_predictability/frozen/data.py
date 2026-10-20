"""Extract completion-preceding features and exact same-choice labels separately."""
from pathlib import Path
import csv, hashlib, importlib.util, json
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid94_predictability'


def loadmod(name,rel):
    sp=importlib.util.spec_from_file_location(name,ROOT/rel);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m


P=loadmod('g94_original','work/grid91_mechanism/policies.py')
I=loadmod('g94_independent','work/grid89_review/reference_logic.py')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csvsave(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def features(order,rows,aliases,subs,limit):
    """No completion result, label or date is accepted by this function."""
    truth={r['pool_id']:r for r in rows};seen=[]
    def observe(i,phase):
        assert i in truth and i not in seen;seen.append(i);return dict(truth[i],phase=phase)
    no=P.run(order,observe,'NO_LOCAL',subs,aliases)
    assert no['completion_budget']>0
    queried=no['outcomes'];eligible=[r for r in queried if r['strict_admissible']]
    incumbent=max(eligible,key=P.key)
    prefix=max((r for r in queried[:min(128,len(order))] if r['strict_admissible']),key=P.key,default=None)
    station=int(subs[incumbent['pool_id']]);targets={int(v) for k,v in aliases.items() if int(subs[k])==station}
    local=[r for r in queried if r['pool_id'] in targets]
    left=[P.key(r)[0] for r in local if r['strict_admissible'] and r['pool_id']!=incumbent['pool_id']]
    right=[P.key(r)[0] for r in eligible if r['pool_id'] not in targets]
    pairs=len(left)*len(right)
    support=sum((a>b)+.5*(a==b) for a in left for b in right)/pairs if pairs else .5
    top=sorted((P.key(r)[0] for r in eligible),reverse=True)
    events=no['events'];last=events[-1];prev=events[-2] if len(events)>1 else None
    last_before=None if prev is None or prev['chosen'] is None else truth[prev['chosen']]
    gains=max(0.,P.key(incumbent)[0]-P.key(last_before)[0]) if last_before is not None else 0.
    batches=events[1:]
    x=dict(incumbent_rho=incumbent['rho'],rho_limit=limit,
        prefix_safe=float(no['prefix_safe_at_entry']),incumbent_changed=float(no['pre_extra_choice']!=no['shortlist_choice']),
        prefix_valid=float(prefix is not None),gain_since_prefix=max(0.,P.key(incumbent)[0]-P.key(prefix)[0]) if prefix else 0.,
        observed_reward_gap=top[0]-top[1] if len(top)>1 else 0.,
        queried_fraction=len(queried)/len(order),remaining_station_fraction=no['completion_budget']/(len(order)-len(queried)),
        observed_admissible_fraction=len(eligible)/len(queried),
        station_observed_admissible_fraction=sum(r['strict_admissible'] for r in local)/len(local) if local else 0.,
        station_support=support,station_support_missing=float(not pairs),last_batch_gain=gains,
        improving_batch_fraction=sum(e['improved'] for e in batches)/len(batches) if batches else 0.)
    return x,no


def prepare():
    OUT.mkdir(exist_ok=True)
    design=read(W/'design.json');sources={}
    def get(p):sources[p.relative_to(ROOT).as_posix()]=sha(p);return read(p)
    refs=get(ROOT/'outputs/grid91_mechanism/design.json')['references']
    subs=get(ROOT/'outputs/grid28/offline.json')['public_subids']
    assert get(ROOT/'outputs/grid91_mechanism/review.json')['passed_saved_data']
    oldroles={}
    q=ROOT/'outputs/frame_completion_roles_20261020/per_call.csv';sources[q.relative_to(ROOT).as_posix()]=sha(q)
    with q.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['population']=='FRAME':oldroles[r['source']]=r
    rows=[];checks=defaultdict(int)
    for ref in refs:
        for path in sorted((ROOT/ref['frame']).glob('call[0-9][0-9][0-9].json')):
            c=get(path);truth={r['pool_id']:r for r in c['outcomes']};aliases={int(k):v for k,v in c['aliases'].items()}
            seen=[]
            def observe(i,phase):
                assert i in truth and i not in seen;seen.append(i);return dict(truth[i],phase=phase)
            full=P.run(c['order'],observe,'FRAME_GUARD',subs,aliases)
            independent,choice,budget=I.replay(c['order'],truth,subs,aliases,'FRAME_GUARD',dict(P.DEFAULT,qmin=0))
            assert seen==independent and choice==full['chosen_pool_id']==c['chosen_pool_id'] and budget==full['completion_budget']
            assert len(seen)==c['public_queries'];checks['frame_reproductions']+=1
            if not budget:continue
            prior=[r for r in full['outcomes'] if r['phase']!='local_completion']
            x,no=features(c['order'],prior,aliases,subs,c['rho_limit'])
            assert list(x)==design['features']
            assert [r['pool_id'] for r in prior]==[r['pool_id'] for r in no['outcomes']]
            # Independently derive most important state/count features directly from prefix.
            incumbent=max((r for r in prior if r['strict_admissible']),key=P.key)
            assert x['incumbent_rho']==incumbent['rho']
            assert x['queried_fraction']==len(prior)/len(c['order'])
            assert x['remaining_station_fraction']==budget/(len(c['order'])-len(prior))
            # Remove the entire suffix. Feature construction must still succeed unchanged.
            xx,nn=features(c['order'],prior,aliases,subs,c['rho_limit'])
            assert xx==x and nn['chosen_pool_id']==no['chosen_pool_id']
            # Perturb every hidden candidate: no effect on any pre-completion feature.
            visible=set(r['pool_id'] for r in prior)
            perturbed=[dict(r) if r['pool_id'] in visible else dict(r,rho=999.,reward=-999.,strict_admissible=False) for r in c['outcomes']]
            xp,np=features(c['order'],perturbed,aliases,subs,c['rho_limit'])
            assert xp==x and np['completion_budget']==budget;checks['hidden_feedback_perturbations']+=1
            y=int(full['chosen_pool_id']==no['chosen_pool_id'])
            before=truth[no['chosen_pool_id']];after=truth[full['chosen_pool_id']]
            gain=P.key(after)[0]-P.key(before)[0];assert gain>=0
            if y:assert before['combined_hash']==after['combined_hash'] and gain==0
            rel=path.relative_to(ROOT).as_posix();old=oldroles[rel]
            assert int(old['completion_queries'])==budget and (old['choice_changed']=='True')==(not y)
            rows.append(dict(row_id=f"{ref['cohort']}|{ref['week']}|{path.stem}",cohort=ref['cohort'],week=ref['week'],
                calendar_date=ref['week'].split('_')[0],call=path.stem,step=c['step'],source=rel,state_hash=c['state_hash'],
                label_skip_same_choice=y,completion_queries=budget,reward_gain=gain,
                before_choice=no['chosen_pool_id'],after_choice=full['chosen_pool_id'],role=old['role'],**x))
            checks['exact_label_checks']+=1
    assert checks['frame_reproductions']==179 and len(rows)==109
    for rel in ['work/grid91_mechanism/policies.py','work/grid89_review/reference_logic.py',
                'work/grid94_predictability/data.py','work/grid94_predictability/design.json']:
        sources[rel]=sha(ROOT/rel)
    csvsave(OUT/'dataset.csv',rows);save(OUT/'dataset.json',rows)
    summary=[]
    for group in ['old13','G1','G2','ALL']:
        rr=[r for r in rows if group=='ALL' or r['cohort']==group]
        summary.append(dict(cohort=group,eligible_calls=len(rr),calendar_groups=len({r['calendar_date'] for r in rr}),
            same_choice=sum(r['label_skip_same_choice'] for r in rr),choice_changes=sum(not r['label_skip_same_choice'] for r in rr),
            completion_queries=sum(r['completion_queries'] for r in rr),
            oracle_same_choice_savings=sum(r['completion_queries'] for r in rr if r['label_skip_same_choice'])))
    save(OUT/'data_summary.json',summary)
    save(OUT/'preflight.json',dict(passed=True,checks=dict(checks),features=design['features'],
        dataset_sha256=sha(OUT/'dataset.json'),sources=sources,
        label_scope='Exact same chosen ID only; no long-term economic labels'))
    print(json.dumps(dict(summary=summary,checks=dict(checks)),ensure_ascii=False,indent=2))


if __name__=='__main__':prepare()
