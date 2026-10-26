"""Inventory action-invariance labels on existing full-teacher suffixes only."""
from pathlib import Path
import collections,json,math,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid186_invariance_labels';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID186_invariance_labels'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    import hashlib
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def native(v):
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=native)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def sealed():
    d=read(R/'outputs/grid100_data_foundation/design.json')
    for row in d['scenario_plan']:
        if row['partition']=='test':
            assert not (R/d['environment']/'chronics'/row['scenario']).exists()
            assert not (R/'outputs/grid100_data_foundation/datasets'/row['scenario']).exists()
def verify():
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    sealed()
def prepare():
    assert not O.exists();O.mkdir(parents=True);sealed();assert read(R/'outputs/grid185_stopping_mechanism/review.json')['passed']
    save('design.json',dict(id='GRID186_EXISTING_SUFFIX_INVARIANCE_LABEL_INVENTORY',authorization='User sustained continue. After185, inventory oldlabels only,not permission for newfit/physics.',
        population='ALLoldGRID102cases:2378trainstarts/3582trainstates/16dates and590oldexposeddevstarts/856states/4dates. Keepdate/scenario/physicalgroup/prefix identifiers;final8sealed.',
        invariant_label='At current prefix j, final observed float32reward/lowIDbest action ID equals currentchosen[j]. This is fullremaininglocalcandidate suffix action invariance on recorded teacher branch only.',
        retain='SeparatechangedactionwithpositivegrossgainbutnonpositivefixedpriceNET,changedactionwithpositiveNET,zero-gainchangedactiontie if any. Keepnext-batchNET,full-suffixNET,oraclebest-prefixNET separate.',
        delayed_opportunity='NextbatchNET<=0 but some later contiguous queriedprefix haspositivecumulativeNET under oldprice; counts only,not RL improvement claim.',
        split='Originaltrain/dev partition andoldGRID121four whole-date12/4folds unchanged. Check both labelclasses in every fit/heldset. No random state splitting/threshold search.',
        correctness='Checkmetadataalignment;nonnegativegains;current/final chosen length;terminalmonotonicity;independentforward/reverse sums;invariant=>zero suffixgain andalllaterchosen identical.',
        information_boundary='Futurechosen/gains are TARGETS only,never new onlinefeatures. No readfinal8,modelforward,fitting or environment. FULLteacherpath differsfromFRAME;olddevexposed,no blind/generalization claim.',
        limits='Action-invariance labels are exact for saved candidatefeedback and comparator;classification errors remain possible. Theyare not wholeweek safety labels or paired physicalcost supervision.',
        prohibited=['training/rerouting/thresholds','newdates/data/physics/final8','copyweeklydeltaontoallbatches','paper/figures']))
    paths=[R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid121_date_transfer/design.json',R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid113_relational_learning/validation_metadata.json',R/'outputs/grid185_stopping_mechanism/summary.json',R/'outputs/grid185_stopping_mechanism/review.json',O/'design.json']
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});verify();archive('frozen');print('GRID186_LABEL_INVENTORY_FROZEN')
def inspect():
    verify();assert not (O/'review.json').exists();cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());task=read(R/'outputs/grid102_search_rl/design.json');fee=task['cost_per_query'];scale=task['gain_scale'];rows=[];checks=collections.Counter()
    for c in cases:
        g=np.asarray(c['gains'],float);q=np.asarray(c['queries'],int);h=len(g);assert h==len(q)==len(c['states']) and len(c['chosen'])==h+1;assert np.all(g>=0) and np.all(q>0) and np.all(q<=64)
        rg=np.r_[np.cumsum(g[::-1])[::-1],0.];rq=np.r_[np.cumsum(q[::-1])[::-1],0]
        for j in range(h):
            gain=math.fsum(g[j:]);queries=int(q[j:].sum());assert abs(gain-rg[j])<1e-12 and queries==rq[j]
            unchanged=c['chosen'][j]==c['chosen'][-1]
            if unchanged:assert gain==0. and all(v==c['chosen'][j] for v in c['chosen'][j:]);checks['invariance_exact']+=1
            if gain>0:assert not unchanged
            suffix_net=(gain-fee*queries)/scale;next_net=(g[j]-fee*q[j])/scale;prefix=(np.cumsum(g[j:])-fee*np.cumsum(q[j:]))/scale;oracle=max(0.,float(prefix.max()));delayed=next_net<=0 and oracle>0
            category='suffix_action_unchanged' if unchanged else 'changed_action_zero_reward_tie' if gain==0 else 'changed_action_positive_suffix_NET' if suffix_net>0 else 'changed_action_nonpositive_suffix_NET'
            rows.append(dict(sample_id=c['sample_id'],position=j,partition=c['partition'],date=c['date'],scenario=c['scenario'],physical_state_group=c['physical_state_group'],current_chosen=c['chosen'][j],final_observed_chosen=c['chosen'][-1],invariant=unchanged,category=category,next_batch_queries=int(q[j]),remaining_queries=queries,remaining_batches=h-j,remaining_reward_gain=gain,next_batch_NET=next_net,full_suffix_NET=suffix_net,best_observed_prefix_NET=oracle,negative_first_positive_later=bool(delayed)))
    summary={};folds=[]
    def aggregate(rr):
        assert rr;bydate={d:[x for x in rr if x['date']==d] for d in sorted({x['date'] for x in rr})};bygroup={g:[x for x in rr if x['physical_state_group']==g] for g in sorted({x['physical_state_group'] for x in rr})}
        return dict(states=len(rr),starts=len({x['sample_id'] for x in rr}),dates=len(bydate),scenarios=len({x['scenario'] for x in rr}),physical_groups=len(bygroup),classes=dict(collections.Counter(x['category'] for x in rr)),invariant=sum(x['invariant'] for x in rr),changed=sum(not x['invariant'] for x in rr),date_macro_invariance=float(np.mean([np.mean([x['invariant'] for x in a]) for a in bydate.values()])),physical_group_macro_invariance=float(np.mean([np.mean([x['invariant'] for x in a]) for a in bygroup.values()])),next_NET_nonpositive=sum(x['next_batch_NET']<=0 for x in rr),delayed_opportunity_states=sum(x['negative_first_positive_later'] for x in rr),delayed_opportunity_starts=sum(x['negative_first_positive_later'] for x in rr if x['position']==0),per_date=[dict(date=d,states=len(a),invariant=sum(x['invariant'] for x in a),changed=sum(not x['invariant'] for x in a),delayed_states=sum(x['negative_first_positive_later'] for x in a)) for d,a in bydate.items()])
    for part,n in [('train',3582),('validation',856)]:
        rr=[x for x in rows if x['partition']==part];assert len(rr)==n;meta=read(R/f'outputs/grid113_relational_learning/{part}_metadata.json');lookup={(x['sample_id'],x['position']):x for x in rr};ordered=[]
        for m in meta:
            x=lookup[m['sample_id'],m['position']];assert x['date']==m['date'] and x['physical_state_group']==m['physical_state_group'] and x['next_batch_queries']==m['next_queries'];ordered.append(x);checks['metadata_rows']+=1
        summary[part]=aggregate(ordered);np.savez_compressed(O/f'{part}_labels.npz',action_invariant=np.array([x['invariant'] for x in ordered]),next_batch_NET=np.array([x['next_batch_NET'] for x in ordered]),full_suffix_NET=np.array([x['full_suffix_NET'] for x in ordered]),negative_first_positive_later=np.array([x['negative_first_positive_later'] for x in ordered]))
    train=[x for x in rows if x['partition']=='train'];validation=[x for x in rows if x['partition']=='validation'];assert not {x['date'] for x in train}&{x['date'] for x in validation}
    for f in read(R/'outputs/grid121_date_transfer/design.json')['folds']:
        fit=[x for x in train if x['date'] in f['fit_dates']];held=[x for x in train if x['date'] in f['held_dates']];assert len(fit)+len(held)==len(train) and not set(f['fit_dates'])&set(f['held_dates'])
        folds.append(dict(fold=f['fold'],fit=aggregate(fit),held=aggregate(held),both_classes_in_fit=len({x['invariant'] for x in fit})==2,both_classes_in_held=len({x['invariant'] for x in held})==2))
    save('label_rows.json',rows);save('summary.json',summary);save('existing_date_fold_coverage.json',folds);verify();save('review.json',dict(passed=True,checks=dict(checks),all4438states_retained=len(rows)==4438,all4folds_retained=True,new_fits=0,new_model_forwards=0,new_physics=0,final8_absent=True));print('GRID186_LABEL_INVENTORY_ANALYSED',json.dumps(summary,ensure_ascii=False,default=native))
if __name__=='__main__':globals()[sys.argv[1]]()
