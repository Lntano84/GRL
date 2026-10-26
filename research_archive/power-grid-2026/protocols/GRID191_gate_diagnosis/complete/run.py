"""Frozen linear-score readback and conditional-support diagnosis; no new fit."""
from pathlib import Path
import collections,hashlib,json,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid191_gate_diagnosis';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID191_gate_diagnosis';P=R/'outputs/grid190_extension_gate'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item())+'\n',encoding='utf8',newline='\n')
def sources():return [R/'outputs/grid168_one_sided_transfer',*[R/f'outputs/grid170_remaining_date_folds/fold{f}' for f in [1,2,3]]]
def sealed():
    d=read(R/'outputs/grid100_data_foundation/design.json')
    for v in d['scenario_plan']:
        if v['partition']=='test':assert not (R/d['environment']/'chronics'/v['scenario']).exists() and not (R/'outputs/grid100_data_foundation/datasets'/v['scenario']).exists()
def verify():
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    sealed()
def archive(phase):
    dst=A/phase;dst.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dst/p.name)
    (dst/'archive_manifest.json').write_text(json.dumps({p.relative_to(dst).as_posix():sha(p) for p in dst.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def prepare():
    assert not O.exists();O.mkdir(parents=True);sealed();assert read(P/'complete.json')['passed']
    save('design.json',dict(id='GRID191_GATE_DIAGNOSIS',authorization='User:好样的继续. Existing gate diagnosis before another minimal intervention.',posthoc=True,
        questions=['Fit-side vs held weighted AUC/recall/BCE/Brier and constant fit-prior baseline','Calibration with fixedbins0/.1/.2/.5/.8/1 without thresholdselection','Hit/waste/missed onebatch reward/query decomposition and missed-gain callconcentration','Does conditioning gatefit on in-sample NETSTOP remove most known profitable fitstates?'],
        method='Reconstitute existing LR logits on saved fitfeatures;held logits already cached. No NETforward/newfeature extraction/optimizer. Each pool date/call/prefix weights; all12pools/16dates/3NETseeds retained. Fit and held STOPsets differ due baseline fit/held errors;do not call allgap pure gateoverfit.',
        limits='Descriptive development evidence,not a validated trigger or proof supportaugmentation works. No newthreshold/label/parameter/gate/physics/test8.'))
    files=[W/'run.py',O/'design.json',P/'fits.json',P/'gate_predictions.json',P/'family_summary.json',P/'complete.json',R/'outputs/grid189_baseline_extension/fit_stop_rows.json',R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']+list((P/'fit_features').iterdir())+list((P/'models').glob('*.npz'))+list(P.glob('held_*.npz'))
    for p in sources():files.append(p/'fold_arrays.npz')
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('frozen');print('GRID191_FROZEN')
def weights(rows):
    dates=sorted({r['date'] for r in rows});groups={d:{r['physical_state_group'] for r in rows if r['date']==d} for d in dates};cnt=collections.Counter(r['physical_state_group'] for r in rows)
    return np.array([1/(len(dates)*len(groups[r['date']])*cnt[r['physical_state_group']]) for r in rows])
def auc(y,l,w):
    pos=float(w[y].sum());neg=float(w[~y].sum());order=np.argsort(l,kind='stable');before=0.;total=0.;start=0
    while start<len(order):
        end=start+1
        while end<len(order) and l[order[end]]==l[order[start]]:end+=1
        ii=order[start:end];p=float(w[ii][y[ii]].sum());n=float(w[ii][~y[ii]].sum());total+=p*(before+.5*n);before+=n;start=end
    return total/(pos*neg) if pos and neg else None
def stats(y,l,w,prior):
    p=np.exp(-np.logaddexp(0.,-l));go=l>0;tp=float(w[y&go].sum());pred=float(w[go].sum());positive=float(w[y].sum())
    return dict(rows=len(y),weighted_prevalence=positive,weighted_append_rate=pred,weighted_precision=tp/pred if pred else None,weighted_recall=tp/positive if positive else None,weighted_AUC=auc(y,l,w),weighted_BCE=float(w@(np.logaddexp(0.,l)-y*l)),weighted_Brier=float(w@((p-y)**2)),fit_prior=prior,constant_fitprior_BCE=float(w@(-y*np.log(prior)-(~y)*np.log(1-prior))),constant_fitprior_Brier=float(w@((prior-y)**2)),count_append=int(go.sum()),count_hit=int((go&y).sum()))
def analyze():
    verify();assert not (O/'summary.json').exists();meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'};pred=read(P/'gate_predictions.json');fitrows=read(R/'outputs/grid189_baseline_extension/fit_stop_rows.json');receipt=read(P/'fits.json')['records'];task=read(R/'outputs/grid102_search_rl/design.json');price=task['cost_per_query'];nc=collections.Counter(c['date'] for c in cases.values());pools=[];cal=[];support=[]
    for f in range(4):
        fi=np.load(sources()[f]/'fold_arrays.npz')['fit_ids'];all_profit=sum(cases[meta[i]['sample_id']]['gains'][meta[i]['position']]>price*cases[meta[i]['sample_id']]['queries'][meta[i]['position']] for i in fi);all_events=sum(cases[meta[i]['sample_id']]['chosen'][meta[i]['position']]!=cases[meta[i]['sample_id']]['chosen'][meta[i]['position']+1] for i in fi)
        for seed in range(3):
            z=np.load(P/'fit_features'/f'f{f}_s{seed}.npz');cp=np.load(P/'models'/f'f{f}_s{seed}.npz');l=z['standard']@cp['coef'][0]+cp['intercept'][0];rr=[v for v in fitrows if v['fold']==f and v['seed']==seed];assert [v['metadata_index'] for v in rr]==z['ids'].tolist();rec=next(x for x in receipt if x['fold']==f and x['seed']==seed);assert abs(float(l.min())-rec['fit_logit_min'])<1e-10 and abs(float(l.max())-rec['fit_logit_max'])<1e-10
            yy=z['y'].astype(bool);w=z['mass'];prior=float(w@yy);fs=stats(yy,l,w,prior);hh=[v for v in pred if v['fold']==f and v['seed']==seed];h=np.load(P/f'held_f{f}_s{seed}.npz');assert [meta[i]['sample_id'] for i in h['ids']]==[v['sample_id'] for v in hh];hy=np.array([v['actual_next_change'] for v in hh]);hw=weights(hh);hs=stats(hy,h['logit'],hw,prior)
            assert hs['count_append']==sum(v['gate_continue'] for v in hh);assert hs['count_hit']==sum(v['gate_continue'] and v['actual_next_change'] for v in hh)
            fitgain=np.array([v['private_label']['next_gain'] for v in rr]);heldgain=np.array([v['actual_next_gain'] for v in hh]);fitprofit=np.array([v['private_label']['next_NET']>0 for v in rr]);heldprofit=np.array([v['actual_next_NET']>0 for v in hh]);fs.update(weighted_profitable_fraction=float(w@fitprofit),weighted_next_gain=float(w@fitgain));hs.update(weighted_profitable_fraction=float(hw@heldprofit),weighted_next_gain=float(hw@heldgain))
            pools.append(dict(fold=f,seed=seed,fit=fs,held=hs));support.append(dict(fold=f,seed=seed,all_fit_states=len(fi),all_fit_change_events=int(all_events),all_fit_profitable_nextbatches=int(all_profit),conditional_STOP_states=len(rr),conditional_positive_events=int(yy.sum()),conditional_profitable_nextbatches=int(fitprofit.sum()),profitable_fit_states_excluded_by_STOP_selection=int(all_profit-fitprofit.sum())))
            for partition,ys,ls,ws in [('fit',yy,l,w),('held',hy,h['logit'],hw)]:
                ps=np.exp(-np.logaddexp(0.,-ls));bins=[0.,.1,.2,.5,.8,1.]
                for j in range(len(bins)-1):
                    mask=(ps>=bins[j])&((ps<=bins[j+1]) if j==len(bins)-2 else (ps<bins[j+1]));mass=float(ws[mask].sum())
                    cal.append(dict(fold=f,seed=seed,partition=partition,bin=f'{bins[j]}..{bins[j+1]}',rows=int(mask.sum()),mass=mass,positive_mass=float(ws[mask]@ys[mask]),prediction_mass=float(ws[mask]@ps[mask])))
    means={part:{k:float(np.mean([p[part][k] for p in pools if p[part][k] is not None])) for k in pools[0][part] if k not in ['rows','count_append','count_hit']} for part in ['fit','held']};account=collections.defaultdict(lambda:dict(records=0,paid_query_mean=0.,potential_gain_mean=0.,groups=set()));missgroups=collections.defaultdict(float)
    for p in pred:
        key='HIT' if p['gate_continue'] and p['actual_next_change'] else 'WASTE' if p['gate_continue'] else 'MISS' if p['actual_next_change'] else 'CORRECT_SKIP';v=account[key];w=1/(3*16*nc[p['date']]);v['records']+=1;v['potential_gain_mean']+=w*p['actual_next_gain'];v['paid_query_mean']+=w*p['actual_next_queries']*p['gate_continue'];v['groups'].add(p['physical_state_group'])
        if key=='MISS':missgroups[p['physical_state_group']]+=w*p['actual_next_gain']
    for v in account.values():v['physical_groups']=len(v.pop('groups'))
    assert account['HIT']['records']==84 and account['WASTE']['records']==194 and account['MISS']['records']==528
    assert abs(account['HIT']['paid_query_mean']+account['WASTE']['paid_query_mean']-read(P/'family_summary.json')['GATE']['added_queries'])<1e-12
    concentration=sorted([dict(group=g,missed_gain_contribution=x) for g,x in missgroups.items()],key=lambda v:-v['missed_gain_contribution']);total=sum(missgroups.values())
    calibration=[]
    for part in ['fit','held']:
        for b in sorted({v['bin'] for v in cal}):
            rr=[v for v in cal if v['partition']==part and v['bin']==b];m=sum(v['mass'] for v in rr);calibration.append(dict(partition=part,bin=b,rows=sum(v['rows'] for v in rr),pool_equal_mass=m/12,mean_probability=sum(v['prediction_mass'] for v in rr)/m if m else None,observed_change_fraction=sum(v['positive_mass'] for v in rr)/m if m else None))
    save('pool_metrics.json',pools);save('conditional_support.json',support);save('calibration.json',calibration);save('opportunity_accounting.json',dict(categories=dict(account),all_missed_groups=concentration,top3_missed_gain_fraction=sum(v['missed_gain_contribution'] for v in concentration[:3])/total));save('summary.json',dict(fit_held=means,conditional_profitable_range=[min(v['conditional_profitable_nextbatches'] for v in support),max(v['conditional_profitable_nextbatches'] for v in support)],all_fit_profitable_range=[min(v['all_fit_profitable_nextbatches'] for v in support),max(v['all_fit_profitable_nextbatches'] for v in support)],fit_linear_rows=16565,held_cached_rows=4681))
    verify();save('review.json',dict(passed=True,all12pool_score_extrema_reproduce_training_receipts=True,all4681held_native_actions_reproduced=True,all_categories_and_query_contributions_conserved=True,all_fit_memberships_labels_and_weights_reused=True,new_fits=0,new_NET_forwards=0,fit_linear_readback_rows=16565,held_cached_rows=4681,new_physics=0,thresholds_unchanged=True,final8_absent=True));archive('analyzed');print('GRID191_ANALYZED',json.dumps(dict(means=means,support_range=[min(v['all_fit_profitable_nextbatches'] for v in support),max(v['all_fit_profitable_nextbatches'] for v in support)],account=dict(account)),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
