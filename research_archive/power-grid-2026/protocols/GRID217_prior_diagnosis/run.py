"""Saved prior strength versus cold cancellation: diagnosis only, no fits/rules."""
from pathlib import Path
import collections,hashlib,json,pickle,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid217_prior_diagnosis';G=R/'outputs/grid216_cancellation_gate';S=R/'outputs/grid214_inner_head_generation';BASE=R/'outputs/grid208_public_amount';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID217_prior_diagnosis'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    for root,names in [(G,['trained_freeze.json','scored_freeze.json']),(S,['trained_freeze.json','scored_freeze.json','labelled_freeze.json'])]:
        for name in names:
            for p,h in read(root/name).items():assert sha(root/p)==h,p

def auc(y,x,w):
    y=np.asarray(y,bool);x=np.asarray(x,float);w=np.asarray(w,float);a=w[y];b=w[~y]
    if not len(a) or not len(b) or a.sum()==0 or b.sum()==0:return None
    delta=x[y,None]-x[None,~y];return float(np.sum(a[:,None]*b[None,:]*((delta>0)+.5*(delta==0)))/(a.sum()*b.sum()))

def masses(rows):
    dates={r['date'] for r in rows};groups={d:{r['call'] for r in rows if r['date']==d} for d in dates};counts=collections.Counter(r['call'] for r in rows);w=np.array([1/(len(dates)*len(groups[r['date']])*counts[r['call']]) for r in rows]);assert abs(w.sum()-1)<1e-10;return w

def stats(rows):
    n=np.array([r['net'] for r in rows]);w=np.array([r['mass'] for r in rows]);w/=w.sum();p=np.array([r['router_score'] for r in rows]);rat=np.array([r['ratio'] for r in rows]);c=w*np.abs(n);y=n>0
    def quant(v):return [float(q) for q in np.quantile(v,[0,.25,.5,.75,1])] if len(v) else None
    return dict(rows=len(rows),positive_rows=int(y.sum()),positive_fraction=float(y.mean()),positive_calls=len({r['call'] for r in rows if r['net']>0}),positive_structural_mass=float(w@y),positive_absNET_cost_share=float(c@y/c.sum()),prior_ratio_AUC_structural=auc(y,rat,w),prior_ratio_AUC_cost=auc(y,rat,c),router_AUC_structural=auc(y,p,w),router_AUC_cost=auc(y,p,c),positive_prior_ratio_five_quantiles=quant(rat[y]),negative_prior_ratio_five_quantiles=quant(rat[~y]),positive_router_five_quantiles=quant(p[y]),negative_router_five_quantiles=quant(p[~y]),max_router_score=float(p.max()),positive_router_kept=int((y&(p>=.5)).sum()))

def main():
    assert not O.exists();O.mkdir();save('design.json',dict(stage='GRID217',scope='Read frozen scores/labels and reconstruct216FITforest probabilities only. No model fits/NN forwards/new heldscores/physical or newpolicy replay.',question='Does208prior decision-strength rank profitable entries, does216 rank them despite threshold collapse, and how do innerhead-trained versus outerheld selected domains differ?',prior_strength='R=(nativeeventP*conditionalmu)/known_public_charge. R>1 was originalcontinue rule. This is not a calibrated payoff/probability; same ratio form on214 inner heads and208held.',weights='Primary date/call/prefix source masses conditioned on originalbaseCONT domain. Held rebuild same structural mass on originalNETSTOP cases beforeCONT filtering. Additional absNET-weighted AUC and cost-share describe target. Do not reweight or train on held outcomes.',bins='Exactly4 public-ratio quartiles perouterseed using only selected training ratios, no gain/heldlabels tosetcuts; held applysamecuts, outoftrainrange separately. Descriptivebins not candidatepolicy/thresholdsearch.',limits='Head training8dates versus deployed12 creates selection/model shift, not isolated labelshift. Fit216scores are in-sample; innerhead scores head-onlycrossfitted with baseNETin-sample. AUC descriptive, prefixes/seeds not independent. Compare withinmodelpool, not treat pooled610 as independent. No causal proof of features/capacity/calibration, no new cancel/keeprule.'))
    paths=[W/'run.py',O/'design.json',G/'complete.json',G/'independent_audit.json',G/'gate_predictions.json',G/'fit_and_transfer_diagnosis.json',G/'configs.json',S/'bundles.json',BASE/'gate_predictions.json',R/'outputs/grid113_relational_learning/train_metadata.json']+[G/n for n in ['trained_freeze.json','scored_freeze.json']]+[S/n for n in ['trained_freeze.json','scored_freeze.json','labelled_freeze.json']]+list((G/'fit_inputs').glob('*.npz'))+list((G/'models').glob('*.pkl'))+list((S/'labelled_samples').glob('*.npz'))
    save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');inner={}
    for b in read(S/'bundles.json'):
        z=np.load(S/'labelled_samples'/f"{b['bundle']}.npz")
        for k,i in enumerate(z['meta_ids']):
            if z['original_STOP_mask'][k] and z['continue4'][k]:inner[b['outer_fold'],b['seed'],int(i)]=(float(z['event_p'][k]*z['mu4'][k]/z['public_charge'][k]),float(z['public_charge'][k]),float(z['event_p'][k]),float(z['mu4'][k]),float(z['actual_NET'][k]))
    fit=collections.defaultdict(list);held=collections.defaultdict(list);fullheld=collections.defaultdict(list);base={(r['seed'],r['sample_id']):r for r in read(BASE/'gate_predictions.json')}
    for r in read(G/'gate_predictions.json'):
        b=base[r['seed'],r['sample_id']];assert r['base_continue']==b['gate_continue'] and abs(r['actual_next_NET']-b['actual_next_NET'])<1e-12;ratio=b['predicted_normalized_gain']/b['known_charge'];a=dict(pool=f"f{r['fold']}_s{r['seed']}",phase='held',index=None,sample_id=r['sample_id'],position=r['position'],date=r['date'],call=r['physical_state_group'],net=r['actual_next_NET'],ratio=ratio,router_score=r['router_keep_score'],base_continue=r['base_continue'],new_continue=r['new_continue']);fullheld[a['pool']].append(a)
    for pool,rows in fullheld.items():
        m=masses(rows)
        for row,w in zip(rows,m):
            if row['base_continue']:row['mass']=float(w);held[pool].append(row)
    for pool in read(G/'configs.json'):
        f,s=[int(v[1:]) for v in pool.split('_')];z=np.load(G/'fit_inputs'/f'{pool}.npz');model=pickle.loads((G/'models'/f'{pool}.pkl').read_bytes())
        if isinstance(model,dict):p=np.ones(len(z['ids']))
        else:
            terms=[]
            for tree in model.estimators_:v=tree.tree_.value[tree.apply(z['X']),0,:];terms.append(v[:,1]/v.sum(1))
            p=np.mean(terms,axis=0)
        for k,i in enumerate(z['ids']):
            i=int(i);m=meta[i];ratio,charge,pp,mu,net=inner[f,s,i];assert ratio>1 and abs(net-z['actual_NET'][k])<1e-12;fit[pool].append(dict(pool=pool,phase='fit',index=i,sample_id=m['sample_id'],position=m['position'],date=m['date'],call=m['physical_state_group'],net=net,ratio=ratio,router_score=float(p[k]),mass=float(z['structural_mass'][k]),base_continue=True))
    assert sum(map(len,fit.values()))==3466 and sum(map(len,held.values()))==610
    summary={};bins=[];identity=[]
    for pool in sorted(fit):
        a,b=fit[pool],held[pool];fs,hs=stats(a),stats(b);cuts=np.quantile([r['ratio'] for r in a],[.25,.5,.75]);fit_min=min(r['ratio'] for r in a);fit_max=max(r['ratio'] for r in a)
        summary[pool]=dict(fit=fs,held=hs,quantile_cuts=cuts.tolist(),held_below_training_min=sum(r['ratio']<fit_min for r in b),held_above_training_max=sum(r['ratio']>fit_max for r in b),keep_constant_pool=pool=='f0_s1')
        for phase,rows in [('fit',a),('held',b)]:
            for group in range(4):
                rr=[r for r in rows if np.searchsorted(cuts,r['ratio'],side='right')==group];ww=sum(r['mass'] for r in rr);plus=sum(r['mass']*max(r['net'],0) for r in rr);minus=sum(r['mass']*max(-r['net'],0) for r in rr);bins.append(dict(pool=pool,phase=phase,bin=group+1,rows=len(rr),positive_rows=sum(r['net']>0 for r in rr),positive_calls=len({r['call'] for r in rr if r['net']>0}),positive_value_mass=plus/ww if ww else None,negative_cost_mass=minus/ww if ww else None,public_cut_values=cuts.tolist(),not_a_policy=True))
            identity.extend(rows)
    def meanfield(phase,field,exclude_constant=True):
        vals=[v[phase][field] for pool,v in summary.items() if not(exclude_constant and pool=='f0_s1') and v[phase][field] is not None];return dict(mean=float(np.mean(vals)) if vals else None,pools=len(vals))
    averages={phase:{field:meanfield(phase,field) for field in ['prior_ratio_AUC_structural','prior_ratio_AUC_cost','router_AUC_structural','router_AUC_cost','positive_fraction','positive_structural_mass','positive_absNET_cost_share']} for phase in ['fit','held']}
    save('row_diagnosis.json',identity);save('quartile_descriptions.json',bins);save('per_pool_summary.json',summary);save('summary.json',dict(average_nonconstant_pools=averages,fit_rows=3466,held_baseCONT_rows=610,fit_positive_rows=sum(r['net']>0 for rr in fit.values() for r in rr),held_positive_rows=sum(r['net']>0 for rr in held.values() for r in rr),new_fits=0,new_NN_forwards=0,new_held_model_scores=0,new_policy_evaluations=0,new_physics=0));verify();print('GRID217_ANALYSIS',json.dumps(averages,ensure_ascii=False))
if __name__=='__main__':main()
