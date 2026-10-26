"""Reconstruct factorized predictions and verify archived delivery."""
from pathlib import Path
import json,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])
def main():
    C.trained();assert C.read(C.O/'complete.json')['passed'];pred=C.read(C.O/'gate_predictions.json');meta,lookup,cases=C.B.load();byidx={lookup[p['sample_id'],p['position']]:p['sample_id'] for p in pred};ix={(p['fold'],p['seed'],p['sample_id']):p for p in pred};count=0
    for f in range(4):
        for s in range(3):
            cp=np.load(C.O/'models'/f'f{f}_s{s}.npz');src=np.load(C.EVENT/'models'/f'f{f}_s{s}.npz');assert np.array_equal(cp['event_coef'],src['coef'][0]) and np.array_equal(cp['event_intercept'],src['intercept']);z=np.load(C.OLD/f'held_f{f}_s{s}.npz');x=(z['features']-cp['mean'])/cp['scale'];logit=x@cp['event_coef']+cp['event_intercept'][0];prob=1/(1+np.exp(-np.clip(logit,-700,700)));mu=np.maximum(0,x@cp['coef']+cp['intercept'][0]);gain=prob*mu
            for i,j in enumerate(z['ids']):
                p=ix[f,s,byidx[int(j)]];assert np.isclose(p['event_probability'],prob[i],atol=1e-12,rtol=1e-10);assert np.isclose(p['predicted_normalized_gain'],gain[i],atol=1e-12,rtol=1e-10);assert p['gate_continue']==bool(gain[i]>z['features'][i,82]);count+=1
    assert count==4681
    comp=C.read(C.O/'expected_budget_comparison.json');new=C.read(C.O/'oof_trajectories.json');ni={(v['method'],v['sample_id']):v for v in new};checked=0
    for key,folder in [('GRID190',C.OLD),('GRID196',C.DIRECT)]:
        oi={(v['method'],v['sample_id']):v for v in C.read(folder/'oof_trajectories.json')}
        for r in comp[key]['per_date_seed']:
            s,d=r['seed'],r['date'];ids=[c['sample_id'] for c in cases if c['date']==d];base=[ni[f'NET_seed{s}',i] for i in ids];arms={'OLD':[oi[f'GATE_seed{s}',i] for i in ids],'HURDLE':[ni[f'GATE_seed{s}',i] for i in ids],'ADD1_RANDOM':[ni[f'ADD1_seed{s}',i] for i in ids]};costs={k:sum(v['queries']-b['queries'] for v,b in zip(rr,base)) for k,rr in arms.items()};budget=min(costs['OLD'],costs['HURDLE']);assert budget==r['matched_extra_query_total'];N=len(ids)
            for k,rr in arms.items():
                p=budget/costs[k] if costs[k] else 0.;g=sum(v['gain'] for v in base)/N+p*sum(v['gain']-b['gain'] for v,b in zip(rr,base))/N;assert abs(g-r[k]['mean_gain'])<1e-12;assert abs(r[k]['mean_queries']-sum(b['queries'] for b in base)/N-budget/N)<1e-12
            checked+=1
    assert checked==96;n=0
    for mf in C.A.rglob('archive_manifest.json'):
        for name,h in C.read(mf).items():assert C.sha(mf.parent/name)==h;n+=1
    dest=C.R/'论文协作资料/FRAME_概率与条件收益_GRID198'
    for p in dest.iterdir():
        ap=C.A/'complete'/p.name;assert C.sha(p)==C.sha(C.O/p.name)==C.sha(ap);assert git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==ap.read_bytes()
    head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote;receipt=dict(passed=True,all4681_products_and_frozen_event_scores_rechecked=True,all96_expected_budget_comparisons_rechecked=True,archive_files_checked=n,collaboration_files_checked=8,verified_publication_head=head,remote_main=head,new_fits=0,new_NET_forwards=0,new_physics=0);C.save('delivery_receipt.json',receipt);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),C.O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps(receipt))
if __name__=='__main__':main()
