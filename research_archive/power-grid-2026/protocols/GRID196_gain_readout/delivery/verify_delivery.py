"""Reconstruct saved gate decisions and expected-query comparisons independently."""
from pathlib import Path
import json,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])
def main():
    C.trained();assert C.read(C.O/'complete.json')['passed'];pred=C.read(C.O/'gate_predictions.json');index={(r['fold'],r['seed'],r['sample_id']):r for r in pred};meta,lookup,cases=C.B.load();byid={lookup[r['sample_id'],r['position']]:r['sample_id'] for r in pred};checked=0
    for f in range(4):
        for s in range(3):
            z=np.load(C.OLD/f'held_f{f}_s{s}.npz');m=np.load(C.O/'models'/f'f{f}_s{s}.npz');value=((z['features']-m['mean'])/m['scale'])@m['coef']+m['intercept'][0]
            for i,idx in enumerate(z['ids']):
                p=index[f,s,byid[int(idx)]];assert abs(p['unclipped_normalized_gain']-float(value[i]))<1e-10;assert p['gate_continue']==(max(float(value[i]),0)>z['features'][i,82]);checked+=1
    assert checked==4681
    new=C.read(C.O/'oof_trajectories.json');nx={(r['method'],r['sample_id']):r for r in new};comp=C.read(C.O/'expected_budget_comparison.json');task=C.read(C.R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];comparisons=0
    for name,folder in [('GRID190',C.OLD),('GRID193',C.R/'outputs/grid193_gate_precondition')]:
        old=C.read(folder/'oof_trajectories.json');ox={(r['method'],r['sample_id']):r for r in old}
        for r in comp[name]['per_date_seed']:
            s,d=r['seed'],r['date'];ids=[c['sample_id'] for c in cases if c['date']==d];N=len(ids);base=[nx[f'NET_seed{s}',i] for i in ids];meanq=sum(v['queries'] for v in base)/N;meang=sum(v['gain'] for v in base)/N;matches=sum(v['action_match'] for v in base)/N
            arms={'OLD':[ox[f'GATE_seed{s}',i] for i in ids],'GAIN':[nx[f'GATE_seed{s}',i] for i in ids],'ADD1_RANDOM':[nx[f'ADD1_seed{s}',i] for i in ids]};budget=min(sum(v['queries']-b['queries'] for v,b in zip(arms[k],base)) for k in ['OLD','GAIN']);assert budget==r['matched_extra_query_total']
            for key,arm in arms.items():
                denom=sum(v['queries']-b['queries'] for v,b in zip(arm,base));prob=budget/denom if denom else 0.;gain=prob*sum(v['gain']-b['gain'] for v,b in zip(arm,base))/N;match=prob*sum(v['action_match']-b['action_match'] for v,b in zip(arm,base))/N
                assert abs(r[key]['mean_queries']-(meanq+budget/N))<1e-12 and abs(r[key]['mean_gain']-meang-gain)<1e-12 and abs(r[key]['action_match_rate']-matches-match)<1e-12
            comparisons+=1
    assert comparisons==96
    count=0
    for mf in C.A.rglob('archive_manifest.json'):
        for name,h in C.read(mf).items():assert C.sha(mf.parent/name)==h;count+=1
    dest=C.R/'论文协作资料/FRAME_收益幅度门控_GRID196';names=[]
    for p in dest.iterdir():
        ap=C.A/'complete'/p.name;assert C.sha(p)==C.sha(C.O/p.name)==C.sha(ap);assert git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==ap.read_bytes();names.append(p.name)
    head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote
    receipt=dict(passed=True,all4681_model_scores_and_decisions_reconstructed=True,all96_matched_query_comparisons_reconstructed=True,archive_files_checked=count,collaboration_files_checked=len(names),committed_bytes_match=True,verified_publication_head=head,remote_main=head,new_fits=0,new_NET_forwards=0,new_physics=0)
    C.save('delivery_receipt.json',receipt);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),C.O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps(receipt))
if __name__=='__main__':main()
