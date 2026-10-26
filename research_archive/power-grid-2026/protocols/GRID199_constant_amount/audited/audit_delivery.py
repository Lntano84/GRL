"""Independent constant predictions, budgets and source-preserving interpretation."""
from pathlib import Path
import json,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])
def audit():
    C.qualified();assert C.read(C.O/'complete.json')['passed'];assert not (C.O/'independent_audit.json').exists();pred=C.read(C.O/'gate_predictions.json');meta,lookup,cases=C.B.load();byidx={lookup[p['sample_id'],p['position']]:p['sample_id'] for p in pred};ix={(p['fold'],p['seed'],p['sample_id']):p for p in pred};count=0
    for f in range(4):
        for s in range(3):
            lab=np.load(C.PARENT/'fit_labels'/f'f{f}_s{s}.npz');mu=float(sum(float(w)*float(y) for w,y in zip(lab['sample_weight'],lab['target']))/sum(float(w) for w in lab['sample_weight']));m=np.load(C.O/'models'/f'f{f}_s{s}.npz');assert abs(m['conditional_constant'][0]-mu)<1e-12;z=np.load(C.OLD/f'held_f{f}_s{s}.npz');x=(z['features']-m['mean'])/m['scale'];logit=x@m['event_coef']+m['event_intercept'][0];p=np.exp(-np.logaddexp(0.,-logit));value=p*mu
            for i,j in enumerate(z['ids']):
                q=ix[f,s,byidx[int(j)]];assert np.isclose(q['predicted_normalized_gain'],value[i],atol=1e-12,rtol=1e-10);assert q['gate_continue']==bool(value[i]>z['features'][i,82]);count+=1
    assert count==4681;new=C.read(C.O/'oof_trajectories.json');ni={(r['method'],r['sample_id']):r for r in new};comp=C.read(C.O/'expected_budget_comparison.json');checked=0
    for name,folder in [('GRID198',C.PARENT),('GRID190',C.OLD)]:
        oi={(r['method'],r['sample_id']):r for r in C.read(folder/'oof_trajectories.json')}
        for row in comp[name]['per_date_seed']:
            s,d=row['seed'],row['date'];ids=[c['sample_id'] for c in cases if c['date']==d];N=len(ids);base=[ni[f'NET_seed{s}',i] for i in ids];arms={'OLD':[oi[f'GATE_seed{s}',i] for i in ids],'CONSTANT':[ni[f'GATE_seed{s}',i] for i in ids],'ADD1_RANDOM':[ni[f'ADD1_seed{s}',i] for i in ids]};costs={k:sum(r['queries']-b['queries'] for r,b in zip(rr,base)) for k,rr in arms.items()};budget=min(costs['OLD'],costs['CONSTANT']);assert budget==row['matched_extra_query_total']
            for k,rr in arms.items():
                p=budget/costs[k] if costs[k] else 0.;g=sum(b['gain'] for b in base)/N+p*sum(r['gain']-b['gain'] for r,b in zip(rr,base))/N;assert abs(row[k]['mean_gain']-g)<1e-12;assert abs(row[k]['mean_queries']-sum(b['queries'] for b in base)/N-budget/N)<1e-12
            checked+=1
    assert checked==96
    primary=comp['GRID198'];ma=primary['macro'];net=ma['OLD']['mean_net_utility']-ma['CONSTANT']['mean_net_utility'];match=100*(ma['OLD']['action_match_rate']-ma['CONSTANT']['action_match_rate']);per=primary['per_seed'];allaction=all(r['match_delta_pp']<0 for r in per);positive_reward_seeds=sum(r['gain_delta']<0 for r in per);assert allaction and positive_reward_seeds==2
    text='## 本轮判读\n\n'+f'在预先固定的同预期候选查询比较下，学习条件幅度相对拟合正例均值的动作一致率平均高{match:.4f}个百分点，三个种子方向一致；奖励/NET在两个种子更好、一个更差，平均NET差{net:+.8f}。因此保留条件幅度头带来的小幅选择改善，不能由此声称该模块普遍必要。\n\n'+'条件头在正收益状态上的归一化幅度MSE为5.7875，固定正例均值为1.8863；它的正例幅度估计尚未优于简单均值。当前决策上的作用可能包含对零收益/低价值状态的再筛选，不能把选择改善直接表述为准确预测了收益大小。这只是需要继续核查的解释，不是因果结论。\n\n'+'GRID198对原GRID190与同预期查询随机ADD1的平均优势仍未建立。完整方法的竞争力与条件头相对这个简单均值的贡献，是两个不同结论，全部保留。\n\n'
    original=C.O/'GRID199_report.md';body=original.read_text(encoding='utf8');head,rest=body.split('\n',1);(C.O/'GRID199_report_v2.md').write_text(head+'\n\n'+text+rest,encoding='utf8',newline='\n')
    C.save('independent_audit.json',dict(passed=True,all4681_constant_predictions_and_96_expected_budgets_rechecked=True,conditional_learning_match_advantage_all3seeds=allaction,conditional_learning_reward_advantage_seeds=positive_reward_seeds,learned_minus_constant_match_pp=match,learned_minus_constant_NET=net,original_report_preserved=True,original_report_sha256=C.sha(original),new_optimization_fits=0,derived_training_means=12,new_NET_forwards=0,new_physics=0));C.archive('audited');dest=C.R/'论文协作资料/FRAME_条件收益均值对照_GRID199'
    for n in ['GRID199_report_v2.md','independent_audit.json']:shutil.copy2(C.O/n,dest/n)
    p=C.REPO/'docs/DECISIONS.md';p.write_text(p.read_text(encoding='utf8')+'\nGRID199判读：198学习条件幅度对固定正例均值，同预期预算动作三个种子均改善（平均+'+f'{match:.4f}pp），奖励两好一差；但正例幅度MSE5.7875劣于均值1.8863。保留决策筛选线索，不声称准确幅度估计或全方法同预算优势。\n',encoding='utf8',newline='\n')
    p=C.REPO/'docs/NEXT_STEPS.md';p.write_text('# Next steps — GRID199判读后\n\n1. 保留198相对正例均值的跨种子动作选择改善，以及190/196/199全部权衡和反例；全方法对190/随机追加的平均同预算优势仍未建立。\n2. 优先只读核查条件头的作用：正例幅度误差未优于均值，决策提升是否主要来自筛掉零收益/低价值补查；不能把筛选作用直接叫准确幅度学习。\n3. 不自动换损失、阈值、均值、费用或组合；新训练先通知范围，不扩物理/日期/最终8或论文。\n',encoding='utf8',newline='\n')
    p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8').replace('protocols/GRID199_constant_amount/complete/GRID199_report.md','protocols/GRID199_constant_amount/audited/GRID199_report_v2.md',1),encoding='utf8',newline='\n');print('GRID199_INDEPENDENT_AUDIT',net,match)
def delivery():
    C.qualified();assert C.read(C.O/'independent_audit.json')['passed'];n=0
    for mf in C.A.rglob('archive_manifest.json'):
        for name,h in C.read(mf).items():assert C.sha(mf.parent/name)==h;n+=1
    dest=C.R/'论文协作资料/FRAME_条件收益均值对照_GRID199';count=0
    for p in dest.iterdir():
        ap=C.A/'audited'/p.name;assert C.sha(p)==C.sha(C.O/p.name)==C.sha(ap);assert git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==ap.read_bytes();count+=1
    head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote;rec=dict(passed=True,archive_files_checked=n,collaboration_files_checked=count,committed_bytes_match=True,verified_publication_head=head,remote_main=head,new_optimization_fits=0,new_physics=0);C.save('delivery_receipt.json',rec);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),C.O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps(rec))
if __name__=='__main__':globals()[sys.argv[1]]()
