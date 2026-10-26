"""Independent forest and expected-budget checks; joint GRID200/201 delivery."""
from pathlib import Path
import importlib.util,json,pickle,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])
def audit():
    C.trained();assert C.read(C.O/'complete.json')['passed'];assert not (C.O/'independent_audit.json').exists();sp=importlib.util.spec_from_file_location('grid201_prior_role_check',C.R/'work/grid200_amount_role/run.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P);P.verify();assert P.read(P.O/'complete.json')['passed']
    pred=C.read(C.O/'gate_predictions.json');meta,lookup,cases=C.B.load();byidx={lookup[p['sample_id'],p['position']]:p['sample_id'] for p in pred};ix={(p['fold'],p['seed'],p['sample_id']):p for p in pred};checked=0
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';cp=np.load(C.O/'models'/f'{name}.npz');z=np.load(C.CACHE/f'held_{name}.npz');forest=pickle.loads((C.O/'models'/f'{name}.pkl').read_bytes());x=((z['features']-cp['mean'])/cp['scale']).astype(np.float32);leafmean=np.zeros(len(x))
            for tree in forest.estimators_:leafmean+=tree.tree_.value[tree.apply(x),0,0]/64
            full=(z['features']-cp['mean'])/cp['scale'];logit=full@cp['event_coef']+cp['event_intercept'][0];prob=np.exp(-np.logaddexp(0.,-logit));value=leafmean*prob
            for i,j in enumerate(z['ids']):
                p=ix[f,s,byidx[int(j)]];assert np.isclose(p['conditional_normalized_gain'],leafmean[i],atol=1e-10,rtol=1e-10);assert np.isclose(p['predicted_normalized_gain'],value[i],atol=1e-10,rtol=1e-10);assert p['gate_continue']==bool(value[i]>z['features'][i,82]);checked+=1
    assert checked==4681;comparisons=C.read(C.O/'expected_budget_comparison.json');new=C.read(C.O/'oof_trajectories.json');ni={(p['method'],p['sample_id']):p for p in new};budgetchecks=0
    for name,folder in [('GRID198',C.LINEAR),('GRID199',C.CONSTANT)]:
        oi={(p['method'],p['sample_id']):p for p in C.read(folder/'oof_trajectories.json')}
        for r in comparisons[name]['per_date_seed']:
            s,d=r['seed'],r['date'];ids=[c['sample_id'] for c in cases if c['date']==d];N=len(ids);base=[ni[f'NET_seed{s}',i] for i in ids];arms={'OLD':[oi[f'GATE_seed{s}',i] for i in ids],'TREE':[ni[f'GATE_seed{s}',i] for i in ids],'ADD1_RANDOM':[ni[f'ADD1_seed{s}',i] for i in ids]};costs={k:sum(p['queries']-b['queries'] for p,b in zip(rr,base)) for k,rr in arms.items()};budget=min(costs['OLD'],costs['TREE']);assert budget==r['matched_extra_query_total']
            for k,rr in arms.items():
                chance=budget/costs[k] if costs[k] else 0.;gain=sum(b['gain'] for b in base)/N+chance*sum(p['gain']-b['gain'] for p,b in zip(rr,base))/N;assert abs(gain-r[k]['mean_gain'])<1e-12 and abs(r[k]['mean_queries']-sum(b['queries'] for b in base)/N-budget/N)<1e-12
            budgetchecks+=1
    assert budgetchecks==96;primary=comparisons['GRID198'];both=sum(p['gain_delta']>0 and p['match_delta_pp']>0 for p in primary['per_seed']);assert both==3
    random=[]
    for s in range(3):
        rr=[p for p in primary['per_date_seed'] if p['seed']==s];random.append(dict(seed=s,tree_minus_random_NET=float(np.mean([p['TREE']['mean_net_utility']-p['ADD1_RANDOM']['mean_net_utility'] for p in rr])),positive_dates=sum(p['TREE']['mean_net_utility']>p['ADD1_RANDOM']['mean_net_utility'] for p in rr)))
    C.save('per_seed_random_reference.json',dict(all_seeds_retained=True,per_seed=random));ma=primary['macro'];net=ma['TREE']['mean_net_utility']-ma['OLD']['mean_net_utility'];match=100*(ma['TREE']['action_match_rate']-ma['OLD']['action_match_rate']);pm=C.read(C.O/'predictive_metrics.json')['all12_mean'];oldpm=C.read(C.LINEAR/'predictive_metrics.json')['all12_mean'];oldcmse=oldpm['conditional_positive_MSE']
    interpretation='## 本轮判读\n\n'+f'在预先固定的同预期候选查询比较下，树头对线性条件头三个种子的奖励和动作一致率方向均为正，平均动作一致率增加{match:.4f}个百分点、NET增加{net:+.8f}。各个种子改善幅度明显不同，尤其种子0的奖励改善较小，不能称为跨日期普遍获胜或统计显著。\n\n'+f'正收益状态的条件幅度MSE为{pm["positive_conditional_MSE"]:.4f}，线性条件头为{oldcmse:.4f}，固定正例均值为{pm["positive_constant_MSE"]:.4f}。因此此次不仅保留了查询选择的小改善，也获得更好的幅度误差证据；模型容量与归纳偏置同时改变，不能将其全归于正值约束。\n\n'+'相同预期查询下随机追加的各个种子差异另存per_seed_random_reference.json，不省略不利种子。相对基础学习NET的代理效用平均只小幅为正，仍有两个种子为负；保留内部低查询候选，不据此宣布完整控制优势。没有新原FRAME闭环对比。\n\n'
    original=C.O/'GRID201_report.md';head,body=original.read_text(encoding='utf8').split('\n',1);(C.O/'GRID201_report_v2.md').write_text(head+'\n\n'+interpretation+body,encoding='utf8',newline='\n');C.save('independent_audit.json',dict(passed=True,GRID200_frozen_sources_verified=True,all4681_forest_leaf_products_reconstructed=True,all96_expected_query_budgets_reconstructed=True,positive_gain_and_match_seeds_vs198=both,all_random_seed_differences_retained=True,original_report_preserved=True,new_fits=0,new_NET_forwards=0,new_physics=0));C.archive('audited');dest=C.R/'论文协作资料/FRAME_浅树条件收益_GRID201'
    for n in ['GRID201_report_v2.md','independent_audit.json','per_seed_random_reference.json']:shutil.copy2(C.O/n,dest/n)
    msg=f'GRID201判读：同预期查询对198，三个种子奖励/动作均正，平均动作+{match:.4f}pp；正例条件MSE{pm["positive_conditional_MSE"]:.4f}优于线性{oldcmse:.4f}/均值{pm["positive_constant_MSE"]:.4f}。幅度不均衡，基础学习NET对比两种子仍负，保留低查询内部改善，不宣布完整控制优势或新FRAME比较。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/name;t=p.read_text(encoding='utf8');t+='\n\n### GRID201核验判读\n\n'+msg+'\n';p.write_text(t,encoding='utf8',newline='\n')
    p=C.REPO/'docs/NEXT_STEPS.md';p.write_text('# Next steps — GRID201核验后\n\n1. 按用户要求继续改进学习内部：保留201低查询树头，以及198/196较高奖励版本，不启动新原FRAME闭环对比。\n2. 正例幅度误差已有改善；下一步先只读区分漏掉高收益机会中的事件概率低估与条件幅度低估，保留同预算三个种子的小正向，不要求全日期获胜。\n3. 不盲目扩大网络或扫描阈值/费用。新训练先通知固定范围，不自动扩日期/物理/最终8/论文。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8').replace('protocols/GRID201_tree_amount/complete/GRID201_report.md','protocols/GRID201_tree_amount/audited/GRID201_report_v2.md',1),encoding='utf8',newline='\n');print('GRID201_INDEPENDENT_AUDIT',json.dumps(dict(match_pp=match,NET_delta=net,random_per_seed=random)))
def delivery():
    C.trained();assert C.read(C.O/'independent_audit.json')['passed'];head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote;stages=[]
    for out,arch,collab,phase in [('grid200_amount_role','GRID200_amount_role','FRAME_条件收益作用_GRID200','complete'),('grid201_tree_amount','GRID201_tree_amount','FRAME_浅树条件收益_GRID201','audited')]:
        src=C.R/'outputs'/out;a=C.REPO/'research_archive/power-grid-2026/protocols'/arch;dest=C.R/'论文协作资料'/collab;count=0
        for mf in a.rglob('archive_manifest.json'):
            for p,h in C.read(mf).items():assert C.sha(mf.parent/p)==h;count+=1
        files=0
        for p in dest.iterdir():
            ap=a/phase/p.name;assert C.sha(p)==C.sha(src/p.name)==C.sha(ap);assert git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==ap.read_bytes();files+=1
        stages.append(dict(stage=arch,archive_files_checked=count,collaboration_files_checked=files,committed_bytes_match=True))
    rec=dict(passed=True,stages=stages,verified_publication_head=head,remote_main=head,new_fits=0,new_physics=0);C.save('delivery_receipt.json',rec);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),C.O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps(rec))
if __name__=='__main__':globals()[sys.argv[1]]()
