"""Reconstruct saved decisions; verify joint GRID202/203 publication and copies."""
from pathlib import Path
import json,pickle,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,read=C.R,C.O,C.read

def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])

def audit():
    assert not (O/'independent_audit.json').exists();C.trained();assert read(O/'complete.json')['passed'];meta,lookup,cases=C.B.load();pred=read(O/'gate_predictions.json');index={(p['fold'],p['seed'],lookup[p['sample_id'],p['position']]):p for p in pred};checked=0
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';z=np.load(C.CACHE/f'held_{name}.npz');cp=np.load(O/'models'/f'{name}.npz');x=((z['features']-cp['mean'])/cp['scale']).astype(np.float32);event=pickle.loads((O/'models'/f'{name}_event.pkl').read_bytes());amount=pickle.loads((O/'models'/f'{name}_amount.pkl').read_bytes());p=np.zeros(len(x));mu=np.zeros(len(x))
            for tree in event.estimators_:
                v=tree.tree_.value[tree.apply(x),0,:];p+=v[:,1]/v.sum(axis=1)/64
            for tree in amount.estimators_:mu+=tree.tree_.value[tree.apply(x),0,0]/64
            for j,i in enumerate(z['ids']):
                row=index[f,s,int(i)];assert np.isclose(row['event_probability'],p[j],atol=1e-11,rtol=1e-10) and np.isclose(row['conditional_normalized_gain'],mu[j],atol=1e-10,rtol=1e-10);assert row['gate_continue']==bool(p[j]*mu[j]>z['features'][j,82]);checked+=1
    assert checked==4681
    # Recompute same-query expectation directly from saved per-start net utilities.
    ni={(r['method'],r['sample_id']):r for r in read(O/'oof_trajectories.json')};comparisons=read(O/'expected_budget_comparison.json');count=0
    for name,folder in [('GRID201',C.OLD),('GRID198',C.LINEAR)]:
        oi={(r['method'],r['sample_id']):r for r in read(folder/'oof_trajectories.json')}
        for row in comparisons[name]['per_date_seed']:
            s,d=row['seed'],row['date'];ids=[c['sample_id'] for c in cases if c['date']==d];base=[ni[f'NET_seed{s}',i] for i in ids];arms={'OLD':[oi[f'GATE_seed{s}',i] for i in ids],'NEW':[ni[f'GATE_seed{s}',i] for i in ids],'ADD1_RANDOM':[ni[f'ADD1_seed{s}',i] for i in ids]};costs={k:sum(a['queries']-b['queries'] for a,b in zip(rr,base)) for k,rr in arms.items()};budget=min(costs['OLD'],costs['NEW']);assert budget==row['matched_extra_query_total']
            for k,rr in arms.items():
                chance=budget/costs[k] if costs[k] else 0.;utility=np.mean([b['net_utility']+chance*(a['net_utility']-b['net_utility']) for a,b in zip(rr,base)]);assert abs(utility-row[k]['mean_net_utility'])<1e-10
            count+=1
    assert count==96
    diag=read(O/'interpretation.json');total=sum(r['macro_delta_utility'] for r in diag['groups'].values());top=max(diag['per_call_contributions'],key=lambda r:r['macro_delta_utility']);remainder=total-top['macro_delta_utility'];rows=read(O/'interpretation_rows.json');positive=[r for r in rows if r['call']==top['call'] and r['group']=='NEW_ONLY' and r['gain']>0];assert len(positive)>0
    desc=dict(total_actual_utility_delta=total,largest_positive_call=top['call'],largest_contribution=top['macro_delta_utility'],share_of_net_improvement=top['macro_delta_utility']/total,remaining_signed_contribution=remainder,positive_NEW_ONLY_pairs_in_call=len(positive),positive_NEW_ONLY_unique_prefixes=len({r['sample_id'] for r in positive}),represented_seeds=sorted({r['seed'] for r in positive}),local_gain_min=min(r['gain'] for r in positive),local_gain_max=max(r['gain'] for r in positive),query_min=min(r['queries'] for r in positive),query_max=max(r['queries'] for r in positive),not_an_independent_retest=True)
    C.save('contribution_concentration.json',desc)
    text='''# GRID203结果判读

这次有改善，值得保留新版继续研究。只更换事件概率读出后，原本被低估的一组有利补查被找回，同时取消了一些查询多、奖励小的追加。

'''+f"整体实际代理净效用改善{total:+.9f}，其中{top['call']}贡献{top['macro_delta_utility']:+.9f}，占净改善的{100*desc['share_of_net_improvement']:.2f}%。其余调用的贡献相加仍为{remainder:+.9f}，但幅度小得多。这个比例以正负相抵后的净改善为分母，是集中程度描述，不是独立样本的成功率。\n\n"+f"该调用的收益恢复出现在种子{desc['represented_seeds']}，共{len(positive)}个种子—起点配对、{desc['positive_NEW_ONLY_unique_prefixes']}个相关起点；追加{desc['query_min']}–{desc['query_max']}次候选预测，局部奖励增量为{desc['local_gain_min']:.8f}。这是同一物理状态的相关起点，不能称10个独立成功场景。原概率在这些起点很低，新概率跨过既定费用条件；幅度模型和费用未变。\n\n"+'''这正是值得继续追踪的小范围机制：为什么相同隐藏表示下，浅树概率读出能够保留这组收益，而线性概率读出几乎排除了它。先解释训练支持、相近正反例和预测路径，再决定下一次学习改动。不能直接把这个调用、查询范围或事后收益写成触发规则。

相同预期查询下对GRID201仍是两个种子净效用提高、一个下降；动作一致率平均下降0.0250个百分点。对基础学习NET三个种子都提高。两种比较的参照不同，全部保留。新版不因此取代全部历史候选，也不因集中于少数调用而被否定。

本轮没有新物理回放、最终测试或原FRAME比较。报告中的局部奖励和代理净效用不等于完整周运行质量或实际时延。
'''
    (O/'结果判读.md').write_text(text,encoding='utf8',newline='\n');C.save('independent_audit.json',dict(passed=True,all4681_leaf_products_and_decisions_reconstructed=True,all96_expected_query_utility_mixtures_reconstructed=True,all_counterexamples_and_concentration_retained=True,new_fits=0,new_NET_forwards=0,new_physics=0));C.archive('audited');dest=R/'论文协作资料/FRAME_浅树概率_GRID203'
    for n in ['结果判读.md','contribution_concentration.json','independent_audit.json']:shutil.copy2(O/n,dest/n)
    msg=f"GRID203改善集中：{top['call']}占实际净效用改善{100*desc['share_of_net_improvement']:.2f}%，其余调用贡献合计{remainder:+.9f}。该调用5相关起点在2种子恢复；保留改善但不作跨场景稳定性结论。下一步只读解释相同表示下概率头何以分歧，再定训练。"
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+'\n\n### GRID203集中程度与核验\n\n'+msg+'\n',encoding='utf8',newline='\n')
    p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8').replace('[报告](protocols/GRID203_tree_event/complete/GRID203_report.md)','[报告](protocols/GRID203_tree_event/audited/GRID203_report.md)；[判读](protocols/GRID203_tree_event/audited/结果判读.md)',1),encoding='utf8',newline='\n');print('GRID203_INDEPENDENT_AUDIT',json.dumps(desc,ensure_ascii=False))

def delivery():
    C.trained();assert read(O/'independent_audit.json')['passed'];head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote;stages=[]
    for out,arch,collab,phase in [('grid202_factor_bottleneck','GRID202_factor_bottleneck','FRAME_概率幅度瓶颈_GRID202','complete'),('grid203_tree_event','GRID203_tree_event','FRAME_浅树概率_GRID203','audited')]:
        src=R/'outputs'/out;a=C.REPO/'research_archive/power-grid-2026/protocols'/arch;dest=R/'论文协作资料'/collab;count=0
        for mf in a.rglob('archive_manifest.json'):
            for p,h in read(mf).items():assert C.sha(mf.parent/p)==h;count+=1
        for p in dest.iterdir():
            assert p.is_file();ap=a/phase/p.name;assert C.sha(p)==C.sha(src/p.name)==C.sha(ap);assert git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==ap.read_bytes()
        for p,h in read(src/'frozen_inputs.json').items():assert C.sha(R/p)==h
        stages.append(dict(stage=arch,archive_files_checked=count,collaboration_files_checked=len(list(dest.iterdir())),committed_bytes_match=True))
    rec=dict(passed=True,stages=stages,verified_publication_head=head,remote_main=head,new_fits=0,new_physics=0);C.save('delivery_receipt.json',rec);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');shutil.copy2(O/'delivery_receipt.json',R/'论文协作资料/FRAME_浅树概率_GRID203/delivery_receipt.json');print(json.dumps(rec))

if __name__=='__main__':globals()[sys.argv[1]]()
