"""Independent native and mixture checks, then joint diagnostic/model delivery."""
from pathlib import Path
import json,pickle,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,read=C.R,C.O,C.read
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])
def audit():
    C.trained();assert read(O/'complete.json')['passed'] and not (O/'independent_audit.json').exists();meta,lookup,cases,raw=C.raw_inputs();pred=read(O/'gate_predictions.json');ix={(p['fold'],p['seed'],lookup[p['sample_id'],p['position']]):p for p in pred};checked=0
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';cp=np.load(O/'models'/f'{name}.npz');hz=np.load(C.CACHE/f'held_{name}.npz');x=np.c_[(hz['features']-cp['mean'])/cp['scale'],(raw[hz['ids'].astype(int)]-cp['raw_mean'])/cp['raw_scale']].astype(np.float32);event=pickle.loads((O/'models'/f'{name}_event.pkl').read_bytes());amount=pickle.loads((O/'models'/f'{name}_amount.pkl').read_bytes());prob=np.zeros(len(x));mu=np.zeros(len(x))
            for tree in event.estimators_:v=tree.tree_.value[tree.apply(x),0,:];prob+=v[:,1]/v.sum(1)/64
            for tree in amount.estimators_:mu+=tree.tree_.value[tree.apply(x),0,0]/64
            for j,i in enumerate(hz['ids']):
                p=ix[f,s,int(i)];assert np.isclose(prob[j],p['event_probability'],atol=1e-10,rtol=1e-10) and np.isclose(mu[j],p['conditional_normalized_gain'],atol=1e-10,rtol=1e-10);assert p['gate_continue']==bool(prob[j]*mu[j]>hz['features'][j,82]);checked+=1
    assert checked==4681;ni={(r['method'],r['sample_id']):r for r in read(O/'oof_trajectories.json')};comparisons=read(O/'expected_budget_comparison.json');budgetchecks=0
    for name,folder in [('GRID206',C.OLD),('GRID203',C.PRIOR)]:
        oi={(r['method'],r['sample_id']):r for r in read(folder/'oof_trajectories.json')}
        for row in comparisons[name]['per_date_seed']:
            s,d=row['seed'],row['date'];ids=[c['sample_id'] for c in cases if c['date']==d];base=[ni[f'NET_seed{s}',i] for i in ids];arms={'OLD':[oi[f'GATE_seed{s}',i] for i in ids],'NEW':[ni[f'GATE_seed{s}',i] for i in ids],'ADD1_RANDOM':[ni[f'ADD1_seed{s}',i] for i in ids]};costs={k:sum(a['queries']-b['queries'] for a,b in zip(rr,base)) for k,rr in arms.items()};budget=min(costs['OLD'],costs['NEW']);assert budget==row['matched_extra_query_total']
            for k,rr in arms.items():
                chance=budget/costs[k] if costs[k] else 0.;values=[(b['net_utility']+chance*(a['net_utility']-b['net_utility']),b['queries']+chance*(a['queries']-b['queries']),float(b['action_match'])+chance*(int(a['action_match'])-int(b['action_match']))) for a,b in zip(rr,base)];v=np.mean(values,axis=0);assert np.allclose(v,[row[k]['mean_net_utility'],row[k]['mean_queries'],row[k]['action_match_rate']],atol=1e-10,rtol=1e-10)
            budgetchecks+=1
        assert len(comparisons[name]['per_seed'])==3 and all(r['NET_delta']>0 and r['gain_delta']>0 and r['match_delta_pp']>0 for r in comparisons[name]['per_seed'])
    assert budgetchecks==96;decision=read(O/'decision.json');assert decision['prior_main_recovery_retained']==decision['prior_main_recovery_total']==10;assert decision['previous_unprofitable_positive_stopped']==7 and decision['previous_profitable_positive_retained']==2
    new=read(O/'family_summary.json')['GATE'];old=read(C.OLD/'family_summary.json')['GATE'];nw=read(O/'wasted_query_audit.json');ow=read(C.OLD/'wasted_query_audit.json');pm=read(O/'predictive_metrics.json')['all12_mean'];opm=read(C.OLD/'predictive_metrics.json')['all12_mean'];zero=dict(old_fraction=ow['date_macro_zero_gain_query_fraction'],new_fraction=nw['date_macro_zero_gain_query_fraction'],old_absolute=ow['date_macro_zero_gain_queries'],new_absolute=nw['date_macro_zero_gain_queries']);C.save('absolute_and_fractional_waste.json',zero)
    lines=['# GRID208结果判读：保留一个更省查询的学习候选','', '这轮只改条件收益模块的输入，固定上一轮已改善的概率模型。结果支持保留一个小而明确的内部改善：三个种子的实际候选查询都下降；相同预期查询量下，对GRID206的三个种子奖励和动作一致率都提高。种子1的净效用提高较小，不据此要求全日期获胜，也不称统计显著。','',f"每起点候选查询{old['mean_queries']:.5f}→{new['mean_queries']:.5f}，代理净效用{old['mean_net_utility']:.9f}→{new['mean_net_utility']:.9f}。同预期查询对206的净效用平均变化{decision['matched']['GRID206']['NET_delta']:+.9f}、动作一致率变化{decision['matched']['GRID206']['match_delta_pp']:+.5f}个百分点；对203的三个种子也全部为正，但后者涉及前两轮累计改动，不是单模块因果对照。",'', '## 这次具体改进了什么','', '此前主要恢复调用的10个相关配对全部保留。上轮新增10个正收益却不回本配对中，这次7个不再启动，而2个新增回本配对均保留。这符合希望收益模块更好地区分大小增量的方向，不过仍是开发教师路径上的案例证据。', '', f"正例条件幅度MSE从{opm['positive_conditional_MSE']:.6f}到{pm['positive_conditional_MSE']:.6f}，事件AUC保持{pm['event_AUC']:.6f}。概率权重字节完全一致，所以本轮概率识别能力没有新增训练。",'', '## 同时保留的代价与反例','',f"实际固定策略的动作一致率由{100*old['action_match_rate']:.5f}%降至{100*new['action_match_rate']:.5f}%。种子0实际净效用变化{decision['actual_per_seed'][0]['NET_delta']:+.9f}，其余两种子为正；不能把同预期预算的三种子正向写成实际策略全赢。",'', f"日期宏平均零收益追加查询的绝对量{zero['old_absolute']:.6f}→{zero['new_absolute']:.6f}，但在追加查询中的占比{100*zero['old_fraction']:.3f}%→{100*zero['new_fraction']:.3f}%。绝对开销减少与占比上升可以同时发生，两个指标均保留。",'', '所有被取消的旧追加中，仍有4个实际回本配对，因此收益筛选尚不可靠；不能只报告那7个有效取消。新版新增5个回本配对，整体回本次数有所增加，但不同起点共享教师调用，不能当独立实验。', '', '## 下一步','', '保留GRID208作为较低查询候选，也保留GRID206较高实际动作一致率版本。下一步只读检查被取消的4个回本配对：它们是否与已保留的高收益机会共享特征和训练支持，幅度估低出现在哪些状态。再据此固定一次有针对性的改动，不先增加网络或改费用门槛。', '', '没有新物理回放、最终测试或原FRAME闭环。当前奖励和查询代理效用不等于完整周成本、存活或实际控制时间。']
    (O/'结果判读与下一步.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.save('independent_audit.json',dict(passed=True,all4681_native_leaf_products_decisions_reconstructed=True,all96_expected_query_utility_and_match_mixtures_reconstructed=True,all3_matched_seeds_gain_and_match_positive_vs206_and203=True,all_adverse_actual_cases_and_waste_fraction_retained=True,new_fits=0,new_NET_forwards=0,new_physics=0));C.archive('audited');dest=R/'论文协作资料/FRAME_公开摘要收益_GRID208'
    for n in ['结果判读与下一步.md','independent_audit.json','absolute_and_fractional_waste.json']:shutil.copy2(O/n,dest/n)
    msg='GRID208独立核验：同预期查询对206/203三个种子奖励与动作均正，主10恢复全保留，旧10正收益未回本取消7且2回本全保留。实际查询三种子均降但种子0实际净效用负、整体动作略降；旧取消中另有4回本损失，零收益查询绝对量降但占比升，反例不删。下一步只读查4个丢失回本配对，不自动训练或新FRAME。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+'\n\n### GRID208小改善与反例核验\n\n'+msg+'\n',encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID208核验后\n\n1. 保留208低查询候选及206较高实际动作一致率，不新FRAME闭环。\n2. 下一步只读查被取消的4个回本配对和保留机会的特征/幅度支持；不忽略种子0实际净效用负、零收益绝对量与占比不同方向。\n3. 需要新训练先通知固定单改动；不调费用/阈值/批次，不扩物理/最终8/论文。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8').replace('[报告](protocols/GRID208_public_amount/complete/GRID208_report.md)','[报告](protocols/GRID208_public_amount/audited/GRID208_report.md)；[判读](protocols/GRID208_public_amount/audited/结果判读与下一步.md)',1),encoding='utf8',newline='\n');print('GRID208_INDEPENDENT_AUDIT',json.dumps(dict(checked=checked,budgetchecks=budgetchecks,matched=decision['matched'],waste=zero)))
def delivery():
    C.trained();head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote;stages=[]
    for out,arch,collab in [('grid207_amount_cost','GRID207_amount_cost','FRAME_收益幅度与成本_GRID207'),('grid208_public_amount','GRID208_public_amount','FRAME_公开摘要收益_GRID208')]:
        src=R/'outputs'/out;a=C.REPO/'research_archive/power-grid-2026/protocols'/arch;dest=R/'论文协作资料'/collab;assert read(src/'complete.json')['passed'] and read(src/'independent_audit.json')['passed'];count=0
        for mf in a.rglob('archive_manifest.json'):
            for p,h in read(mf).items():assert C.sha(mf.parent/p)==h;count+=1
        for p in dest.iterdir():
            assert C.sha(p)==C.sha(src/p.name);matches=[ap for ap in a.rglob(p.name) if ap.is_file() and C.sha(ap)==C.sha(p)];assert matches;assert any(git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==p.read_bytes() for ap in matches)
        for p,h in read(src/'frozen_inputs.json').items():assert C.sha(R/p)==h,p
        stages.append(dict(stage=arch,archive_files_checked=count,collaboration_files_checked=len(list(dest.iterdir())),committed_bytes_match=True))
    rec=dict(passed=True,verified_publication_head=head,remote_main=head,stages=stages,new_fits=0,new_physics=0);C.save('delivery_receipt.json',rec);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');shutil.copy2(O/'delivery_receipt.json',R/'论文协作资料/FRAME_公开摘要收益_GRID208/delivery_receipt.json');print(json.dumps(rec))
if __name__=='__main__':globals()[sys.argv[1]]()
