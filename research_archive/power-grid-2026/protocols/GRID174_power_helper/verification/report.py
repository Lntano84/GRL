"""Retain all paired outcomes and trace old wins/losses, without an adoption gate."""
import collections,json,math,shutil
import run as D
def main():
    D.verify_trained();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'];assert not (O/'complete.json').exists()
    family=D.read(O/'family_summary.json');key=D.read(O/'key_changes.json');pairs=D.read(O/'paired_summary.json');events=D.read(O/'all7134_paired_events.json');tr=D.read(O/'oof_trajectories.json');by={(r['method'],r['sample_id']):r for r in tr};prior={(r['seed'],r['sample_id']):r for r in D.read(D.T.O/'all7134_paired_events.json')}
    traces=[];grouped=collections.defaultdict(list)
    for e in events:
        seed=e['seed'];sid=e['sample_id'];a=by[f'POWER8_GATE_seed{seed}',sid];b=by[f'ZERO8_GATE_seed{seed}',sid];F=by[f'FACTOR_seed{seed}',sid];old=by[f'OLD_GATE_seed{seed}',sid]
        cat='UNCHANGED' if a['batches']==b['batches'] else ('SAME_GAIN_LESS_QUERY' if a['queries']<b['queries'] and a['gain']==b['gain'] and a['chosen_pool_id']==b['chosen_pool_id'] else ('RESTORED_GAIN_NET_POSITIVE' if a['gain']>b['gain'] and e['net_delta']>0 else ('OTHER_NET_POSITIVE' if e['net_delta']>0 else 'NET_NEGATIVE_OR_ZERO')))
        traces.append(dict(e,category=cat,power_gain=a['gain'],zero_gain=b['gain'],factor_gain=F['gain'],power_queries=a['queries'],zero_queries=b['queries'],factor_queries=F['queries'],old_net_delta=prior[seed,sid]['net_delta'],
            improved_old_loss=(e['old_category']=='LOST_GAIN_TOO_MUCH' and a['net_utility']>old['net_utility']),old_loss_fully_restored=(e['old_category']=='LOST_GAIN_TOO_MUCH' and a['gain']==F['gain'] and a['chosen_pool_id']==F['chosen_pool_id']),
            original_same_gain_saving_preserved=(e['old_category']=='SAME_GAIN_LESS_QUERY' and a['gain']==F['gain'] and a['chosen_pool_id']==F['chosen_pool_id'] and a['queries']<F['queries'])))
        grouped[e['physical_state_group']].append(e)
    categories=[]
    for cat in sorted({e['category'] for e in traces}):
        rr=[e for e in traces if e['category']==cat];categories.append(dict(category=cat,pairs=len(rr),starts=len({e['sample_id'] for e in rr}),physical_groups=len({e['physical_state_group'] for e in rr}),net_contribution=math.fsum(e['weight']*e['net_delta'] for e in rr)))
    assert abs(sum(e['net_contribution'] for e in categories)-key['net_delta_power_vs_zero'])<1e-12
    recurrence=[]
    for g,rr in sorted(grouped.items()):
        if not any(r['net_delta']!=0 for r in rr):continue
        recurrence.append(dict(physical_state_group=g,positive_seeds=sorted({r['seed'] for r in rr if r['net_delta']>0}),negative_seeds=sorted({r['seed'] for r in rr if r['net_delta']<0}),net_contribution=math.fsum(r['weight']*r['net_delta'] for r in rr)))
    counts=dict(old15_losses_improved=sum(e['improved_old_loss'] for e in traces),old15_losses_fully_restored=sum(e['old_loss_fully_restored'] for e in traces),old226_same_gain_action_savings_preserved=sum(e['original_same_gain_saving_preserved'] for e in traces),
        positive_pairs=sum(e['net_delta']>0 for e in traces),negative_pairs=sum(e['net_delta']<0 for e in traces),positive_physical_groups=sum(bool(g['positive_seeds']) for g in recurrence),positive_groups_atleast2seeds=sum(len(g['positive_seeds'])>=2 for g in recurrence),positive_groups_all3seeds=sum(len(g['positive_seeds'])==3 for g in recurrence))
    D.save('mechanism_counts.json',counts);D.save('all7134_explained_events.json',traces);D.save('event_summary.json',categories);D.save('physical_group_recurrence.json',recurrence)
    positive=key['net_delta_power_vs_zero']>0
    lines=['# GRID174：给便宜帮助模型补入当前电气状态','',
        f"相对同容量ZERO8，POWER8组成的单向规则在16个留出日期上的平均局部净效用变化{key['relative_power_vs_zero']:+.2%}；{key['positive_seeds']}/3个种子、{key['positive_dates']}/16个日期平均正向。相对原FACTOR，净效用变化{key['relative_power_vs_factor']:+.2%}、查询减少{key['query_reduction_power_vs_factor']:.2%}。",'',
        f"此前15条有利查询漏停中，{counts['old15_losses_improved']}条改善，{counts['old15_losses_fully_restored']}条完全恢复原FACTOR的收益和动作；此前226条同收益同动作节省中，{counts['old226_same_gain_action_savings_preserved']}条仍被保留。全部新增改善与损失均保留。",'',
        '## 只改了哪一处','',
        '原FACTOR、费用路由模型的折外权重和单向停止规则均固定，只重新训练便宜MLP帮助模型。两组都是50—64—64—2、7554个总参数；原42列来自相同随机初始化，新增8列初始权重均为0。ZERO8的新增输入始终为0，POWER8提供GRID173核验的当前负载、站内功率及已知候选分区描述。',
        '', '同架构总参数相同不等于活跃输入维度相同：ZERO8的8列梯度恒零。此对照检验的是整个输入块的作用，不证明每一项必要，也不能将重复最大负载率当成新信息。没有更换标签、费用、候选顺序、阈值或采样。',
        '', '每折8项的归一化只由12个拟合日期计算；7250次更新、batch32、原Adam和NET Huber目标不变。三种子、四折、两组24份最终权重在留出评分前统一推送。原4个开发日期没有提取新输入或评分，最终8未生成。','',
        '## 全部留出结果','', '| 方案 | 净效用 | 平均查询 | 完整补查查询比例 | 收益保留率 | 调用均衡效用 |','|---|---:|---:|---:|---:|---:|']
    for method in ['FACTOR','OLD_GATE','ZERO8_GATE','POWER8_GATE','ZERO8_HELPER','POWER8_HELPER']:
        f=family[method];lines.append(f"| {method} | {f['mean_net_utility']:.8f} | {f['mean_queries']:.3f} | {f['query_fraction']:.2%} | {f['gain_capture']:.2%} | {f['physical_group_mean_utility']:.8f} |")
    lines+=['','GATE是接入原单向规则后的完整补查策略；HELPER是小模型单独决定，单列用于诊断，不能混用两者成绩。','',
        f"ZERO8对原COST42的全部训练/留出状态决定，有{key['zero8_decision_differences_from_old_cost']}行不同，实际分数差保存于control_reproduction.json，不假定扩大零列后优化必然完全一致。",'',
        '| 种子 | POWER8 − ZERO8净效用 |','|---|---:|']
    for p in pairs['pairs']:lines.append(f"| {p['seed']} | {p['net_delta']:+.8f} |")
    lines+=['','| 日期 | 三种子平均净效用变化 |','|---|---:|']
    for p in pairs['per_date']:lines.append(f"| {p['date']} | {p['mean_delta']:+.8f} |")
    lines+=['','## 变化来自哪里','','| 类型 | 种子×起点对 | 起点 | 物理状态组 | 宏平均效用贡献 |','|---|---:|---:|---:|---:|']
    for e in categories:lines.append(f"| {e['category']} | {e['pairs']} | {e['starts']} | {e['physical_groups']} | {e['net_contribution']:+.8f} |")
    lines+=['',f"正向变化涉及{counts['positive_physical_groups']}个物理组，{counts['positive_groups_atleast2seeds']}组至少两个种子重现，{counts['positive_groups_all3seeds']}组在三个种子都重现。前缀和种子相关，不能当独立场景或统计显著性。",'',
        '## 如何继续','',
        '小幅、重复出现的正向变化保留，不用全部种子全部日期获胜的门槛抹去；新增漏收益与无利多查也逐条保留。优先解释同一公共输入块在哪些有利机会恢复了继续、在哪些无收益位置仍能停止，不马上增加新网络或追逐阈值。',
        '', '当前母线不平衡描述只把已知动作与当前功率联系起来，不是对未来电网的潮流求解。它的含义依据Grid2Op观测字段定义，特征本身是本项目的固定描述：[官方观测文档](https://grid2op.readthedocs.io/en/latest/user/observation.html)。',
        '', f"24次拟合总耗时{D.read(O/'fits.json')['total_wall_s']:.1f}秒。独立NumPy分数、全部优化器步数/抽样、仅拟合日期归一化和逐搜索查询上界通过。仍是同网历史数据的局部NET开发实验，不把查询下降解释成完整控制提速或整周质量保证，也不把旧GRID171计时直接套到新模型。无新物理、新数据或论文改动。"]
    (O/'GRID174_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    D.save('decision.json',dict(positive_mean=positive,positive_seeds=key['positive_seeds'],positive_dates=key['positive_dates'],small_recurring_signals_retained=True,adopted=False,ready_to_scale=False,next='Read-only actual restored opportunities and retained query savings under the fixed block; no further feature/loss/threshold scan.'))
    D.save('complete.json',dict(passed_audit=True,new_fits=24,new_physics=0,new_data=0,final8_absent=True,all_adverse_results_retained=True));D.archive('complete')
    collab=R/'论文协作资料/FRAME_电气输入帮助模型_GRID174';collab.mkdir(exist_ok=False)
    for n in ['GRID174_report.md','key_changes.json','family_summary.json','mechanism_counts.json','event_summary.json','paired_summary.json','review.json','decision.json','complete.json']:shutil.copy2(O/n,collab/n);assert D.sha(collab/n)==D.sha(O/n)
    msg=f"GRID174仅cheaphelper POWER8vsZERO8，同架构7554总参数/初始化/目标/采样/四折三种子24拟合，F/routed/单向规则不变，全权重先推送。POWERvsZERO净效用{key['relative_power_vs_zero']:+.2%}，{key['positive_seeds']}/3种子、{key['positive_dates']}/16日期正；相对F{key['relative_power_vs_factor']:+.2%}、查询少{key['query_reduction_power_vs_factor']:.2%}。旧15损失改善{counts['old15_losses_improved']}全恢复{counts['old15_losses_fully_restored']}，旧226正确省保留{counts['old226_same_gain_action_savings_preserved']}。不假装每项必要或活跃参数相同；重复maxrho明示。全部反例保留，无物理/原4dev新评分/最终8，不修改论文。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID174电气输入帮助模型\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        p.write_text(old,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID174固定输入块解释\n\n1. 保留POWER8与ZERO8全部四折、种子与正反变化，复核恢复旧有利机会和保留原查询节省的位置，不以全赢门槛否定小改善。\n2. 只读解释既有8项块在实际决定中的作用，不追加字段、门槛、损失或网络搜索。新训练前给针对性方案和通知。\n3. 局部NET/收益/查询与完整周质量/控制时间分开；新物理/数据扩大先给范围，原4dev/最终8和论文不动。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(f'# 最新：GRID174电气输入帮助模型\r\n\r\n[报告](protocols/GRID174_power_helper/complete/GRID174_report.md)；POWER8对同容量ZERO8效用{key["relative_power_vs_zero"]:+.2%}，全部四折结果保留。\r\n\r\n'.encode('utf8')+p.read_bytes())
    checked=0
    for stage in D.A.iterdir():
        for n,h in D.read(stage/'archive_manifest.json').items():assert D.sha(stage/n)==h;checked+=1
    D.save('publication_verification.json',dict(passed=True,archive_files_checked=checked,model_checkpoints_checked=24,collaboration_copies=9));dest=D.A/'verification';dest.mkdir();shutil.copy2(O/'publication_verification.json',dest/'publication_verification.json');shutil.copy2(__file__,dest/'report.py')
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:D.sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID174_COMPLETE',msg)
if __name__=='__main__':main()
