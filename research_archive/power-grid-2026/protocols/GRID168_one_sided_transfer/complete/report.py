"""Publish all results of the prespecified first date fold, without choosing a winner."""
import json,math,shutil
import run as D

def main():
    D.verify_trained();O,R,REPO=D.O,D.R,D.REPO
    assert D.read(O/'review.json')['passed'];assert not (O/'complete.json').exists()
    design=D.read(O/'design.json');fam=D.read(O/'family_summary.json');paired=D.read(O/'paired_summary.json');held=paired['held'];fit=paired['fit'];f=fam['held']
    positive_dates=sum(r['mean_delta']>0 for r in held['per_date']);positive_pairs=sum(r['net_delta']>0 for p in held['pairs'] for r in p['per_date'])
    unchanged_pairs=sum(r['net_delta']==0 for p in held['pairs'] for r in p['per_date'])
    delta=held['mean_net_delta'];relative=held['relative_change'];gainpp=100*(f['ONE_SIDED']['gain_capture']-f['FACTOR']['gain_capture'])
    D.save('key_changes.json',dict(held_net_delta=delta,held_net_relative=relative,held_mean_query_reduction=held['query_reduction'],held_gain_capture_percentage_points=gainpp,positive_seeds=held['positive_seeds'],positive_dates=positive_dates,positive_seed_dates=positive_pairs,unchanged_seed_dates=unchanged_pairs,fit_net_relative=fit['relative_change']))
    direction='提高' if delta>0 else ('降低' if delta<0 else '不变')
    lines=['# GRID168：固定单向停止规则的第一折日期迁移','',
        f"相对同折重新训练的FACTOR，留出日期的平均补查净效用{direction}{abs(relative):.2%}，平均每个起点查询减少{held['query_reduction']:.2%}。{held['positive_seeds']}/3个种子效用提高，{positive_dates}/4个日期的种子平均提高，{positive_pairs}/12个种子×日期组合提高、{unchanged_pairs}个不变。",'',
        '本轮先固定规则，再重训组成模型。它检查上轮发现的小改善是否能转移到未参与本轮拟合的日期，不要求全部种子、全部日期获胜才有探索价值。所有结果和反例均保留。','',
        '## 固定了什么','',
        '只使用GRID121早已定义的第0折：12个日期拟合，4个日期留出。每个种子重新训练FACTOR、费用路由FACTOR和小费用MLP，合计9份；未加载任何旧拟合权重。原4个开发日期没有进入本轮前向预测，最终8测试未生成。',
        '', '留出日期：'+ '、'.join(design['fold']['held_dates'])+'。',
        '',f"拟合{design['population']['fit_states']}个保存状态/{design['population']['fit_starts']}个补查起点；留出{design['population']['held_states']}个状态/{design['population']['held_starts']}个起点。前缀、同日期重复场景及模型种子不能当成独立物理场景。",
        '', '原始41项公共特征的均值和尺度只由12个拟合日期计算，复现旧第0折归一化。抽样概率只保留拟合行再归一化，三个模型在每个种子上使用完全相同的7250×32条索引流，留出行抽样为0。NET目标、每查询费用、收益尺度、优化器和更新数不变。9份最终权重已在留出日期预测前统一推送。',
        '', '单向规则始终为：原FACTOR继续 AND（费用路由FACTOR继续 OR 小费用MLP继续）。因此只有两位帮助模型都建议停时才提前停，原FACTOR的停止决定不会被重新打开。没有阈值、权重、日期或规则搜索。','',
        '## 留出日期结果','',
        '| 策略 | 日期宏平均净效用 | 平均查询数 | 完整补查查询比例 | 收益保留率 | 物理调用均衡效用 |',
        '|---|---:|---:|---:|---:|---:|']
    for a in D.ARMS+['ONE_SIDED']:
        v=f[a];lines.append(f"| {a} | {v['mean_net_utility']:.8f} | {v['mean_queries']:.3f} | {v['query_fraction']:.2%} | {v['gain_capture']:.2%} | {v['physical_group_mean_utility']:.8f} |")
    lines += ['', f"收益保留率相对FACTOR变化{gainpp:+.4f}个百分点。净效用同时考虑局部收益与查询费用；少查不等于收益、动作或完整周质量无损。",'',
        '| 种子 | FACTOR效用 | 单向规则效用 | 差值 |','|---|---:|---:|---:|']
    for p in held['pairs']:lines.append(f"| {p['seed']} | {p['factor']:.8f} | {p['one_sided']:.8f} | {p['net_delta']:+.8f} |")
    lines+=['','| 留出日期 | 三种子平均效用差 |','|---|---:|']
    for p in held['per_date']:lines.append(f"| {p['date']} | {p['mean_delta']:+.8f} |")
    lines+=['','## 改善和损失分别来自哪里','',
        '| 变化类型 | 种子×起点对 | 不同起点 | 物理状态组 | 宏平均效用贡献 |','|---|---:|---:|---:|---:|']
    cn={'UNCHANGED':'不变','SAME_GAIN_LESS_QUERY':'同收益而少查','GAIN_LOSS_NET_POSITIVE':'少量收益损失、节省费用仍净正','LOST_GAIN_TOO_MUCH':'漏掉收益过多、净效用损失'}
    for p in held['event_summary']:lines.append(f"| {cn[p['category']]} | {p['pairs']} | {p['starts']} | {p['physical_groups']} | {p['net_contribution']:+.8f} |")
    events=D.read(O/'held_paired_events.json');changed=[e for e in events if e['saved_queries']]
    if changed:
        lines+=['',f"共{len(changed)}对搜索在原本继续的位置提前停止；其中下一批真实增益为零的有{sum(e['next_gain']==0 for e in changed)}对，下一批真实净收益为正的有{sum(e['next_net']>0 for e in changed)}对。这里的未来增益仅用于解释，未输入模型或停止规则。"]
    bad=[e for e in changed if e['net_delta']<0]
    if bad:
        lines+=['','净损失案例逐条保存在held_paired_events.json，下面列出损失最大的前5对（完整记录全部保留）：']
        for e in sorted(bad,key=lambda r:r['net_delta'])[:5]:
            lines.append(f"- 种子{e['seed']}，{e['date']}，起点{e['sample_id']}，位置{e['first_difference_position']}：少查{e['saved_queries']}，局部收益变化{e['gain_delta']:.8g}，净效用变化{e['net_delta']:+.8f}。")
    lines+=['','## 拟合侧与核验','',f"拟合日期的平均净效用变化为{fit['relative_change']:+.2%}；此项只用于诊断，不代替留出结果。9次拟合总耗时{D.read(O/'fits.json')['total_wall_s']:.1f}秒。",
        '',f"核验：{D.read(O/'review.json')['checks']['numpy_rows']}行独立NumPy前向（结构模型覆盖全部留出状态）、全部保存搜索独立遍历、逐状态不重开STOP、逐搜索查询不增加、拟合日期归一化及完整抽样流、9份优化器最终步数和权重封存均通过。无新物理预测或独立AC重求解。",
        '', '## 结论边界与下一步','',
        '这是一组固定日期折上的局部补查迁移核查。数据与日期均已在历史开发中暴露；费用与收益尺度仍是历史16日期定义的固定任务单位，因此不能称作新的盲测或完整嵌套泛化。',
        '', '三模型推理及结构特征处理的实际时间尚未纳入旧查询费用，不能把查询下降直接写成控制提速。单向规则保证同起点、同候选顺序下一次补查查询不增加，不保证后续物理轨迹或整周成本。',
        '', '保留本轮小改善及其损失，优先解释跨种子重复出现的正确停止位置与漏收益位置。后续若核查其它已存在的日期折，仍固定规则、全部报告，不按第一折表现选择日期；新数据或物理扩大另行给出范围。']
    (O/'GRID168_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    D.save('decision.json',dict(developmental_positive_mean=delta>0,positive_seeds=held['positive_seeds'],positive_dates=positive_dates,query_bound_verified=True,adopted=False,ready_to_scale=False,interpretation='Small recurring improvements retained; not an all-seed/date adoption gate.',next='Trace first differences across all held dates and seeds before another fixed-fold check; no new rule search or physics.'))
    D.save('complete.json',dict(passed_audit=True,new_fits=9,new_data=0,new_physics=0,original4dev_not_scored=True,final8_unmaterialized=True,all_unfavorable_results_retained=True));D.archive('complete')
    collab=R/'论文协作资料/FRAME_单向停止日期迁移_GRID168';collab.mkdir(exist_ok=False)
    names=['GRID168_report.md','design.json','key_changes.json','family_summary.json','paired_summary.json','review.json','decision.json','complete.json']
    for n in names:shutil.copy2(O/n,collab/n)
    msg=f"GRID168固定旧第0折12fit/4held，9次全新拟合，归一化仅fit/三族同索引流/全部权重预测前推送。单向规则相对同折FACTOR留出净效用{relative:+.2%}、均查询减少{held['query_reduction']:.2%}，{held['positive_seeds']}/3种子正、{positive_dates}/4日期平均正、{positive_pairs}/12种子日期正。全部反例保留；已暴露同网旧局部NET任务，历史费用单位固定，非盲测/整周质量/实际提速。继续追小改善及损失，不用全赢门槛否定，不自动扩物理或新数据。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        path=REPO/'docs'/name;old=path.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID168单向停止日期迁移\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        path.write_text(old,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID168固定规则迁移\n\n1. 解释第0折单向早停的正反变化，全部种子/日期保留；不再寻找投票规则或阈值。\n2. 下一轮若用其它既有日期折，先通知再重新拟合，沿用原拆分、任务单位、特征、抽样和规则，全部报告。小改善有探索价值，不要求全赢。\n3. 不自动扩大新数据/物理，不生成最终8，不改论文；实际推理成本与整周质量另行核验。\n',encoding='utf8',newline='\n')
    index=REPO/'research_archive/power-grid-2026/README.md'
    prefix=f'# 最新：GRID168单向停止日期迁移\r\n\r\n[报告](protocols/GRID168_one_sided_transfer/complete/GRID168_report.md)；固定第0折留出效用{relative:+.2%}、均查询减少{held["query_reduction"]:.2%}，全部种子日期保留。\r\n\r\n'
    index.write_bytes(prefix.encode('utf8')+index.read_bytes());checked=0
    for phase in D.A.iterdir():
        for p,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/p)==h;checked+=1
    for n in names:assert D.sha(collab/n)==D.sha(O/n)
    D.save('publication_verification.json',dict(passed=True,archive_files_checked=checked,collaboration_files_checked=len(names)))
    dest=D.A/'verification';dest.mkdir();shutil.copy2(O/'publication_verification.json',dest/'publication_verification.json')
    (dest/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(dest/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n')
    print('GRID168_REPORT_COMPLETE',msg)

if __name__=='__main__':main()
