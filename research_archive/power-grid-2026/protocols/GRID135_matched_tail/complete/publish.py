"""Render numerical result and preserve immutable artifacts, no fitting."""
from common135 import *
import shutil
def main():
    verify();assert read(O/'review.json')['passed'];assert not (O/'complete.json').exists()
    D=read(O/'design.json');F=read(O/'fits.json');S=read(O/'validation_results.json');fam=read(O/'families.json');dec=read(O/'decision.json');diag=read(O/'diagnostics.json')
    lines=['# GRID135：完整候选范围与动作反馈配对的匹配学习','',
        '四组各三个种子的12份模型均按固定终点训练并核验。所有模型预测同一个问题：完整同站补查是否还能改善已观察最佳奖励；这次是监督分类，没有强化学习或新电网回放。','',
        '## 对照保持了什么','',
        'FIRST_JOINT：只看前64个待查候选，保留动作—反馈配对；FIRST_FACTOR：只看前64个，先汇总已购反馈；FULL_JOINT：全部待查候选，保留配对；FULL_FACTOR：全部待查候选，先汇总反馈。',
        '四组同5938个命名参数、同初始化/抽样流、445条拟合和193条校准前缀，原12/4日期不变。原37维摘要、当前最佳动作、全部已购同站反馈和完整剩余数量/代理查询费相同。FIRST也知道完整数量，因此范围对照只增加公开接线计划。',
        '全局输入仅原37维信息，历史41槽中末4维归零；规范化仅用445拟合数据。原68维动作与8维已购反馈编码不变，最多237个候选在训练/校准来源中全部保留。',
        '每模型901步、每批32，Adam学习率0.0003，固定末步；由ceil(7250×445/3582)=901折算旧实验的每样本期望重复次数，不按验证成绩延长。不等同旧任务或旧实验总计算量；本轮四组更新/样本预算匹配，候选多的组计算可能更多。',
        f"12份拟合总耗时{F['wall_s']:.2f}秒。全部权重与经验校准门槛封存后，才为原4个已暴露验证日期构建完整公开候选输入并评分。",'',
        '## 四日期结果','', '| 方法 | 补查增量保留 | 同站查询节省 | 动作差异 | 局部净效用 |','|---|---:|---:|---:|---:|']
    for name in ['FULL_LOCAL','BATCH1','GLOBAL','GROUP','EXP_PUBLIC37','EXP_COUNT1','EXP_RF37']:
        v=S[name];m=v['macro'];lines.append(f"| {name} | {100*m['gain_capture']:.5f}% | {100*m['query_saving']:.3f}% | {v['total_action_disagreements']} | {m['utility']:.8f} |")
    for arm in ARMS:
        v=fam[arm];m=v['macro'];lines.append(f"| {arm} 三种子均值 | {100*m['gain_capture']:.5f}% | {100*m['query_saving']:.3f}% | {v['total_disagreements_by_seed']} | {m['utility']:.8f} |")
    lines+=['',
        '比例先按日期计算再宏平均，家族再平均三个种子；不同前缀共享物理调用，不能当独立场景。这里的增量保留是补查预测奖励增量，查询节省仅同站阶段，净效用含继承代理费，均不是整周经济成本或实际时延。','',
        '## 主要机制对照','']
    for name in ['range_primary','association_primary','range_factor','association_first']:
        c=dec['contrasts'][name];lines.append(f"- {name} ({c['a']}−{c['b']})：查询节省百分点差{[round(100*v,4) for v in c['query_saving_deltas_by_seed']]}；两家族大组无动作差异={c['both_families_expensive_zero_disagreement']}；预定机制信号={c['mechanism_signal']}。")
        lines.append(f"  各日期种子均值百分点差：{ {d:round(100*v,4) for d,v in c['perdate_seed_mean_deltas'].items()} }")
    lines+=[f"- 配对效应的全尾减首批交互差(百分点)：{[round(100*v,4) for v in dec['factorial_interaction_by_seed']]}。只描述本架构/本数据，未作显著性推断。",'',
        '机制方向需三个配对种子均节省更多查询、至少3/4日期同向，并且对照两家族的大补查都无动作差异；不能只看总体省查询。','',
        '## 所有种子与拟合表现','']
    for rec in F['records']:
        name=rec['name'];v=S[name];pr=dec['practical_by_model'][name];a=diag[name]
        lines.append(f"- {name}：增量保留{100*v['macro']['gain_capture']:.5f}%，查询省{100*v['macro']['query_saving']:.3f}%，大组动作差异{v['expensive_disagreements']}，大组跳过{v['expensive_skipped']}；逐日期优于GROUP且大组无差异={pr['conditional_screen']}。")
        lines.append(f"  AUC拟合/校准/验证={a['fit']['auc']:.4f}/{a['calibration']['auc']:.4f}/{a['validation']['auc']:.4f}，前/后100步CE={rec['first100_loss']:.5f}/{rec['last100_loss']:.5f}；配对错位后的门控变化{a['binding_shift']['calibrated_decision_changes']}，其中正例{a['binding_shift']['positive_case_decision_changes']}。")
        for d,p in v['per_date'].items():lines.append(f"  {d}：增量保留{100*p['gain_capture']:.5f}%，查询省{100*p['query_saving']:.3f}%，动作差异{p['action_disagreements']}。")
    lines+=['', '## 判读与边界','',
        '原小补查分支保持不变，旧7个动作差异仍如实报告。主筛查针对本轮可改变的大补查：新增差异为零且逐日期增量/查询共同优于GROUP；旧全量零差异另列为历史筛查，不假装它能由不变小分支实现。这个条件筛查不取代整周质量标准，也不授权自动扩大。',
        '每模型门槛来自4个校准日期的全部有用见证，校准零误停是构造性质，不是统计风险保证。全部种子和不利日期保留，不挑最好模型、重训、换门槛或增加数据救结果。',
        '旧GRID98/99/110结构统计和113/116/119关系模块的失败仍有效。本轮改变了任务目标和训练人口，不能把相对旧模型的所有变化解释为结构作用；只有本轮四组内部受控差异用于检验范围与配对。',
        '当前模型不采用、不扩大；最终8测试仍未生成，没有新物理预测/AC重求解/论文图表。']
    (O/'GRID135_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save('complete.json',dict(passed=True,new_fits=12,adopted=False,ready_to_scale=False,report='GRID135_report.md'))
    collab=R/'论文协作资料/FRAME_完整尾部匹配学习_GRID135';collab.mkdir(parents=True,exist_ok=False)
    for name in ['GRID135_report.md','review.json','decision.json','families.json','diagnostics.json','design.json']:
        shutil.copy2(O/name,collab/name);assert sha(O/name)==sha(collab/name)
    save('collaboration_receipt.json',dict(directory=collab.relative_to(R).as_posix(),hashes={p.name:sha(p) for p in collab.iterdir()}))
    save('state.json',dict(stage='complete',fits=12))
    dest=A/'complete';dest.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file() and p.suffix not in ['.pkl']:shutil.copy2(p,dest/p.name)
    shutil.copytree(O/'models',dest/'models')
    with (dest/'archive_manifest.json').open('w',encoding='utf8') as f:json.dump({p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()},f,indent=2);f.write('\n')
    for phase in ['frozen','complete']:
        root=A/phase
        for n,h in read(root/'archive_manifest.json').items():assert sha(root/n)==h
    family_text='; '.join(f"{arm}:增量{100*v['macro']['gain_capture']:.5f}%/省查询{100*v['macro']['query_saving']:.3f}%/大组差异{v['expensive_disagreements_by_seed']}" for arm,v in fam.items())
    entry='\n\n## 2026-10-23 — GRID135完整尾部/配对的十二份匹配分类完成\n\n'+family_text+'。四组同5938参数/901步/445拟合193校准/原12+4日期/三种子配对初始化和抽样，首批组同样知道完整Q和费用；全部权重/门槛先封存。目标完整补查有无增量，非RL。主机制范围信号='+str(dec['contrasts']['range_primary']['mechanism_signal'])+'，全尾配对信号='+str(dec['contrasts']['association_primary']['mechanism_signal'])+'。全部种子/日期/错位诊断保留，独立NumPy前向、901优化器状态、7080保存决策及不变便宜分支核验通过。不采用或扩大，零新数据/物理/最终测试。原7小组差异另列，不用已知无法由本轮改变的总量零差异替代大组条件筛查。报告：research_archive/power-grid-2026/protocols/GRID135_matched_tail/complete/GRID135_report.md\n'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+entry,encoding='utf8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — 解释匹配学习的结果\n\n1. GRID135十二份匹配分类已完成；先按保存的逐日期/种子/校准覆盖和错位诊断解释，不选择最好种子或追加训练。\n2. 只有本轮四组内部差异用于判断完整候选范围/配对；相对旧网络同时改变目标和人口，不作单因素归因。\n3. 保留GROUP与原FRAME参照。局部开发条件不当整周质量或提速，最终8测试仍封存；不自动扩大、做物理回放或修改论文。\n\n'+family_text+'\n',encoding='utf8')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8')+'\n- [GRID135完整尾部与配对匹配分类](protocols/GRID135_matched_tail/complete/GRID135_report.md)：四组同容量/目标/抽样/预算，全部三种子和日期保留，原7便宜分支差异与新大组差异分别报告。\n',encoding='utf8')
    verify();save('state.json',dict(stage='complete',fits=12));print('GRID135_COMPLETE '+family_text,flush=True)
if __name__=='__main__':main()
