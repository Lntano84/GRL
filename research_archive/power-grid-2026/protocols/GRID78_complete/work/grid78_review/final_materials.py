"""One readable comparison table and Chinese contribution note, retaining prior drafts."""
from pathlib import Path
import csv,gzip,json,statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid78'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    r=read(OUT/'review.json');i=read(OUT/'interpretation.json');assert r['passed_saved_data']
    weeks=read(OUT/'design.json')['weeks'];rows=[]
    sources=[(65,'FULL'),(70,'FRAME_GUARD'),(76,'CONTEXT_GATE'),(78,'META_CONTEXT0'),(78,'META_CONTEXT1')]
    full={s['week']:s for s in read(ROOT/'outputs/grid65/finished.json')['summaries'] if s['method']=='FULL' and s['repeat']==0}
    for stage,method in sources:
        ss=[s for s in read(ROOT/f'outputs/grid{stage}/finished.json')['summaries'] if s['method']==method and s['repeat']==0 and s['week'] in weeks];assert len(ss)==7
        for s in ss:
            f=full[s['week']];costdelta=s['cost']/f['cost']-1 if s['complete'] and f['complete'] else None
            path=ROOT/f'outputs/grid{stage}/runs'/f"r0__{s['week']}__{method}"/'steps.jsonl.gz'
            with gzip.open(path,'rt',encoding='utf-8') as h:latency=[json.loads(line)['decision_s'] for line in h]
            assert len(latency)==s['steps']
            survive=bool(s['complete']) if f['complete'] else s['steps']>=f['steps']
            rows.append(dict(method=method,date=s['week'],stage=stage,steps=s['steps'],complete=s['complete'],quality_pass=bool(survive and (costdelta is None or costdelta<=.005)),cost=s['cost'],cost_delta_full=costdelta,queries=s['n1_forecasts'],controller_s=s['controller_s'],decision_p95_s=float(np.percentile(latency,95)),timing_scope='Existing single repeat, not a fresh paired timing comparison.'))
    with (OUT/'all_methods_per_date.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for stage,method in sources:
        ss=[x for x in rows if x['method']==method]
        summary.append(dict(method=method,quality=sum(x['quality_pass'] for x in ss),complete=sum(x['complete'] for x in ss),queries=sum(x['queries'] for x in ss),max_cost_delta=max(x['cost_delta_full'] for x in ss if x['cost_delta_full'] is not None),controller_macro_s=statistics.mean(x['controller_s'] for x in ss)))
    assert next(s for s in summary if s['method']=='CONTEXT_GATE')['queries']==10171
    for s in r['by_seed']:assert next(x for x in summary if x['method']==s['method'])['queries']==s['queries']
    (OUT/'all_methods_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    p=OUT/'GRID78_report.md';previous=OUT/'GRID78_report_initial_interpretation.md';assert not previous.exists();previous.write_bytes(p.read_bytes())
    lines=['# GRID78：相同信息与核查权限下的会议适配比较','',
        '本轮完成14条闭环回放和保存数据核验。普通运行上下文门控在七个既定日期均满足当前质量条件；相对会议适配的两个种子，候选预测查询总量分别减少36.18%和10.47%。','',
        '## 主要结果','',
        '| 方法 | 质量通过 | 完成整周 | 候选预测查询 | 最大完整周成本增幅 |','|---|---:|---:|---:|---:|']
    names={'FULL':'完整搜索 FULL','FRAME_GUARD':'固定核查 FRAME_GUARD','CONTEXT_GATE':'普通上下文门控','META_CONTEXT0':'会议适配，种子0','META_CONTEXT1':'会议适配，种子1'}
    for s in summary:lines.append(f"| {names[s['method']]} | {s['quality']}/7 | {s['complete']}/7 | {s['queries']} | {100*s['max_cost_delta']:+.4f}% |")
    lines += ['',
        '质量条件为：FULL完成的日期也完成，且完整周成本增幅≤0.5%；FULL提前终止的日期不更早终止。五种方法在这七个日期的终止长度相同，因此查询总量没有混入少跑步骤的优势。两个共享提前终止日期单列，完整周成本只在完成日期上比较。','',
        '门控相对固定核查少查询12.45%。在此前全部13日期中，它满足质量条件13/13、完成11/13；相对FULL，12个有候选搜索且终止长度相同日期的候选查询宏平均减少66.47%。','',
        '## 对照设置','',
        '会议适配保留ICAPS2022式DoubleDQN训练内核，使用相同24维公开电网与候选反馈信息，并增加模式可用位。动作包括返回、继续查询64个候选、直接调用完整FRAME_GUARD核查。两种子共享一个校准费用配置，原有控制器、动作库、排序及公共缓存保持一致。','',
        '训练使用129份既有付费轨迹、校准22份，排除全部13个评价日期家族。六个固定拟合全部保留。没有费用配置使两个种子的校准均值与最大单步遗憾同时达到固定核查参照，因此按事前规则选用最低遗憾的共同配置，并披露这一校准缺口。该基线是会议方法在电网搜索任务中的适配。','',
        '## 现在可以形成的贡献表述','',
        '核心是按状态分配候选核查计算：先获取固定前缀反馈，再利用公开运行状态判断是否继续完整核查，执行过程中复用已付费候选反馈。相同信息和核查权限的对照补强了这一流程在当前电网任务中的实用价值。','',
        '普通上下文森林是辅助决策器。结构38维版本没有显示额外收益，GNN不作为这轮计算优势的核心。论文应突出闭环质量约束下的候选预测节省、核查流程和应用结果。停止与运行时参数控制属于已有研究，成熟随机森林本身不列作新模型创新。','',
        '## 时间与证据范围','',
        '完整控制时间与逐决策P95已整理到统一逐日期表。当前运行来自不同批次的单次重复，时间只作描述；候选查询减少66.47%不等于完整控制时间减少66.47%。新的端到端提速主张需要同批平衡计时。七日期是已暴露开发日期，本轮也不替代新场景确认。','',
        '核验覆盖25,466个保存物理步骤、70次合法候选搜索、70份由观测重建的公开上下文、258次模型决策、动作向量和逐步成本账目。未独立重求解AC。','',
        '交付：all_methods_per_date.csv为五方法×七日期的统一表；per_date.csv为两种子逐日期原始比较；fit_audit.json与review.json保留详细核验。旧报告和全部不利结果均保留。','']
    p.write_text('\n'.join(lines),encoding='utf-8')
    note=OUT/'本轮结论与贡献定位.md'
    note.write_text('''# 本轮结论与贡献定位

当前最有依据的主线是：在保留现有电网控制器与候选动作库的基础上，按运行状态与已查询反馈分配昂贵核查计算。

当前可用版本：控制器提出候选 → 查询128个候选 → 普通上下文门控 → 必要时调用FRAME_GUARD → 执行动作。所有候选反馈只付费一次。

新增实证：13日期质量条件13/13，候选预测查询相对FULL宏平均减少66.47%；旧七日期相对具备相同信息与核查权限的两个会议适配种子少查询36.18%和10.47%，质量通过7/7对6/7。

GNN与结构特征不是新增优势的关键。辅助模型使用成熟随机森林，结构版本结果保留。论文重点应放在核查流程、计算节省与电网闭环效果。

还需区别：相对FULL的候选查询节省、相对会议方法适配的质量/查询关系、同批计时的完整控制加速。这一轮补强前两项；第三项需要单独冻结平衡计时。

算法新颖性需要依靠与既有停止/参数控制研究的具体差异来论证。当前结果支持继续完善这条应用框架，不保证新规划算法的普适优势或会议录用。
''',encoding='utf-8')
    print(json.dumps(dict(passed=True,unified_rows=len(rows),methods=len(summary),source_audit_passed=True,independent_timing_confirmation=False)))
if __name__=='__main__':main()
