"""Separate quality/horizon eligibility from raw query totals in the final reading."""
from pathlib import Path
import json,math
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid78'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    r=read(OUT/'review.json');assert r['passed_saved_data'];full={s['week']:s for s in read(ROOT/'outputs/grid65/finished.json')['summaries'] if s['method']=='FULL' and s['repeat']==0}
    g76=read(ROOT/'outputs/grid76/review.json');rows=[]
    for s in r['by_seed']:
        ds=[x for x in r['dates'] if x['method']==s['method']];same=all(x['steps']==full[x['date']]['steps'] and x['complete']==full[x['date']]['complete'] for x in ds)
        rows.append(dict(**s,all_equal_horizon=same,all_quality_pass=s['quality_dates']==7,
            interpretation='Same-horizon raw query comparison; whole-week quality does not match on all dates.' if same and s['quality_dates']<7 else 'Same-horizon and all-date quality qualified.' if same else 'Raw totals have unequal horizons; not a query-efficiency comparison.'))
    result=dict(passed_saved_data=True,seeds=rows,context_quality_dates=7,context_complete_dates=5,context_queries=10171,calibration_quality_matched=r['calibration_quality_matched'],
        claim='Ordinary operating-context verification development. Structural increment was not demonstrated. No matched fresh timing, independent new dates or full original conference task reproduction.')
    (OUT/'interpretation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    old=OUT/'GRID78_report.md';copy=OUT/'GRID78_saveddata_initial_report.md'
    assert old.exists() and not copy.exists();copy.write_bytes(old.read_bytes())
    lines=['# GRID78 相同上下文与核查权限：完整结果','',
        '本轮补齐基于ICAPS2022训练内核的电网适配对照：与上下文门控共享24维公开信息，并可直接调用完整FRAME_GUARD核查；额外具备继续查询64个候选的动作。两个种子共享同一校准配置。','',
        '| 方法 | 质量通过日期 | 完成日期 | 候选查询总量 |','|---|---:|---:|---:|',
        '| 普通上下文门控（GRID76） | 7/7 | 5/7 | 10171 |',
        '| FRAME_GUARD（GRID70） | 7/7 | 5/7 | 11618 |']
    for s in rows:lines.append(f"| {s['method']} | {s['quality_dates']}/7 | {s['complete_dates']}/7 | {s['queries']} |")
    lines+=['','质量通过：FULL完成的日期也完成且完整周成本增幅≤0.5%；FULL提前终止的日期不更早终止。未完成日期不能使用截短累计成本解释完整周质量。','',
        '## 每个种子与日期','',
        '| 方法 | 日期 | 步数 | 相对FULL完整周成本 | 查询 | 质量通过 |','|---|---|---:|---:|---:|---|']
    for x in r['dates']:
        delta='未完成' if x['cost_delta_full'] is None else f"{100*x['cost_delta_full']:+.4f}%"
        lines.append(f"| {x['method']} | {x['date']} | {x['steps']} | {delta} | {x['queries']} | {x['quality_pass']} |")
    lines+=['','## 可以说明什么','',
        '普通上下文门控在此前13日期质量13/13、完成11/13；相对FULL，12个有搜索且同终止长度日期的候选预测查询宏平均减少66.4680%。它比统一FRAME_GUARD在旧七日期少查询12.455%。这是本次新增可用的工程结果。','']
    for s in rows:
        lines.append(f"- {s['method']}：{s['interpretation']} 原始查询总量相对下，门控变化为{-100*s['context_query_reduction_vs_meta']:+.3f}%；最大完整周成本增幅{100*s['max_complete_cost_delta']:+.4f}%。")
    lines+=['','六个会议适配模型均保留；没有配置使两个种子的校准均值和最大单步遗憾同时不劣于FRAME_GUARD，故按事前规则选择最低遗憾的共同费用配置，并披露校准质量未匹配。不能把适配不足包装成已经战胜原论文的普适SOTA。','',
        '## 对创新性定位的影响','',
        '有效的对象是条件核查流程：初始候选反馈与公开电网运行状态决定是否调用完整核查，缓存确保已查询候选不重复付费。固定全局上下文森林能够辅助该流程；结构38维版本没有额外优势，不能强行声称结构/GNN优越。','',
        '论文主张可集中为电网闭环中减少昂贵候选预测并保留质量的框架与实证；停止、继续、运行时模式控制本身已有会议研究，成熟随机森林本身也不是新模型。当前比较未建立新规划算法或结构化学习贡献。','',
        '这是七个已暴露日期的开发比较，单次计时只报告描述值。新的完整控制加速需同批平衡计时，不把66.468%的候选查询节省写成完整控制时间节省。保存向量、反馈、逐步成本、公共特征及模型决策均复算；未独立重求解AC。','']
    old.write_text('\n'.join(lines),encoding='utf-8')
    note=ROOT/'outputs/grid78/本轮结论与贡献定位.md'
    note.write_text('''# 本轮结论与贡献定位

工作目标是减少计算并保住电网闭环质量。首先修复了新日期上的质量退化，再用普通运行上下文减少不必要核查，并补齐相同信息和核查权限的会议方法适配。所有模型、负面结果、日期和工程修正均已保留。

可落地的当前版本是：原控制器和候选库 → 128个候选反馈 → 普通上下文门控 → 必要时委托FRAME_GUARD核查 → 执行动作。GNN不是这轮结果的关键，结构特征的额外价值未建立。

明确区别三层证据：相对FULL少查询；相对同信息会议适配的质量/查询关系；完整控制时间是否更快。前两层见GRID76和GRID78；第三层本轮没有新配对计时证明。

论文能够围绕电网任务中的条件核查框架组织贡献。但现有证据不能保证算法新颖性或录用，也不能把采用成熟停止控制/随机森林本身作为原创。
''',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='seeds'}))
if __name__=='__main__':main()
