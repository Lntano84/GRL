"""Readable final timing/accounting materials; no checkpoint or threshold selection."""
from pathlib import Path
import datetime as dt,json,statistics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid79'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    r=read(OUT/'review.json');assert r['passed_saved_data'];D=read(OUT/'design.json');groups=r['groups'];rows=r['rows'];summary=[]
    for m in D['methods']:
        own=[x for x in groups if x['method']==m['method']]
        v=dict(method=m['method'],quality_dates=sum(x['quality_pass'] for x in own),complete_dates=sum(x['complete'] for x in own),queries=sum(x['queries'] for x in own),max_complete_cost_delta=max(x['cost_delta_full'] for x in own if x['cost_delta_full'] is not None))
        for k in ['controller_s','controller_cpu_s','loop_wall_s','decision_p95_s','optimizer_s','forecast_s','n1_nonforecast_s','remainder_s']:v[k]=statistics.mean(x[k] for x in own)
        summary.append(v)
    (OUT/'method_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    names={'FULL':'完整搜索','CONTEXT_GATE':'固定核查＋普通上下文门控','META_CONTEXT0':'会议适配，种子0','META_CONTEXT1':'会议适配，种子1'}
    lines=['# GRID79：同批反向配对的完整控制计时','',
        '四个固定方法、七个既有日期，每方法每日期两次新进程回放，共56条。第二次反转日期顺序和各日期的方法顺序，模型、控制器、候选排序和质量条件保持一致。没有新增训练。','',
        '## 主要结果','',
        '| 方法 | 质量通过 | 完成整周 | 候选查询总量¹ | 控制时间²（秒/日期） | P95决策延迟³（毫秒） |','|---|---:|---:|---:|---:|---:|']
    for s in summary:lines.append(f"| {names[s['method']]} | {s['quality_dates']}/7 | {s['complete_dates']}/7 | {s['queries']} | {s['controller_s']:.3f} | {1000*s['decision_p95_s']:.3f} |")
    lines+=['','¹ 查询总量按七个日期计一次，两次重复得到相同查询及轨迹，不把重复当作新样本。² 先平均每日期两次完整控制总时间，再平均七日期。³ 各次逐决策P95先在日期内平均，再宏平均日期。','',
        '质量条件：FULL完成的日期也完成且完整周成本增幅≤0.5%；FULL提前终止的日期不更早终止。完成整周与质量通过分开报告。','',
        '## 门控相对各对照的实测变化','',
        '| 对照 | 控制时间宏平均减少 | 汇总秒数减少 | 正向日期 | 重复0 / 重复1宏平均减少 | P95宏平均减少 | 预设稳定5%筛查 |','|---|---:|---:|---:|---:|---:|---|']
    for c in r['comparisons']:lines.append(f"| {names[c['baseline']]} | {100*c['macro_controller_reduction']:+.3f}% | {100*c['pooled_controller_reduction']:+.3f}% | {c['positive_dates']}/7 | {100*c['repeat_macro_reductions'][0]:+.3f}% / {100*c['repeat_macro_reductions'][1]:+.3f}% | {100*c['macro_p95_reduction']:+.3f}% | {c['stable_5percent_screen']} |")
    lines+=['','筛查在回放前固定：控制宏平均减少至少5%，至少4/7日期正向，两次重复各自宏平均均为正；这是投资筛查，不是显著性或等价检验。','',
        '## 逐日期控制时间和重复波动','',
        '| 日期 | 方法 | 两次控制时间均值（秒） | 最小—最大（秒） | 完整周成本增幅 | 候选查询 |','|---|---|---:|---:|---:|---:|']
    for w in D['weeks']:
        for s in [x for x in groups if x['date']==w]:
            delta='共享提前终止，未比较整周成本' if s['cost_delta_full'] is None else f"{100*s['cost_delta_full']:+.4f}%"
            lines.append(f"| {w} | {names[s['method']]} | {s['controller_s']:.3f} | {s['controller_min_s']:.3f}—{s['controller_max_s']:.3f} | {delta} | {s['queries']} |")
    if (OUT/'recovery_01.json').exists():
        recovery=read(OUT/'recovery_01.json');intervals=[read(OUT/f'interval_{j:02d}.json') for j in range(56)]
        active=sum((dt.datetime.fromisoformat(x['ended_utc'])-dt.datetime.fromisoformat(x['started_utc'])).total_seconds() for x in intervals)
        gap=(dt.datetime.fromisoformat(intervals[51]['started_utc'])-dt.datetime.fromisoformat(intervals[50]['ended_utc'])).total_seconds()
        lines+=['','## 中断与恢复','',f"前51条完整结果保留，下一条最后保存的进度为512步，随后发现进程已停止，没有最终结果或Python错误栈。原父/子进程均已确认停止，冻结432项哈希一致；未完成目录及日志归档后，只从固定索引51—55续跑。正式56条完整回放之外，另有1次未完成尝试，未混入计时均值。已完成的51条没有重跑。",'',f"完整56条工作区间合计{active:.1f}秒；索引50与51条完成区间之间有{gap:.1f}秒间隔，包含未完成尝试和停顿。第二轮在该处发生中断，因此不称连续无中断测量，也不删掉跨中断的日期。该边界与全部重复波动一并报告。",'']
        (OUT/'timing_recovery_summary.json').write_text(json.dumps(dict(completed_runs=56,incomplete_attempts=1,completed_preserved=51,complete_worker_intervals_s=active,interruption_gap_before_index51_s=gap,selection_unchanged=True),indent=2),encoding='utf-8')
    lines+=['','## 不重叠的控制计算分解','',
        '| 方法 | 优化器（秒） | 全部预测（秒） | 候选搜索非预测部分（秒） | 其余控制（秒） | 合计（秒） |','|---|---:|---:|---:|---:|---:|']
    for s in summary:lines.append(f"| {names[s['method']]} | {s['optimizer_s']:.3f} | {s['forecast_s']:.3f} | {s['n1_nonforecast_s']:.3f} | {s['remainder_s']:.3f} | {s['controller_s']:.3f} |")
    lines+=['','优化器为控制期间显式DC/safeQP/unsafeQP调用；全部预测为控制期间环境预测调用；候选搜索非预测时间为完整候选搜索计时扣掉其内部预测；余项由完整控制时间减去前三项。每条回放均检查分解和等于控制总时间。公共缓存对所有方法相同，不归为GNN或门控独有贡献。','',
        '完整控制时间包含状态特征构建、首次模型读取/推理、优化和预测。环境真实步进、初始化、记录与落盘另列于per_run.csv；loop_wall_s含控制、步进与记录开销。候选查询节省率与控制时间节省率分别报告。','',
        '## 核验与证据边界','',
        f"保存数据核验覆盖{r['checks']['physical_rows']}步、{r['checks']['causal_calls']}次合法搜索、{r['checks']['public_contexts']}份公共特征重建。每方法每日期的动作/观测哈希、成本和查询量与原冻结运行逐步一致；本轮仅改变计时安排。没有独立AC重求解。",'',
        '这是同机、同批、已暴露日期的反向配对重复。两个时间重复不是独立日期，尚不构成跨机器或新场景确认。会议适配沿用ICAPS2022式训练内核，使用相同公开上下文和完整核查动作；此前共同校准未完全达到固定核查质量参照，该限制保持披露。','',
        '全部更慢日期、未过筛查的比较及原先结构特征未获增益的结果保留。本轮不因计时结果再改模型、补重复或换日期。','']
    (OUT/'GRID79_report.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'final_materials.json').write_text(json.dumps(dict(passed=True,methods=4,per_run_rows=len(rows),per_date_rows=len(groups),source_review_passed=True),indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,rows=len(rows),all_timing_screens_passed=r['all_timing_screens_passed'])))
if __name__=='__main__':main()
