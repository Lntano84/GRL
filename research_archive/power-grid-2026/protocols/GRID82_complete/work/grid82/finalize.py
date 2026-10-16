"""Separate quality eligibility from matched controller-time comparisons."""
from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid82';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    R=read(OUT/'review.json');D=read(OUT/'design.json');guard_ok=R['guard_quality_dates']==6;decisions=[]
    for c in R['comparisons']:
        meta_ok=c['quality_dates']==6
        category='GUARD_QUALITY_SCREEN_FAILED' if not guard_ok else 'GUARD_PASS_META_QUALITY_FAIL' if not meta_ok else 'SUPPORTED_EQUAL_QUALITY_SPEED' if c['stable_5percent_screen'] else 'BOTH_PASS_TIMING_MIXED_OR_SMALL'
        decisions.append(dict(baseline=c['baseline'],guard_quality_dates=R['guard_quality_dates'],baseline_quality_dates=c['quality_dates'],equal_quality_eligible=bool(guard_ok and meta_ok),same_quality_speed_supported=bool(guard_ok and meta_ok and c['stable_5percent_screen']),category=category,macro_controller_reduction=c['macro_controller_reduction'],repeat_macro_reductions=c['repeat_macro_reductions']))
    result=dict(decisions=decisions,new_fits=0,method='FRAME_GUARD',scope='Final-version comparison on six already exposed generated same-grid scenarios. ICAPS2022-core task adaptation, not original SOTA task reproduction. Calibration mismatch retained; timing repeats/seeds not independent dates.')
    (OUT/'decision.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# GRID82：最终固定核查版与会议方法适配的同批比较','', '36条正式回放，六个GRID81场景，FRAME_GUARD与会议适配两个冻结种子，各两次反向顺序计时。模型、控制器、候选排序、日期与质量条件未改；新增拟合0。', '',
      '| 方法 | 质量通过 | 完成整周 | 搜索查询合计¹ | 控制秒/场景² | 最大完整周成本增幅 |', '|---|---:|---:|---:|---:|---:|']
    for m in D['methods']:
        name=m['method'];gs=[g for g in R['groups'] if g['method']==name];cost=[g['cost_delta_full'] for g in gs if g['cost_delta_full'] is not None]
        lines.append(f"|{name}|{sum(g['quality_pass'] for g in gs)}/6|{sum(g['complete'] for g in gs)}/6|{sum(g['queries'] for g in gs)}|{statistics.mean(g['controller_s'] for g in gs):.3f}|{max(cost)*100:+.4f}%|")
    lines+=['','¹各场景仅计一次，两次计时重复轨迹和查询相同。²先在每场景平均两次，再平均六场景；实际相对节省按逐场景比例宏平均，另列汇总秒数比例。质量参照为冻结GRID81 FULL，原FULL时间不并入本轮同批计时。', '',
      '| 固定核查版相对参照 | 控制时间宏平均减少 | 汇总秒数减少 | 正向场景 | 两次重复分别 | 查询宏平均减少³ | 同质量提速判据 |','|---|---:|---:|---:|---|---:|---|']
    for c,d in zip(R['comparisons'],decisions):
        q=c['macro_query_reduction_equal_horizons'];qs='不适用' if q is None else f'{q*100:+.3f}%'
        lines.append(f"|{c['baseline']}|{c['macro_controller_reduction']*100:+.3f}%|{c['pooled_controller_reduction']*100:+.3f}%|{c['positive_dates']}/6|"+' / '.join(f'{v*100:+.3f}%' for v in c['repeat_macro_reductions'])+f"|{qs}|{d['category']}|")
    lines+=['','³仅在两方法终止长度相同且参照搜索非零时计算；场景数和逐项结果见review.json。不完整轨迹仍全部保留。', '', '## 逐场景完整结果','', '| 场景 | 方法 | 完成/步数 | 完整周成本相对FULL | 质量通过 | 搜索查询 | 控制秒：均值[最小,最大] |','|---|---|---|---:|---|---:|---|']
    for w in D['weeks']:
      for m in D['methods']:
        g=next(g for g in R['groups'] if g['date']==w and g['method']==m['method']);r=next(r for r in R['rows'] if r['date']==w and r['method']==m['method']);delta=g['cost_delta_full'];cs='未双方完成，不比较完整周成本' if delta is None else f'{delta*100:+.4f}%'
        lines.append(f"|{w}|{m['method']}|{g['complete']}/{r['steps']}|{cs}|{g['quality_pass']}|{g['queries']}|{g['controller_s']:.3f}[{g['controller_min_s']:.3f},{g['controller_max_s']:.3f}]|")
    lines+=['','## 公平条件与解释','',
      '三方法共享静态电网、基础动作与连续优化/恢复控制、候选库/去重、STATIC_ID顺序、公开预测接口以及DC/safeQP和公共元数据缓存。两会议模型沿用相同24维公开上下文与模式可用位，支持返回、追加64个全局候选、委托固定核查；冻结种子0/1都报告。固定核查版不运行新增学习门控，仍承担同批公共上下文构建成本。',
      '对照源自Bhatia等ICAPS2022方法的DoubleDQN内核，任务已改为电网搜索预算控制。GRID77共同校准质量匹配为False；这里不重训或按新结果选费用、种子。如果对手未通过质量条件，不把全场景时间差描述为同质量加速。质量合格场景子集的时间可单列描述，但不代替预设全六场景判据。',
      '固定核查版全部轨迹需逐步重现GRID81既有固定核查结果。审计核对保存向量、公开反馈、逐步成本、两次重复和互斥时间分解，没有独立AC重求解。六场景已在GRID81暴露，本轮不算新的独立确认；两个模型种子和两次计时不增加独立场景数。',
      '所有失败和较差结果保留。不进行新增训练、成本配置选择、额外计时或日期筛选。工作1的最终方法、伪代码和贡献证据表位于outputs/grid82_method。']
    (OUT/'GRID82_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
