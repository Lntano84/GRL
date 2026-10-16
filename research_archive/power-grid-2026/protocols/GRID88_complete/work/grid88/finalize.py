"""Interpret feedback-vs-fixed mechanism comparison with quality/horizon checks."""
from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid88'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    R=read(OUT/'review.json');D=read(OUT/'design.json');assert R['passed_saved_data'];decisions=[]
    for c in R['comparisons']:
        equal=sum(p['equal_horizon'] for p in c['dates']);quality=R['guard_quality_dates']==13 and c['quality_dates']==13
        q=c['macro_query_reduction_equal_horizons'];query_adv=q is not None and q>=.05
        fixed_time_extra=statistics.mean(p['baseline_s']/p['context_s']-1 for p in c['dates'])
        query_ratios=[];query_infinite=False
        for p in c['dates']:
            if not p['equal_horizon']:continue
            a=next(g for g in R['groups'] if g['method']=='FRAME_GUARD' and g['date']==p['date'])
            b=next(g for g in R['groups'] if g['method']==c['baseline'] and g['date']==p['date'])
            if not a['queries'] and not b['queries']:continue
            if not a['queries']:query_infinite=True
            else:query_ratios.append(b['queries']/a['queries'])
        fixed_query_ratio=statistics.mean(query_ratios) if query_ratios and not query_infinite else None
        speed=bool(quality and equal==13 and c['stable_5percent_screen'] and query_adv)
        competitive=bool(quality and equal==13 and fixed_time_extra<=.02 and fixed_query_ratio is not None and fixed_query_ratio<=1.02)
        category='GUARD_QUALITY_FAILED' if R['guard_quality_dates']!=13 else 'FIXED_QUALITY_FAILED' if c['quality_dates']!=13 else 'DIFFERENT_SURVIVAL_HORIZONS' if equal!=13 else 'FEEDBACK_COMPUTE_ADVANTAGE' if speed else 'FIXED_BASELINE_COMPETITIVE' if competitive else 'BOTH_PASS_MIXED_OR_BUDGET_TRADEOFF'
        decisions.append(dict(baseline=c['baseline'],guard_quality_dates=R['guard_quality_dates'],fixed_quality_dates=c['quality_dates'],same_horizon_dates=equal,
           supported_feedback_computational_advantage=speed,fixed_baseline_competitive=competitive,category=category,
           macro_controller_reduction=c['macro_controller_reduction'],macro_query_reduction_equal_horizons=q,
           fixed_relative_control_extra=fixed_time_extra,fixed_relative_query_ratio=fixed_query_ratio))
    overall='GUARD_QUALITY_FAILED' if R['guard_quality_dates']!=13 else 'FIXED_RULE_ALREADY_COMPETITIVE' if any(d['fixed_baseline_competitive'] for d in decisions) else 'FEEDBACK_ADVANTAGE_OVER_FIXED512' if all(d['supported_feedback_computational_advantage'] for d in decisions) else 'QUALITY_DIFFERENCE_OR_INCOMPLETE_COMPUTE_EVIDENCE'
    result=dict(overall=overall,decisions=decisions,new_fits=0,new_dates=0,scope=D['scope'])
    (OUT/'decision.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# GRID88：反馈核查与固定查询预算的闭环比较','',
      '原13个已暴露日期，FRAME_GUARD、FIXED_512各两次反向计时，共52条串行回放；没有新增训练、调参或选择预算。','',
      '| 方法 | 质量通过 | 完成整周 | 搜索查询合计¹ | 完整控制秒/日期² | 最大可比完整周成本增幅 |','|---|---:|---:|---:|---:|---:|']
    for m in D['methods']:
        gs=[g for g in R['groups'] if g['method']==m['method']];cost=[g['cost_delta_full'] for g in gs if g['cost_delta_full'] is not None]
        cs='不适用' if not cost else f'{max(cost)*100:+.4f}%'
        lines.append(f"|{m['method']}|{sum(g['quality_pass'] for g in gs)}/13|{sum(g['complete'] for g in gs)}/13|{sum(g['queries'] for g in gs)}|{statistics.mean(g['controller_s'] for g in gs):.3f}|{cs}|")
    lines+=['','¹每日期仅统计一次查询，两次重复轨迹相同。²每日期先平均两次，再平均十三日期；相对变化按日期比例宏平均另列。','',
       '| FRAME相对固定预算 | 控制时间宏平均减少 | 匹配时域查询宏平均减少 | 正向日期 | 两次时间重复 | 同时域日期 | 判读 |','|---|---:|---:|---:|---|---:|---|']
    for c,d in zip(R['comparisons'],decisions):
        q=c['macro_query_reduction_equal_horizons'];qs='不适用' if q is None else f'{q*100:+.3f}%'
        lines.append(f"|{c['baseline']}|{c['macro_controller_reduction']*100:+.3f}%|{qs}|{c['positive_dates']}/13|"+' / '.join(f'{v*100:+.3f}%' for v in c['repeat_macro_reductions'])+f"|{d['same_horizon_dates']}/13|{d['category']}|")
    lines+=['',f'冻结总体判读：**{overall}**。','',
       '完整十三日期的同质量提速主张同时要求两方法均通过质量、十三日期存活/终止长度一致、控制时间宏平均减少至少5%、至少9/13日期同向、两次计时均同向，以及匹配活跃时域查询减少至少5%。质量失败或存活时域不同时，时间只作描述，不把提前失败解释为提速。',
       '固定预算竞争力的预设判据：双方质量与时域均通过，固定版的日期宏平均控制额外开销不超过2%，匹配活跃时域查询比例不超过FRAME的102%。本轮仅固定512；固定256既有全部结果保留，不新增预算。阈值用于投资筛查，不构成统计等价检验。','',
       '## 策略定义与公平条件','',
       '固定版实际查询STATIC_ID顺序前min(512,M)个代表，不因奖励或安全阈值提前停止，也不进行同站补查；按相同严格可接纳规则、float32奖励和ID破并列交付。没有可接纳候选时返回空给共同底层控制器，不追加隐含查询。',
       'FRAME沿用GRID70最终规则，所有日期同一套参数。两臂共用基础控制器、候选库/别名、公开预测接口和公共DC/safeQP/元数据缓存；共有上下文构造和首载成本计入控制耗时。',
       '质量/完整周成本参照冻结GRID65/68 FULL；本轮没有FULL新计时，不能由其旧时间计算本轮同批提速。FRAME需逐步重现GRID70已有轨迹。','',
       '## 逐日期成本、查询、时间与延迟','',
       '| 日期 | 方法 | 完成/步数 | 完整周成本相对FULL | 质量通过 | 查询 | 控制秒均值[最小,最大] | P95秒 |','|---|---|---|---:|---|---:|---:|---:|']
    for w in D['weeks']:
      for m in D['methods']:
        g=next(g for g in R['groups'] if g['date']==w and g['method']==m['method']);r=next(r for r in R['rows'] if r['date']==w and r['method']==m['method']);delta=g['cost_delta_full'];cs='未双方完成' if delta is None else f'{delta*100:+.4f}%'
        lines.append(f"|{w}|{m['method']}|{g['complete']}/{r['steps']}|{cs}|{g['quality_pass']}|{g['queries']}|{g['controller_s']:.3f}[{g['controller_min_s']:.3f},{g['controller_max_s']:.3f}]|{g['decision_p95_s']:.4f}|")
    lines+=['','## 查询使用与计算组成','', '| 方法 | 预测秒/日期 | 搜索非预测秒/日期 | 优化秒/日期 | 其余控制秒/日期 |','|---|---:|---:|---:|---:|']
    for m in D['methods']:
        gs=[g for g in R['groups'] if g['method']==m['method']]
        lines.append('|'+m['method']+'|'+ '|'.join(f"{statistics.mean(g[k] for g in gs):.3f}" for k in ['forecast_s','n1_nonforecast_s','optimizer_s','remainder_s'])+'|')
    lines+=['','完整逐步控制延迟、P95/P99、均值及离散范围见per_run.csv与per_date.csv。查询和组成的改变可能伴随闭环轨迹改变，不将所有组成差异归为单模块严格因果贡献。','',
       '## 研究解释','',
       '本轮直接检验GRID85留下的机制问题：反馈核查是否比固定查询数量带来额外闭环价值。一步FULL选择一致率不替代闭环质量。固定256在GRID87质量通过9/13且查询较少；本轮检验统一提高到512能否保持质量并达到FRAME的计算效率。',
       '保存数据审计核对动作/观测向量、合法反馈、固定预算与最优已观测选择、逐步成本、重复轨迹、共有上下文和互斥计时分解，没有独立AC重求解。十三日期为同一电网的生成时序且已暴露，计时重复不算独立场景。所有失败、较差结果和原稿保留。']
    (OUT/'GRID88_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('COMPLETE:52 fixed-budget mechanism runs and saved-data audit finished; all dates and both timing repeats retained.\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
