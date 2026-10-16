"""Readable results for both authorized questions; retain failures and denominators."""
from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid81';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    R=read(OUT/'review.json');D=read(OUT/'design.json');groups=R['groups'];comparisons={c['baseline']:c for c in R['comparisons']}
    fresh=comparisons['FULL'];queries=fresh['macro_query_reduction_equal_horizons']
    quality=R['context_quality_dates']==6
    confirmed=bool(quality and queries is not None and queries>=.5 and fresh['stable_5percent_screen'])
    ablations=[]
    for method in ['RANDOM_GATE0','RANDOM_GATE1']:
        c=comparisons[method];eligible=[x for x in c['dates'] if x['query_reduction'] is not None]
        query_adv=bool(c['macro_query_reduction_equal_horizons'] is not None and c['macro_query_reduction_equal_horizons']>=.05 and sum(x['query_reduction']>0 for x in eligible)>=4)
        passed=quality and (c['quality_dates']<6 or query_adv)
        ablations.append(dict(baseline=method,quality_dates=c['quality_dates'],targeting_screen=passed,query_screen=query_adv))
    gate_support=all(x['targeting_screen'] for x in ablations)
    result=dict(fresh_confirmation_screen=confirmed,context_quality_dates=R['context_quality_dates'],targeting_screen_both_random_seeds=gate_support,random=ablations,scope='Investment screen on six generated same-grid dates; two repeats and random seeds not independent scenarios; no independent AC re-solve.')
    (OUT/'decision.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# GRID81：新场景确认与门控消融','',f"正式回放 {R['runs']} 条，六个预先冻结的2036生成场景；每个方法两次反向计时重复。新增拟合0。",'',f"新场景联合筛查：{confirmed}；随机门控两种子识别增量筛查：{gate_support}。这些是冻结投资判据，不是显著性/等价检验。",'', '| 方法 | 质量通过场景 | 完成场景 | 搜索查询合计（每场景只计一次） | 控制秒/场景 |','|---|---:|---:|---:|---:|']
    for m in D['methods']:
        name=m['method'];g=[x for x in groups if x['method']==name]
        lines.append(f"|{name}|{sum(x['quality_pass'] for x in g)}/6|{sum(x['complete'] for x in g)}/6|{sum(x['queries'] for x in g)}|{statistics.mean(x['controller_s'] for x in g):.3f}|")
    lines+=['','| 门控相对参照 | 控制时间宏平均减少 | P95宏平均减少 | 同时域活跃搜索场景查询宏平均减少 | 两次计时分别 |','|---|---:|---:|---:|---|']
    for c in R['comparisons']:
        q=c['macro_query_reduction_equal_horizons'];s='不适用' if q is None else f'{q*100:+.3f}%'
        lines.append(f"|{c['baseline']}|{c['macro_controller_reduction']*100:+.3f}%|{c['macro_p95_reduction']*100:+.3f}%|{s}|"+' / '.join(f'{v*100:+.3f}%' for v in c['repeat_macro_reductions'])+'|')
    lines+=['','## 逐场景质量与时间','', '|场景|方法|完成/步数|完整周成本相对FULL|查询|控制秒，两次范围|','|---|---|---|---:|---:|---|']
    for w in D['weeks']:
        for m in D['methods']:
            g=next(x for x in groups if x['date']==w and x['method']==m['method']);r=next(x for x in R['rows'] if x['date']==w and x['method']==m['method']);delta=g['cost_delta_full']
            s='不同完成条件，不作成本比' if delta is None else f'{delta*100:+.4f}%'
            lines.append(f"|{w}|{m['method']}|{g['complete']}/{r['steps']}|{s}|{g['queries']}|{g['controller_s']:.3f} [{g['controller_min_s']:.3f}, {g['controller_max_s']:.3f}]|")
    lines+=['','## 解释与边界','',
        '当前门控与始终核查共享冻结模型外围、控制器及缓存；随机门控保留同一个支持范围守卫。跳过概率4/13来自旧校准集的4个跳过与13个范围内状态，测试中的实际查询量没有强制相等。随机门控两种子全部报告，不按输赢选择。',
        '六条时序由官方Chronix2Grid产生，使用相同电网与静态物理配置；这提供了未用于本地模型开发的新随机时序，不是第二电网或真实生产数据。生成器与控制器依赖隔离，日期、生成种子、模型、阈值均在闭环前固定。外部作者模型原始训练覆盖仍未知。',
        '质量判据：不早于同场景FULL终止，双方完成时完整周成本不超过FULL的0.5%。不完整场景保留存活/步数，不能拿短轨迹的总查询优势当同等工作量加速。查询比仅在相同时域且参照搜索非零的场景计算；实际控制时间按全部场景报告，并同时给出质量资格。',
        '保存向量、公开反馈、逐步成本与互斥时间分解重算通过；没有独立重做AC仿真或重拟合模型。所有失败、较差日期、两个随机种子及计时重复均保留。会议适配模型沿用GRID79已有对照，本轮没有新增外部基线。']
    (OUT/'GRID81_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
