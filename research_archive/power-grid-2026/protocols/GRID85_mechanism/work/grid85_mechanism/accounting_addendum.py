"""Further split prediction time without changing the completed main analysis."""
from pathlib import Path
import hashlib,json,shutil,statistics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid85_mechanism'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    src=ROOT/'outputs/grid81/finished.json';summaries=json.loads(src.read_text(encoding='utf-8'))['summaries'];rows=[]
    for method in ['FULL','FRAME_GUARD']:
        rs=[r for r in summaries if r['method']==method];assert len(rs)==12
        avg=lambda k:statistics.mean(r[k] for r in rs)
        rows.append(dict(method=method,search_forecast_s=avg('n1_forecast_s'),all_forecast_s=avg('all_forecast_s'),
                         other_forecast_s=statistics.mean(r['all_forecast_s']-r['n1_forecast_s'] for r in rs),controller_s=avg('controller_s')))
    delta=rows[0]['controller_s']-rows[1]['controller_s'];searchdelta=rows[0]['search_forecast_s']-rows[1]['search_forecast_s'];alldelta=rows[0]['all_forecast_s']-rows[1]['all_forecast_s']
    result=dict(source=src.relative_to(ROOT).as_posix(),sha256=sha(src),rows=rows,mean_controller_s_difference=delta,
                mean_search_forecast_s_difference=searchdelta,search_forecast_difference_share=searchdelta/delta,
                net_all_forecast_difference_share=alldelta/delta,
                interpretation='Additive observed accounting, not an intervention holding closed-loop state trajectories fixed. Includes complete same-batch six dates and both timing repeats.',new_runs=0)
    p=OUT/'compute_breakdown_detail.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    text=f'''# 计算来源的进一步拆分

主报告把全部预测耗时单列，这里继续区分搜索内部预测与其余预测。沿用GRID81同批六日期、两次反向计时；按秒数算术平均，不替代逐日期相对节省宏平均。

| 组成 | FULL秒/日期 | FRAME秒/日期 | FULL减FRAME |
|---|---:|---:|---:|
| 搜索内部预测 | {rows[0]['search_forecast_s']:.3f} | {rows[1]['search_forecast_s']:.3f} | {searchdelta:+.3f} |
| 其余预测 | {rows[0]['other_forecast_s']:.3f} | {rows[1]['other_forecast_s']:.3f} | {rows[0]['other_forecast_s']-rows[1]['other_forecast_s']:+.3f} |
| 完整控制 | {rows[0]['controller_s']:.3f} | {rows[1]['controller_s']:.3f} | {delta:+.3f} |

搜索内部预测耗时减少约7.845秒，占完整控制均值差9.362秒的约{searchdelta/delta*100:.1f}%。其余预测稍有增加，所以主报告中全部预测的净减少占比约81.3%，两个数的口径不同。

这份对账更直接地显示计算收益集中于候选搜索预测。闭环动作与后续状态会变化，其他组成的增减仍需按实测解释，不作为严格的逐模块因果归因。没有新增模拟或重计时。
'''
    p2=OUT/'计算来源细分.md';assert not p2.exists();p2.write_text(text,encoding='utf-8')
    added=[p,p2,Path(__file__).resolve()]
    arc=ROOT/'work/GRL-results-integration/research_archive/power-grid-2026/protocols/GRID85_mechanism'
    for f in added:
        target=arc/f.relative_to(ROOT);assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
        if f.parent==OUT:shutil.copy2(f,ROOT/'论文协作资料/FRAME_机制分析_20261016'/f.name)
    (arc/'accounting_addendum_receipt.json').write_text(json.dumps(dict(files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in added]),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
