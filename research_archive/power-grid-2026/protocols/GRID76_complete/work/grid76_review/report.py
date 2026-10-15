"""Finalize current-context development result without claiming new timing confirmation."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid76'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    r=read(OUT/'review.json');a=read(OUT/'all_context_review.json')
    assert r['passed_saved_data'] and a['passed'] and r['runs']==13
    oldweeks=read(ROOT/'outputs/grid65/design.json')['weeks']
    rows=[x for x in r['dates'] if x['date'] in oldweeks]
    assert len(rows)==7
    total=sum(x['queries'] for x in rows)
    comparison=dict(context_queries=total,guard_queries=11618,old_meta_seed0_queries=10305,old_meta_seed1_queries=11201,
        reduction_guard=1-total/11618,reduction_old_meta0=1-total/10305,reduction_old_meta1=1-total/11201,
        caveat='Old META adapter lacks same operating context and guard macro. Not a fair same-information victory or matched timing comparison.')
    (OUT/'old_seven_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# GRID76 普通上下文门控：13 日期闭环开发检查','',
        '使用固定的普通24维上下文森林；结构38维版本未通过校准，不进入回放。原FRAME、FRAME_GUARD及所有较差结果保留。','',
        '| 日期 | 步数 | 相对FULL完整周成本 | 查询 |','|---|---:|---:|---:|']
    for x in r['dates']:
        delta='未完成，不能比较完整周成本' if x['cost_delta_full'] is None else f"{100*x['cost_delta_full']:+.4f}%"
        lines.append(f"| {x['date']} | {x['steps']} | {delta} | {x['queries']} |")
    eligible=[x['query_reduction_full'] for x in r['dates'] if x['query_reduction_full'] is not None]
    lines += ['',f"质量条件13/13；完成11/13，与FULL相同；完成日期最大成本增幅+0.3720%。12个有搜索的同终止长度日期查询宏平均减少{100*sum(eligible)/len(eligible):.4f}%。",'',
        f"旧七日期查询{total}，FRAME_GUARD为11618，旧会议适配对照两个种子为10305/11201；查询分别减少{100*comparison['reduction_old_meta0']:.4f}%/{100*comparison['reduction_old_meta1']:.4f}%。旧会议适配缺少相同上下文和完整核查模式，不能以此直接宣布战胜公平会议对照。",'',
        f"已核验保存反馈、成本账目及合法逐步决策，并从观测重建{a['public_contexts_reconstructed']}份公共上下文，复算{a['gate_decisions_recomputed']}次门控、{a['early_returns']}次提前返回；不声称独立AC重求解。",'',
        '这是13个已暴露日期上的开发回归。每日期单次新计时，仅报告绝对时间；没有新配对计时，不宣称实际控制提速。结构特征增量仍未成立。下一步仅补齐会议适配的相同信息与核查权限。','']
    (OUT/'GRID76_report.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('COMPLETE:13/13 physical runs; saved-feedback and public-context audits passed. Quality13/13, complete11/13. Not independent confirmation or matched timing.\n',encoding='utf-8')
    print(json.dumps(comparison))
if __name__=='__main__':main()
