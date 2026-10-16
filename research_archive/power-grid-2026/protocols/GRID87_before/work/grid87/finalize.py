"""Report every original date; no historical wall-clock ratios."""
from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid87'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def main():
 R=read(OUT/'review.json');assert R['passed_saved_data'];D=read(OUT/'design.json');C=R['comparisons']
 q=R['macro_fixed_query_ratio_guard'];zero_excess=any(c['equal_horizon_guard'] and c['guard_queries']==0 and c['fixed_queries']>0 for c in C)
 cheap=bool(R['fixed_quality_dates']==13 and q is not None and q<=1.02 and not zero_excess)
 category='FIXED256_ACCEPTED_CHEAP_BASELINE' if cheap else 'FIXED256_QUALITY_FAILURE_GUARD_PASSES' if R['fixed_quality_dates']<13 and R['guard_quality_dates']==13 else 'MIXED_OR_QUERY_CRITERION_NOT_MET'
 result=dict(overall=category,fixed_quality_dates=R['fixed_quality_dates'],guard_quality_dates=R['guard_quality_dates'],fixed_complete_dates=R['fixed_complete_dates'],full_complete_dates=R['full_complete_dates'],same_guard_horizon_dates=R['equal_horizon_guard_dates'],macro_query_ratio_guard=q,query_zero_excess=zero_excess,accepted_cheap_baseline=cheap,no_timing_comparison=True,scope=D['scope'])
 (OUT/'decision.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 def pc(v):return '未双方完成' if v is None else f'{100*v:+.4f}%'
 lines=['# GRID87：固定256在原13日期的质量回归','',f'冻结裁决：**{category}**。','',f"FIXED_256质量通过 **{R['fixed_quality_dates']}/13**，完整周完成 **{R['fixed_complete_dates']}/13**；FULL完整周完成{R['full_complete_dates']}/13，FRAME_GUARD质量通过{R['guard_quality_dates']}/13。",'',
 '本轮只新增13条FIXED_256回放，规则逐字复用GRID86。原2035日期全数保留，未训练、调参、改预算或追加种子。FULL为GRID65/68冻结记录；FRAME_GUARD为GRID70冻结记录。','',
 '| 日期 | 固定256完成/步数 | FULL步数 | 固定256质量 | 固定256成本相对FULL | GUARD成本相对FULL | 固定256查询 | GUARD查询 | FULL查询 |','|---|---|---:|---|---:|---:|---:|---:|---:|']
 for c in C:lines.append(f"|{c['date']}|{c['fixed_complete']}/{c['fixed_steps']}|{c['full_steps']}|{c['fixed_quality']}|{pc(c['fixed_cost_delta_full'])}|{pc(c['guard_cost_delta_full'])}|{c['fixed_queries']}|{c['guard_queries']}|{c['full_queries']}|")
 lines+=['','## 查询与质量判读','',f"同GUARD时域日期{R['equal_horizon_guard_dates']}/13；有非零参照查询的可比日期{R['query_comparable_guard_dates']}个。固定256/GUARD查询比例的日期宏平均："+('不适用' if q is None else f'{q:.5f}')+'。',f"查询合计：固定256 {sum(c['fixed_queries'] for c in C)}，GUARD {sum(c['guard_queries'] for c in C)}，FULL {sum(c['full_queries'] for c in C)}。合计可能包含不同存活时域，不直接用合计比例主张加速。",'',
 '质量门槛沿用：不早于FULL终止；双方完成整周时成本不超过FULL的100.5%。固定256若13/13通过且匹配活跃时域查询宏平均不超过GUARD的102%，接受便宜固定方案，收口当前额外核查优势主张。质量失败时保留具体日期和幅度，并检查首次动作分歧，不改预算补救。','',
 '## 首次动作分歧','', '| 日期 | 首次分歧步 | 相同动作前缀步数 | 双方搜索调用 | 同观测/底层动作 | GUARD查询/固定查询 |','|---|---:|---:|---|---|---|']
 for v in R['first_divergences']:
  lines.append(f"|{v['date']}|{v['first_action_divergence_step']}|{v['shared_action_prefix_steps']}|{v.get('both_have_search_call','—')}|{str(v.get('same_observation','—'))}/{str(v.get('same_base_action','—'))}|{v.get('guard_queries','—')}/{v.get('fixed_queries','—')}|")
 lines+=['','详细选择、首次共同状态中已揭示奖励、双方额外查询ID见first_divergences.json。只比较共同前缀实际揭示的反馈；首次分歧后的累计成本可能受到闭环连锁影响，不能全部归因到某一次查询。','',
 '## 本轮时间与验证范围','',f"新回放完整控制耗时平均{statistics.mean(r['controller_s'] for r in R['rows']):.3f}秒/日期。逐日期P95、P99及预测/优化/搜索/其余互斥分解保存在per_run.csv。只有单次计时，不与历史FULL/GUARD控制时间计算提速，也不估计时间波动。",'',
 f"保存数据核对：{R['checks']['physical_rows']}行物理记录，{R['checks']['causal_calls']}次合法搜索决策，{R['checks']['public_contexts']}次公开上下文重建；固定查询前min(256,M)代表、float32奖励/最低ID破并列、无可接纳候选返回空，均逐次核对。",'',
 '原13日期均已暴露，属于同一公开电网的开发回归。保存向量和逐步成本通过核对，没有独立AC重求解。全部较差结果与历史稿件保留；该实验不建立跨电网泛化或一般质量保证。']
 (OUT/'GRID87_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 (OUT/'RUN_STATE.md').write_text('COMPLETE:13 original-date FIXED_256 runs and saved-data audit; every date retained, no historical timing ratios.\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
