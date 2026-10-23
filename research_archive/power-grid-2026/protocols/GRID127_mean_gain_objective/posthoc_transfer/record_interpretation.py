"""Editorial interpretation correction from audited risk components,no scientific changes."""
from common127 import *

def main():
    T=O/'posthoc_transfer';assert not (T/'风险解释修正.md').exists();risk=read(T/'risk_parts.json')['summary'];verify();assert read(T/'review.json')['passed']
    for arms in risk.values():
        assert arms['POS_KL']['missed_positive_value']>arms['NET_ZERO']['missed_positive_value']
        assert arms['POS_KL']['wasted_query_value']<arms['NET_ZERO']['wasted_query_value']
    note=['# 风险解释修正：训练总指标改善不等于高收益保护改善','',
      '两组训练的原始结果与判读不变。补充拆解后，应把“训练侧总遗憾下降约73%”解释为主要减少无利继续查询，而不能解释为已经更会识别高收益机会。','',
      '| 评价侧 | 原模型漏正收益代价 | KL漏正收益代价 | 原模型无利继续代价 | KL无利继续代价 |',
      '|---|---:|---:|---:|---:|']
    for scope,arms in risk.items():
        a,b=arms['NET_ZERO'],arms['POS_KL'];note.append(f'| {scope} | {a["missed_positive_value"]:.8f} | {b["missed_positive_value"]:.8f} | {a["wasted_query_value"]:.8f} | {b["wasted_query_value"]:.8f} |')
    note+=['','这些是同一原状态权重下的单状态代价，不是整周经济损失。训练侧KL漏掉正收益的代价已经略增，留出侧进一步增大，所以后续问题同时包括保守停止的取舍及其迁移，不能单独归为“训练完全学会但泛化失败”。',
      '保留原净收益回归；没有采用KL或选择单种子。今后若继续设计学习，应明确保护少数重要继续机会的目标及可迁移证据，而不是仅继续降低总训练风险。已做过的MSE与代价加权分类不重复包装为新方案。']
    (T/'风险解释修正.md').write_text('\n'.join(note)+'\n',encoding='utf8');summary=O/'GRID126_127_阶段汇总.md';old=summary.read_bytes();shutil.copy2(summary,T/'阶段汇总_风险拆解前.md')
    with summary.open('a',encoding='utf8') as f:f.write('\n## 追加：风险分项后的解释修正\n\n训练总遗憾下降主要来自减少无利查询；训练侧漏掉正收益的代价已略增，留出侧增幅更大。因此不能将73%的改善解释为高收益保护成功，也不能把问题只归为泛化。详见补充的风险分项表；模型、评价及采用结论均未改变。\n')
    save(T/'interpretation_correction.json',dict(posthoc=True,source_risk_parts_sha256=sha(T/'risk_parts.json'),original_stage_summary_sha256=__import__('hashlib').sha256(old).hexdigest(),updated_stage_summary_sha256=sha(summary),no_model_metric_or_criteria_changed=True,new_fits=0))
    dest=A/'posthoc_transfer'
    for p in [W/'record_interpretation.py',T/'风险解释修正.md',T/'阶段汇总_风险拆解前.md',T/'interpretation_correction.json']:shutil.copy2(p,dest/p.name)
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='archive_manifest.json'})
    complete=A/'complete';shutil.copy2(summary,complete/summary.name);save(complete/'archive_manifest.json',{p.relative_to(complete).as_posix():sha(p) for p in complete.rglob('*') if p.is_file() and p.name!='archive_manifest.json'})
    c=R/'论文协作资料/FRAME_均值收益目标学习_GRID127';shutil.copy2(summary,c/summary.name)
    for p in [T/'风险解释修正.md',T/'interpretation_correction.json']:shutil.copy2(p,c/'posthoc_transfer'/p.name)
    docs('## 2026-10-23 — GRID127 interpretation narrowed after risk decomposition\n\nTrainKLtotalregret-72.75%mostlyavoidsunprofitableCONT;positiveforegonevalueincreasesinfulltrain/foldfitandheld. Fulltrainmiss.00048547->.00090642,waste.00555638->.00073994;validationmiss.00857824->.02294631,waste.00924708->.00236179. Thusnotproofhighvalueprotectionlearned,norsolelyperfectfitthenfailedtransfer. Models/metrics/criteriaunchanged;retainNETbaseline,no candidateadoption.',
      '# Next steps — 聚焦重要机会的误停保护与可迁移信息\n\n1. 汇报GRID126/127共60次训练：已知费用、非负链接、MSE/KL均未稳定超过原回归。\n2. 训练总风险改善主要来自省无利查询，不等于高收益保护成功；下一设计必须同时明确重要误停的代价与跨日期信息依据。\n3. 用既有公开结构/逐调用记录推进证据，不重复换损失、择种子或盲目扩大网络。\n4. 保留原模型；若需扩大独立日期覆盖，先给出固定规模方案，不动最终8测试，不自动物理回放。\n\n汇总：research_archive/power-grid-2026/protocols/GRID127_mean_gain_objective/complete/GRID126_127_阶段汇总.md\n解释修正：research_archive/power-grid-2026/protocols/GRID127_mean_gain_objective/posthoc_transfer/风险解释修正.md\n')
    print('RISK_INTERPRETATION_CORRECTION_RECORDED',flush=True)

if __name__=='__main__':main()
