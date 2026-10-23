"""Close read-only audit, specify label-use contract, archive all outcomes."""
from run import *
def main():
    verify();assert read(O/'review.json')['passed'] and read(O/'support_detail.json')['passed'] and read(O/'historical_bridge_review.json')['passed'];assert not (O/'publication_receipt.json').exists()
    h=read(O/'historical_bridge_summary.json');s=read(O/'support_detail.json');m=read(O/'missed_opportunity_summary.json')
    summary='''# GRID136阶段结论

这轮没有重新训练。当前最应优先解决的是学习任务与质量目标的对应，而不是继续换网络。

## 1. 三种量没有统一

- 作者候选奖励为2−最大负载率，补查增量表示这个预测量能改善多少。
- 旧强化学习还扣除启发式查询费，优化计算价值；价格不是真实秒数或货币。
- 论文最终看整周完成情况、运行成本及实际计算。前两种量不能直接当成这一质量标准。

旧56条单次取消的真实回放已提供反证：17条即时G>0里，5条原标签为SKIP、11条MIXED、1条KEEP。即时降低最大负载率，不等于整周保留补查更好。原标签、时域、试点边界和全部不利结果保持不变。

## 2. 当前误停不全是同一种问题

7条新大补查误停来自同一次搜索。按保护任何正增量的任务，7条都错；按旧查询费任务，4条在全部保存批次上都不值得继续，另3条查第一批值得、继续查完反而亏。

因此必须区分“有收益”“值得付费”和“值得查到哪”。一次跳过/查完的门控不能表达中间停止；但旧逐批学习在自己的扣费指标上也曾输给固定一批，不能只恢复原RL就当解决。

## 3. 两种直觉解释被排除了

大补查拟合445条和校准193条都在48号站；误停也在该站，37项摘要没有单项越界，不能解释成没见过这个站。

大补查正例中，≤4ULP只有拟合5/155、校准1/46；微小正例不是多数，不能说整批数据被数值噪声污染。少数极端正例约束校准覆盖是事实，但不能据此删除它们、放宽门槛或认定真实运行没损失。

待查动作也不是完全不变：237个代表编号中236个有2–3种已记录的站内接线计划。动作结构不是没有变化；本轮只表明补齐它并未形成稳定增益。它仍缺少运行状态与效果的完整联系，不能据此宣称加全网观测一定有效。

## 我建议的下一步

最终目标继续以论文原整周质量为准。先把已有56条真实标签的使用条件和证据缺口定清；2378条教师前缀用于即时收益辅助研究，不能直接替代整周标签。

已另存《后续学习标签使用规范》：不改旧标签、不选新容忍度，明确单次取消、逐批停止和多次在线门控需要不同的监督证据。先完成这个任务对应，再确定是否需要新的业务分支采集与训练；当前不扩大、不使用最终测试、不修改论文。
'''
    (O/'阶段结论.md').write_text(summary,encoding='utf8')
    contract='''# 后续学习标签使用规范（研究建议，未改现有数据）

## 主任务建议

主任务以论文原质量要求为准：在指定场景、共同评价终点和原控制器下保持完成情况与完整周运行成本质量，再比较真实控制计算。最大负载率改进和历史代理查询费可作局部辅助量，不能替代主评价。

本规范只规定如何使用已有证据，没有生成新标签、改变0.5%阈值、拟合模型或宣布策略通过。

## 两层数据保持分开

第一层：GRID100/102的2378训练前缀，来自FULL教师物理路径，能给出已选站内各批次的预测最大负载率/奖励改进与查询数。它们可以支持即时或多批局部搜索价值学习，不能独自给出替代动作后的整周经济结果。

第二层：GRID95/96的56条单次取消记录，能给出在相同干预前缀后、随后继续原FRAME时的存活、共同终点成本、查询与风险结果。它们覆盖少量已暴露旧场景，原标签44SKIP/11MIXED/1KEEP；其中主要非试点没有新KEEP。它们不是大量独立有效训练样本，也不能与2037前缀按行直接混合。

## 保留原始多维事实

每条业务记录保留：干预前状态与动作/反馈、干预时刻、后续基准策略、外部事件对照、双方是否达到预定终点、存活步数、共同终点成本差、真实候选/全部预测查询、过载及非法/异常等信息。

成本只在双方达到原共同终点时比较。提前失败、外部事件不同、基准缺失的记录单列；不把它们强填为某种经济收益。MIXED保留其多维分歧，不能一律视为SKIP或KEEP。

若要按论文原FULL参照+0.5%成本标准派生一个学习目标，应先确认该状态对应的FULL基准和相同时域资料齐备，并在新的数据版本中写明规则。不能把相对FRAME的成本差直接当成相对FULL的质量差，也不能按当前验证错例反推一个能过的容忍度。

## 动作与监督必须对应

- 研究“取消这一次整个补查”时，旧56条是相应的单次干预证据；局部G只作旁证。
- 研究“查下一批后停”时，必须区分这批本身和后续批次。现有保存序列能给局部预测收益，旧56条SKIP/整段补查的业务标签不能复制到每个64候选批次。
- 研究在线连续多次取消时，单次取消的收益不能相加；状态分布及之后搜索会变化，需要另外验证。

## 新训练前的准入材料

需要一份明确的任务/动作/标签/最终评价映射，以及按日期与物理调用隔离的方案。说明哪些是可直接核验的监督，哪些只是辅助目标或未知；固定校准/费用/质量标准和全部参照，继承旧失败与反例。

真实业务标签若覆盖不足，应先提出具体采集方案和预算再决定是否扩大，不能用更大的网络弥补缺少的反事实。当前没有授权自动生成最终8测试或据本规范启动新的物理回放。
'''
    (O/'后续学习标签使用规范.md').write_text(contract,encoding='utf8')
    receipt=dict(passed=True,main_review=sha(O/'review.json'),support_review=sha(O/'support_detail.json'),historical_bridge_review=sha(O/'historical_bridge_review.json'),
        existing_numeric_labels_models_cutoffs_unchanged=True,new_fits=0,new_model_forwards=0,new_physical_replays=0,final_tests_absent=True,
        known_checker_error='Historical bridge first expected passed;legacy GRID96 uses passed_saved_data. Failure before metrics preserved under interrupted_checks;corrected schema only.')
    save('publication_receipt.json',receipt)
    collab=R/'论文协作资料/FRAME_学习目标与标签核查_GRID136';collab.mkdir(parents=True,exist_ok=False)
    for n in ['阶段结论.md','后续学习标签使用规范.md','GRID136_report.md','动作结构覆盖与目标取舍.md','即时收益与历史物理标签对照.md','review.json','support_detail.json','historical_bridge_review.json','publication_receipt.json']:
        shutil.copy2(O/n,collab/n);assert sha(O/n)==sha(collab/n)
    save('collaboration_receipt.json',dict(directory=collab.relative_to(R).as_posix(),hashes={p.name:sha(p) for p in collab.iterdir()}))
    dest=A/'publication';dest.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*[O/n for n in ['阶段结论.md','后续学习标签使用规范.md','publication_receipt.json','collaboration_receipt.json']]]:shutil.copy2(p,dest/p.name)
    shutil.copytree(O/'interrupted_checks',dest/'interrupted_checks')
    manifest={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()}
    (dest/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    for phase in ['frozen','complete','support_detail','historical_bridge','publication']:
        root=A/phase
        for n,h in read(root/'archive_manifest.json').items():assert sha(root/n)==h,(phase,n)
    entry='''

## 2026-10-23 — GRID136目标、收益尺度与公开支持只读核查完成

原2378训练前缀逐项核对，零拟合/模型前向/新阈值/新仿真。大组正例155拟合/46校准中≤4ULP仅5/1，并非多数标签被噪声污染；只描述精度，不删除小增量。正例按旧固定费的任意保存前缀均不回本99/155、18/46，另15/3仅中途停止能回本；完整收益符号与扣费价值不是同一标签。已知7大组误停全来自1调用，其中4任意批次都不回本，3第一批有利、查完转亏；保留135原失败判定，不把不同目标混为一谈。全部大组445/193都站48，误停37摘要无单项越界，排除新站完全无支持的解释但不保证联合覆盖。96257待查动作行中237代表编号有236个出现2–3接线变体，不支持完全静态的简单解释。源码核实MaxRhoReward=2-maxrho，非周成本；候选记录无替代分支整周成本。另对旧96全部56真实单次取消标签复核：17个G>0为5SKIP/11MIXED/1KEEP，说明即时正增益不等于业务保留收益；旧95短期省成本后提前失败反例保留。没有新物理/新标签/合并旧新数据；原目标/模型/门槛不变。保存标签使用规范建议，区分即时辅助、单次干预及逐批/多次门控证据。归档与协作哈希通过；旧96审阅字段passed_saved_data的检查器接口错误及原日志保留，未更动科学结果。阶段结论：research_archive/power-grid-2026/protocols/GRID136_objective_support/publication/阶段结论.md
'''
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+entry,encoding='utf8')
    (REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps — 按业务质量定义后续学习

1. GRID136已确认当前几种目标不同：最大负载率增益、固定查询费净收益、整周完成/成本质量不能互换。保留135全部三种子、旧127同目标输固定一批的结果，不重训救结果。
2. 标签规范建议已保存：2378教师前缀作局部收益辅助证据；56既有单次物理干预标签作有限业务证据，17即时正G仍含5SKIP/11MIXED/1KEEP。不要直接混合/重命名为可靠整周监督。
3. 下一步先把主任务、动作范围与现有标签可用条件映射具体；若派生FULL+0.5%质量目标，须核对对应基准和共同终点，不按验证例子选容忍度。原56标签不能复制给每批或相加当在线多次收益。
4. 大组全在48站，已知误停摘要无单项越界；不是新站陌生，也无已证明的全状态不可学习结论。动作计划有2–3种变体，完整计划不等于全网状态。
5. 本轮不扩大数据/物理回放/改论文。需要新业务标签采集时先提出明确方案和预算；最终8测试封存，旧模型/标签/价格/校准门槛均不变。
''',encoding='utf8')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8')+'\n- [GRID136学习目标与标签核查](protocols/GRID136_objective_support/publication/阶段结论.md)：只读奖励精度、旧代理费下分批机会、同站支持及56真实物理标签；区分局部增益与整周业务目标，保存后续标签使用规范建议，无新拟合/仿真。\n',encoding='utf8')
    verify();print('GRID136_PUBLISHED zero fits/forwards/physical,new labels0',flush=True)
if __name__=='__main__':main()
