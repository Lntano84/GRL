"""Publish the full diagnostic, including weak/negative and historical results."""
from pathlib import Path
import collections,datetime,json,pickle,shutil
import numpy as np
from extract import R,O,W,read,save,sha
from structure import NAMES
REPO=R/'work/GRL-results-integration'
def main():
    assert not (O/'complete.json').exists()
    review=read(O/'review.json');assert review['passed']
    for n in ['input_manifest.json','verification_input_manifest.json']:
        assert all(sha(R/p)==h for p,h in read(O/n).items())
    assert sha(O/'diagnostic_descriptors.npz')==read(O/'audit_recovery_01.json')['descriptor_array_sha256']
    C=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());D=read(R/'outputs/grid102_search_rl/design.json')
    rows=read(O/'rows.json');X=np.load(O/'diagnostic_descriptors.npz')['X'];Y=np.concatenate([c['gains']-D['cost_per_query']*c['queries'] for c in C])
    pressure={}
    for part in ['train','validation']:
        p=np.array([r['partition']==part for r in rows])&(Y>0);m=p&(X[:,9]<1)
        pressure[part]=dict(positive_states=int(p.sum()),positive_calls=len({r['physical_state_group'] for i,r in enumerate(rows) if p[i]}),
            positive_states_no_incident_overload=int(m.sum()),calls_with_such_positive_states=len({r['physical_state_group'] for i,r in enumerate(rows) if m[i]}))
    save('pressure_counterexamples.json',dict(partitions=pressure,threshold='rho=1 is the physical overload definition, not a fitted trigger.',
        interpretation='Observed immediate forecast net gain can be positive while all current live incident lines are below rating. This refutes unconditional skip-if-local-not-overloaded; it does not prove pressure features have no predictive use or establish whole-week gain.'))
    comp=read(O/'comparison.json')['cohorts'];crit={r['context']:r for r in read(O/'critical_structure.json')}
    profiles=read(O/'descriptor_profile.json')
    decision=dict(saved_structure_recoverable=True,all_reachable_states_supported=True,historical_structural_feature_attempt_acknowledged=True,
        stable_benefit_of_appending11_summaries_established=False,all_local_pressure_only_skip_rule_refuted=True,
        all_structural_representations_useless=False,current41_inherently_unlearnable_proven=False,
        direct52_feature_training_recommended=False,learning_inputs_or_weights_changed=False,ready_to_scale=False,
        recommendation='Before another fit, inspect action-to-feedback correspondence and which terminal connections differ, rather than adding another set of unordered distance/pressure averages. No further extraction/fit/rollout automatically scheduled.')
    save('decision.json',decision)
    lines=['# GRID110：候选结构能否还原，能否解释补查收益','',
        '**现有记录足够还原候选的站内连接意图，无需为此重新采集；但把连接差异和本地压力压成几项汇总数，目前没有显示稳定的区分改善，不建议直接拼成52维再训练。**','',
        '## 先把历史接上','',
        'GRID98已经在56个已暴露分支案例中补过连接分区距离、同站反馈和线路压力，GRID99用15维/35维比较了逻辑回归、浅层森林和TabPFN，未见实质改善。那时预测的是“取消一次补查后物理轨迹是否相同”，本轮关注“下一批预测收益是否覆盖查询费”，两种标签不能混用。',
        '因此，“缺少结构信息”不是全新的发现，也不是已经验证的改进。本轮复用可解释、可检查的结构描述方式，核对新数据是否具备这些信息，以及它们能否帮助理解最近的漏查。所有结果都属于已暴露开发数据上的诊断。','',
        '## 1. 数据能可靠还原到什么程度','',
        '对32个训练、8个验证场景涉及的278次物理搜索、4438个可达补查状态全部核对。使用原先保存的观测/动作空间反序列化向量，不建立电网环境，不调用reset、step、simulate，也不运行控制器或优化器。',
        '- 观测4460维、动作1605维，556次保存向量往返一致；静态118站、186线、540个拓扑位置及端点归属与历史模板一致。',
        '- 公共909个候选逐项对齐原动作库；278次候选资格、断线端点屏蔽与别名合并均重现。',
        '- 244545个组合动作哈希与原记录一致；实际用于诊断的站内连接均可从公共命令与当前拓扑还原，没有填猜测值。',
        '- 4438个状态逐项检查未来反馈隔离、母线编号交换不变，以及拒绝把未来收益注入描述函数。',
        '这里还原的是**动作的站内连接意图**，不是执行后的电网潮流。站外某些重连端点的母线不能仅靠当前向量确定，它们没有被猜测或用于本地分区描述，也不能因此宣称已经还原完整的未来电网状态。','',
        '## 2. 固定检查了哪些描述','',
        '检查11项：下一批最多64个候选与当前最佳动作的分区差异（最小/均值/最大）、与已查同站动作的最近分区差异（均值/最大）、新分区占比、相对当前站内分区的平均差异，以及全网当前最大负载率、过载量均值、本站接线最大负载率、本站占全网过载线的比例。',
        '分区差异按设备对是否同母线计算，交换母线1/2编号结果不变。它们只描述命令与当前状态，不使用待查候选的仿真结果。原41维、标签、候选顺序、模型权重全部未改；这11项仅保存在独立诊断文件中。',
        '还查出两个冗余/弱信息问题：current_max_rho在全部4438状态上与原rho_limit完全相同；新分区占比在全部856个验证状态上恒为1。可见“增加11列”并不意味着增加11种有效信息，更不能拿列数说明表示变强。','',
        '## 3. 结构描述是否把正负情况分开了','',
        '先只用训练日期算诊断标准化，再分别计算原41维、结构11维、合并52维的距离。每个邻域保留最近5次不同训练物理搜索，每次只取最近的一个前缀；没有拟合邻居分类器、调权重、定阈值或挑选特征。',
        f"- 全部41个验证正收益状态中，最近5次训练搜索没有正例的数量：原41维为{comp['positive_validation']['original41']['zero_positive_among5']}，结构11维为{comp['positive_validation']['structure11']['zero_positive_among5']}，合并52维为{comp['positive_validation']['combined52']['zero_positive_among5']}。",
        f"- 13个新增漏查状态中，同一数量分别为{comp['newly_missed']['original41']['zero_positive_among5']}、{comp['newly_missed']['structure11']['zero_positive_among5']}、{comp['newly_missed']['combined52']['zero_positive_among5']}。",
        '- 第552步单独使用结构描述时，最近5次训练搜索中出现2个正例；合入原41维后仍为0。RF路径552也相同。',
        '- 第108步原本是负净收益，合并52维反而把其邻域正例数从1增到2；这提醒我们不能只展示552附近看起来更好的变化。',
        '这些变化有好有坏：合并后全部正收益的邻域覆盖略变好，但新增漏查的空白数没有减少；只看结构时，新漏查附近略改善，全部正收益附近却变差。距离和邻居标签不是预测性能，因此本轮既没有建立“加上就有效”，也没有证明所有结构表示都无效。','',
        '## 4. 最明确的反例：本站不拥堵，也可能值得补查','',
        f"验证的41个正净收益状态中，{pressure['validation']['positive_states_no_incident_overload']}个当时本站所有在线接入线路均未过载，来自{pressure['validation']['calls_with_such_positive_states']}次物理搜索。训练数据也有{pressure['train']['positive_states_no_incident_overload']}/{pressure['train']['positive_states']}个类似状态。这些前缀高度相关，不能当作相同数量的独立场景。",
        f"第552步全网最大负载率为{crit['MARCH552']['descriptor_values']['current_max_rho']:.6f}，本站接线最大负载率仅{crit['MARCH552']['descriptor_values']['station_incident_max_rho']:.6f}，但下一批归一化预测净收益为{crit['MARCH552']['normalized_target']:.6f}。",
        '十月19日第二份时序第1964步，本站接线最大负载率约0.470，下一批仍有正净收益；这个调用还包含三个种子共同新增漏查的前缀。相反，第108步本站接线最大负载率约0.509，下一批没有奖励改善。',
        '**所以不能把“本站接线没有过载”当成取消补查的充分条件，也不能用“本地压力更低”直接推断更不值得查。** 本地拓扑动作可能改变全网潮流分布；上述数据证明的是这类补查正收益确实存在，尚未定位具体通过哪些线路产生作用。',
        '这里的正收益指原一步预测奖励增益减去原固定查询费，不是整周经济收益，更不是运行安全保证。','',
        '## 5. 我的判断','',
        '目前可以把两件事分开了：第一，记录里有可用的结构和物理状态，工程上不用重新采集；第二，现成的分区距离均值、局部压力等汇总量，没有给出足以支持新训练方案的稳定证据。',
        '先不直接训练52维，也不扩大网络。下一步更值得核对的是：**哪些连接改变、哪些已查动作表现好，以及两者是否对应。** 把这些关系都压成“平均距离”和“奖励方差”，可能丢掉了真正需要的组合信息；这仍是待检查的解释，不是本轮已证明的原因。',
        '需要区分“结构是否可恢复”“某种结构描述是否有信息”“模型是否能利用信息”三个问题。本轮完成第一个、对第二个给出弱且混合的诊断，没有执行第三个，更不据此宣称学习改进或闭环提速。最终8个测试场景保持未生成。','',
        '## 核验与工程记录','',
        f"独立标量循环复算{review['independent_arithmetic_contexts']}个上下文、{review['independent_values']}个数值，包括全部正收益动作变化状态、三个关键实际上下文及20个日期的确定性覆盖；对全部856个验证状态的三种距离、三个邻居数量共{review['all_validation_neighbour_checks']}项检查通过。原41维结果逐项重现GRID109。",
        '首次独立核验误把“站内描述需要已知连接”的断言扩到了全网站外重连端点，因此停止。原错误、代码和清单保存在interrupted_attempts/audit_scope_01。恢复只把核验范围限定回原定义的站内位置，重新核验后通过；原4438×11描述数组哈希完全不变，没有改数据或标签。',
        '完整输入/源码哈希、公开命令核对、诊断数组、所有状态和邻居结果以及不利结果均保留。无新模型拟合、推理、物理观测、预测或回放。','',
        '## 文件','',
        '- plan.json、input_manifest.json：提取前的固定描述、范围与来源哈希。',
        '- extraction_receipt.json、provenance.json、decoded_schema.json：全部公共结构还原与向量核验。',
        '- diagnostic_descriptors.npz、rows.json：仅用于诊断的11项数值与关联信息；ID不是模型标量输入。',
        '- descriptor_profile.json：全部描述的常量、范围和与原通道的精确重复。',
        '- validation_comparison.json、flip_structure_details.json：全部856个验证状态及正收益变化状态，不挑有利邻域。',
        '- critical_structure.json：108、552及RF路径552的实际上下文。',
        '- pressure_counterexamples.json、review.json、decision.json：反例、核验与当前判断。']
    (O/'GRID110_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    archive=REPO/'research_archive/power-grid-2026/protocols/GRID110_structure_audit/complete';archive.mkdir(parents=True,exist_ok=False)
    collab=R/'论文协作资料/FRAME_结构核查_GRID110';collab.mkdir(parents=True,exist_ok=False)
    save('complete.json',dict(passed=True,report='GRID110_report.md',review='review.json',archive=str(archive),collaboration=str(collab),
        new_fits=0,new_inference=0,new_physical_steps=0,learner_inputs_changed=False,ready_to_scale=False))
    for p in O.rglob('*'):
        if p.is_file():dest=archive/p.relative_to(O);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in W.glob('*.py'):shutil.copy2(p,archive/p.name)
    for n in ['GRID110_report.md','decision.json','review.json','complete.json','pressure_counterexamples.json']:shutil.copy2(O/n,collab/n)
    (archive/'archive_manifest.json').write_text(json.dumps({p.relative_to(archive).as_posix():sha(p) for p in archive.rglob('*') if p.is_file()},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    note='\n\n## 2026-10-23 — GRID110 structural-information audit\n\nAll278 saved train/validation calls and4438 reachable states decoded with serialized spaces, no environment construction.556 vector roundtrips,244545 combined-command hashes, eligibility/mask/alias reproduction and4438 suffix/bus-label/forbidden-key checks pass.11 fixed next-batch partition-distance/current-pressure diagnostic descriptors reuse GRID98 ideas; GRID99 historical negative trajectory-equivalence result acknowledged, target differs. No existing learner input/weight changed. current_max_rho duplicates oldrho_limit exactly; novelty fraction constant1 across validation. Nearest5 distinct-train-call zero-positive counts on all41 validation positives: original41/structure11/combined52=18/22/16; on13 new misses=8/6/8. Mixed descriptive changes, no predictive benefit established or metric/feature selection.37/41 positive validation states (15 physical calls) have no current incident overload;552stationrho0.890648 and October1964rho0.470043 still positive, refuting unconditional local-not-overloaded skip. Independent37-context407-value and7704 neighbour checks pass. First independent audit overreached to station-external unknown reconnection buses; preserved audit_scope_01 and narrow scope-only repair, original descriptor array unchanged. Inputs are command intentions, not future solved grid state. No fit/inference/forecast/replay/test materialization; do not directly launch52-input training. Next candidate diagnosis: retain action-connection/observed-feedback correspondence rather than adding more unpaired averages.\n'
    for n in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write(note)
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID110结构核查完成\n\n1. 汇报：已有记录足够还原结构，无需为此重采；现成11项结构/压力摘要没有稳定区分改善，不直接训练52维。\n2. 若继续诊断，先核对动作连接变化与已查反馈的对应关系如何保留；不能用本站未过载直接取消补查。尚未执行进一步提取或新训练。\n3. 不加训练量/模型/物理回放，不扩大规模；最终8测试场景封存。\n\n报告：research_archive/power-grid-2026/protocols/GRID110_structure_audit/complete/GRID110_report.md\n',encoding='utf8')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:
        f.write('\n- [GRID110结构核查](protocols/GRID110_structure_audit/complete/GRID110_report.md)：全部4438状态结构可还原，11项摘要区分变化混合，本地未过载不是取消补查条件；无新训练/回放。\n')
    print('COMPLETE',archive,flush=True)
if __name__=='__main__':main()
