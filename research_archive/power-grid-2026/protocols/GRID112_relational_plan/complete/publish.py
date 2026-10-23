"""Document a reviewable proposal, not a trained model or accepted method."""
from pathlib import Path
import json,shutil
from prepare import R,O,W,REPO,read,save,sha
def main():
    assert not (O/'complete.json').exists();check=read(O/'schema_check.json');assert check['passed']
    plan=read(O/'plan.json');assert plan['status']=='proposal_for_user_review_no_training'
    assert all(sha(R/p)==h for p,h in read(O/'input_manifest.json').items())
    refs=[dict(title='Deep Sets',authors='Manzil Zaheer et al.',venue='NeurIPS 2017',url='https://arxiv.org/abs/1703.06114',read='Abstract: functions on sets and invariance to element permutations.',
        use='Shared per-element encoding and order-invariant aggregation; not a guarantee of generalization in grid control.'),
        dict(title='Conditional Neural Processes',authors='Marta Garnelo et al.',venue='ICML 2018',url='https://proceedings.mlr.press/v80/garnelo18a.html',pdf='https://proceedings.mlr.press/v80/garnelo18a/garnelo18a.pdf',read='Section2.2 equations1-3 and Section2.3.',
        use='Jointly encode observed input-output pairs before aggregation. Our targets/sampling/loss are original search-gain protocol, not CNP likelihood training.'),
        dict(title='Attentive Neural Processes',authors='Hyunjik Kim et al.',venue='ICLR 2019',url='https://arxiv.org/html/1901.05761v2',read='Sections2.1,2.2,3, including scaled dot-product attention and query-specific context.',
        use='Query-dependent access to paired context. Proposed single-head deterministic gain model has no latent-variable path, ELBO, or predictive uncertainty; not a reproduction of full ANP.')]
    save('primary_sources.json',refs)
    notes=['# 阅读依据与本次取舍','',
        '[Deep Sets](https://arxiv.org/abs/1703.06114)研究集合上的置换不变模型。这里借用共享编码与集合聚合，避免已查记录的存放顺序改变输出；不改变原候选查询顺序。','',
        '[Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a.html)先联合编码已观测的输入—输出对，再据此条件化目标输入。这里对应“已查连接方案—已查预测反馈”；没有采用原文的随机上下文训练和概率似然目标。','',
        '[Attentive Neural Processes](https://arxiv.org/html/1901.05761v2)让目标输入关注相关上下文。这里借用单头跨注意力，保留待查动作对已查记录的条件化查询；没有采用潜变量、自注意力堆叠、ELBO或不确定性输出。','',
        '本方案是针对已有补查任务的小型结构假设，不是新集合模型、CNP/ANP原论文复现或已验证贡献。是否能改善补查决策，要由后续严格匹配的JOINT/FACTOR比较回答。']
    (O/'阅读与取舍_中文.md').write_text('\n'.join(notes)+'\n',encoding='utf8')
    doc=['# 下一轮学习方案：保留连接组合与已查反馈的对应关系','',
        '**建议先做一次小型监督学习比较，回答“把动作和反馈成对编码，是否比各自汇总更会判断补查价值”。本阶段只完成方案与输入格式核对，尚未实现或训练神经网络。**','',
        '## 1. 模型具体看什么','',
        '每次补查决策接收三部分信息：',
        '- 已查记录集合：每个同站动作的完整连接分区，以及该动作已经获得的预测反馈。失败、非法或终止记录也保留状态标记，不只挑表现好的动作。',
        '- 待查候选集合：原顺序下下一批最多64个动作的已知连接分区，不提供它们的预测结果。',
        '- 当前搜索状态：沿用原41维汇总、当前最好动作的分区，以及按原费用计算的下一批已知查询成本。',
        '这里的“完整连接分区”保留哪些设备同母线，而不是只给平均距离。原电网每站最多17个设备，因此先用紧凑的固定长度表示：17位规范母线分区、17位相对当前最好的变化、17位在线标记、17位设备存在标记，共68维。第一台在线设备所在母线作为规范母线0，另一母线为1；交换实际母线编号不改变表示。',
        '该表示不丢失当前两母线电网的站内分区信息，神经编码器仍可能压缩信息；不能把输入无损说成模型必然学会所有组合。保留公共设备槽位顺序，尚不具备任意设备重排不变性，也不据此主张跨电网泛化。候选ID、站点ID、日期只做索引，不作为数值输入。',
        '每条反馈用8项表示：相对当前最好动作的奖励差，以及奖励有效性、合法反馈、可接纳、终止、非法、歧义、异常标记。奖励差除以原收益尺度后做asinh单调压缩；原先拟用的硬截断会让287740个反馈项饱和，已在输入检查后修订并保留旧稿。修订没有用到训练成绩或未来目标。','',
        '## 2. 网络怎样使用这些信息','',
        '使用小型条件集合网络，暂以JOINT标记实验组，不另起算法名称。对每条已查记录，先编码完整动作分区与反馈，再把两者组合起来。每个待查动作查询与自己相关的已查记录，得到自己的条件表示；最后把这一批候选的表示按均值和最大值汇总，结合原搜索状态，输出停止/继续两个价值。',
        '注意力加权的是学习到的中间表示，最后通过线性价值头输出，并非平均已查原始奖励，因此不受GRID111中“预测不超过已查最大值”的凸插值上界约束。这只消除了那一种结构限制，不保证正确外推。',
        '借鉴来源是[Deep Sets](https://arxiv.org/abs/1703.06114)的集合处理、[CNP](https://proceedings.mlr.press/v80/garnelo18a.html)的配对上下文，以及[ANP](https://arxiv.org/html/1901.05761v2)的目标条件注意力。只采用小型确定性结构，不宣称复现原论文，也没有不确定性估计保证。','',
        '## 3. 最关键的对照怎么做','',
        '只比较两版新网络：',
        '- JOINT：先把每个动作与自己的反馈成对编码，再汇总。',
        '- FACTOR：先独立编码全部反馈并求均值，再让动作接收同一个反馈集合摘要。',
        '两版都能看到全部已查动作和反馈，都使用非线性反馈编码器。区别在于是否保留逐动作对应关系；不采用挑选某次打乱结果的办法。它们使用相同模块、参数量、初始权重、样本序列、优化器和更新次数。',
        '该对照检验这套网络中“联合编码”相对“先分开汇总”的增量，不是信息论上的唯一完美消融，也不能外推到所有结构模型。原41维直接回归、原gamma0 DQN的全部种子，以及固定一批、停止、完整补查结果保留作参照；原网络参数量和结构不同，不把历史对比称为纯输入消融。','',
        '## 4. 学习目标保持什么','',
        '保持原来的即时净收益目标：停止价值为0；继续价值为“下一批使最好可接纳预测奖励增加的量，减去原单次查询费乘本批查询数，再除以原收益尺度”。',
        f"原查询费为{plan['target']['cost_per_query']:.17g}，原收益尺度为{plan['target']['gain_scale']:.17g}，均不调整。",
        '继续价值高于停止价值才查询下一批，严格相等则停止。所有已买反馈和公共控制规则保持原样，返回最好已观察动作，不增加候选、不改候选排序或补查站。',
        '第一轮仍用监督回归和原Huber损失，只检验输入与联合编码，不同时更换强化学习算法、损失或费用。目标仍是固定教师路径上的即时搜索收益，并非完整周收益，也不声称获得长期规划能力。','',
        '## 5. 规模与预算','',
        '网络隐藏宽度16、注意力键/查询宽度8；按层尺寸计算每版5938个参数，少于原41维MLP的6978。实际模型实现后还须逐张量核验参数数目。',
        '仅用已有32训练/8验证场景，日期组仍为16/4；共3582/856个可达状态。正训练状态来自54次物理调用，前缀并非独立样本，不能按行数夸大数据量。最终8测试场景继续封存。',
        '若后续确认训练，计划为两版×三个固定种子0/1/2，共6次拟合；每次7250次更新、每批32状态/64标量目标，沿用原Adam学习率0.0003、Huber beta1和梯度裁剪10。抽样概率与原GRID107逐项一致。7250沿用最初直接回归预算，不从GRID108多个端点挑验证最优者。',
        '只保留固定终点评价，不挑种子、验证检查点或训练时长。不自动加损失、网络、费用、正例加权等分支。总训练墙钟上限30分钟，超时保留现场和未完成项，不自动延长或重跑完整项。','',
        '## 6. 怎样才算值得继续','',
        '主指标仍是执行完整补查停止序列后的四日期宏平均净效用；同时报告实际买了多少查询、收益保留率、漏掉多少正收益，以及按物理调用均衡的结果。奖励MSE和第552步只作诊断，不作为主结论。',
        '机制信号预先要求：JOINT三种子平均净效用高于FACTOR，三个配对种子都同向，按种子平均后至少3/4日期同向。未达到就完整报告混合或否定结果，不追加变体救结果。三个种子和四个已暴露日期不构成独立确认，也不据此宣称统计显著。',
        '即便通过机制比较，也要超过固定一批及原直接回归、gamma0 DQN两个完整种子家族的平均净效用，才称为值得进一步核查的竞争候选。超过FACTOR但没超过这些对照，只说明当前结构中保留配对有用。',
        '离线通过仍不等于电网整周质量达标或实际提速。后续若安排闭环，需另行封存协议，并把构造结构输入、注意力计算和推理全部计入完整控制时间；不能混用旧计时或把提前结束当提速。本方案不会自动启动闭环或扩大规模。','',
        '## 7. 目前做到了哪里','',
        f"已对全部{check['checks']['states']}个状态完成输入合同检查，核对{check['checks']['lossless_partition_roundtrips']}个动作分区的规范化往返及母线编号不变性；4438次ID重标记和未来字段拒绝检查均通过。最多267条已查同站记录、64个待查候选，未截断记录。",
        '尚未实例化神经网络、执行随机前向、反向或优化器更新。真正训练前，还需实现JOINT/FACTOR，核验5938参数、配对初始权重、集合/填充不变性及采样序列；用合成数据检查梯度时不执行optimizer.step，也不把烟测权重带进研究拟合。',
        '这份方案可供检查，训练尚未开始。','',
        '## 技术规格','',
        '- 共享动作编码器A：68→16→16；反馈编码器F：8→16→16；联合值编码器V：32→16→16。',
        '- 键/查询：动作嵌入16→8，无偏置，单头缩放点积注意力。',
        '- 公共状态编码G：[原41维、最好动作嵌入16、已知查询费1]，58→16→16。',
        '- 候选表示H：[候选嵌入16、注意力上下文16、G16]，48→16→16。',
        '- 输出头：[候选表示均值16、最大值16、G16]，48→16→2。隐藏线性层后均用ReLU，最后两输出保持线性。',
        '- 所有集合聚合都排除填充项；已查记录重排不改变新增集合通道的结果，原41维保持固定。这不等于改变原查询顺序也不影响算法。',
        '- 输入、目标与关联元数据分开保存；未来目标只在forward后进入loss，模型forward不接收未来反馈。']
    (O/'学习模块方案_中文.md').write_text('\n'.join(doc)+'\n',encoding='utf8')
    archive=REPO/'research_archive/power-grid-2026/protocols/GRID112_relational_plan/complete';archive.mkdir(parents=True,exist_ok=False)
    collab=R/'论文协作资料/FRAME_配对学习方案_GRID112';collab.mkdir(parents=True,exist_ok=False)
    save('complete.json',dict(passed=True,scope='Proposal and input schema verification only',training_started=False,
        neural_model_implemented=False,report='学习模块方案_中文.md',archive=str(archive),collaboration=str(collab),ready_to_scale=False))
    for p in O.rglob('*'):
        if p.is_file():d=archive/p.relative_to(O);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
    for p in W.glob('*.py'):shutil.copy2(p,archive/p.name)
    for n in ['学习模块方案_中文.md','阅读与取舍_中文.md','plan.json','schema_check.json','complete.json']:shutil.copy2(O/n,collab/n)
    (archive/'archive_manifest.json').write_text(json.dumps({p.relative_to(archive).as_posix():sha(p) for p in archive.rglob('*') if p.is_file()},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    note='\n\n## 2026-10-23 — GRID112 reviewable relational learning proposal, zero training\n\nUser agreed to define paired action-feedback input/target/control. Read primary Deep Sets, CNP Section2.2/2.3 and ANP Sections2/3; propose deterministic single-head conditional set network, not full NP/ANP replication or uncertainty model. JOINT combines action/feedback embeddings per paid record; parameter-matched FACTOR aggregates feedback first. Each5938 proposed parameters versus old6978. Canonical two-bus full partition68/action,8feedback flags/gap, frozen41+known-cost/anchor shared. Draft clipping would saturate287740 feedback tokens; preserved v0 and revised monotonically to asinh without outcome/model-performance selection. All4438 states and615421 partitions losslessly checked, IDs ignored and future input rejected; no neural forward/optimizer/model fit. Proposed if subsequently authorized: exactly6fits (2arms x3seeds), original7250/Huber/Adam/sampling/fee/targets, fixed endpoints,30min cap. Main whole-search utility with paired seed/date consistency and mandatory BATCH1/all-old-family anchors; no automatic rollout or scaling. Actual neural implementation/parameter/init/mask/gradient checks remain unexecuted pretraining requirements. Proposal, not experiment completion.\n'
    for n in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write(note)
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID112方案与输入合同完成，未训练\n\n1. 将具体JOINT/FACTOR方案交用户检查；当前无训练器/模型实例/优化器更新。\n2. 后续训练阶段获准后，先实现并核验参数/配对初始化/集合与填充不变性/标签隔离/原采样流，再按固定6fits协议执行；不得把合同核验当训练完成。\n3. 不自动追加训练变体、闭环、扩大规模或读取最终8测试场景。\n\n方案：research_archive/power-grid-2026/protocols/GRID112_relational_plan/complete/学习模块方案_中文.md\n',encoding='utf8')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID112配对学习方案](protocols/GRID112_relational_plan/complete/学习模块方案_中文.md)：完整分区—反馈条件化、同参数JOINT/FACTOR比较；输入合同已核验，神经模型与训练尚未执行。\n')
    print('PROPOSAL_PUBLISHED',archive,flush=True)
if __name__=='__main__':main()
