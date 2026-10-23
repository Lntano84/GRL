"""Calibrated interpretation and archived publication; no new experiment."""
from common111 import R,O,W,read,save,sha
import shutil,json
REPO=R/'work/GRL-results-integration'
def main():
    assert not (O/'complete.json').exists();review=read(O/'review.json');assert review['passed']
    for p,h in read(O/'input_manifest.json').items():assert sha(R/p)==h,p
    s=read(O/'summary.json');v=s['validation'];t=s['train'];m=v['date_call_balanced_macro'];profile=read(O/'combination_profile.json')
    improve=1-m['nearest_mse']/m['mean_mse'];nullimprove=1-m['nearest_mse']/m['exact_unpaired_mse']
    decision=dict(action_feedback_pairing_contains_local_reward_similarity_signal=True,
        signal_equals_profitable_continue_decision=False,convex_reward_interpolation_can_predict_positive_improvement=False,
        existing_DQN_or_supervised_network_subject_to_convex_bound=False,
        independent_pairwise_marginal_effects_supported=False,complex_interactions_causally_identified=False,
        shared_relational_gain_representation_worth_designing=True,new_model_training_authorized_or_run=False,
        final_test_ungenerated=True,ready_to_scale=False,
        next='Specify an input that retains candidate connection patterns jointly with their paid feedback and predicts improvement beyond the incumbent; assess complementarity/uncertainty, rather than copying observed rewards or adding isolated marginal contrasts. No new fit automatically scheduled.')
    save('decision.json',decision)
    lines=['# GRID111：动作连接与已查反馈的对应关系','',
        '**对应关系有信号，但“能估计相似动作的奖励”与“能判断未查动作能否超过当前最好”是两件事。** 本轮进一步表明，直接抄近邻奖励、或把单个连接的平均好坏相加，都不足以解释关键补查。','',
        '## 本轮怎么查','',
        '保留每个已查同站动作的连接分区与预测反馈，不再把它们各自压成均值、方差。沿用全部3582个训练、856个验证可达状态、原候选顺序、下一批最多64个、原查询费和原收益标签。',
        '固定比较三种算术估计：最近连接分区对应的已查奖励（精确并列取均值）、全部已查合法奖励均值，以及保持奖励集合不变但打乱动作—奖励对应关系时的精确期望误差。最后一项用全排列期望公式计算，没有抽随机种子、调距离参数或选择最好的近邻数量。',
        '未来教师反馈只进入独立评价函数，公开输入另存；没有拟合模型、调用现有学习权重、部署新规则或新增物理预测。以下均为已暴露开发数据的诊断。','',
        '## 1. 对应关系不是完全没有用','',
        f"验证结果按候选→前缀状态→物理搜索调用→日期逐层平均，最近分区估计的奖励MSE为{m['nearest_mse']:.8f}，直接均值为{m['mean_mse']:.8f}，减少{improve:.2%}；打乱对应关系的精确期望MSE为{m['exact_unpaired_mse']:.8f}，相对减少{nullimprove:.2%}。",
        f"四个验证日期相对直接均值均同向；按物理搜索看，{v['calls_nearest_better_than_mean']}/{v['legal_error_calls']}次优于直接均值，{v['calls_nearest_better_than_exact_unpaired']}/{v['legal_error_calls']}次优于打乱对应关系的期望。其余调用的反向结果保存在summary.json。",
        '训练组相对直接均值的奖励MSE减少7.36%，但3月16日、12月7日两个训练日期反向；不能写成所有日期均有效。',
        '这支持保留“哪个连接方案得到了哪个反馈”：只保留奖励分布会丢掉局部相似性信息。不过，误差是在未结束、合法、有限正负载率的下一批候选上计算；非法/终止结果另计，没有被当成普通奖励拟合。这不是补查门控精度、整周成本或速度改善，更不是新的学习方法成绩。','',
        '## 2. 更关键的限制：平均已查奖励，预测不出超过已查最好的动作','',
        '设当前最好已查合法奖励为r_best。如果用已查奖励的非负权重加权平均估计未查奖励，且权重和为1，则估计值一定不超过r_best。扣除正查询费后，就不能凭这种估计给出正的新增搜索净收益。',
        '这个界在全部4438个状态上逐项核对成立；训练有211个、验证有41个真实的正净收益批次，但这种奖励插值本身无法为它们估计出突破当前最好的正增益。不能因为普通奖励误差降低，就认为已经学会何时继续。',
        '**该限制只针对这里的已查奖励凸插值，不针对此前的DQN或直接收益回归。** 原网络学习跨状态的增益映射，输出没有被限制在已查奖励范围内；本轮没有证明它们理论上不能学会，也没有重训它们。','',
        '## 3. 第552步：单个关系见过，新组合却没有见过','',
        '第552步已有9个合法同站反馈，当前最好为候选141。下一批真正带来改进的候选355，与它结构最接近的已查动作恰好就是141。奖励插值因此给355估计了与141相同的奖励1.05217695，而记录中的实际一步奖励增益为0.00709510，扣查询费后仍为正。',
        '355相对141改变了6个设备对的同母线关系。每个关系的0和1，在已查反馈中都出现过；因此不是这6种单独关系从未观察到。但它们一起构成的连接分区没有被查询过。',
        '如果分别统计“该设备对连在一起/分开时的平均奖励差”，再按355的变化方向取平均，得到-0.203179，方向反而与实际改善相反。RF路径552也呈现相同情况。',
        '这不是严格证明某一项物理交互效应：不同动作同时改变了多个关系，分组均值受其他连接影响，不能当成单独连接的因果效应。它明确说明，在这个案例中不能把单个关系的平均好坏当作可独立相加的贡献。','',
        '## 4. 这类组合问题并不只出现在552','',
        f"41个验证正收益状态里，最好新增动作的分区均未在当时已查动作中出现；其中{profile['partitions']['validation']['positive_without_unknown_changed_relations']}个状态的全部改变关系，都已经分别观察过0和1。训练组对应为145/211。",
        f"但按上述固定的单关系奖励对比方向，只有{v['positive_states_best_next_direction_positive']}/41个验证正收益状态的最好新增动作得到正方向；训练组为{t['positive_states_best_next_direction_positive']}/211。单关系覆盖较全，不等于其组合收益能被正确推出。",
        '“分区没见过”本身也不是触发条件：上一轮验证批次的新分区占比恒为1，负收益候选同样可能是新组合。不能只因新颖就查，也不能只因每个关系都见过就停。',
        '这些状态来自相关前缀和有限物理调用，不能把41当作41个独立场景。当前候选集合内设备在线/断开模式均相同，连接签名没有混淆由候选造成的在线模式变化；它仍只是站内连接意图，不是未来潮流解。','',
        '## 我的判断','',
        '现在可以更具体地定位学习问题：**利用已查动作的连接—反馈对应关系，判断未查的新连接组合是否还能超过当前最佳，以及这种潜在收益是否值得查询成本。**',
        '输入应保留“动作怎样连接”和“这个动作的已查反馈”这一对信息，并让待查候选相对当前最佳动作参与比较。目标应继续围绕新增收益或值得继续的程度，而不能只做绝对奖励的相似性平均。',
        '这比再添加几列平均距离更有根据，也给学习模块一个明确任务。但还不能决定必须用大模型、GNN或强化学习：先固定能保留组合关系的表示和比较目标，再安排一次受控训练。网络形式只是实现选择。',
        '本轮没有新增训练，现有41维模型及其全部不利结果保持不变；没有扩规模，最终8个测试场景未生成。','',
        '## 核验与记录','',
        '全部4438个旧目标重现；独立循环核对34个确定上下文和3个关键实际上下文；18个小型全排列例子验证打乱对应关系的精确期望公式。原始来源哈希不变，母线编号交换、未来字段拒绝检查逐状态通过。',
        '- plan.json / input_manifest.json：固定诊断问题、算术与数据边界。',
        '- public_inputs.jsonl.gz / future_targets.jsonl.gz：公开连接—反馈输入与未来评价目标分离。',
        '- state_results.json / summary.json：全部状态、日期与物理调用，包括所有反向结果。',
        '- flip_details.json / critical_details.json：全部15个正收益动作变化状态、108/552/RF路径552的具体对应关系。',
        '- combination_profile.json / independent_audit_details.json / review.json：组合覆盖与独立复算。',
        '- decision.json：当前判断及尚未建立的结论。']
    (O/'GRID111_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    archive=REPO/'research_archive/power-grid-2026/protocols/GRID111_action_feedback/complete';archive.mkdir(parents=True,exist_ok=False)
    collab=R/'论文协作资料/FRAME_动作反馈对应_GRID111';collab.mkdir(parents=True,exist_ok=False)
    save('complete.json',dict(passed=True,report='GRID111_report.md',review='review.json',archive=str(archive),collaboration=str(collab),
        new_model_fits=0,learned_model_inferences=0,new_physical_calls=0,existing_learner_unchanged=True,ready_to_scale=False))
    for p in O.rglob('*'):
        if p.is_file():d=archive/p.relative_to(O);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
    for p in W.glob('*.py'):shutil.copy2(p,archive/p.name)
    for n in ['GRID111_report.md','decision.json','review.json','complete.json','combination_profile.json']:shutil.copy2(O/n,collab/n)
    (archive/'archive_manifest.json').write_text(json.dumps({p.relative_to(archive).as_posix():sha(p) for p in archive.rglob('*') if p.is_file()},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    note='\n\n## 2026-10-23 — GRID111 action-feedback association diagnosis\n\nAll4438 saved states, no fit/learned inference/physical call. Keep paid candidate partition-reward correspondence: nearest structural reward interpolation validation date/call-balanced MSE0.09339365 versus mean0.10189388 (-8.34%) and exact random-assignment expectation0.12578865 (-25.75%);4/4 validation dates and39/56 calls better than mean,49/56 versus exact null. Training2/16 dates reverse. This is ordinary reward-error association, not continue/stop or closed-loop advantage. Convex interpolation bounded by current incumbent on4438 states, so it cannot itself forecast positive gain on211train/41val profitable batches; this bound does NOT constrain existing DQN/direct gain regressors.552candidate355 nearest to incumbent141: interpolation copies1.05217695 but actual next improvement0.00709510. All6 changed relations have both values observed; unseen joint partition improves despite marginal contrast direction-0.203179.33/41val positive winners have all changed relations observed both ways; isolated marginal contrasts positive only8/41 (train23/211), confounded/not causal effects. Novel partition alone is not a trigger. Recommend defining joint candidate/paid-feedback representation for improvement beyond incumbent before any controlled fit; no model family or scaling automatically chosen. Public inputs and future evaluation targets separately archived. All4438 targets,34 scalar contexts+3probes,18 exact-null tests pass; final8test ungenerated.\n'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write(note)
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID111动作—反馈对应关系诊断完成\n\n1. 汇报：对应关系有局部奖励信号，但奖励插值不能预测超过当前最佳；单关系均值不能直接相加解释组合收益。\n2. 下一步先固定保留动作连接组合与已查反馈对应的输入，以及预测新增收益/值得继续程度的目标和一次受控比较方案；尚未开始新训练或选模型家族。\n3. 不扩大网络/数据/物理回放；最终8测试场景封存。\n\n报告：research_archive/power-grid-2026/protocols/GRID111_action_feedback/complete/GRID111_report.md\n',encoding='utf8')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID111动作—反馈对应关系](protocols/GRID111_action_feedback/complete/GRID111_report.md)：局部奖励关联有信号，但插值有上界、单关系对比不能直接推出组合增益；无新训练或物理回放。\n')
    print('COMPLETE',archive,flush=True)
if __name__=='__main__':main()
