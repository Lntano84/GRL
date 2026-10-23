"""Archive the visibility audit; never train or choose a new stop threshold."""
import argparse
from common122 import *

def sync_docs(note,next_text):
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write('\n\n'+note+'\n')
    (REPO/'docs/NEXT_STEPS.md').write_text(next_text,encoding='utf8')

def archive(phase):
    a=REPO/f'research_archive/power-grid-2026/protocols/GRID122_false_stop_audit/{phase}';a.mkdir(parents=True,exist_ok=False)
    for p in W.glob('*.py'):shutil.copy2(p,a/p.name)
    for p in O.iterdir():
        if p.is_file():shutil.copy2(p,a/p.name)
    save(a/'archive_manifest.json',{p.name:sha(p) for p in a.iterdir() if p.is_file()});return a

def before():
    verify();archive('before')
    sync_docs('## 2026-10-23 — GRID122 visible-evidence/support audit frozen\n\nUser continues announced highcostfalseSTOPdiagnosis. Readonlyall3582OOFstates/3seeds/firstcommonprefixdifferences;NNinoriginal41fold-normalizedspaceusesonly12fitdates,allnearestties,no learnedmetric/k/threshold tuning. Fitcoveragebaselineexcludesentirephysicalcall;fit-only95percentile diagnosticnotstoprule. Describeall41+8fixedpaidfeedback/geometryfieldsandrawgain/knownchargefactorization. Threepriorworstcallsillustrativeonly;no fit/forward/physical/test.',
        '# Next steps — GRID122可见证据核查已封存\n\n1. 完成训练覆盖、首次分歧和公开反馈描述的只读复算。\n2. 核对全部标签与分歧、直接距离见证、物理调用均衡描述及宏平均差额守恒。\n3. 据证据选择下一种监督信号；本轮不训练、不改费用或生成测试。\n')
    print('GRID122_BEFORE_ARCHIVED',flush=True)

def complete():
    assert not (O/'complete.json').exists();verify();assert read(O/'review.json')['passed'];groups=read(O/'group_summaries.json');factor=read(O/'label_factorization.json');coverage=read(O/'fold_coverage.json');assoc=read(O/'all_feature_associations.json');examples=read(O/'top_call_details.json')
    bad=groups['utility0_huber1_nextpositive'];saved=groups['utility0_huber1_nextnonpositive'];ranked=sorted([a for a in assoc if a['standardized_difference'] is not None],key=lambda a:-abs(a['standardized_difference']))
    overlap=sum(a['observed_call_mean_range_overlap'] for a in assoc)
    # These are observations and next hypotheses, not new trained outcomes or decision rules.
    decision=dict(high_cost_false_stops=bad['records'],high_cost_false_stop_calls=bad['distinct_calls'],avoided_unprofitable=saved['records'],avoided_unprofitable_calls=saved['distinct_calls'],
        bad_beyond_fit_distance95=bad['call_balanced_beyond_fit_lco_q95'],saved_beyond_fit_distance95=saved['call_balanced_beyond_fit_lco_q95'],
        bad_nearest_training_positive_fraction=bad['call_balanced_nearest_positive_fraction'],saved_nearest_training_positive_fraction=saved['call_balanced_nearest_positive_fraction'],
        descriptive_feature_range_overlaps=overlap,features_described=len(assoc),
        no_learnability_or_irreducibility_proof=True,no_new_trigger_selected=True,new_fits=0,new_physical_calls=0,ready_to_scale=False,
        next='Keep original net regression; use these coverage and observed-feedback results to define separate improvement-event and amount supervision. Match the control before fitting; do not adopt a nearest-neighbor trigger or infer wholeweek value.')
    save(O/'decision.json',decision)
    lines=['# GRID122：高代价误停的可见证据核查','',
        '**本轮复核已有输入、标签和折外预测，没有训练新模型。近邻和特征差异只用于诊断，没有据此生成补查触发规则。**','',
        '## 查的是什么','',
        'GRID121中，加权分类总体低于原回归。它减少了很多无利查询，但也漏掉一些有价值的补查。本轮覆盖16个原训练日期的全部3582个可达状态、三个种子；具体组别只在两种策略首次实际分歧处统计，不把已经停止后才出现的分歧算进去。',
        '重点比较两组：加权停止、回归继续，且下一批净收益为正的“误停”；同样的策略分歧但下一批不盈利的“省下无利查询”。前者与后者均由已保存标签作事后分组，不能拿这些组标签作为在线输入。','',
        f'- 误停：{bad["records"]}条前缀×种子记录，{bad["unique_states"]}个不同输入状态，来自{bad["distinct_calls"]}次物理搜索调用。',
        f'- 省下无利查询：{saved["records"]}条记录，{saved["unique_states"]}个不同状态，来自{saved["distinct_calls"]}次调用。',
        '两组调用可能重叠，记录也存在重复前缀；以下按物理调用先平均再汇总，不把每条前缀当成独立样本。','',
        '## 一、相似训练状态是否存在','',
        '对每个留出状态，只在对应12个训练日期中查找已有41维摘要的最近状态。使用与原折模型相同的训练侧标准化和截断，41维平方欧氏距离不加权、不挑选维度；所有精确最近并列一起报告标签均值。没有选择k、带宽、距离参数或训练近邻策略。',
        '作为覆盖参照，另外计算训练状态到“其他物理搜索调用”的最近距离，排除同一次调用的所有前缀。用训练侧原抽样概率求该距离的95%分位；它只衡量相对覆盖，不是新的停止门槛。',
        f'- 误停组：最近训练状态为正收益的比例，按调用均衡为{100*bad["call_balanced_nearest_positive_fraction"]:.2f}%；距离超过训练侧95%分位的比例为{100*bad["call_balanced_beyond_fit_lco_q95"]:.2f}%。',
        f'- 无利查询被省下组：对应比例为{100*saved["call_balanced_nearest_positive_fraction"]:.2f}%和{100*saved["call_balanced_beyond_fit_lco_q95"]:.2f}%。',
        f'- 误停组正收益最近邻比非正收益最近邻更远的比例为{100*bad["call_balanced_positive_farther_than_negative"]:.2f}%；另一组为{100*saved["call_balanced_positive_farther_than_negative"]:.2f}%。',
        '各折覆盖情况：']
    for fold,c in coverage.items():lines.append(f'- 折{fold}：留出{c["held_states"]}状态，其中{c["held_beyond_q95"]}个超出覆盖参照；原始41维精确匹配{c["held_exact_raw_matches"]}个，标准化后精确匹配{c["held_exact_normalized_matches"]}个。')
    lines+=['',
        '正收益状态本来较少，最近邻更多为负，不能单独证明信息不足；距离近也不能证明标签可预测。这里既不是学习能力上界，也没有证明这些状态必然无法区分。尤其不能把“有正收益近邻”直接变成应当继续的规则。','',
        '## 二、补查前能看到什么差别','',
        '原41维摘要外，本轮固定描述8项已有公开信息：已知同站动作数、严格可接纳/合法反馈数、合法奖励间隔的均值与标准差、下一批到当前最好动作/已知动作的设备对连接距离，以及下一批与已查连接完全相同的比例。这些信息全部来自已买到的反馈和公开候选计划，没有使用下一批的真实结果。',
        f'共49项字段中，{overlap}项的两组调用均值范围有重叠。下列为绝对标准化均值差最大的前8项，仅作描述；全部字段及反向结果均保存在all_feature_associations.json。']
    for a in ranked[:8]:lines.append(f'- {a["feature"]}：误停组均值{a["bad_mean"]:.6g}，省查询组{a["saved_mean"]:.6g}，标准化差{a["standardized_difference"]:+.3f}，调用均值范围重叠={a["observed_call_mean_range_overlap"]}。')
    lines += ['',
        '分组由实际结果事后确定，且两组调用重叠。这些关联不是独立统计检验、因果关系或可部署规则；不据此挑一个阈值、只保留最有利字段或宣布神经网络应能分开两组。先前GRID111中，已观察的结构—反馈关系可以改善部分奖励相似性估计，但凸插值仍不能预测超过已查最好的新动作；这个限制继续保留，不重复将结构新颖性当成触发条件。','',
        '## 三、三次高损失调用的上下文','',
        '这三次是在前一轮误差记账后选出的例子，只帮助定位，不是独立检验。']
    for ex in examples:
        rows=ex['divergence_rows'];miss=[r for r in rows if r['category']=='utility0_huber1_nextpositive']
        lines.append(f'- {ex["physical_state_group"]}：首次分歧{len(rows)}条，其中正净收益误停{len(miss)}条；先前宏平均负贡献{ex["prior_negative_contribution"]:+.8f}。')
        if miss:lines.append(f'  误停处下一批真实归一化净收益范围{min(r["target_net"] for r in miss):.6g}—{max(r["target_net"] for r in miss):.6g}；模型分类分数范围{min(r["utility_margin"] for r in miss):.6g}—{max(r["utility_margin"] for r in miss):.6g}。这些未来收益只用于诊断。')
    lines+=['','## 四、下一种监督信号应保留什么','',
        '上一轮只学净收益的正负，丢掉收益大小；加权分类虽然保留了错误代价，仍未稳定迁移。可以将原下一批增益与已知查询费用分开：是否出现真实改进，以及改进一旦出现有多大。两项都来自既有教师反馈，不需要增加仿真或改价格。',
        f'原训练状态中，下一批原始增益大于0的有{factor["positive_raw_gain_states"]}个、来自{factor["positive_raw_gain_calls"]}次调用；扣费后净收益大于0的有{factor["positive_net_states"]}个、来自{factor["positive_net_calls"]}次调用。',
        f'按原训练概率，发生原始改进的概率为{factor["train_probability_raw_improvement"]:.6f}，正改进条件均值为{factor["conditional_mean_positive_gain"]:.6f}，两者乘积为平均增益{factor["expected_normalized_gain"]:.6f}；已知平均查询费用为{factor["expected_normalized_charge"]:.6f}。',
        '该乘积分解只是标签恒等式，不保证拆成两个预测头会更准。我的判断是：保留原净收益回归作为主任务，优先检验“发生改进”是否能提供有用的辅助监督，比再做一个只判正负的替代模型更值得尝试。若进一步拟合，必须有相同网络、初始化、更新预算和输入的无辅助监督对照；不能把新损失与新网络同时引入。',
        '本轮没有执行上述训练，也没有确定辅助损失系数或新触发门槛。先把具体匹配方案说明清楚，再检验其收益；强化学习的后续价值或整周信用分配仍是另一个问题。','',
        '## 核验与范围','',
        '独立核验包括全部3582个原标签、全部首次实际分歧与分数、宏平均差额守恒、直接逐维距离见证、排除同次调用的训练覆盖、公开连接描述和调用均衡关联。源文件、冻结权重及训练数据未改变。',
        '本轮没有载入模型权重、前向预测、优化器、拟合、物理仿真或新测试。全部结果限于原16个已暴露训练日期和固定FULL教师路径；历史费用/收益尺度的全16日期依赖保持显式，不将它称为新独立泛化验证。']
    report=O/'GRID122_report.md';report.write_text('\n'.join(lines)+'\n',encoding='utf8');collab=R/'论文协作资料/FRAME_高代价误停证据_GRID122';collab.mkdir(exist_ok=False);a=REPO/'research_archive/power-grid-2026/protocols/GRID122_false_stop_audit/complete'
    save(O/'state.json',dict(stage='complete',new_fits=0,review_passed=True));save(O/'complete.json',dict(passed=True,scope='Read-only evidence diagnosis, no new learning or deployed trigger.',report=str(report),archive=str(a),collaboration=str(collab),new_fits=0,new_physical_calls=0,final_test_materialized=False));archive('complete')
    for name in ['GRID122_report.md','decision.json','group_summaries.json','label_factorization.json','review.json','complete.json']:shutil.copy2(O/name,collab/name)
    sync_docs(f"## 2026-10-23 — GRID122 visible-evidence/support audit complete\n\nReadonly3582OOFstatesandallfirstdivergences. ProfitablefalseSTOP{bad['records']}records/{bad['distinct_calls']}calls;avoidedunprofitable{saved['records']}/{saved['distinct_calls']}. Callbalanced nearestpositivefractionsbad/saved={bad['call_balanced_nearest_positive_fraction']:.6f}/{saved['call_balanced_nearest_positive_fraction']:.6f},beyondfit-onlydistance95={bad['call_balanced_beyond_fit_lco_q95']:.6f}/{saved['call_balanced_beyond_fit_lco_q95']:.6f};descriptiveonlynotlearnability/irreducibility. {overlap}/49callmeanfeaturerangesoverlap. Rawgainpositives{factor['positive_raw_gain_states']}/{factor['positive_raw_gain_calls']}calls,netpositive{factor['positive_net_states']}/{factor['positive_net_calls']}calls;probability-timesamountidentitychecked,notpredictivemodelresult. Independentdistances/publicdescriptors/divergences/conservationauditpasses;zero model/optimizer/fit/physical/test. Keepnetregression,nextcandidateimprovement-eventauxiliarysupervisionneedsmatchedprotocolbeforefit;nochosenNNtrigger.",
        '# Next steps — GRID122高代价误停证据核查完成\n\n1. 汇报训练覆盖、可见反馈差异与标签分解，避免把近邻或事后字段差异直接当规则。\n2. 下一候选保留净收益回归主任务，制定改进事件辅助监督的同容量/同初始化/同预算对照；匹配后再训练。\n3. 不自动改费用、扩大网络、启动新电网回放或生成最终测试。\n\n报告：research_archive/power-grid-2026/protocols/GRID122_false_stop_audit/complete/GRID122_report.md\n')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID122高代价误停的可见证据核查](protocols/GRID122_false_stop_audit/complete/GRID122_report.md)：折内训练覆盖、首次实际分歧和公开反馈描述；无新增训练或触发规则。\n')
    print('GRID122_COMPLETE_ARCHIVED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['before','complete'],required=True);a=p.parse_args();before() if a.phase=='before' else complete()
