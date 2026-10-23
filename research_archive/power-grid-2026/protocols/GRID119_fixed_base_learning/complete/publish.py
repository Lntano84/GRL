"""Selective protocol/result archive and Chinese report; no extra fitting."""
import argparse,shutil
from common119 import *

def copy_archive(phase):
    dest=REPO/f'research_archive/power-grid-2026/protocols/GRID119_fixed_base_learning/{phase}'
    dest.mkdir(parents=True,exist_ok=False)
    for p in W.glob('*.py'):shutil.copy2(p,dest/p.name)
    for p in O.iterdir():
        if p.is_file() and (phase=='complete' or p.name in ['design.json','preflight.json','code_freeze.json','data_freeze.json','state.json']):shutil.copy2(p,dest/p.name)
    if phase=='complete':shutil.copytree(O/'models',dest/'models')
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})
    return dest

def update_docs(note,next_text):
    for n in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write('\n\n'+note+'\n')
    (REPO/'docs/NEXT_STEPS.md').write_text(next_text,encoding='utf8')

def before():
    assert read(O/'preflight.json')['passed'];verify();copy_archive('before')
    update_docs('## 2026-10-23 — GRID119 fixed-base training authorized and frozen\n\nUser authorizes reviewed GRID118. Exactly9newfits,BASE_EXTRA/FIXED_JOINT/FIXED_FACTOR seeds0/1/2; originalGRID107 final bases, fixed6978base+5938correction for newmodule arms, freshAdamall, originalHuber/fee/splits/7250x32; common next-stage sampling,1800s total. Zero-update synthetic gradients/masks/permutations and4438target/inputchecks passed. Allvalidationevaluationafter9fixedendpoints. Firstactualupdateandupstreamflowchecksinsidebudget; nohotstartfit. Noenvironment/test/scaling.',
        '# Next steps — GRID119已封存，按授权训练\n\n1. 串行完成9次固定训练，保留全部种子及部分故障，原1800秒总窗口不延长。\n2. 全部完成后统一评价完整补查序列，核对固定基础权重、优化器范围、抽样、目标和独立前向结果。\n3. 汇报与原模型、额外训练和汇总校正的比较；不自动扩规模、改费用或上电网闭环。\n')
    print('GRID119_BEFORE_ARCHIVED',flush=True)

def complete():
    assert not (O/'complete.json').exists();verify();sealed();assert read(O/'review.json')['passed']
    d=read(O/'decision.json');fits=read(O/'fits.json');val=read(O/'validation_results.json');tr=read(O/'train_results.json');metrics=read(O/'state_metrics.json');sens=read(O/'binding_sensitivity.json')
    formats={'JOINT':'FIXED_JOINT_seed{}','FACTOR':'FIXED_FACTOR_seed{}','EXTRA':'BASE_EXTRA_seed{}','BASE':'DIRECT{}','DQN':'DQN0_{}'}
    metric_keys=['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_call_balanced_utility']
    family_metrics={part:{fam:{k:float(np.mean([res[fmt.format(s)]['all']['macro'][k] for s in range(3)])) for k in metric_keys} for fam,fmt in formats.items()} for part,res in [('train',tr),('validation',val)]}
    decomposed={}
    scale=read(O/'design.json')['unchanged']['gain_scale'];fee=read(O/'design.json')['unchanged']['cost_per_query']
    for reference in ['FACTOR','EXTRA','BASE']:
        a=family_metrics['validation']['JOINT'];b=family_metrics['validation'][reference]
        gain=(a['mean_gain']-b['mean_gain'])/scale;cost=-fee*(a['mean_queries']-b['mean_queries'])/scale
        net=a['mean_net_utility']-b['mean_net_utility'];assert abs(gain+cost-net)<1e-12
        decomposed[reference]=dict(gain_contribution=gain,query_charge_contribution=cost,net_difference=net)
    save(O/'summary_analysis.json',dict(family_metrics=family_metrics,joint_minus_reference_decomposition=decomposed,
        no_new_fits=True,scope='Read-only accounting of fixed outputs; no threshold or model selection.'))
    lines=['# GRID119：固定基础模型的补查校正训练结果','',
        '**九次训练及保存结果核验均已完成。只比较固定教师路径上的补查决策，未运行新的电网整周回放。**','',
        '## 本轮实际执行','',
        f'BASE_EXTRA、FIXED_JOINT、FIXED_FACTOR各三个种子，均完成7250次额外更新，每批32状态、64个标量目标。新增训练总耗时{fits["wall_s"]:.2f}秒，原总窗口1800秒未延长。32训练、8验证场景及16/4日期不变，最终8个测试场景仍未生成。全部九次固定终点完成后才统一计算验证结果。',
        '三个旧基础模型均来自GRID107原固定终点。JOINT和FACTOR中基础权重全程固定，只更新5938参数的校正模块；BASE_EXTRA更新原6978参数。各组重新建立Adam，使用相同后续抽样流，损失、费用、标签、决策门槛均未改。',
        '每组都已有一轮基础预训练，再增加一轮更新。额外更新次数与目标数量相同，不代表参数量、计算量或耗时相同。BASE_EXTRA是重置优化器后的额外训练，不能当作不中断的原训练延长。','',
        '## 验证结果','',
        '主指标为完整补查序列的四日期宏平均归一化净效用，已经扣除查询费用，越高越好。下面均为三个种子平均；查询比例、收益保留也先按日期汇总，再作宏平均。']
    for fam,label in [('BASE','原模型'),('EXTRA','原模型额外训练'),('JOINT','固定基础模型＋配对校正'),('FACTOR','固定基础模型＋汇总校正'),('DQN','历史gamma0 DQN')]:
        m=family_metrics['validation'][fam]
        lines.append(f'- {label}：净效用{m["mean_net_utility"]:.8f}；查询比例{100*m["query_fraction"]:.3f}%；收益保留{100*m["gain_capture"]:.3f}%；物理调用均衡效用{m["physical_call_balanced_utility"]:.8f}。')
    batch=val['BATCH1']['all']['macro'];lines+= [f'- 历史固定一批：净效用{batch["mean_net_utility"]:.8f}；查询比例{100*batch["query_fraction"]:.3f}%；收益保留{100*batch["gain_capture"]:.3f}%。','', '这些是离线搜索效用和查询比例，不是整周运行成本、质量达标率或实际控制提速。','', '## 配对种子与日期','']
    for r in d['paired_seeds']:
        lines.append(f'- 种子{r["seed"]}：JOINT {r["joint"]:.8f}，FACTOR {r["factor"]:.8f}，EXTRA {r["extra"]:.8f}，原模型{r["base"]:.8f}；JOINT−FACTOR {r["joint_minus_factor"]:+.8f}，JOINT−EXTRA {r["joint_minus_extra"]:+.8f}。')
    lines+=['','种子平均后的逐日期JOINT−FACTOR：']
    for date,delta in d['date_joint_minus_factor'].items():lines.append(f'- {date}：{delta:+.8f}。')
    lines+=['','## 收益与查询开销分解','']
    for ref,dc in decomposed.items():lines.append(f'- JOINT相对{ref}：收益项{dc["gain_contribution"]:+.8f}，查询费用项{dc["query_charge_contribution"]:+.8f}，合计净效用差{dc["net_difference"]:+.8f}。')
    lines+=['','## 训练与行为检查','']
    for arm in ['BASE_EXTRA','FIXED_JOINT','FIXED_FACTOR']:
        for part in ['train','validation']:
            ms=[metrics[f'{arm}_seed{s}'][part] for s in range(3)]
            lines.append(f'- {arm}，{part}：抽样加权Huber均值{np.mean([m["sampling_weighted_huber"] for m in ms]):.8f}；单步净收益符号遗憾均值{np.mean([m["sampling_weighted_sign_regret"] for m in ms]):.8f}；正收益误停状态数（各种子）{[m["false_stop"] for m in ms]}；无利继续状态数{[m["unprofitable_continue"] for m in ms]}。')
    lines+=['','预定反馈循环错位诊断只改变动作与已知反馈的对应关系，不改变反馈集合；不用于选模型、门槛或种子。各组验证决策变化数：']
    for arm in ['BASE_EXTRA','FIXED_JOINT','FIXED_FACTOR']:lines.append(f'- {arm}：{[sens[f"{arm}_seed{s}"]["validation"]["decision_sign_changes"] for s in range(3)]}；其中正标签状态变化{[sens[f"{arm}_seed{s}"]["validation"]["positive_sign_changes"] for s in range(3)]}。')
    lines+=['','## 按预定条件判读','',
        f'配对机制信号：{d["mechanism_signal"]}。要求家族均值JOINT高于FACTOR、三个配对种子均为正，且至少3/4日期同向。',
        f'JOINT家族均值超过原模型：{d["incremental_over_base"]}；超过额外训练对照：{d["incremental_over_extra"]}；同时超过全部参照：{d["all_anchors_exceeded"]}；满足继续推进的组合条件：{d["competitive_candidate"]}。',
        '若只有平均提高而配对种子或日期反向，不能把它概括为稳定优势。若汇总校正同样好或更好，则不能把校正的收益归给配对关系。原方案和GRID116反例完整保留，本轮不追加训练救结果。','',
        '训练集完整序列净效用（三种子均值）：'+ '，'.join(f'{k} {family_metrics["train"][k]["mean_net_utility"]:.8f}' for k in formats)+'。',
        '训练损失和单步标签误差只作解释，完整序列效用才是主要判读指标。','',
        '## 核验与范围','',
        '核验包含4438个原目标、全部新模型的优化器步数与参数范围、被冻结的基础权重、后续抽样哈希及计数、预算内前两步梯度流、独立NumPy前向、完整保存序列的收益/查询/最终候选，以及预定三个关键上下文。新增校正的第一步与上游梯度流检查均计入7250次预算，没有额外热身拟合。',
        '这是四个已暴露验证日期、同一电网和固定FULL教师路径上的开发实验；相关前缀不是独立样本。即时净收益不等同整周收益。本轮没有新物理预测或回放，没有统计显著、跨网泛化或在线时延结论。',
        '固定基础模型使校正参照明确，但不保证正确使用动作—反馈关系，也不保证没有错误校正。本轮还有分阶段预训练的区别，不能把相对GRID116的全部变化只归因于冻结。下一路线需据这些结果决定，不自动扩大或读取最终测试。']
    report=O/'GRID119_report.md';report.write_text('\n'.join(lines)+'\n',encoding='utf8')
    collab=R/'论文协作资料/FRAME_固定主干校正训练_GRID119';collab.mkdir(exist_ok=False)
    archive=REPO/'research_archive/power-grid-2026/protocols/GRID119_fixed_base_learning/complete'
    save(O/'complete.json',dict(passed=True,completed_fits=9,review_passed=True,ready_to_scale=False,report=str(report),archive=str(archive),collaboration=str(collab),new_physical_calls=0,final_test_materialized=False))
    copy_archive('complete')
    for n in ['GRID119_report.md','decision.json','summary_analysis.json','review.json','fits.json','complete.json']:shutil.copy2(O/n,collab/n)
    note=f'''## 2026-10-23 — GRID119 fixed-base correction nine fits complete

All9fixedendpoints7250x32, commoncontinuationstreams,freshAdam/originalHuber/fee/data; fixedbasehashes exact in6modulefits, no validationselection. Wall{fits['wall_s']:.2f}s. Validationfamily{d['family_mean_utility']};BATCH1={d['batch1_utility']}. PairedJF={[r['joint_minus_factor'] for r in d['paired_seeds']]};pairedJExtra={[r['joint_minus_extra'] for r in d['paired_seeds']]};datesJF={d['date_joint_minus_factor']}. Mechanismgate={d['mechanism_signal']},competitivecandidate={d['competitive_candidate']}. Alladverseresults preserved. IndependentNumPy/targets/optimizer/sampling/trajectory/frozenbaseauditpassed. Onlyexposedteacherpathoffline,notwholeweekquality/latency/crossgrid. Noautomaticfit/scaling/physical/test.
'''
    update_docs(note,'# Next steps — GRID119训练与核验完成\n\n1. 向用户汇报四组实际结果、配对种子差异和收益/查询费用分解。\n2. 依据本轮结果决定是否保留校正路线；不以挑种子、改费用或继续训练补救结果。\n3. 不自动扩大、运行电网闭环或读取最终测试。\n\n报告：research_archive/power-grid-2026/protocols/GRID119_fixed_base_learning/complete/GRID119_report.md\n')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID119固定基础模型校正训练](protocols/GRID119_fixed_base_learning/complete/GRID119_report.md)：九次固定训练、原模型额外训练对照、完整搜索效用及独立保存结果核验。\n')
    print('GRID119_COMPLETE_ARCHIVED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['before','complete'],required=True);args=parser.parse_args()
    before() if args.phase=='before' else complete()
