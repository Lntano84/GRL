"""Protocol/result archive and report for the bounded objective screen."""
import argparse,shutil
from common120 import *

def docs(note,next_steps):
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write('\n\n'+note+'\n')
    (REPO/'docs/NEXT_STEPS.md').write_text(next_steps,encoding='utf8')

def archive(phase):
    dest=REPO/f'research_archive/power-grid-2026/protocols/GRID120_decision_objective/{phase}';dest.mkdir(parents=True,exist_ok=False)
    for p in W.glob('*.py'):shutil.copy2(p,dest/p.name)
    for p in O.iterdir():
        if p.is_file() and (phase=='complete' or p.name in ['design.json','preflight.json','training_distribution.json','code_freeze.json','data_freeze.json','state.json']):shutil.copy2(p,dest/p.name)
    if phase=='complete':shutil.copytree(O/'models',dest/'models')
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})
    return dest

def before():
    verify();assert read(O/'preflight.json')['passed'];archive('before')
    docs('## 2026-10-23 — GRID120 decision-objective screen authorized and frozen\n\nUser authorizes more learning exploration. Bounded6fits SIGN_BCE/UTILITY_BCE seeds0/1/2, original6978MLP/randominitialization/7250x32sample streams/Adam/clip10/splits/price; Huber old3reuse. UTILITY weight abs(original normalized net)/training Eabsnet, fixed global normalizer,no class balancing/clipping/threshold search. Bothnewarmsuse q1-q0binarylogits,not numerical action values. Updates/stateexposure/capacity matched,not scalar-supervision/gradient scale tooldHuber. Cost-regret identity andpopulationlogisticdecisiontarget derived,notwholeweek/convergenceguarantee. Preflight checked4438targets/inputsandoriginalstreams; nooptimizerupdates. All6finishbeforevalidation,1800s total,zeroenvironment/finaltest/scaling.',
        '# Next steps — GRID120学习目标对照已封存\n\n1. 完成两种分类目标各三个种子的固定训练；全部完成后统一评价，不改价格、种子或阈值。\n2. 核对与原Huber、DQN及固定一批的完整补查净效用、查询费用和逐日期/种子结果。\n3. 结果决定下一条学习路线，不自动扩大、加网络、上物理闭环或生成最终测试。\n')
    print('GRID120_BEFORE_ARCHIVED',flush=True)

def complete():
    assert not (O/'complete.json').exists();verify();assert read(O/'review.json')['passed'];F=read(O/'fits.json');D=read(O/'design.json');dec=read(O/'decision.json');val=read(O/'validation_results.json');tr=read(O/'train_results.json');ms=read(O/'state_metrics.json')
    formats={'UTILITY':'UTILITY_BCE_seed{}','SIGN':'SIGN_BCE_seed{}','HUBER':'DIRECT{}','DQN':'DQN0_{}'}
    keys=['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_call_balanced_utility']
    family={part:{fam:{k:float(np.mean([res[fmt.format(s)]['all']['macro'][k] for s in range(3)])) for k in keys} for fam,fmt in formats.items()} for part,res in [('train',tr),('validation',val)]}
    decomp={}
    for ref in ['SIGN','HUBER','DQN']:
        a,b=family['validation']['UTILITY'],family['validation'][ref]
        gain=(a['mean_gain']-b['mean_gain'])/D['gain_scale'];cost=-D['cost_per_query']*(a['mean_queries']-b['mean_queries'])/D['gain_scale'];net=a['mean_net_utility']-b['mean_net_utility'];assert abs(gain+cost-net)<1e-12
        decomp[ref]=dict(gain_contribution=gain,query_charge_contribution=cost,net_difference=net)
    save(O/'summary_analysis.json',dict(family_metrics=family,utility_minus_reference=decomp,no_additional_fits=True))
    verdict='本轮达到预定开发筛选条件，值得继续核对迁移与闭环；这还不是部署结论。' if dec['useful_candidate'] else '本轮未达到预定开发筛选条件，不能据此认定新目标优于现有方案。'
    lines=['# GRID120：直接学习补查取舍的目标对照','',f'**六次训练和保存结果核验完成。{verdict}**','',
        '## 为什么先做这一轮','',
        'GRID108中换成平方损失或延长训练，没有稳定改善净收益；GRID119中固定基础模型的配对校正，也未超过原模型。因此，本轮暂不增加结构，先检验“训练时惩罚的错误，是否对应补查真正付出的代价”。',
        '普通分类只区分下一批净收益正负，漏掉一次大收益和多花一点查询费各算一次错误。收益代价加权分类按错误损失的净收益大小加权，两种错误都计入，权重直接来自原标签，不是为了这次结果挑出来的正负类别系数。',
        '文献依据是[Zadrozny、Langford和Abe，ICDM 2003](https://hunch.net/~jl/projects/reductions/costing/finalICDM2003.pdf)的样本依赖误分类代价和直接加权思想。本轮没有复现原论文的拒绝采样或集成算法；具体二元损失及本任务对应关系在下面推导。','',
        '## 目标与含义','',
        '仍令y为原下一批即时净收益，已扣原查询费用并采用原收益尺度。标签t=1[y>0]，模型分数s=q1−q0，s>0继续，其他情况停止。普通目标为BCEWithLogits(s,t)；加权目标为|y|/Z×BCEWithLogits(s,t)，其中Z只用原训练概率计算E|y|，全程固定，不按小批次重新归一化。没有截断标签或权重，也没有扫门槛。',
        '对任一二元决定a，max(y,0)−a·y=|y|×1[a≠t]。左边是单步收益遗憾，右边是按收益代价加权的分类错误，因此权重有明确的任务含义。固定输入x时，无约束总体加权逻辑损失的最优概率为E[y+|x]/E[|y||x]；它超过1/2，当且仅当E[y|x]>0（分母非零时）。',
        '这段推导只说明总体目标与期望即时取舍的关系，不保证有限数据、神经网络优化或新日期泛化。训练仍有原梯度范数裁剪，裁剪次数完整保存。加权模型的sigmoid输出是代价倾斜的决策概率，不是普通“正收益概率”，两个输出也不是以收益单位标定的动作价值。','',
        '## 实际执行与公平性','',
        f'两种新目标各用种子0/1/2，共六次训练，均完成7250次更新、每批32状态；总训练耗时{F["wall_s"]:.2f}秒，原1800秒总窗口未延长。原Huber模型和gamma0 DQN直接复用，没有重复拟合。',
        '保持原41→64→64→2网络、6978参数、逐种子原随机初始化、原7250批抽样索引和Adam设置，学习率0.0003、梯度裁剪10。原费用、标准化、候选范围、日期和数据未改变。',
        '新目标每状态是一条二元分类监督，旧回归每状态是两个数值目标。二者匹配更新次数、状态曝光、网络和初始化，不能声称标量监督或梯度尺度完全相同；两种新分类目标则只改变收益权重。',
        '仍是32训练场景/16日期、8验证场景/4日期，3582/856可达状态、2378/590起始前缀。所有训练固定终点完成后才统一评价验证，不挑种子或中间检查点。最终8测试场景未生成，本轮没有新电网预测、回放或安装。','',
        '## 验证结果','',
        '以下为四日期宏平均后，再对三个种子平均。净效用已扣查询费用，越高越好；查询比例与收益保留均相对完整同站补查，先按日期计算，再宏平均。']
    for fam,label in [('HUBER','原Huber回归'),('SIGN','普通正负分类'),('UTILITY','收益代价加权分类'),('DQN','历史gamma0 DQN')]:
        m=family['validation'][fam];lines.append(f'- {label}：净效用{m["mean_net_utility"]:.8f}，查询比例{100*m["query_fraction"]:.3f}%，收益保留{100*m["gain_capture"]:.3f}%，物理调用均衡净效用{m["physical_call_balanced_utility"]:.8f}。')
    b=val['BATCH1']['all']['macro'];lines += [f'- 固定一批：净效用{b["mean_net_utility"]:.8f}，查询比例{100*b["query_fraction"]:.3f}%，收益保留{100*b["gain_capture"]:.3f}%。','',
        '配对种子（加权目标减普通分类、减原Huber）：']
    for p in dec['paired_seeds']:lines.append(f'- 种子{p["seed"]}：加权{p["utility"]:.8f}，普通分类{p["sign"]:.8f}，Huber {p["huber"]:.8f}；差值分别{p["utility_minus_sign"]:+.8f}、{p["utility_minus_huber"]:+.8f}。')
    lines+=['','种子平均后的逐日期“加权减普通分类”：']
    for date,delta in dec['date_utility_minus_sign'].items():lines.append(f'- {date}：{delta:+.8f}。')
    lines+=['','## 差异来自哪里','']
    for ref,dc in decomp.items():lines.append(f'- 加权目标相对{ref}：收益项{dc["gain_contribution"]:+.8f}，查询费用项{dc["query_charge_contribution"]:+.8f}，净差{dc["net_difference"]:+.8f}。')
    for fam,fmt in formats.items():
        for part in ['train','validation']:
            stats=[ms[fmt.format(s)][part] for s in range(3)]
            lines.append(f'- {fam}，{part}：原层次抽样权重下单步收益遗憾均值{np.mean([m["weighted_sign_regret"] for m in stats]):.8f}；正收益误停状态数{[m["positive_misses"] for m in stats]}；无利继续状态数{[m["unprofitable_continues"] for m in stats]}。')
    lines+=['','训练集完整搜索净效用（三种子均值）：'+ '，'.join(f'{k} {family["train"][k]["mean_net_utility"]:.8f}' for k in formats)+'。',
        '单步指标和误停次数不能替代完整搜索效用，误停很多小收益与误停少数大收益的后果也不同。','',
        '## 预定判读与下一步','',
        f'目标信号={dec["objective_signal"]}；同时超过Huber、DQN和固定一批={dec["all_anchors_exceeded"]}；组合开发条件={dec["useful_candidate"]}。',
        '目标信号要求加权家族均值高于普通分类、三个配对种子差异均为正、至少3/4日期同向。组合条件还要求加权家族均值高于原Huber、原gamma0 DQN和固定一批。没有根据本轮成绩放宽条件。',
        '后续学习工作按三个问题展开：第一，目标是否真的减少有代价的误停和多查；第二，哪些可见反馈具有跨日期的预测价值；第三，教师路径上的判断接入FRAME实际路径后是否仍有效。强化学习只有在后续批次价值或闭环信用分配问题被明确定义后再单独设计，不把更换RL算法当作自动补救。',
        '本轮只完成上述第一项的一个有界筛选，不自动执行下一批算法、收集新场景或扩网。所有不利结果和旧模型保留。','',
        '## 核验与适用范围','',
        f'已核对全部26628个新模型状态前向、4438个标签、原初始化和抽样计数、六个优化器的7250步、完整训练/验证搜索序列，以及两种动作下收益遗憾的代数等式。保存结果审计见review.json。',
        '这仍是同一电网的四个已暴露验证日期、固定FULL教师路径和相关前缀。即时搜索收益不是整周运行成本，不报告本轮物理安全率或实际控制提速，也不把四日期的开发比较称为独立最终测试。']
    (O/'GRID120_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    collab=R/'论文协作资料/FRAME_学习目标对照_GRID120';collab.mkdir(exist_ok=False);dest=REPO/'research_archive/power-grid-2026/protocols/GRID120_decision_objective/complete'
    save(O/'complete.json',dict(passed=True,completed_fits=6,review_passed=True,ready_to_scale=False,report=str(O/'GRID120_report.md'),archive=str(dest),collaboration=str(collab),new_physical_calls=0,final_test_materialized=False))
    archive('complete')
    for n in ['GRID120_report.md','decision.json','review.json','summary_analysis.json','complete.json','training_distribution.json','fits.json']:shutil.copy2(O/n,collab/n)
    note=f'''## 2026-10-23 — GRID120 decision-objective screen complete

Exactly6fits SIGN_BCE/UTILITY_BCE seeds0/1/2,original6978network/init/7250x32streams/Adam/fee/data,no modelselection. Trainingwall{F['wall_s']:.2f}s. Validationfamily{dec['family_mean_utility']};BATCH1={dec['batch1_utility']}. Pairedutility-sign={[r['utility_minus_sign'] for r in dec['paired_seeds']]};pairedutility-Huber={[r['utility_minus_huber'] for r in dec['paired_seeds']]};dateutility-sign={dec['date_utility_minus_sign']}. Objectivegate={dec['objective_signal']},usefulcandidate={dec['useful_candidate']}. AllstateNumPy/optimizer/sampling/target/regretidentity/trajectoriesauditpassed. OriginalregressionandRLnegativeevidencekept. ClassificationlogitsnotnumericalQ,weightedprobabilitynotunweightedclassprobability;populationconsistencydoesnotguaranteefinitefit/generalization. Exposedteacherpathofflineonly,no physical/finaltest/autoscaling.
'''
    docs(note,'# Next steps — GRID120学习目标对照完成\n\n1. 向用户解释训练目标对完整补查净效用、查询费用及全部种子/日期的影响。\n2. 结合本轮结果确定目标、可见反馈或教师路径迁移的下一项，不自动扫参或更换更多算法。\n3. 新的强化学习应先明确后续价值/信用分配与匹配对照；最终测试和物理扩大仍不自动启动。\n\n报告：research_archive/power-grid-2026/protocols/GRID120_decision_objective/complete/GRID120_report.md\n')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID120学习目标对照](protocols/GRID120_decision_objective/complete/GRID120_report.md)：普通分类与收益代价加权分类，六个固定训练终点及完整保存结果核验。\n')
    print('GRID120_COMPLETE_ARCHIVED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['before','complete'],required=True);a=p.parse_args();before() if a.phase=='before' else complete()
