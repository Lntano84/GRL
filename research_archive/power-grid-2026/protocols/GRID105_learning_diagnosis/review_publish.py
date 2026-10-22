"""Independently reconcile saved diagnostic labels and publish the finite work unit."""
from pathlib import Path
import collections,gzip,hashlib,json,pickle,shutil,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid105_learning_diagnosis'
REPO=ROOT/'work/GRL-results-integration'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
    assert not (OUT/'complete.json').exists()
    hashes=read(OUT/'input_hashes.json')
    for p,h in hashes.items():assert sha(ROOT/p)==h,p
    # Verify original frozen scientific inputs, not just the files selected for explanation.
    count=0
    for name in ['code_freeze.json','data_freeze.json']:
        for p,h in read(ROOT/'outputs/grid104_learning_closedloop'/name).items():
            assert sha(ROOT/p)==h,p;count+=1
    with (ROOT/'outputs/grid102_search_rl/cases.pkl').open('rb') as f:cases=pickle.load(f)
    d=read(ROOT/'outputs/grid102_search_rl/design.json');fee=d['cost_per_query'];scale=d['gain_scale']
    truth={}
    for c in cases:
        for j in range(len(c['queries'])):
            truth[c['sample_id'],j]=(c['partition'],c['physical_state_group'],float(c['gains'][j])-fee*int(c['queries'][j]))
    with gzip.open(OUT/'offline_predictions.jsonl.gz','rt',encoding='utf8') as f:rows=[json.loads(s) for s in f]
    assert len(rows)==17752
    for r in rows:
        p,g,t=truth[r['sample_id'],r['batch']]
        assert r['partition']==p and r['group']==g and r['target_net']==t and r['target_normalized']==t/scale
        if r['model'].startswith('DQN'):assert r['action']==int(r['q1']>r['q0'])
        else:assert r['action']==int(r['normalized_gain']>fee*r['next_queries']/scale)
    D=read(OUT/'diagnosis.json')
    for model in ['DQN0_0','DQN0_1','DQN0_2','RF']:
        for part in ['train','validation']:
            rs=[r for r in rows if r['model']==model and r['partition']==part]
            m=D['offline'][model][part]['pooled']
            assert len(rs)==m['states'] and sum(r['target_net']>0 for r in rs)==m['positive']
            assert sum(r['target_net']>0 and r['action']==0 for r in rs)==m['false_stop']
            assert sum(r['target_net']<=0 and r['action']==1 for r in rs)==m['unprofitable_continue']
    online=read(OUT/'closedloop_decisions.json')
    assert len(online['known'])==104 and len(online['unknown'])==20
    for r in online['known']:
        assert r['target_net']==r['actual_next_gain']-fee*r['next_queries']
        assert r['best_future_net']>=0 and r['hindsight_myopic_continue']==int(r['target_net']>0)
    # Exact feature collisions: an absence of collisions cannot prove that 41 summaries are sufficient.
    with np.load(ROOT/'outputs/grid102_search_rl/normalization.npz') as z:mean=z['mean'];std=z['scale']
    groups=collections.defaultdict(list)
    for c in cases:
        for j,x in enumerate(c['states']):
            x=np.clip((x-mean)/std,-20,20).astype(np.float32)
            groups[c['partition'],x.tobytes()].append(dict(positive=truth[c['sample_id'],j][2]>0,group=c['physical_state_group']))
    collisions={p:dict(unique_features=sum(k[0]==p for k in groups),conflicting_sign_vectors=sum(k[0]==p and len({r['positive'] for r in v})>1 for k,v in groups.items())) for p in ['train','validation']}
    save(OUT/'supplementary_checks.json',dict(exact_feature_collisions=collisions,
        note='No exact collision is not a sufficiency or out-of-distribution test.'))
    # Earlier paired experiments are evidence about label semantics, never new training data here.
    for p in [ROOT/'outputs/grid95_labels/GRID95_report.md',ROOT/'outputs/grid97_effect_learning/GRID97_report.md',ROOT/'outputs/grid99_feature_learning/GRID99_report.md']:
        hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'input_hashes.json',hashes)
    text='''# GRID105：先解决目标不一致，再解决补查误判

本轮只读取原数据并用固定模型复算，没有训练、物理回放、新场景或价格调整。全部三个DQN种子保留，原GRID104结论没有改变。最终测试集未生成。

## 1. 第一类问题：当前奖励没有直接描述整周质量

源码确认，候选预测奖励是 `2 − 最大线路负载率`。补查学习奖励是“最佳已观察候选的预测奖励增量 − 固定查询费用”，再除以训练集固定尺度。它衡量下一步预测负载率改善与查询数量的权衡。查询价格由训练数据的平均增益与查询量换算，不是实际秒数或运行成本。

论文采用的整周运行成本则由网损、再调度、弃电变化和储能吞吐等累计构成，存活也单独评价。因此，减少这项即时净奖励的损失，并不等于维持整周成本。gamma=1也只延伸到后续补查批次，当前环境没有跨电网控制时刻的转移。

3月RF第108步，下一批64个候选的真实增益为0，费用为0.00079631；后续所有可选停止位置也没有正的即时净收益。RF停止符合当前目标，但整套RF闭环策略成本比FRAME高1.112%。这说明当前目标不能直接充当整周质量标准；**尚不能认定这一次停止造成了全部1.112%的差距**。

这并不是要从头发明一种长期标签。GRID95已经验证“只取消一次补查，之后恢复共同FRAME”的成对回放方法：有的单次取消省成本，有的使控制提前失败。GRID97/99后来主要学习的是轨迹是否相同，不能把这些标签当作运行成本收益。本轮不把历史已暴露案例或当前验证错误塞进训练集。

建议将学习问题改为：先判断少查对存活、后续运行成本有什么影响，再在满足质量要求的选择中减少计算。保留这些量的独立记录，暂不把它们用一个新系数加成总分。单纯降低查询费或把费用设为0，仍不能让一步负载率奖励等同于整周成本。

## 2. 第二类问题：DQN对当前目标也有明显误判

对原数据的4438个可达状态逐一复算固定模型，训练与验证分开统计，未选择模型或检查点。这是对已保存下一批结果的事后诊断，不是每个状态都实际运行一次新策略。

训练3582个状态中，下一批净收益为正的有211个，约5.89%，分布在54个物理搜索调用、15个训练日期。三个DQN分别漏查45、54、59个，正收益召回为78.67%、74.41%、72.04%；同时分别在851、826、463个非正收益状态上继续查询。既有漏查也有多查，不能简单用“提高补查倾向”解决。

验证856个状态中，正净收益有41个。三个DQN分别漏查15、7、17个，召回63.41%、82.93%、58.54%；非正收益继续查询分别为395、368、194次。统计单位是相关的搜索前缀/批次，不能把这些数当作数百个独立电网场景。逐日期和物理调用分组指标见diagnosis.json。

在3月第552步，下一批13个候选的净收益为0.00693335，归一化后为+0.80360；种子0/1却给继续动作估值−0.10889/−0.06504并停止。两者STOP估值接近0，输入也没有触发截断。这里的具体问题是继续价值被低估，不是STOP输出虚高或输入被截断。

在同一套41维标准化特征下，这个状态最接近的五个**不同训练物理调用**的下一批净收益都非正。这个观察与关键正例覆盖不足或当前特征难以区分相似状态相符，但距离度量并不能证明分布外失效。训练集本身也存在错判，所以同样不能把全部原因归给新路径。

当前观测只汇总已查询候选的反馈和候选结构，未直接表征电网负荷、出力、线路运行状态等物理信息。是否需要这些信息，要通过训练数据上的受控比较判断；不是把原始向量全部拼进去。没有原DQN回放缓冲区，不能倒推出每个正例实际被训练了多少次，也不能仅凭固定30000步宣称收敛或不收敛。

## 3. 闭环能核对到什么

四个学习策略在repeat0共有124个补查决策。其中104个能够找到相同的FRAME物理观测、基础动作和全局反馈，并用FRAME已支付的补查结果核对下一批；另外20个没有共同上下文，标签保留未知。repeat1只核对重复决策一致，不增加样本数。

可核对部分中，种子0/1各漏掉1个正净收益批次，均为3月第552步；种子2没有漏掉已知正例。三个种子分别有14、20、8次非正收益继续查询。RF没有漏掉这部分中的正例，但这部分本身没有正例，不能据此称其可靠。没有发现“当前批不划算但后面能补回来”的漏停案例。

这些结果支持分开处理即时目标误判和整周质量，但还不能给20个不同路径状态补上反事实答案，更不能把首处分歧之后的整周差额直接标给该次补查。

## 4. 接下来的最小修复顺序

**先把长期标签接上。** 沿用GRID95的完整前缀重放和单次干预资格流程，针对已定位的3月108/552步分别只取消那一次补查，之后统一恢复原FRAME。核对存活、剩余整周成本和实际查询；一步、一小时、一天只辅助解释，不挑选有利窗口。两处都是已暴露的开发诊断，不作为新测试成绩，也不直接拿来训练。若单次取消没有损害，必须保留该反例，不能继续把整策略差额归给它。

**再隔离训练方式的影响。** 在原训练数据上，用同样41维输入和64×64网络，直接回归已知的下一批净收益，与现有gamma0 DQN比较。gamma0的目标不依赖下一状态Q值，因此这能检查：用完整已知监督目标替代探索采样后，误判能否减少。固定种子、物理调用/日期权重及训练预算，在训练部分确定方案；验证不用于挑检查点。不同时扩网络、换费用或加特征，避免改善后不知道原因。

若同网络的直接回归也在训练物理调用留出上失效，再检查哪些候选结构和物理运行信息缺失。若直接回归明显改善而DQN仍不稳定，优先修训练采样与价值估计；没有必要为了叫强化学习而保留低效训练方式。若长期质量目标尚未建立，即时预测改善也只报告即时预测改善。

本轮到诊断与修复方案为止，**没有执行上述新分支回放或拟合**。先解决目标与估值，仍不扩大日期、数据量、网络或最终测试范围。

## 复核与资料

diagnose.py保存预先声明的诊断范围、17752条固定模型预测、全部匹配/未知在线决策和输入哈希。review_publish.py独立核对收益减费用、动作符号、汇总数量及原冻结文件。各阶段原始数据和模型不改；没有独立AC重求解。
'''
    (OUT/'GRID105_report.md').write_text(text,encoding='utf8')
    audit=read(OUT/'review.json');audit.update(independent_target_checks=len(rows),original_GRID104_frozen_hashes_checked=count,
        exact_feature_collision_check=collisions,posthoc=True)
    save(OUT/'review.json',audit)
    save(OUT/'decision.json',dict(diagnosis_completed=True,original_scale_decision_unchanged=True,
        current_objective_not_week_cost=True,realized_misjudgments_present_in_training=True,
        convergence_cause_identified=False,feature_sufficiency_established=False,
        proposed_next=['Paired single-intervention March108/552 with common continuation, not whole-policy labels.',
            'Matched same-architecture direct next-net-value regression to isolate training mechanism; no simultaneous feature/network/price changes.'],
        new_rollouts_or_fits_executed=False,final_test_materialized=False))
    dest=REPO/'research_archive/power-grid-2026/protocols/GRID105_learning_diagnosis'
    collab=ROOT/'论文协作资料/FRAME_学习目标与误判诊断_GRID105_20261022'
    assert not dest.exists() and not collab.exists();dest.mkdir(parents=True);collab.mkdir(parents=True)
    for p in list(OUT.iterdir())+list(Path(__file__).parent.glob('*.py')):
        if p.is_file():shutil.copy2(p,dest/p.name)
    for name in ['GRID105_report.md','diagnosis.json','review.json','decision.json']:
        shutil.copy2(OUT/name,collab/name)
    save(dest/'archive_manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
    save(OUT/'complete.json',dict(passed=True,diagnostic_only=True,new_fits=0,new_forecasts=0,new_physical_steps=0,
        final_test_materialized=False,archive=str(dest),collaboration=str(collab),report_sha256=sha(OUT/'GRID105_report.md')))
    shutil.copy2(OUT/'complete.json',dest/'complete.json')
    # Keep manifest complete after adding completion receipt.
    save(dest/'archive_manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='archive_manifest.json'})
    note='\n\n## 2026-10-22 — GRID105 objective and value-error diagnosis\n\nUser prioritizes objective alignment and DQN errors. Read-only4438 reachable states/17752 fixed-model predictions: train211 profitable next batches, DQN false stops45/54/59; validation41, misses15/7/17. Online124 decisions,104 exact FRAME-context labels and20 unknown; repeats not independent samples. Source objective is max-rho improvement minus proxy query fee, not whole-week operational cost. March552 continuation undervalued by seeds0/1; nearest five distinct training-call groups have nonpositive realized net value, descriptive not causal OOD proof. Train errors rule out attributing everything solely to new closed-loop paths; convergence/feature insufficiency not proven. No fits/forecasts/rollouts/price changes or final-test access. Proposed next: reuse GRID95 single-intervention/common-continuation labels for March108/552, then matched same-architecture direct net-value regression to isolate training mechanism. Neither proposed experiment executed this stage; GRID104 scaling failure unchanged.\n'
    for name in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/name).open('a',encoding='utf8') as f:f.write(note)
    (REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps — GRID105两类问题诊断完成

1. 目标对齐：沿用GRID95单次干预、共同FRAME续行的标签流程，检验3月108/552步本身的长期价值；不得用整策略差额给每次补查贴标签。
2. 估值误判：在原训练数据上准备同41维/64×64网络的直接净收益回归对照，隔离DQN采样与训练影响；不同时调费用、特征和规模。尚未执行新回放或拟合。
3. 原GRID104扩大资格仍为False。四个验证日期已暴露；最终8场景未生成，不恢复旧自动化，不自动扩大数据/网络/电网。

当前报告：research_archive/power-grid-2026/protocols/GRID105_learning_diagnosis/GRID105_report.md
''',encoding='utf8')
    print(json.dumps(dict(passed=True,frozen_checks=count,prediction_checks=len(rows),feature_collision_check=collisions,report=str(OUT/'GRID105_report.md')),ensure_ascii=False))
if __name__=='__main__':main()
