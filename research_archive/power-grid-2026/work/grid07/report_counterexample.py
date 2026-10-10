"""Write counterexample interpretation and an ordinary scientific diagnostic plot."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid07'
SCENE='2035-01-08_5'
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def lines(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
audit=load(OUT/'GRID07_counterfactual_audit.json');assert audit['passed']
offline=load(OUT/'GRID07_offline_summary.json')
first=load(OUT/'first_common_state_comparison.json')
base=ROOT/'outputs/grid06/runs'
old={p:lines(base/f'{SCENE}__{p}'/'steps.jsonl') for p in ['NN20','FALLBACK90','NN352']}
cf=lines(OUT/'runs'/f'{SCENE}__SINGLE_EXPAND_221'/'steps.jsonl')
assert len(cf)==len(old['NN20'])==2017 and audit['completed_native_week']
paths={'NN20':old['NN20'],'FALLBACK90':old['FALLBACK90'],'SINGLE_EXPAND_221':cf}
vectors={p:np.array([r['raw_operational_cost'] for r in rows]) for p,rows in paths.items()}
delta={p:np.cumsum(v-vectors['NN20']) for p,v in vectors.items()}
assert np.all(delta['SINGLE_EXPAND_221'][:220]==0)
costdiff=audit['counterfactual_raw_cost']-audit['archived_nn20_raw_cost']
whole=offline['raw_costs']['FALLBACK90']-offline['raw_costs']['NN20']
first_positive=next(i+1 for i,x in enumerate(delta['FALLBACK90']) if x>0 and np.all(delta['FALLBACK90'][i:]>0))
effect_fraction=costdiff/whole
summary=dict(stage='GRID07',engineering_and_saved_data_audit_passed=True,
    native_new_physical_steps=2017,native_new_simulations=24185,model_fits=0,
    same_state_first_decision={'step':221,'old_topology_forecast_rho':first['NN20']['topology_forecast_rho'],
        'expanded_topology_forecast_rho':first['FALLBACK90']['topology_forecast_rho'],
        'old_actual_after_rho':first['NN20']['actual_after_rho'],
        'expanded_actual_after_rho':first['FALLBACK90']['actual_after_rho'],
        'old_native_cost':first['NN20']['raw_operational_cost'],
        'expanded_native_cost':first['FALLBACK90']['raw_operational_cost']},
    first_sustained_positive_full_rule_cumulative_cost_difference_step=first_positive,
    single_intervention_week_relative_cost_change_vs_nn20=audit['relative_cost_change_vs_archived_nn20'],
    first_intervention_extra_cost=costdiff,full_rule_extra_cost=whole,
    arithmetic_ratio_of_extra_costs=effect_fraction,
    ratio_warning='Not a causal share decomposition: later decisions and trajectories interact; no additivity or independent mechanisms inferred.',
    interpretation='The first expansion is cheaper immediately but a single expanded decision followed by original NN20 has higher week cost. Delayed consequences are present in this exposed context. The controller surrogate and cumulative dispatch-state handling deserve cheap non-learning checks before any neural training.',
    training_readiness=False,reason_no_training='One exposed week, archived baseline, no demonstrated legal predictive policy advantage over an economic or restoration baseline, no novelty confirmation.',
    next_work_not_executed='Qualify a bounded safety-checked redispatch restoration baseline using only present observations. Compare to controller-aware candidate evaluation before designing or training a predictor.')
(OUT/'GRID07_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,1,figsize=(9,6.5),sharex=True)
days=np.arange(1,2018)*5/1440
styles={'NN20':('Original NN20','#333333'), 'FALLBACK90':('Full fallback rule','#c44e52'),
    'SINGLE_EXPAND_221':('Expand only at step 221','#4c72b0')}
for p in ['FALLBACK90','SINGLE_EXPAND_221']:
    label,color=styles[p]
    axs[0].plot(days,delta[p]/1e6,label=label,color=color,lw=1.7)
for p,rows in paths.items():
    label,color=styles[p]
    axs[1].plot(days,[r['ledger']['redispatch_mwh']*12 for r in rows],label=label,color=color,lw=1.1,alpha=.8)
for ax in axs:
    ax.axvline(221*5/1440,color='gray',ls=':',lw=1)
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
axs[0].axhline(0,color='black',lw=.7)
axs[0].set_ylabel('Cumulative extra native cost\nvs NN20 (million cost units)')
axs[1].set_ylabel('Actual dispatch L1 (MW)')
axs[1].set_xlabel('Elapsed days of the exposed January week')
fig.suptitle('A locally cheaper expansion can have a more costly continuation')
fig.tight_layout()
fig.savefig(OUT/'january_counterexample.png',dpi=170)
plt.close(fig)

n,f=first['NN20'],first['FALLBACK90'];fees=offline['descriptive_fee_difference']
report=f'''# GRID07：一月反例的离线分析与单次干预诊断

结论：反例不是“第一次扩展搜索立即选了更贵的动作”。在第221步，扩展得到更小的拓扑预测rho，并交付更便宜的当步控制动作；但此后仅继续原NN20策略，整周原生成本仍比原NN20高 **{100*audit['relative_cost_change_vs_archived_nn20']:.4f}%**。整套FALLBACK90高25.4377%。这提供了该已暴露周中的延迟后果证据，不是可部署学习优势。

## 反例的精确定义

场景2035-01-08_5、种子0、相同作者模型和352动作库、相同物理环境及连续控制。NN20只枚举20个神经候选；FALLBACK90在候选不能达到原0.9阈值时补查332个。三条GRID06一月轨迹都完整到2017步且无实际非法动作；原生整周成本分别为NN20 6,707,772.520813、FALLBACK90 8,414,074.452332、NN352 8,494,068.365540。

这不等于“搜索一般有害”。也不能把所有差距归于首个动作。本轮先查描述账目，再做只改变一个决策的有限诊断。

## 第一处分歧：立即看并不贵

前220步动作、状态和成本逐项一致。第221步是2035-01-08 18:25，同一决策前最大rho={n['before_rho']:.9f}。

| 项目 | NN20 | 扩展搜索 |
|---|---:|---:|
| 拓扑候选预测最大rho | {n['topology_forecast_rho']:.9f} | {f['topology_forecast_rho']:.9f} |
| 加连续控制后的实际最大rho | {n['actual_after_rho']:.9f} | {f['actual_after_rho']:.9f} |
| 实际当步原生成本 | {n['raw_operational_cost']:.6f} | {f['raw_operational_cost']:.6f} |
| 实际再调度L1，MW | {n['action_metrics']['actual_dispatch_l1']:.6f} | {f['action_metrics']['actual_dispatch_l1']:.6f} |

原首选动作的动作库行号190被替换为255，预测最大rho只改善约0.001391。两者都仍高于0.9，所以都调用同一个连续优化模块。QP状态均为optimal_inaccurate。扩展方案当步费用少约250.603，甚至到第288步整套回退的累计成本还低8,850.965。此后整套回退的累计差在第{first_positive}步转正并保持为正。

拓扑预测与最终交付动作不是同一个动作。不能用上述预测rho与最终实际rho差值，直接宣称神经模型预测错误；这里的预测来自物理simulate，神经网络只负责候选顺序。

## 晚期费用的描述账目

FALLBACK90减NN20：再调度费用 **+{fees['redispatch_cost']:,.6f}**、损耗费用 **{fees['loss_cost']:,.6f}**、储能费用 **+{fees['storage_cost']:,.6f}**、弃能变化项 **{fees['curtailment_change_cost']:,.6f}**。合计与native差额在沿用的float32容差内一致。主要是较大的持续再调度计费量，不是更多simulate直接计入原生电网成本。

NN20有1855步、FALLBACK90有1844步没有新增再调度指令，但实际再调度状态不为零。各自在这些步的再调度费用约431.76万、603.40万。这个划分是描述性的，不是“这些步可以免费撤销再调度”的证据。

计费边际系数并非常数：观察到140、148、149，NN20与FALLBACK90有140步不同。本分析保留逐步真实系数。不能把总费用差全部解释为等价的能量差。

![累计费用差与再调度状态]({(OUT/'january_counterexample.png').as_posix()})

## 代码揭示的具体局限：控制增量和持续状态是两种量

1. 原拓扑GreedyModule按MaxRhoReward=2−最大rho排序，符合资格后取最大奖励；没有按交付动作的原生运营成本或后续成本排序。
2. 原unsafe QP在DC近似中惩罚当步再调度增量的平方、弃能、储能和线路越限量；原生费用则逐步惩罚actual_dispatch的L1，以及损耗等项。小增量不保证已有再调度状态逐步回到零。
3. Grid2Op的再调度目标会累加指令。发出零增量不等于撤销先前再调度；这在保存向量中可直接观察。
4. 作者代码存在compute_optimum_safe，目标包含恢复dispatch等项；当前LJNAgentTopoNN.act的安全分支只做拓扑恢复，没有调用它，optim.get_act调用的是unsafe求解。不能把“函数存在”当作部署控制器已经做了持续再调度恢复。
5. 神经候选器当前只直接输入186维rho。增加合法控制状态也许有用，但rho可以间接包含相关信息，本轮不证明表达不可辨识，更不证明GNN会利用这些信息获得优势。

这些是已读代码和测得轨迹支持的局限。它们不单独证明哪一条造成全部成本差。尤其unsafe QP的有限迭代返回也未认证为精确最优解。

## 单次干预：把首个决定与后续回退分开

开始前冻结counterfactual_design.json。只有一次新尝试，只运行同一个已暴露周：第1–220步原NN20；第221步使用原FALLBACK90扩展；第222步起恢复原NN20函数与20候选设置，保留各自合法演化的内部控制状态。没有切换后续阈值、模型、连续优化或N−1模块。

工程门槛：前220步对照原NN20保存动作/前后观测哈希完全重现，第221步最终动作/前后观测哈希完全重现原FALLBACK90。通过后才解释后续轨迹。所有实际步账目、向量、native调用与事件计数另行重算。

| 策略 | 完整步数 | 整周原生成本 | 相对原NN20 |
|---|---:|---:|---:|
| 历史NN20 | 2017 | {audit['archived_nn20_raw_cost']:.6f} | 参照 |
| 只在221步扩展，后续NN20 | 2017 | {audit['counterfactual_raw_cost']:.6f} | +{100*audit['relative_cost_change_vs_archived_nn20']:.4f}% |
| 历史整套FALLBACK90 | 2017 | {audit['archived_fallback_raw_cost']:.6f} | +25.4377% |

单次干预差额为{costdiff:,.6f}，整套规则差额为{whole:,.6f}，数值之比为{100*effect_fraction:.2f}%。**这个比例不是因果贡献占比**：后续决策会随状态改变，不能假定干预效应线性可加，也不能把剩余差额归为独立机制。

它支持一个较窄结论：在这次完整重现前缀的诊断中，一次立即较便宜的扩展足以产生较昂贵的原策略续接。它不说明所有扩展或所有场景如此；基线来自历史运行，没有新跑整周同期控制组，后续跨运行的求解器噪声尚未量化。

## 工程和证据边界

单次过程墙钟{audit['process_wall_s']:.3f}秒；2017次实际推进、24,185次native simulate；2017个落盘向量和物理账目通过，最大账目残差{audit['max_ledger_abs_residual']:.9f}，满足原容差。实际输出约40.36MB，资源包络通过。实际非法/歧义/异常步均为0；模拟候选中有{audit['exception_simulation_calls']}次异常反馈，不能把控制台打印的模拟异常当成真实电网已停电。没有重试、训练、安装、使用其他封存场景，也未修改GRID06原轨迹或裁决。

查询进程状态的工具另有一次约298秒才返回；子进程持久记录的实际过程为202.791秒并正常完成。两种时间分开记录，不归因于主机休眠、PyTorch或求解器，也不把工具等待时间冒充算法推理时间。

离线脚本初版误设“边际系数恒149”的断言，读取原始数据后失败。更正为逐步记录140/148/149及其分布，原失败日志保留，未改变数据、策略或诊断设计。该错误前提不是候选方法的负面结果。

## 研究决策：训练门槛仍未达到

这次不支持继续把“是否多模拟332项”直接包装成GRL课题。更具体的未回答问题是：在保持安全的条件下，候选拓扑及持续再调度状态能否被低成本方法一起评价或恢复。

下一项必要检查应先给非学习方法机会：限定现有安全恢复逻辑的合法调用条件，核对物理安全、不可行反馈和完整成本，然后与控制器后续处理一致的候选评价比较。先做接口资格检查，不能直接把未调用过的safe函数接到整周控制器并宣称可用。本轮尚未执行这项新方法。

即使便宜规则失败，仍需证明不同合法观测下有稳定的策略差异、可预测收益能超过强固定/搜索对照，再考虑监督学习；RL还需要额外的多步策略价值证据。本轮只有一个已暴露周及事后选出的干预点，不提供这些结论，也没有论文新意认证。

代码依据：work/grid02/ljn_nn/modules/base_module.py、rewards.py、convex_optim.py、LJNAgent.py、make_agent.py；安装的Grid2Op action/baseEnv与L2RPNSandBoxScore定义。正式执行源码与复用作者文件哈希保持不变。当前模型是作者MLP，不是我们的GNN或GRL。

用户要求：常规结果保存，不为每次小实验询问确认；仅工程受阻或准备训练时通知。当前工程通过，但还未准备训练。没有后台训练或自动化。
'''
(OUT/'GRID07_report.md').write_text(report,encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('# GRID07\n\nCOUNTEREXAMPLE_ANALYSIS_COMPLETE: offline reconstruction plus one first-decision intervention audited. Engineering passed; training readiness NOT established.\nNo pending processes, retries, training, installations or automation. Next non-learning qualification described, not executed.\n',encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
