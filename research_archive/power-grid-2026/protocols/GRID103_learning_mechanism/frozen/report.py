"""Render all prespecified results without fitting, tuning, or selecting a seed."""
from common import OUT,read

def main():
    assert read(OUT/'review.json')['passed']
    R=read(OUT/'validation_results.json');D=read(OUT/'decision.json');P=read(OUT/'preflight.json')
    lines=['# GRID103：学习补查的收益预测与后续收益机制','',
      '本轮在既有训练/验证反馈上，检查DQN的优势来自哪里。新增一个下一批收益回归对照，并将DQN折扣因子从1改为0，其他网络、种子和训练步数保持一致。原三个DQN不重训。没有新增电网物理回放。',
      '', '## 设置与比较含义','',
      '训练2378个起始前缀、16日期；验证590个前缀、4日期。单批448、多批142，最多4批。前缀、同站范围、候选顺序、每批最多64和41项可见特征不变。所有日期内前缀相关；同时报告物理搜索调用等权的敏感性结果。',
      '', 'MYOPIC_RF64使用64棵深度4随机森林预测下一批即时增益，沿用既有森林规模，不选择参数。预测收益超过同一固定查询费用就继续，否则停止。训练目标是收益大小，没有将少数有收益事件统一压成一个类别。',
      '', 'DQN_G1沿用GRID102三个最终模型；DQN_G0每个种子各训练30000步，两层64节点、同一训练分区和探索设置，只取消后续奖励项。两者的实际探索轨迹会随策略分歧而不同，并不具有相同回放缓冲区。主评价统一使用未折扣的累计净效用。这个对照检验后续奖励训练是否有帮助，不代表比较所有强化学习算法。',
      '', '查询单价1.244234864e-05，训练增益尺度0.008627818251，均继承GRID102。价格是训练数据设定的代理，不是秒数或货币。不调整价格、阈值或模型来改善本轮成绩。',
      '', 'BATCH1固定查一批后停止；STOP不补查；FULL_LOCAL查完整站剩余候选。ORACLE_OPTIMAL事后选累计净效用最大的停止点；ORACLE_MYOPIC事先知道下一批真实增益，但只在该批净收益为正时继续。两个ORACLE仅用于诊断上界与跨批机会，不能部署。',
      '', '## 全部前缀：四日期宏平均','',
      '| 方法 | 净效用 | 查询占完整补查 | 即时增益覆盖 | 物理调用等权净效用 |',
      '|---|---:|---:|---:|---:|']
    methods=['STOP','FULL_LOCAL','BATCH1','COST_LOGISTIC','MYOPIC_RF64']+[f'DQN_G{g}_{s}' for g in [1,0] for s in [0,1,2]]+['ORACLE_MYOPIC','ORACLE_OPTIMAL']
    for m in methods:
        r=R[m]['all']['macro'];lines.append(f"| {m} | {r['mean_net_utility']:.6f} | {r['query_fraction']:.2%} | {r['gain_capture']:.2%} | {r['group_balanced_utility']:.6f} |")
    for g in [1,0]:
        r=D['seed_means'][f'DQN_G{g}_MEAN']['all'];lines.append(f"| DQN_G{g}三种子平均 | {r['mean_net_utility']:.6f} | {r['query_fraction']:.2%} | {r['gain_capture']:.2%} | {r['group_balanced_utility']:.6f} |")
    lines+=['','所有查询/增益比例先按日期计算再平均，不能与直接合并所有记录的比值混用。净效用是奖励增量减查询代理费用后除以固定尺度；不是整周运行成本。','','## 单批与多批','',
      '| 分组 | 前缀数 | FULL_LOCAL | MYOPIC_RF64 | DQN_G1三种子 | DQN_G0三种子 | 最优事后上界 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for st in ['single','multi','horizon2','horizon3','horizon4']:
        lines.append(f"| {st} | {R['FULL_LOCAL'][st]['cases']} | {R['FULL_LOCAL'][st]['macro']['mean_net_utility']:.6f} | {R['MYOPIC_RF64'][st]['macro']['mean_net_utility']:.6f} | {D['seed_means']['DQN_G1_MEAN'][st]['mean_net_utility']:.6f} | {D['seed_means']['DQN_G0_MEAN'][st]['mean_net_utility']:.6f} | {R['ORACLE_OPTIMAL'][st]['macro']['mean_net_utility']:.6f} |")
    contrast=D['gamma1_minus_gamma0']['multi']
    lines+=['','## 后续收益对照判读','',f"多批前缀上，G1减G0的平均净效用差为{contrast['mean_delta']:.6f}，{contrast['positive_seeds']}/3个配对种子同向。"]
    lines.append('本轮满足预先规定的探索性同向条件；仍须检查逐日期反例，不能据此声称已证实普遍的长期规划优势。' if D['sequential_support_exploratory'] else '本轮未满足预先规定的探索性同向条件，不能声称DQN通过考虑后续批次建立了优势。')
    for r in contrast['seeds']:
        lines.append(f"- 种子{r['seed']}：净效用差{r['delta_net']:.6f}，物理调用等权差{r['delta_group_balanced_net']:.6f}；各日期差值："+'；'.join(f'{d} {v:+.6f}' for d,v in r['per_date'].items())+'。')
    delayed=D['delayed_opportunities']
    lines+=['',f"事后观察到下一批净收益非正、但后续某个停止点总净收益为正的状态共{delayed['states']}个，来自{delayed['physical_groups']}个物理调用、{len(delayed['dates'])}个日期。它们只是相关前缀中的机会，不是独立样本，也不说明可见特征足以预测它们。"]
    for st in ['all','multi']:
        gap=R['ORACLE_OPTIMAL'][st]['macro']['mean_net_utility']-R['ORACLE_MYOPIC'][st]['macro']['mean_net_utility']
        lines.append(f'- {st}：完整事后停止上界减下一批事后贪心的净效用差为{gap:.6f}。')
    lines+=['','## 收益预测是否准确','', '| 模型 | 下一批收益MAE | 下一批收益RMSE |','|---|---:|---:|']
    for m,e in D['regression_errors'].items():lines.append(f"| {m} | {e['macro']['mae']:.6f} | {e['macro']['rmse']:.6f} |")
    lines+=['','这里收益按固定增益尺度标准化，误差在每日期所有可达状态计算再平均。预测误差与策略效用必须同时看，不能只凭一个指标判断机制成立。','','## 10月日期失败分解','',
      '| 方法 | 负净效用前缀 | 平均增益 | 平均查询费用 | 净效用 |','|---|---:|---:|---:|---:|']
    for m in ['STOP','FULL_LOCAL','MYOPIC_RF64']+[f'DQN_G{g}_{s}' for g in [1,0] for s in [0,1,2]]:
        r=D['october_failure'][m];lines.append(f"| {m} | {r['negative_cases']}/{r['cases']} | {r['mean_normalized_gain']:.6f} | {r['mean_normalized_query_cost']:.6f} | {r['mean_net']:.6f} |")
    lines+=['','增益与费用使用同一尺度，净效用为两者之差。负数表示实际补查收益没有覆盖本轮查询价格；并不等于整周运行恶化。详细错停、无收益继续与逐状态Q值保存在decision.json、state_predictions.json和visited_decisions.json。',
      '', '## 核验与边界','',
      f"独立重建{P['reconstructed_states']}个可达状态，逐一扰动未查询反馈并复算下一批增益；原GRID102的3540条STOP/FULL/LR/DQN轨迹重现。新拟合四次：一个随机森林、三个G0 DQN；没有重训G1或选择最好种子。保存轨迹、预测和宏平均核验见review.json。",
      '', '本轮是已经暴露的四日期开发分析，不是独立确认，也不根据结果推迟报告不利种子。事后上界是特权诊断，不能作为可部署输入。所有观察仍来自FULL教师路径，接入FRAME后会遇到不同状态；即时搜索收益不等于整周收益。',
      '', '未增加网络、训练步数或数据规模；未扫描费用/预算/特征/阈值；无新物理仿真、安装、最终测试生成或论文修改。完成后交用户决定是否进入开发场景闭环，不自动启动下一轮。']
    (OUT/'GRID103_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__':main()
