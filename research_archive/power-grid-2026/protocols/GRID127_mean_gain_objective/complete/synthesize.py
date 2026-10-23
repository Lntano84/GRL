"""Conserved two-stage synthesis from audited endpoints only."""
from common127 import *

def main():
    assert not (O/'GRID126_127_阶段汇总.md').exists();verify();assert read(O/'review.json')['passed'];assert read(BASE/'review.json')['passed'];labels=[('NET_ZERO','原净收益回归',O),('GAIN_DIRECT','直接预测改进量',BASE),('COST_POSITIVE','非负链接＋Huber',BASE),('POS_MSE','非负链接＋MSE',O),('POS_KL','非负链接＋广义KL',O)];table={}
    for arm,label,path in labels:
        table[label]={}
        for dom in ['full_train','validation','oof']:
            rr=read(path/f'{dom}_results.json');table[label][dom]={k:float(np.mean([rr[f'{arm}_seed{s}']['all']['macro'][k] for s in range(3)])) for k in ['mean_net_utility','query_fraction','gain_capture','physical_call_balanced_utility','mean_queries','mean_gain']}
    M=read(O/'state_metrics.json');fold=[]
    for f in range(4):
        for seed in range(3):fold.append(M[f'f{f}__POS_KL_seed{seed}']['fit']['sign_regret']-M[f'f{f}__NET_ZERO_seed{seed}']['fit']['sign_regret'])
    old=float(np.mean([M[f'full__NET_ZERO_seed{s}']['train']['sign_regret'] for s in range(3)]));new=float(np.mean([M[f'full__POS_KL_seed{s}']['train']['sign_regret'] for s in range(3)]));changes={dom:dict(net_relative=table['非负链接＋广义KL'][dom]['mean_net_utility']/table['原净收益回归'][dom]['mean_net_utility']-1,query_fraction_relative=table['非负链接＋广义KL'][dom]['query_fraction']/table['原净收益回归'][dom]['query_fraction']-1,gain_capture_difference=table['非负链接＋广义KL'][dom]['gain_capture']-table['原净收益回归'][dom]['gain_capture']) for dom in ['full_train','validation','oof']}
    save(O/'two_stage_synthesis.json',dict(metrics=table,KL_training_sign_regret_reduction=1-new/old,KL_fold_training_regret_improves=sum(x<0 for x in fold),fold_pairs=12,KL_changes=changes,new_fits_this_user_turn=60,untrained_original_model=False,new_RL_fits=0,new_physical_calls=0))
    lines=['# GRID126—127：连续两组收益学习的阶段汇总','',
      '本次授权下连续完成两组、共60次新训练；两个阶段共用原15个净收益回归控制。GRID126产生的15个非负Huber模型在GRID127直接复用，没有重复训练。所有种子和反向日期保留，没有训练强化学习或扩大数据。','',
      '## 完整局部补查结果','',
      '三个种子平均。净效用越高越好；查询比例和收益保留率先按日期计算，再宏平均。下表只描述固定FULL教师路径上的局部搜索，不是整周经济成本或真实计算时间。','',
      '| 方法 | 原验证净效用 | 查询比例 | 收益保留 | 折外净效用 | 查询比例 | 收益保留 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for label,doms in table.items():
        a,b=doms['validation'],doms['oof'];lines.append(f'| {label} | {a["mean_net_utility"]:.8f} | {100*a["query_fraction"]:.2f}% | {100*a["gain_capture"]:.2f}% | {b["mean_net_utility"]:.8f} | {100*b["query_fraction"]:.2f}% | {100*b["gain_capture"]:.2f}% |')
    lines+=['','## 这两组分别回答了什么','',
      '第一组否定了“只把收益变为非负就会改善”的预期。已知费用外置的直接回归略低于原方法；非负链接配Huber更偏向停止，节省了很多查询，同时漏掉较多收益。它提供了可检查的梯度敏感性问题，没有支持采用这一配置。',
      '第二组保持相同链接，直接检验均值目标。KL在原验证日期上相对非负Huber、MSE的三个种子全部更好，四日期全部更好；折外相对两者三个种子均更好，但日期方向仍有反例，所以严格特异条件仍未通过。',
      f'训练侧，KL相对原回归的单状态决策遗憾下降{100*(1-new/old):.2f}%，四折训练的12个配对模型全部下降；完整训练搜索效用从{table["原净收益回归"]["full_train"]["mean_net_utility"]:.8f}提高到{table["非负链接＋广义KL"]["full_train"]["mean_net_utility"]:.8f}。这说明它能在训练日期上学到有代价的继续/停止选择，不应再把问题笼统叫作“完全学不会”。',
      f'但到原验证日期，KL相对原回归净效用变化{100*changes["validation"]["net_relative"]:.2f}%；折外变化{100*changes["oof"]["net_relative"]:.2f}%。查询比例分别相对降低{100*(-changes["validation"]["query_fraction_relative"]):.2f}%和{100*(-changes["oof"]["query_fraction_relative"]):.2f}%，收益保留率却分别减少{100*(-changes["validation"]["gain_capture_difference"]):.2f}和{100*(-changes["oof"]["gain_capture_difference"]):.2f}个百分点。它形成计算—收益取舍，尚未形成同质量优势。',
      '相对原回归，KL原验证三个种子中仅一个净效用为正，折外三个种子全部为负。不能选这个单种子替代原模型，也不能因查询少就声称完整控制更快。','',
      '## 当前判断','',
      '**保留原净收益回归。KL是一个有训练机制证据、但尚无稳定跨日期收益的开发方向。** 对现有数据，继续只换损失或把网络加大，已经缺少足够理由。',
      '下一步研究重点应转向：训练日期上学到的高收益机会，为什么不能迁移到留出日期。需要区分输入摘要/结构反馈的可迁移性、稀有收益训练支持，以及教师路径与真实控制分布；当前结果不能单独证明其中任何一项是唯一原因。',
      '这不是要求立即增加场景。任何数据扩大应另设固定方案，保持最终测试封存；在此之前，先用现有公开结构与逐调用记录定位训练/留出差距，不重复已经否定的纯损失替换。',
      '数学上的均值一致性、训练侧改善、局部搜索效用、物理整周质量是四个不同层次，本轮完成前两个层次的部分机制核查和第三层的反向检验，没有执行第四层。','',
      '## 可复核材料','',
      '- [GRID126完整报告](https://github.com/Lntano84/GRL/blob/main/research_archive/power-grid-2026/protocols/GRID126_known_cost_gain/complete/GRID126_report.md)：收益/费用外置与非负链接。',
      '- [GRID127完整报告](GRID127_report.md)：相同非负链接下的MSE/KL目标，全部种子、日期和查询/收益分解。',
      '- 两阶段review.json、权重、曲线、抽样曝光、predictions.npz、全部局部搜索轨迹及首次分歧记账均已保留。',
      '- 同网已暴露开发数据，重复前缀/种子不是新独立场景；历史费用/尺度的全训练日期依赖不变。']
    (O/'GRID126_127_阶段汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('TWO_STAGE_SYNTHESIS_SAVED',changes,flush=True)

if __name__=='__main__':main()
