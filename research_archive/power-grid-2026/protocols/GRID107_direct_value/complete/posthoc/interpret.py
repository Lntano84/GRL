"""Postfit descriptive loss statistics and calibrated interpretation; no further fit."""
import inspect,shutil
import numpy as np
import torch.nn.functional as F
from common107 import ROOT,OUT,REPO,ARCHIVE,read,save,sha
def main():
    assert not (OUT/'overall_interpretation.json').exists()
    with np.load(OUT/'train.npz') as z:y=z['target'].astype(float);w=z['sampling_probability']
    # Descriptive weighted Huber location: score equation, not a deployed or evaluated model.
    low,high=float(y.min()),float(y.max())
    for _ in range(100):
        mid=(low+high)/2
        if float(np.dot(w,np.clip(mid-y,-1,1)))<0:low=mid
        else:high=mid
    center=(low+high)/2;mean=float(w@y)
    huber=lambda c:float(w@np.where(np.abs(c-y)<1,.5*(c-y)**2,np.abs(c-y)-.5))
    assert abs(float(w@np.clip(center-y,-1,1)))<1e-12 and huber(center)<=huber(mean)
    d=read(OUT/'decision.json');m=read(OUT/'state_metrics.json');probes=read(OUT/'critical_probes.json')
    last={}
    for seed in [0,1,2]:
        t=read(OUT/f'trace_seed{seed}.json');a=next(r for r in t if r['update']==5000);b=t[-1]
        last[str(seed)]=dict(loss5000=a['train_weighted_huber'],loss7250=b['train_weighted_huber'],
            relative_loss_drop=(a['train_weighted_huber']-b['train_weighted_huber'])/a['train_weighted_huber'])
    stats=dict(weighted_target_mean=mean,weighted_huber_location=center,weighted_positive_mass=float(w[y>0].sum()),
        weighted_target_above_one_mass=float(w[y>1].sum()),score_residual=float(w@np.clip(center-y,-1,1)),
        scope='All training next-net labels under prespecified sampling masses, one-dimensional descriptive locations. Not an estimate conditioned on552 features, not another trained predictor.',
        derivation='For beta1 Huber, derivative is sum(weight*clip(prediction-target,-1,1)); squared loss minimum is weighted mean.',
        official_formula='https://docs.pytorch.org/docs/2.8/generated/torch.nn.SmoothL1Loss.html',
        installed_function_source=inspect.getsourcefile(F.smooth_l1_loss),installed_source_sha256=sha(__import__('pathlib').Path(inspect.getsourcefile(F.smooth_l1_loss))))
    save(OUT/'loss_shape_diagnostic.json',stats)
    direct=d['all_seed_means']['DIRECT'];old=d['all_seed_means']['DQN0'];ratio=(direct['mean_net_utility']/old['mean_net_utility']-1)
    conclusion=dict(overall_value_or_quality_advantage_established=False,critical552_misses_fixed=False,
        direct_mean_utility=direct['mean_net_utility'],dqn_mean_utility=old['mean_net_utility'],relative_utility_change=ratio,
        prespecified_weak_condition=d['training_mechanism_evidence_at_fixed_config'],
        why_not_accept='Two paired seeds improve but third loses more; family mean does not improve, critical552 action pattern unchanged0/0/1.',
        train_loss_plateau_established=False,loss_trace=last,
        loss_shape_is_next_hypothesis_not_proven_cause=True,additional_model_fits=0,automatic_followup=False)
    save(OUT/'overall_interpretation.json',conclusion)
    lines=['# GRID107训练结果：关键漏查尚未解决','',
        '**这轮对照没有解决第552步的关键错误，也没有建立比DQN更好的平均收益。** 它完成了3个固定种子，保留了所有结果，没有选最优模型。','',
        '## 实际改善与没有改善的部分','',
        '训练集中211个正净收益状态，三个直接回归模型分别漏查30、27、38个，原DQN为45、54、59个。误停数量有所减少，但并不是所有价值误差都改善：三个新模型的继续价值RMSE均高于原DQN。',
        '验证集中41个正净收益状态，新模型漏查13、14、13个，原DQN为15、7、17个。平均漏查数没有降低，不能用训练集的改善代替泛化成绩。',
        f"四日期宏平均净效用，三个种子的平均值：直接回归{direct['mean_net_utility']:.8f}，原DQN{old['mean_net_utility']:.8f}，变化{ratio:+.3%}；固定一批为{d['batch1_utility']:.8f}。与固定一批的平均差非常小，不作实质优势判断。",
        f"即时增益覆盖从{old['gain_capture']:.2%}变为{direct['gain_capture']:.2%}，查询比例从{old['query_fraction']:.2%}变为{direct['query_fraction']:.2%}。覆盖更多增益仍没有带来更高平均净效用。",
        '预设的较弱诊断条件确实通过：训练日期宏平均符号遗憾下降、2/3配对种子的验证净效用上升。但第三种子的下降幅度更大，使全部种子均值略降。因此，该条件只保留为局部训练机制线索，不当作模型接受或扩大资格。','',
        '## 第552步仍然没有修好','',
        '原FRAME路径和RF路径的第552步，三个新模型都仍为“停止、停止、继续”，与原DQN一样。它们没有使用这些案例的标签训练，也没有针对其调整费用。第108步则三个模型都继续，未学会此前单次干预中更省成本和查询的取消。',
        '这次所有3582个训练状态及全部正收益状态均实际被抽到过，但固定更新预算下仍有明显误判。可以排除“只要同预算完整监督就能解决”的简单解释；不能由此排除优化未充分、特征信息不足或训练覆盖不足。尤其关键状态最相近的训练调用多数无收益，完整访问旧数据并不会创造缺少的区分信息。','',
        '## 损失函数与训练充分性的新线索','',
        '本轮为了公平，沿用了原DQN的Huber/SmoothL1损失。该损失对小误差使用平方项，对大误差改为线性项，减弱大误差的影响。[PyTorch 2.8官方公式](https://docs.pytorch.org/docs/2.8/generated/torch.nn.SmoothL1Loss.html)。',
        f"仅描述原训练标签及固定抽样权重：正净收益权重约{stats['weighted_positive_mass']:.2%}，大于1的归一化净收益权重约{stats['weighted_target_above_one_mass']:.2%}；加权均值为{mean:+.6f}，同一标签分布的Huber中心却为{center:+.6f}。",
        '由官方分段公式可推导：平方损失的常数最优值是均值，Huber中心满足加权截断残差和为0。上述符号差说明，偏斜的少数高收益标签下，稳健中心和期望收益可能方向不同。这是总体标签的描述性统计，不是第552步条件收益的估计，更没有证明Huber就是该处漏判的原因。',
        '训练末段也未显示可以认定的收敛：从5000到7250更新，三个种子的训练加权损失仍下降约'+ '/'.join(f"{last[str(s)]['relative_loss_drop']:.1%}" for s in [0,1,2])+'。因此，既不能宣称再训练没用，也不能把失败全归给特征。按照原方案，本轮没有追加更新或选择更有利的中间检查点。','',
        '## 下一步建议','',
        '先保持这轮结果，不扩大数据或网络。更值得做的是把损失形式与训练充分性分开检查：若下一阶段比较损失，只改变这一项，沿用固定数据、种子、采样和更新预算；若研究收敛，另固定只看训练侧的停止规则。不能同时换损失、延长训练、加特征，再将改善归给某一个机制。',
        '本轮没有自动执行这些新实验，也没有启动闭环。最终测试仍未生成；三处已暴露上下文不是新增独立测试。']
    (OUT/'训练结果解读_中文.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    dest=ARCHIVE/'complete';post=dest/'posthoc';post.mkdir(exist_ok=False)
    shutil.copy2(__import__('pathlib').Path(__file__),post/'interpret.py')
    for name in ['loss_shape_diagnostic.json','overall_interpretation.json','训练结果解读_中文.md']:
        shutil.copy2(OUT/name,dest/name)
    collab=__import__('pathlib').Path(read(OUT/'complete.json')['collaboration'])
    for name in ['训练结果解读_中文.md','overall_interpretation.json','loss_shape_diagnostic.json']:shutil.copy2(OUT/name,collab/name)
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='archive_manifest.json'})
    note='\n\n## 2026-10-22 — GRID107 calibrated interpretation\n\nTraining positive misses improve30/27/38 versus DQN45/54/59, but validation misses13/14/13 versus15/7/17. Family macro utility direct0.16292381 vs DQN0.16319774 (-0.168%), BATCH1 0.16286891; critical552 action pattern remains0/0/1, no rescue. Prespecified weak2-of3 condition=True is not acceptance: third seed decline outweighs gains, no overall advantage or scaling. All training states were sampled. Loss still decreases over final2250 updates; convergence not established. Training-label descriptive weighted mean+0.030487 vs Huber location-0.008537 motivates checking robust loss under skew gains, not proof of conditional552 cause. No extra fits/physical rollouts/price tuning/final-test materialization. Further loss or convergence studies must be separate controlled designs, not automatic continuation.\n'
    for n in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write(note)
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID107完成，关键552仍未修复\n\n1. 向用户汇报：训练漏查减少，验证均值未提高，关键552仍0/0/1；不按最优种子或弱2/3条件宣称接受。\n2. 后续考虑分开检查损失形式与训练充分性；不同时改多因素，不自动追训旧模型/新加数据或物理闭环。\n3. 最终8测试场景保持封存，不扩大网络/日期/电网。\n\n解读：research_archive/power-grid-2026/protocols/GRID107_direct_value/complete/训练结果解读_中文.md\n',encoding='utf8')
    print(conclusion,flush=True)
if __name__=='__main__':main()
