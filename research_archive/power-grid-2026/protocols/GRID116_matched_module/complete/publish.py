"""Archive frozen execution or audited all-seed results; no research fitting."""
import shutil
from common116 import *
def archive(dest,final):
    dest.mkdir(parents=True,exist_ok=False)
    allowed={'design.json','preflight.json','code_freeze.json','data_freeze.json','state.json'}
    for p in O.rglob('*'):
        if not p.is_file() or p.suffix=='.tmp' or (not final and p.name not in allowed):continue
        d=dest/p.relative_to(O);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
    for p in W.glob('*.py'):shutil.copy2(p,dest/p.name)
    for src,name in [('work/grid115_matched_module_plan/matched_model.py','matched_model115.py'),('work/grid113_relational_learning/model.py','relational_model113.py')]:shutil.copy2(R/src,dest/name)
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})
def docs(note,nexttext):
    for name in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\n'+note+'\n')
    (REPO/'docs/NEXT_STEPS.md').write_text(nexttext+'\n',encoding='utf8')
def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['frozen','complete'],required=True);phase=parser.parse_args().phase
    verify();base=REPO/'research_archive/power-grid-2026/protocols/GRID116_matched_module'
    if phase=='frozen':
        assert not (O/'training_started.json').exists();archive(base/'frozen',False)
        docs('## 2026-10-23 — GRID116 matched module training authorized and frozen\n\nUser authorizes matched GRID115 six fits. Preserve original6978parameter41-64-64-2 backbone plus5938correction,12916eachJOINT/FACTOR,alltrainfromscratch,zero final correction layer,paired old base/full initialization/indexstreams. Synthetic backward finite,base/finalcorrection gradients nonzero and initial upstreamzero expected,no optimizer updates. All4438 targets/known inputs/sourcefreezes verified. Exactly6serial7250fits,32states,originalHuber/Adam/fee/splits,1800s total. First twoactualsteps count insidebudget;checkupstreamflowafterfirst. OldBASEresults reused,no repeatedfits. Training/validation/test boundaries and alladverseresults preserved;noautophysical/scaling.',
            '# Next steps — GRID116六次训练已获准封存\n\n1. 推送封存后仅执行既定6次7250更新，前两步计入预算并核验梯度流。\n2. 全部六次完成后统一离线完整序列评价、配对敏感性与审计，保留全部种子/日期。\n3. 不变损失/费用/种子/输入/预算，不自动扩大、闭环或最终测试。')
        print('FROZEN_ARCHIVED',flush=True);return
    assert read(O/'review.json')['passed'];assert not (O/'complete.json').exists();D=read(O/'decision.json');F=read(O/'fits.json');vr=read(O/'validation_results.json');tr=read(O/'train_results.json');M=read(O/'state_metrics.json');B=read(O/'binding_sensitivity.json')
    family_train={k:float(np.mean([tr[fmt.format(s)]['all']['macro']['mean_net_utility'] for s in range(3)])) for k,fmt in [('JOINT','BASE_JOINT_seed{}'),('FACTOR','BASE_FACTOR_seed{}'),('BASE','DIRECT{}')]}
    lines=['# GRID116：保留原主干后的配对模块训练','',
        f"**六次固定训练和离线审计完成。配对机制判据：{'达到' if D['mechanism_signal'] else '未达到'}；配对模块平均净效用超过原BASE：{'是' if D['incremental_over_base'] else '否'}；同时超过全部竞争参照并满足机制判据：{'是' if D['competitive_candidate'] else '否'}。**",'',
        '本轮按GRID115方案，保留原41→64→64→2主干，在其两项价值上增加配对模块JOINT或先汇总模块FACTOR的修正。两组均12916参数，初始张量逐项一致，修正输出层初始化为0，起始输出与原BASE相同。全部从随机初始化联合训练，没有旧模型热启动或额外预训练。',
        f"新增两组各三个种子0/1/2，共六次；每次7250更新、32状态/64标量目标、原Huber/Adam/抽样序列/费用/标签。总训练墙钟{F['wall_s']:.1f}秒，原上限1800秒。首两步计入预算，上游梯度在第一步后开始流动，没有额外热身。BASE直接复用GRID107三个固定终点。",'',
        '训练仍为3582状态、2378起始前缀、32场景/16日期；验证856状态、590起始前缀、8场景/4日期。全部来自同一电网，验证日期已暴露，相关前缀不算独立场景。最终8测试场景未生成。','',
        '## 完整补查序列的主结果','',
        '净效用为获得的奖励增量减去实际查询费用，再除以原收益尺度；先按每个日期的起始前缀平均，再对日期平均。此处是保存教师路径上的离线补查停止决策，不是整周运行成本。','']
    for label in ['JOINT','FACTOR','BASE','DQN']:lines.append(f"- {label}三个种子平均验证净效用：{D['family_mean_utility'][label]:.12f}。")
    lines += [f"- 固定一批BATCH1：{D['batch1_utility']:.12f}。",'']
    for a in ['BASE_JOINT','BASE_FACTOR']:
        lines.append(a+'逐种子结果：')
        for s in range(3):
            name=f'{a}_seed{s}';v=vr[name]['all']['macro'];m=M[name]['validation']
            lines.append(f"- 种子{s}：净效用{v['mean_net_utility']:.12f}，物理调用均衡净效用{v['physical_call_balanced_utility']:.12f}，收益保留率{v['gain_capture']*100:.3f}%，查询比例{v['query_fraction']*100:.3f}%，正收益漏停{m['false_stop']}/41，无利继续{m['unprofitable_continue']}。")
        lines.append('')
    lines += ['## 匹配对照与预设判读','']
    for p in D['paired_seeds']:lines.append(f"- 种子{p['seed']}：JOINT−FACTOR={p['joint_minus_factor']:.12f}，JOINT−原BASE={p['joint_minus_base']:.12f}，FACTOR−原BASE={p['factor_minus_base']:.12f}。")
    lines.append('')
    for date,delta in D['date_joint_minus_factor'].items():lines.append(f'- {date}：三种子平均JOINT−FACTOR={delta:.12f}。')
    lines += ['',
        '配对机制信号事前要求家族均值更高、三个配对种子都同向、至少3/4日期同向。仅高于FACTOR却未超过原BASE，不称为有用增强；若两种新增模块接近，只能说新增模块的整体作用，不能单独归因于配对关系。',
        'JOINT与FACTOR在参数、初始化、原始信息、更新和目标曝光上匹配。原BASE仅6978参数，故对BASE的比较检验增加模块后的实际价值，并非同参数的因果证明。相同更新次数也不等于相同FLOPs或在线时延。','',
        '## 训练与验证取舍','',
        f"完整训练序列三种子平均净效用：JOINT {family_train['JOINT']:.12f}，FACTOR {family_train['FACTOR']:.12f}，BASE {family_train['BASE']:.12f}。",'']
    for a in ['BASE_JOINT','BASE_FACTOR']:
        lines.append(f"- {a}训练正收益漏停：{[M[f'{a}_seed{s}']['train']['false_stop'] for s in range(3)]}/211；原抽样加权Huber：{[round(M[f'{a}_seed{s}']['train']['sampling_weighted_huber'],9) for s in range(3)]}。")
    lines += ['',
        '这些数值与完整序列收益一起判读，不能把更低回归误差或更少查询单独当成学习成功。全部单批/多批、逐日期、逐种子、输入状态预测与逐序列查询均保留。','',
        '## 配对敏感性与固定难例','',
        '沿用事前固定的一次循环移位：只错开同站已查动作与反馈的对应关系，原41维、动作集合、反馈多重集和待查集合保持不变。此处仅为已训练权重的输入敏感性，不训练扰动模型、不将结果挑成部署策略。','']
    for a in ['BASE_JOINT','BASE_FACTOR']:
        lines.append(f"- {a}正训练/验证状态决策变化数：{[(B[f'{a}_seed{s}']['train']['positive_sign_changes'],B[f'{a}_seed{s}']['validation']['positive_sign_changes']) for s in range(3)]}；正验证状态优势绝对变化中位数{[round(B[f'{a}_seed{s}']['validation']['positive_median_abs_change'],8) for s in range(3)]}。")
    lines.append('')
    for p in read(O/'critical_probes.json'):
        lines.append(f"- {p['context']}，已知下一批归一化净收益{p['known_batch_normalized_net']:.8f}，JOINT继续决策{[p['scores'][f'BASE_JOINT_seed{s}']['action'] for s in range(3)]}，FACTOR{[p['scores'][f'BASE_FACTOR_seed{s}']['action'] for s in range(3)]}（1继续，0停止）。")
    lines += ['',
        '难例此前均已暴露，仅用于解释，不用于训练加权、挑种子或整轮接受。一次配对扰动也不能证明全部条件依赖或其因果价值。','',
        '## 审计与结论边界','',
        '全部4438目标、原抽样索引流和逐状态计数、六份Adam7250步收据、首两步梯度与冻结哈希通过核对。独立NumPy实现分别计算原主干与修正模块再相加，核对全部正收益状态和固定间隔状态及三处难例；所有训练/验证完整序列重新计费。详见review.json。',
        '保持原主干结构和初始输出，不保证训练后策略不退步。当前仍优化原即时净收益Huber目标，GRID114发现的损失与决策错位没有在本轮同时修改。新模型尚未进入真实控制，特征构造和推理的完整控制开销未经本轮验证，不能声称实际提速、整周质量或跨网泛化。',
        '本轮不根据中间成绩调整模型、目标或阈值，不追加种子或训练时长；全部不利结果保留。后续是否继续学习、扩大规模或做闭环，由用户根据本轮完整结果决定。']
    report=O/'GRID116_report.md';report.write_text('\n'.join(lines)+'\n',encoding='utf8')
    collab=R/'论文协作资料/FRAME_保留主干模块结果_GRID116';collab.mkdir(exist_ok=False)
    save(O/'complete.json',dict(passed=True,experiment_completed=True,new_fits=6,report=str(report),archive=str(base/'complete'),collaboration=str(collab),
        mechanism_signal=D['mechanism_signal'],competitive_candidate=D['competitive_candidate'],ready_to_scale=False,new_physical_calls=0,final_test_materialized=False))
    archive(base/'complete',True)
    for n in ['GRID116_report.md','decision.json','review.json','complete.json','validation_results.json','train_results.json','state_metrics.json','binding_sensitivity.json','critical_probes.json']:shutil.copy2(O/n,collab/n)
    note=f"## 2026-10-23 — GRID116 matched preserved-backbone module experiment complete\n\nExactly6fresh12916parameterfits,original64widebackbone retained,paired zero-correction initialization,7250x32/Huber/Adam/fee/oldindexstreams,first2stepsincluded/upstreamflowqualified. Totaltraining{F['wall_s']:.1f}s. Validation familyJOINT/FACTOR/BASE/DQN={D['family_mean_utility']},BATCH1={D['batch1_utility']:.12f}.PairedseedJF{[p['joint_minus_factor'] for p in D['paired_seeds']]},dateJF{D['date_joint_minus_factor']}.Mechanismgate{D['mechanism_signal']},incrementoverBASE{D['incremental_over_base']},competitivecandidate{D['competitive_candidate']}.Train family{family_train}.All adverse seeds/dates preserved.IndependentNumPybase+delta/targets/trajectories/optimizer/sampling/freezeauditpassed.Onlyofflineexposedteacherpath,notphysicalquality/time/crossgridproof.Noautomatictraining/scaling/replay/test."
    docs(note,'# Next steps — GRID116固定训练与审计完成\n\n1. 汇报全部种子/日期及BASE、FACTOR、BATCH1、DQN参照，按预设机制判据与实际增量分开判读。\n2. 不自动改损失/阈值/数据或追加训练，不扩大、闭环或读取最终测试。\n\n报告：research_archive/power-grid-2026/protocols/GRID116_matched_module/complete/GRID116_report.md')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID116保留原主干的模块训练](protocols/GRID116_matched_module/complete/GRID116_report.md)：固定六次JOINT/FACTOR训练、全部种子与原BASE参照、完整序列评价及审计。\n')
    save(O/'state.json',dict(stage='published_waiting_remote_verification',fits=6));print('PUBLISHED',flush=True)
if __name__=='__main__':main()
