"""Finish read-only score decomposition and preserve all effective sources."""
from pathlib import Path
import collections,json,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
def main():
    C.verify();assert C.read(O/'analysis_complete.json')['passed'] and not (O/'complete.json').exists()
    rows=C.read(O/'rows.json');changes=[r for r in rows if r['changed']];decomp=[]
    for r in changes:
        old,new=r['arms']['STOP_ONLY'],r['arms']['MIXED'];d0=old['endpoint']-r['b'];d1=new['endpoint']-r['b'];a0,a1=old['alpha'],new['alpha']
        direction=.5*(a0+a1)*(d1-d0);step=.5*(d0+d1)*(a1-a0);delta=new['score']-old['score'];assert abs(direction+step-delta)<1e-10
        decomp.append(dict(seed=r['seed'],sample_id=r['sample_id'],position=r['position'],call=r['call'],actual_NET=r['actual_NET'],old_alpha=a0,new_alpha=a1,old_direction=d0,new_direction=d1,score_delta=delta,symmetric_endpoint_contribution=direction,symmetric_step_contribution=step,larger_component='step' if abs(step)>abs(direction) else 'endpoint',descriptive_not_counterfactual_policy=True))
    assert len(decomp)==47
    pool=C.read(O/'pool_scales.json');limits=C.read(O/'limiting_fit_rows.json');focus=C.read(O/'focus_cases.json');major=next(r for r in decomp if r['seed']==0 and r['actual_NET']>4);majorrow=next(r for r in focus if r['sample_id']==major['sample_id'] and r['seed']==0)
    blocks=C.read(O/'component_shifts.json');summary=C.read(O/'summary.json')
    numeric_ok=all(r['arms'][a]['numeric']['status']!='inconclusive_diagnostic' for r in rows for a in ['STOP_ONLY','MIXED']);assert numeric_ok
    fit_summary=[]
    for p in pool:
        l0=[r for r in limits if r['pool']==p['pool'] and r['arm']=='STOP_ONLY'];l1=[r for r in limits if r['pool']==p['pool'] and r['arm']=='MIXED']
        sample=next((r for r in rows if r['pool']==p['pool']),None)
        if sample is not None:
            fit_summary.append(dict(pool=p['pool'],fallback=p['fallback'],old_alpha=sample['arms']['STOP_ONLY']['alpha'],new_alpha=sample['arms']['MIXED']['alpha'],old_limit_calls=sorted({r['call'] for r in l0}),new_limit_calls=sorted({r['call'] for r in l1})))
    C.save('direction_step_decomposition.json',decomp);C.save('fit_step_summary.json',fit_summary)
    contributions=dict(collections.Counter(r['larger_component'] for r in decomp));C.save('conclusions.json',dict(numerical_boundary_not_supported_as_main_explanation=True,floating_bounds_not_formal_certificates=True,all610_sealed_decisions_checked=True,score_change_dominant_component=contributions,major_case=major,latent_largest_block_changes=summary['changed_largest_block'],no_policy_counterfactual_evaluated=True,no_new_fits=True,next='Qualify training-only control of correction strength before another fit. Compare ownSTOP constraints/endpoint changes and bounded public input eligibility;do not pick held-informed step/threshold or evaluate capped policy from these outcomes.'))
    lines=['# GRID228：补查判断为什么改变','','本轮只核查已保存模型与预测，没有训练、调步长或运行新策略。比较覆盖原208会追加的610条记录，保留混合经验相对222的全部47处变化。','',
    '最值得继续研究的线索，是修正方向和保护步长之间的相互作用。加入辅助经验后，训练正例上的修正变弱，允许采用的保护步长可能反而变大。因此，训练正例得到保护，并不代表未参与拟合的状态也受到同样程度的保护。','',
    '## 一次大损失是怎样发生的','',
    f"2037-01-05_101 call001、种子0的一条回本机会，旧模型分数为{majorrow['arms']['STOP_ONLY']['score']:.6f}，新模型为{majorrow['arms']['MIXED']['score']:.6f}，从继续变成停止。其即时净收益标签为{major['actual_NET']:.6f}。",
    f"这里并非学习终点变得更悲观：原始修正由{major['old_direction']:.6f}变为{major['new_direction']:.6f}，方向有所改善。但保护步长由{major['old_alpha']:.6f}升至{major['new_alpha']:.6f}，最终采用了更强的向下修正。分数总变化{major['score_delta']:.6f}，对称代数分解中，方向变化贡献{major['symmetric_endpoint_contribution']:+.6f}，步长变化贡献{major['symmetric_step_contribution']:+.6f}。",
    '两次拟合都由同一个训练正例约束步长：2037-11-16_101 call001。该训练正例的终点负分减轻，原经验保护规则便允许走得更远。这个解释来自保存的训练约束和分数，不是重新选择一个更好的步长。','',
    '## 小改善也包含同一种作用','',
    '此前两处在种子1、2重复省掉3—4个无效查询的状态仍保留。对于种子2，两个状态的原始修正方向实际上都变得更温和，最终却因步长增大而转为停止。种子1的其中一处，原始方向几乎没变，主要也是步长增大使它停止。',
    '因此，大损失与小节省不能简单归为两种不同模块造成：同一种修正强度变化既可能省下无效查询，也可能误停回本机会。不能只消除大损失对应的现象，同时假定节省必然保留。',
    '11月16日的两条回本恢复则相反：方向改善与步长减小都把分数往继续推，但同一调用的两条不回本记录也一并恢复，额外查询仍须计入。','',
    '## 数值与输入核查','',
    '610条记录的旧、新决定均通过固定强凸目标下的梯度误差诊断；其中11条是常量KEEP。全部47处变化以及5条回本变化都没有出现诊断界无法区分的情况。当前证据不支持把这些行为变化主要解释成求解器没有收敛。该结论依赖浮点梯度和数学误差上界，是数值诊断，不是严格区间证书。',
    f"在47处变化中，按最终分数变化的输入块分解，前83维块是{summary['changed_largest_block'].get('latent83',0)}处的最大项，41维公共摘要是{summary['changed_largest_block'].get('public41',0)}处的最大项。该块含缓存隐表示及原派生量，不能统称全部是神经网络隐向量。",
    '辅助经验在前83维块中出现很大的标准化值，最大约2088511；41维公共摘要在这批输入中最大约32.39。两者尺度差异支持进一步检查输入限制，但不能证明删掉前83维就会改善，因为重复省查询也使用了这个块。不能根据这批外层结果挑特征或设置范围阈值。','',
    '## 下一步','',
    '先用各模型自身的训练记录核对：辅助经验如何改变正例约束、修正方向与可用步长。再固定一个有限的修正强度对照或输入限制方案，保留原学习先验与全部种子。现阶段不直接把某个旧步长套回去计算成绩，也不按1月5日这个例子设置例外规则。',
    '原208仍是主要内部参考，222保留为低查询折中，227保留局部改善线索。当前均为已暴露开发数据上的局部代理分析，未验证整周成本或实际控制时延；没有读取最终8场景，也没有新增物理仿真。']
    (O/'GRID228_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    # Delivery integrity checks; do not rerun completed model or trajectory audits.
    manifests=[]
    for stage in ['grid225_context_support','grid226_context_numeric','grid227_context_newton']:
        m=C.module(stage,R/'work'/stage/'run.py')
        for manifest in sorted(m.A.glob('*/archive_manifest.json')):
            for rel,h in C.read(manifest).items():assert C.sha(manifest.parent/rel)==h,str(manifest.parent/rel)
            manifests.append(str(manifest))
    assert len(manifests)==10
    collaboration=R/'论文协作资料/FRAME_辅助经验对照_GRID227';copies=[]
    for p in collaboration.iterdir():
        assert C.sha(p)==C.sha(C.S/p.name);copies.append(p.name)
    assert len(copies)==8
    C.verify();C.save('review.json',dict(passed=True,all610_sealed_scores_reconstructed=True,all47_component_decompositions_checked=True,all100_synthetic_algebra_checks_passed=True,all_10_prior_archive_manifests_verified=True,prior_8_collaboration_copies_identical=True,source225_226_remain_stopped_unscored=True,source_models_predictions_unchanged=True,no_refitting_or_new_policy_evaluation=True))
    C.save('complete.json',dict(passed=True,new_fits=0,new_scalars=0,new_NN_forwards=0,new_policy_evaluations=0,new_physics=0,final8_absent=True,rows=610,changes=47))
    C.archive('completed');dest=R/'论文协作资料/FRAME_补查机制诊断_GRID228';dest.mkdir(exist_ok=False)
    for n in ['GRID228_report.md','conclusions.json','direction_step_decomposition.json','fit_step_summary.json','focus_cases.json','review.json','complete.json']:shutil.copy2(O/n,dest/n)
    message='GRID228只读核查610记录/47变化，原新决定均通过梯度强凸误差诊断（浮点非区间证明）。Jan5大损失的终点方向改善，但alpha .138295→.275001使分数.396647→-.044154；对称分解方向+.057133、步长-.497934。两处重复节省在seed2也主要受步长增大推动，不能只保留节省而假定消除损失。保留208/222/227全部反例。下步先训练内资格核对修正方向与步长的耦合，不按已暴露外层调alpha或挑特征；暂不新增拟合。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID228修正方向与保护步长\n\n'+message+'\n'
        if n=='RESEARCH_STATE.md':first,rest=t.split('\n',1);t=first+'\n'+entry+'\n'+rest
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID228后\n\n1. GRID225/226历史数值失败保持零评分；227、228已完成，不重做。\n2. GRID229训练内资格：只用自身外层STOP/辅助CONT及已拟合父模型，核对端点改善如何放宽全局保护步长。先明确可检验目标，再固定一个有限的匹配学习/修正强度方案；不按已暴露外层选择步长、阈值、惩罚或种子。\n3. 原41公共摘要+b的42输入修正器仅为候选资格方向，尚未训练或证明有效。前83维同时影响节省和损失，不直接按本批外层挑特征。\n4. 每阶段至多22拟合/校准、600秒，不盲扫、不延长。维持原费用/标签/全日期种子/原控制器、排序、别名、锁站、128/64/停滞2/.9；控制复用。最终8、物理/电网/论文/新FRAME不动。用户允许持续自主推进直至喊停。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID228修正方向与步长诊断\n\n[报告](protocols/'+A.name+'/completed/GRID228_report.md)；大损失和重复节省都涉及步长强度，下一步仅训练内资格。\n\n').encode('utf8')+p.read_bytes())
    print('GRID228_COMPLETED',json.dumps(dict(step_decomposition=contributions,major=major),ensure_ascii=False))
if __name__=='__main__':main()
