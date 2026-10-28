"""Independent weighted-rank audit and a zero-correction interface proof."""
from pathlib import Path
import collections,json,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,R,G,S,BASE,REPO,A=C.O,C.R,C.G,C.S,C.BASE,C.REPO,C.A

def rank_auc(y,score,w):
    y=np.asarray(y,bool);score=np.asarray(score);w=np.asarray(w);pos=w[y].sum();neg=w[~y].sum()
    if pos==0 or neg==0:return None
    numer=0.;lessnegative=0.
    for value in np.unique(score):
        hit=score==value;pp=w[hit&y].sum();nn=w[hit&~y].sum();numer+=pp*(lessnegative+.5*nn);lessnegative+=nn
    return float(numer/(pos*neg))

def main():
    assert not (O/'complete.json').exists();C.verify();rows=C.read(O/'row_diagnosis.json');summary=C.read(O/'per_pool_summary.json');group=collections.defaultdict(list)
    for r in rows:group[r['pool'],r['phase']].append(r)
    assert len(rows)==4076 and len(group)==24;rankchecks=0
    for (pool,phase),rr in group.items():
        y=np.array([r['net']>0 for r in rr]);w=np.array([r['mass'] for r in rr]);w/=w.sum();cost=w*np.abs([r['net'] for r in rr]);x=np.array([r['ratio'] for r in rr]);p=np.array([r['router_score'] for r in rr]);ref=summary[pool][phase]
        for field,score,weight in [('prior_ratio_AUC_structural',x,w),('router_AUC_structural',p,w),('prior_ratio_AUC_cost',x,cost),('router_AUC_cost',p,cost)]:
            value=rank_auc(y,score,weight);assert (value is None)==(ref[field] is None)
            if value is not None:assert abs(value-ref[field])<1e-12,(pool,phase,field);rankchecks+=1
        assert int(y.sum())==ref['positive_rows'];assert len(rr)==ref['rows'];assert np.all(x>1)
    fitq={r['pool']:r for r in C.read(G/'fit_qualification.json')['records']}
    for pool in summary:
        a={r['date'] for r in group[pool,'fit']};b={r['date'] for r in group[pool,'held']};assert a<=set(fitq[pool]['fit_dates']) and b<=set(fitq[pool]['evaluation_dates']);assert not a&b
    # Rejoin identity and scores to independently verify every selected held row.
    hmap={(r['seed'],r['sample_id']):r for r in C.read(BASE/'gate_predictions.json')};newmap={(r['seed'],r['sample_id']):r for r in C.read(G/'gate_predictions.json')}
    for r in rows:
        if r['phase']!='held':continue
        seed=int(r['pool'].split('_s')[1]);b=hmap[seed,r['sample_id']];n=newmap[seed,r['sample_id']];assert b['gate_continue'];assert abs(r['ratio']-b['predicted_normalized_gain']/b['known_charge'])<1e-12;assert r['router_score']==n['router_keep_score'] and r['net']==n['actual_next_NET']
    # The candidate interface is proved only at zero correction on real saved points.
    # Nonzero toy corrections are not scored as any new policy.
    def proof(gain,charge,base):
        gain=np.asarray(gain);charge=np.asarray(charge);base=np.asarray(base,bool);assert (charge>0).all() and (gain>=0).all();b=(gain-charge)/(gain+charge);assert np.isfinite(b).all() and (abs(b)<=1).all();assert np.array_equal(base,gain>charge);new=base&(b+np.zeros(len(b))>=0);assert np.array_equal(new,base);return len(b)
    hd=list(hmap.values());heldchecks=proof([r['predicted_normalized_gain'] for r in hd],[r['known_charge'] for r in hd],[r['gate_continue'] for r in hd]);innerchecks=0
    for file in (S/'labelled_samples').glob('*.npz'):
        z=np.load(file);mask=z['original_STOP_mask'];innerchecks+=proof((z['event_p']*z['mu4'])[mask],z['public_charge'][mask],z['continue4'][mask])
    assert (heldchecks,innerchecks)==(4681,16565)
    toy_g=np.array([0.,1.,2.,4.,10.]);toy_c=np.array([1.,1.,1.,2.,2.]);base=toy_g>toy_c;b=(toy_g-toy_c)/(toy_g+toy_c)
    for d in [-100.,0.,100.]:assert np.all((base&(b+d>=0))<=base)
    C.save('zero_correction_proof.json',dict(passed=True,formula='b=(predicted_gain-known_charge)/(predicted_gain+known_charge); output=original_CONT and (b+delta>=0)',held208_points=heldchecks,inner_head_STOP_points=innerchecks,all_zero_delta_actions_exactly_unchanged=True,no_rescue_even_with_toy_nonzero_delta=True,trained_correction=False,new_policy_metrics=False,limit='Initial equality only; does not guarantee useful or safe learned corrections.'))
    keys=[k for k,v in summary.items() if k!='f0_s1' and v['fit']['prior_ratio_AUC_structural'] is not None and v['held']['prior_ratio_AUC_structural'] is not None];assert len(keys)==10
    paired={phase:{metric:float(np.mean([summary[k][phase][metric] for k in keys])) for metric in ['prior_ratio_AUC_structural','router_AUC_structural','prior_ratio_AUC_cost','router_AUC_cost']} for phase in ['fit','held']};improved=[dict(pool=k,positive_rows=summary[k]['held']['positive_rows'],positive_calls=summary[k]['held']['positive_calls'],prior_AUC=summary[k]['held']['prior_ratio_AUC_structural'],router_AUC=summary[k]['held']['router_AUC_structural']) for k in keys if summary[k]['held']['router_AUC_structural']>summary[k]['held']['prior_ratio_AUC_structural']];assert {r['pool'] for r in improved}=={'f0_s0','f1_s1'}
    C.save('paired_rank_summary.json',dict(paired_defined_learned_pools=keys,averages=paired,pools_with_improved_held_ranking=improved,all_other_pools_retained=True,unpaired_fit_pool='f2_s0:held_no_positive',constant_pool='f0_s1:not_fitted'))
    avg=C.read(O/'summary.json')['average_nonconstant_pools'];decision=dict(retain_GRID208=True,do_not_adopt216=True,no_claim_capacity_alone_caused_failure=True,fit_ranking_high_but_half_threshold_underprotects=True,held_ranking_and_target_composition_shift_observed=True,do_not_only_lower_threshold=True,preserve_two_pools_positive_rank_signals=improved,next_direction='Qualify a baseline-anchored correction with zero-correction exact208 behavior, then freeze matched direct-vs-anchored input/capacity/objective/budget before a preannounced training.',new_training=False,no_new_policy_selected_from_quartiles=True);C.save('decision.json',decision)
    lines=['# GRID217：原判断与取消分类器的排序、读出和迁移核查','','本轮只读已封存记录，并重新核验固定森林的拟合输出。没有拟合新模型、运行新的外层模型预测、改阈值或执行新的取消策略。','', '## 先修正问题判断','','GRID216在训练中只保留19/90条回本记录，但这不等于训练中完全没有学到区分信息。取消分数在拟合内排序很好；固定0.5读出却保留很少。到外层日期，排序也明显不稳定。应同时考虑拟合内分数与决策边界的关系，以及外层排序的变化，不能仅归因于模型太小或仅归因于阈值。','', '原判断强度定义为R=(原事件输出×条件增益输出)/已知查询费用；原模型在R>1时追加。R是原决策的相对强度，不是实际回本概率。拟合侧来自8日期内层头，外层侧来自12日期原GRID208，模型与入口均有变化。','', '## 同10个可比较学习池的排序','','使用原日期/调用/前缀结构权重，逐池计算回本与不回本记录之间的加权AUC，再对同10池平均。常量池f0_s1保留但不算学习池；f2_s0外层没有回本例，AUC不定义，全部记录仍保留。','', '|分数|拟合内AUC|外层AUC|','|---|---:|---:|',f"|原判断R|{paired['fit']['prior_ratio_AUC_structural']:.6f}|{paired['held']['prior_ratio_AUC_structural']:.6f}|",f"|新取消分类器保留分数|{paired['fit']['router_AUC_structural']:.6f}|{paired['held']['router_AUC_structural']:.6f}|",'', f"若再按|净收益|加权，原判断外层AUC为{paired['held']['prior_ratio_AUC_cost']:.6f}，新分类器为{paired['held']['router_AUC_cost']:.6f}。AUC只描述排序，不证明阈值、查询预算、完整轨迹或整周收益；这些记录相关，未作独立样本显著性推断。",'', '新分类器的外层平均结构加权AUC接近0.5，各池范围很大。原判断的区分能力仍有保留，直接用冷启动分类器接管整个保留判断，损失了这一已有基础。这是后续测试继承原判断的依据，不是该方案必然有效的证明。','', '## 两处有用线索也保留','','并非每个学习池的排序都更差。以下2个池优于原判断，但原0.5判定未将它们兑现为追加。这是诊断中的开发线索，不能据此选种子或只保留这些池。','', '|池|回本行/调用|原判断AUC|新分数AUC|','|---|---:|---:|---:|']
    for r in improved:lines.append(f"|{r['pool']}|{r['positive_rows']}/{r['positive_calls']}|{r['prior_AUC']:.6f}|{r['router_AUC']:.6f}|")
    lines+=['','## 标签代价分布也不同','',f"11个学习池等权平均，回本行比例从拟合侧{avg['fit']['positive_fraction']['mean']:.2%}变为外层{avg['held']['positive_fraction']['mean']:.2%}；按原结构权重计算，正净收益占总绝对净收益代价的比例从{avg['fit']['positive_absNET_cost_share']['mean']:.2%}变为{avg['held']['positive_absNET_cost_share']['mean']:.2%}。这说明这批拟合域的保护代价远低于对应外层域，不能只看90条和55条正例数量。",'', '这些差异来自不同日期、8日期与12日期模型的筛选、样本权重和已实现收益等共同作用；不能当成已识别的纯标签先验漂移，更不能用外层标签估计新类别权重再回报同一批外层效果。', '', '610条外层追加中，只有2条原判断R低于本池拟合范围、2条高于范围。原判断单一数值范围大多重叠，并不排除多维输入或条件分布变化。四分位描述的分界仅用拟合侧R确定，未用于执行策略或选择阈值。','', '## 可验证的局部修正接口','','先把原判断写成有限分数b=(预测增益-费用)/(预测增益+费用)，再考虑学习修正量delta。保留判断为：原模型准备追加，且b+delta>=0。','', '在delta=0时，它对4681个原外层停止点和16565个内层停止点完全复现原决定；原本不追加的点不会由这个取消模块变成追加。只做了零修正等价和玩具非零修正的结构检查，没有评估任何已学习修正量。零初始化保证起点一致，不保证训练后保持质量。','', '下一次若训练，应以同样输入、容量、目标与训练预算比较直接读出和带原判断的读出，单独识别继承原判断的作用。原模型强度可以作为公共输入，但8日期头与12日期头的入口差异仍需报告。此处未选择网络、训练轮数、修正强度或新阈值。','', '当前继续保留GRID208，GRID216失败模型和两个排序改善线索同时保留。下一步先固定局部修正的输入与匹配对照，再通知新训练；不扩大物理仿真、不动最终8、论文或新FRAME对照。']
    (O/'GRID217_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.verify();C.save('independent_audit.json',dict(passed=True,rank_AUC_checks_reconstructed_by_sorted_tie_algorithm=rankchecks,all4076_row_identity_and_label_checks=True,all610_held_ratios_scores_joined_to_original=True,ownouter_fit_dates_disjoint_held=True,zero_delta_checks=21246,no_new_fits=True,no_new_held_model_scores=True,no_new_policy_metrics=True));C.save('complete.json',dict(passed=True,scope='saved_prior_and_cold_cancellation_diagnosis_only',fit_rows=3466,held_selected_rows=610,paired_AUC_pools=10,improved_rank_pools=2,new_fits=0,new_NN_forwards=0,new_physics=0,new_policy_evaluations=0))
    dest=R/'论文协作资料/FRAME_判断继承诊断_GRID217';dest.mkdir(exist_ok=False)
    for name in ['GRID217_report.md','summary.json','per_pool_summary.json','paired_rank_summary.json','zero_correction_proof.json','decision.json','independent_audit.json','complete.json']:shutil.copy2(O/name,dest/name)
    A.mkdir(parents=True,exist_ok=False)
    for p in [*Path(__file__).parent.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,A/p.name)
    for p in dest.iterdir():assert C.sha(p)==C.sha(O/p.name)==C.sha(A/p.name)
    (A/'archive_manifest.json').write_text(json.dumps({p.relative_to(A).as_posix():C.sha(p) for p in A.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
    msg='GRID217只读诊断：同10学习池，原判断R拟合/外层AUC .7373/.7233，新取消分数.9878/.4952；训练71误停不等于无排序学习，外层排序也不稳，不能仅降阈值或仅加容量。11池正代价占比拟合2.98%/外层52.84%，多源域差异非纯标签先验漂移。f0_s0/f1_s1外层排序有改善，保留不择种子。零修正接口在21246点完全复现原决定，仅结构证明未训练/评估修正策略。保留208，先固定继承原判断的匹配对照，再通知训练。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID217原判断与取消读出诊断\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID217后\n\n1. 保留208与216全部失败及两池排序改善线索。新分类器拟合排序好、外层不稳且分数低，不只归因容量或阈值。\n2. 固定继承原判断的局部修正与相同输入/容量/目标/预算的直接读出对照；零修正应完全复现208。当前仅通过结构证明，未定训练参数或新训练。\n3. 公共原判断输入和8日期→12日期入口差异须明确。不得用已看外层标签校正类别先验再把同一批当独立确认；不跨外层拼OOF。\n4. 新训练先通知；不扫阈值/选择种子/追加物理/最终8/论文/新FRAME。\n',encoding='utf8',newline='\n');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID217原判断继承诊断\n\n[报告](protocols/'+A.name+'/GRID217_report.md)；保留两处排序改善线索，零修正结构证明，未新训练。\n\n').encode('utf8')+p.read_bytes());print('GRID217_AUDITED',json.dumps(dict(paired=paired,improved=improved,rankchecks=rankchecks,zero_delta_points=21246),ensure_ascii=False))
if __name__=='__main__':main()
