"""Audit fixed forests and saved-sequence effects; preserve the adverse result."""
from pathlib import Path
import collections,json,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,R,REPO,A=C.O,C.R,C.REPO,C.A

def main():
    C.scored();assert C.read(O/'review.json')['passed'] and not (O/'complete.json').exists();params=C.read(O/'configs.json');fitdiag=[];fit_rows=fitpos=fitkeep=0;native_checks=0;allheldbelow=True
    raw=np.load(C.S/'public_raw41.npz')['values']
    for key,config in params.items():
        z=np.load(O/'fit_inputs'/f'{key}.npz');model=pickle.loads((O/'models'/f'{key}.pkl').read_bytes());x,y,w=z['X'],z['y'].astype(bool),z['weight'];fitted=not isinstance(model,dict);source=np.load(C.FULL/'fit_features'/f'{key}.npz');idx={int(i):j for j,i in enumerate(source['ids'])};assert np.array_equal(x,source['standard'][[idx[int(i)] for i in z['ids']]])
        if fitted:
            assert model.get_params()==config and model.n_features_in_==124 and np.array_equal(model.classes_,[0,1]);manual=[]
            for tree in model.estimators_:
                leaves=tree.apply(x);v=tree.tree_.value[leaves,0,:];manual.append(v[:,1]/v.sum(1));assert tree.tree_.max_depth<=4
                for leaf in np.unique(leaves):
                    mask=leaves==leaf;assert mask.sum()>=10;value=tree.tree_.value[leaf,0,:];assert abs(value[1]/value.sum()-float(w[mask]@y[mask]/w[mask].sum()))<1e-8
            p=np.mean(manual,axis=0);assert np.allclose(p,model.predict_proba(x)[:,1],atol=1e-10,rtol=1e-10)
        else:assert key=='f0_s1' and not y.any() and model['type']=='KEEP';p=np.ones(len(x))
        h=np.load(C.CACHE/f'held_{key}.npz');cp=np.load(C.BASE/'models'/f'{key}.npz');hx=np.c_[(h['features']-cp['mean'])/cp['scale'],(raw[h['ids'].astype(int)]-cp['raw_mean'])/cp['raw_scale']];saved=np.load(O/'held_predictions'/f'{key}.npz');hp=np.ones(len(hx))
        if fitted:
            hp=np.mean([t.tree_.value[t.apply(hx),0,1]/t.tree_.value[t.apply(hx),0,:].sum(1) for t in model.estimators_],axis=0);allheldbelow=allheldbelow and bool((hp<.5).all())
        assert np.array_equal(saved['ids'],h['ids']) and np.allclose(hp,saved['keep_score'],atol=1e-10,rtol=1e-10) and np.array_equal(saved['keep'],hp>=.5);keep=p>=.5;mass=z['structural_mass'];net=z['actual_NET'];plus=float(mass@np.maximum(net,0));miss=float(mass@(np.maximum(net,0)*~keep));waste=float(mass@(np.maximum(-net,0)*keep));
        fitdiag.append(dict(pool=key,fitted=fitted,rows=len(y),positive_rows=int(y.sum()),positive_kept=int((keep&y).sum()),negative_kept=int((keep&~y).sum()),fit_max_keep_score=float(p.max()),held_max_keep_score=float(hp.max()),positive_value_mass=plus,missed_positive_value_mass=miss,false_keep_cost_mass=waste,zero_positive_default_KEEP=not fitted))
        if fitted:fit_rows+=len(y);fitpos+=int(y.sum());fitkeep+=int((keep&y).sum())
        native_checks+=len(hp)
    assert fitpos==90 and fitkeep==19 and allheldbelow and native_checks==4681
    pub=C.read(O/'pre_score_publication.json');join=C.read(O/'pre_join_publication.json')
    for receipt,phase,name in [(pub,'trained','trained_freeze.json'),(join,'scored','scored_freeze.json')]:assert C.git('show',receipt['git_sha']+':'+(A/phase/name).relative_to(REPO).as_posix())==(O/name).read_bytes()
    tr=C.read(O/'oof_trajectories.json');base={(r['method'],r['sample_id']):r for r in tr if r['method'].startswith('GATE')};changed=[]
    for n in tr:
        if not n['method'].startswith('CANCEL'):continue
        seed=int(n['method'].split('seed')[-1]);b=base[f'GATE_seed{seed}',n['sample_id']];assert n['queries']<=b['queries'] and n['gain']<=b['gain']
        if n['append']!=b['append']:changed.append(n['sample_id'])
    removed=C.read(O/'removed_batches.json');assert len(changed)==len(removed)==599;byseed=collections.defaultdict(list)
    for p in C.read(O/'gate_predictions.json'):byseed[p['seed']].append(p)
    removedsummary=[]
    for seed,rr in sorted(byseed.items()):
        cut=[r for r in rr if r['base_continue'] and not r['new_continue']];removedsummary.append(dict(seed=seed,baseline_CONT=sum(r['base_continue'] for r in rr),retained_CONT=sum(r['new_continue'] for r in rr),removed=len(cut),profitable_removed=sum(r['actual_next_NET']>0 for r in cut),unprofitable_removed=sum(r['actual_next_NET']<=0 for r in cut),zero_gain_removed=sum(r['actual_next_gain']==0 for r in cut),profitable_removed_calls=len({r['physical_state_group'] for r in cut if r['actual_next_NET']>0})))
    fam=C.read(O/'family_summary.json');pairs=C.read(O/'paired_summary.json');metrics=C.read(O/'search_metrics.json');dates=[]
    for date in sorted({r['date'] for r in removed}):
        deltas=[next(d for d in p['per_date'] if d['date']==date)['NET_delta'] for p in pairs];dates.append(dict(date=date,mean_NET_delta=float(np.mean(deltas)),seed_NET_deltas=deltas))
    diagnostic=dict(fit_pools=fitdiag,learned_pool_fit_rows=fit_rows,learned_pool_positive_rows=fitpos,learned_pool_positive_kept=fitkeep,learned_pool_positive_missed=fitpos-fitkeep,all11_learned_pools_all_held_scores_below_half=allheldbelow,only_fallback_pool_retains_batches=True,removed_by_seed=removedsummary,per_date_NET_differences=dates,interpretation='Major loss already visible in fit:19/90 positives kept. Held learned scores allbelow.5; only constantKEEP acts. This rules out solely unseen-date failure, but does not isolate capacity, feature support, selection shift or tree averaging causally. No threshold/model changes executed.')
    C.save('fit_and_transfer_diagnosis.json',diagnostic);counts=C.read(O/'gate_counts.json');querypct=100*(fam['CANCEL']['mean_queries']/fam['GATE']['mean_queries']-1)
    decision=dict(adopt_cancellation_gate=False,retain_GRID208=True,do_not_discard_broader_support_direction=True,readout_failed_to_protect_gain=True,fit_underprotection_and_held_collapse_both_observed=True,no_claim_solely_transfer_or_insufficient_data=True,no_auto_thresholdsearch_or_retraining=True,next='Read-only compare frozen208 predicted decision strength and cancellation fit errors; design baseline-preserving correction before any next training, rather than reset full decision from rare labels.')
    C.save('decision.json',decision)
    lines=['# GRID216：取消补查分类器固定实验','','按预先固定方案训练11个森林和1个KEEP常量。全部权重发布后生成外层4681个停止状态预测，预测发布后连接标签；使用原16个已暴露日期、全部3种子。','', '## 结果','','固定查询费用下，取消模块明显缩减补查，但误删了多数回本补查，三个种子的净效用均下降。该分类器不替换GRID208。该结果仅针对当前入口、目标和固定浅森林，保留更广训练入口的探索方向。','', '|指标（日期宏平均后对种子平均）|GRID208|加取消模块|','|---|---:|---:|']
    for key,label in [('mean_queries','候选查询'),('mean_gain','即时增益'),('mean_net_utility','固定费代理净效用'),('gain_capture','局部收益保留比例'),('action_match_rate','与完整局部搜索动作一致率'),('physical_group_mean_utility','调用组均衡净效用')]:
        aa,bb=fam['GATE'][key],fam['CANCEL'][key];lines.append(f'|{label}|{aa:.9f}|{bb:.9f}|')
    lines+=['',f"查询变化{querypct:.2f}%；局部收益保留比例下降{100*(fam['GATE']['gain_capture']-fam['CANCEL']['gain_capture']):.3f}个百分点，动作一致率下降{100*(fam['GATE']['action_match_rate']-fam['CANCEL']['action_match_rate']):.3f}个百分点。这里没有测量整周运行成本或完整控制时间。",'',f"原610次追加中保留11次、取消599次。取消的547次不回本，其中540次零即时增益；但55次原本回本的追加也被取消52次，涉及15个物理调用。余下3次回本追加全部由无正例池的KEEP常量保留，不能归功于学习分类器。重复前缀和种子记录不独立。",'', '|种子|净效用变化|查询变化|动作一致率变化（百分点）|','|---|---:|---:|---:|']
    for p in pairs:dd=p['macro_delta'];lines.append(f"|{p['seed']}|{dd['mean_net_utility']:.9f}|{dd['mean_queries']:.6f}|{100*dd['action_match_rate']:.6f}|")
    lines+=['','## 问题发生在哪里','','在11个学习池自身的拟合样本里，90条回本记录仅保留19条，已有71条被误停。到外层日期，11个森林的全部保留分数都低于固定0.5，最终保留追加的只有事先规定的KEEP常量。故不能把问题完全归因于外层新日期。','', 'GRID215提供了更多保护例，但原结构权重下保护价值在多数池中仍稀疏，浅树的加权叶子平均并未形成可靠保留区域。观察到的是训练内保护不足与外层分数进一步降低共同存在；这不证明增加深度、放大类别权重或降低阈值一定有效，也未区分特征、模型容量、入口变化和集成平均各自的作用。','', '本轮仍使用原费用与m|z|权重，没有为得到好结果而改经济目标。按收益符号分类、代价敏感分类与原概率乘幅度模型属于不同读出方式；加权分类输出不是实际回本概率。','', '## 下一步','','保留GRID208提供的继续判断，先只读核查其已有判断强度与新分类器误停的关系，再设计保留原判断的局部修正。现成模型已经有有效的回本识别能力，下一次改进应明确如何继承这部分能力；本轮不再追加分类器、扫描深度或调整0.5。','', '所有失败模型、种子、日期、连续收益、查询和标签均保留。当前是旧教师路径上的离线单批追加重放，不能替代闭环整周效果、实际时延或跨电网泛化。最终8个测试场景、论文和新FRAME对照未动。']
    (O/'GRID216_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.scored();C.save('independent_audit.json',dict(passed=True,all11_configurations_and_weighted_leaf_means_reconstructed=True,one_constantKEEP_no_fit_verified=True,all4681_native_predictions_reconstructed_after_seal=True,all14268_control_and_cancel_trajectories_reviewed=True,all599_cancellations_correspond_to_removed_batches=True,publication_barriers_verified=True,all_source_and_model_freezes_unchanged=True,new_NN_fits=0,new_NN_forwards=0,new_physics=0));C.save('complete.json',dict(passed=True,new_forests=11,new_trees=704,constant_KEEP=1,held_STOP_predictions=4681,trajectory_rows=14268,retain_GRID208=True,scope='fixed_exposed_teacher_path_cancellation_experiment',final8_absent=True));C.archive('audited')
    dest=R/'论文协作资料/FRAME_取消分类器_GRID216';dest.mkdir(exist_ok=False)
    for name in ['GRID216_report.md','family_summary.json','paired_summary.json','fit_and_transfer_diagnosis.json','gate_counts.json','decision.json','review.json','independent_audit.json','complete.json']:shutil.copy2(O/name,dest/name)
    msg='GRID216固定11浅森林+f0_s1常量KEEP完成；全部权重先发、4681外层预测先发后连标签。查询13.4405→12.9114，净效用.09609252→.08902488，三种子均负；610原追加取消599，547无利/52回本，55回本仅常量保留3。训练90回本仅保留19，11学习池外层全部分数<.5，非单纯新日期失败。保留208、全部反例及更宽入口方向；下一步只读检查既有判断强度与误停，拟定保留原判断的局部修正，不自动扫阈值/加深或重训。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID216取消分类器固定实验\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID216后\n\n1. 保留208；216三种子代理净效用均负，所有模型/日期/反例保留。学习入口扩大有保护例，不能当成分类已有效。\n2. 只读核对原208判断强度、训练内71回本误停和外层52回本误删；区分拟合保护不足与迁移变化。11学习池外层全停，不归因单一原因。\n3. 下一次拟定保留原模型有效判断的局部修正，再通知新训练。当前不调0.5、类别权重或扫容量，不继续做同一分类器。\n4. 无新物理/最终8/论文/新FRAME比较；完整控制时间与整周质量未测。\n',encoding='utf8',newline='\n');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID216取消分类器固定实验\n\n[报告](protocols/'+A.name+'/audited/GRID216_report.md)；11森林+1KEEP，保护不足，保留208与全部反例。\n\n').encode('utf8')+p.read_bytes());print('GRID216_AUDITED',json.dumps(dict(fit_positive_kept=fitkeep,fit_positive_total=fitpos,counts=counts,retain208=True),ensure_ascii=False))
if __name__=='__main__':main()
