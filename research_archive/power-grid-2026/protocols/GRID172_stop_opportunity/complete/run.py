"""Hindsight opportunity within each cross-fitted FACTOR path; not a deployable rule."""
from pathlib import Path
import collections,hashlib,importlib.util,json,math,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid172_stop_opportunity';REPO=R/'work/GRL-results-integration'
A=REPO/'research_archive/power-grid-2026/protocols/GRID172_stop_opportunity'
sp=importlib.util.spec_from_file_location('g172_oof',R/'work/grid170_remaining_date_folds/run.py');T=importlib.util.module_from_spec(sp);sp.loader.exec_module(T)
read=T.read;sha=T.sha;H=T.H
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def main():
    assert not O.exists();O.mkdir(parents=True);T.verify_trained();assert read(T.O/'complete.json')['passed_audit']
    save('design.json',dict(id='GRID172_BASELINE_PATH_STOP_OPPORTUNITY',authorization='User sustained work: quantify how recurring small improvements could grow.',
        question='How much cost-only opportunity remains within the already-cross-fitted FACTOR search path? Where does current fixed correction capture or miss it?',
        scope='All7134seed-startpairs,4fixedfolds/16dates,oldlocalNETtask. Also audit immediate versus actual-baseline-continuation labels on every baselineCONTstate before proposing any target change. No newmodel/forward/fit/physical.',
        bound='At each FACTOR endpointB, hindsight earliest j<=B with identical totalgain AND chosenaction defines zero-lossquery-saving upperbound. A separate maxNETprefix oracle may sacrificegain. Both use futurefeedback;not deployable or newtest.',
        aggregate='Date macro thenseed;allphysicalgroup andseed recurrence,immediate/delayedlosses,all adverse results. No threshold/model selection.',
        prohibition=['newtrigger/fit/forward','futurelabels asfeatures','newdata/physics/final8','paper']))
    files=[T.O/n for n in ['design.json','review.json','complete.json','all_oof_trajectories.json','all7134_paired_events.json','family_summary.json','key_changes.json','trained_freeze.json']]+[R/'outputs/grid102_search_rl/cases.pkl',Path(__file__),O/'design.json']
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in files})
    tr=read(T.O/'all_oof_trajectories.json');by={(r['method'],r['sample_id']):r for r in tr};events=read(T.O/'all7134_paired_events.json');cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'}
    d=read(T.O/'design.json');fee=d['cost_per_query'];scale=d['gain_scale'];records=[];oracletr=[];netoracletr=[];label_rows=[]
    for e in events:
        seed=e['seed'];c=cases[e['sample_id']];b=by[f'FACTOR_seed{seed}',c['sample_id']];a=by[f'ONE_SIDED_seed{seed}',c['sample_id']];B=b['batches']
        q=np.r_[0,np.cumsum(c['queries'][:B])];gain=np.r_[0,np.cumsum(c['gains'][:B])];assert int(q[B])==b['queries'] and abs(float(gain[B])-b['gain'])<1e-12
        earliest=next(j for j in range(B+1) if float(gain[j])==b['gain'] and c['chosen'][j]==b['chosen_pool_id'])
        net=(gain-fee*q)/scale;best=int(np.argmax(net));assert net[best]>=b['net_utility']-1e-12
        bound_queries=int(q[B]-q[earliest]);same=a['gain']==b['gain'] and a['chosen_pool_id']==b['chosen_pool_id']
        realized=int(b['queries']-a['queries']) if same else 0;assert 0<=realized<=bound_queries
        row=dict(e,baseline_batches=B,baseline_queries=b['queries'],baseline_gain=b['gain'],same_gain_action=same,earliest_same_gain_action=earliest,zero_loss_saved_queries_bound=bound_queries,realized_same_gain_action_saved_queries=realized,unrealized_zero_loss_queries=bound_queries-realized,
            zero_loss_net_gain_bound=fee*bound_queries/scale,best_net_prefix=best,best_net_improvement_bound=float(net[best]-net[B]))
        if e['saved_queries']:
            j=a['batches'];kinds=[p for p in range(j,B) if c['gains'][p]>0]
            row['missed_gain_kind']='NO_GAIN' if not kinds else ('IMMEDIATE' if c['gains'][j]>0 else 'DELAYED')
            row['next_profitable']=bool((c['gains'][j]-fee*c['queries'][j])>0)
            immediate=(float(c['gains'][j])-fee*int(c['queries'][j]))/scale;tail=(b['gain']-a['gain']-fee*e['saved_queries'])/scale
            assert abs(tail+e['net_delta'])<1e-12
            row['changed_stop_label_same_sign']=bool(immediate>0)==bool(tail>0)
        for j in range(B):
            immediate=(float(c['gains'][j])-fee*int(c['queries'][j]))/scale;tail=(float(gain[B]-gain[j])-fee*int(q[B]-q[j]))/scale
            label_rows.append(dict(seed=seed,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],position=j,immediate_net=immediate,baseline_continuation_net=tail,same_sign=bool(immediate>0)==bool(tail>0)))
        records.append(row)
        for j,method,dest in [(earliest,'ZERO_LOSS_ORACLE',oracletr),(best,'NET_ORACLE',netoracletr)]:
            go=np.array([p<j for p in range(len(c['states']))]);dest.append(H.trajectory(c,go,f'{method}_seed{seed}',fee,scale))
    save('all7134_opportunity_records.json',records);save('baseline_continuation_label_audit.json',label_rows)
    summary={};family={}
    for method,rr in [('FACTOR',[r for r in tr if r['method'].startswith('FACTOR_')]),('ONE_SIDED',[r for r in tr if r['method'].startswith('ONE_SIDED_')]),('ZERO_LOSS_ORACLE',oracletr),('NET_ORACLE',netoracletr)]:
        perseed=[H.summarize([r for r in rr if r['method']==f'{method}_seed{s}']) for s in [0,1,2]];summary[method]=perseed
        family[method]={k:float(np.mean([p['macro'][k] for p in perseed])) for k in ['mean_net_utility','mean_queries','query_fraction','gain_capture','physical_group_mean_utility']}
    groups=[]
    for g in sorted({r['physical_state_group'] for r in records}):
        rr=[r for r in records if r['physical_state_group']==g];positive=[r for r in rr if r['net_delta']>0];negative=[r for r in rr if r['net_delta']<0]
        groups.append(dict(physical_state_group=g,dates=sorted({r['date'] for r in rr}),positive_seeds=sorted({r['seed'] for r in positive}),negative_seeds=sorted({r['seed'] for r in negative}),positive_pairs=len(positive),negative_pairs=len(negative),zero_loss_opportunity_pairs=sum(r['zero_loss_saved_queries_bound']>0 for r in rr),net_contribution=math.fsum(r['weight']*r['net_delta'] for r in rr),opportunity_contribution=math.fsum(r['weight']*r['zero_loss_net_gain_bound'] for r in rr)))
    save('physical_group_recurrence.json',groups);save('family_summary.json',family);save('oracle_summary.json',summary)
    potential=math.fsum(r['weight']*r['zero_loss_saved_queries_bound'] for r in records);realized=math.fsum(r['weight']*r['realized_same_gain_action_saved_queries'] for r in records)
    base=family['FACTOR']['mean_net_utility'];bound=family['ZERO_LOSS_ORACLE']['mean_net_utility']-base;assert abs(bound-fee*potential/scale)<1e-12
    loss=[r for r in records if r['net_delta']<0];positive=[r for r in records if r['net_delta']>0];same=[r for r in records if r['saved_queries'] and r['same_gain_action']]
    key=dict(held_pairs=len(records),zero_loss_opportunity_pairs=sum(r['zero_loss_saved_queries_bound']>0 for r in records),zero_loss_opportunity_physical_groups=sum(g['zero_loss_opportunity_pairs']>0 for g in groups),
        potential_same_gain_action_queries_saved=potential,realized_same_gain_action_queries_saved=realized,zero_loss_opportunity_capture=realized/potential if potential else None,
        zero_loss_net_utility_upper_improvement=bound,zero_loss_net_utility_upper_relative=bound/base,actual_net_utility_change=family['ONE_SIDED']['mean_net_utility']-base,
        correct_same_gain_action_pairs=len(same),positive_pairs=len(positive),negative_pairs=len(loss),positive_physical_groups=sum(bool(g['positive_seeds']) for g in groups),recurring_positive_groups_atleast2seeds=sum(len(g['positive_seeds'])>=2 for g in groups),recurring_positive_groups_all3=sum(len(g['positive_seeds'])==3 for g in groups),mixed_sign_groups=sum(bool(g['positive_seeds']) and bool(g['negative_seeds']) for g in groups),
        loss_immediate=sum(r.get('missed_gain_kind')=='IMMEDIATE' for r in loss),loss_delayed=sum(r.get('missed_gain_kind')=='DELAYED' for r in loss),loss_next_profitable=sum(r.get('next_profitable',False) for r in loss),
        baseline_continue_state_records=len(label_rows),baseline_tail_vs_immediate_label_sign_differences=sum(not r['same_sign'] for r in label_rows),loss_cases_with_changed_label_sign=sum(not r.get('changed_stop_label_same_sign',True) for r in loss))
    save('key_changes.json',key)
    lines=['# GRID172：同收益、同动作的节省，还有多少空间','',
        f"在四折16日期的原FACTOR路径内，事后最多还能在保持相同局部收益和动作的前提下，平均每起点少查{potential:.3f}个候选，对应净效用提升上界{key['zero_loss_net_utility_upper_relative']:.2%}。当前单向规则取得其中{key['zero_loss_opportunity_capture']:.2%}的同收益同动作查询节省。",'',
        '这个上界沿每条原FACTOR实际会走的路径，找到最终动作首次已经获得的位置。之后再查而收益和动作不变的部分，才计为可无损省掉的查询。它使用后续反馈，只是研究空间的核算，不能部署，也不是模型已经学会的结果。','',
        '| 策略或事后上界 | 局部净效用 | 平均查询 | 收益保留率 |','|---|---:|---:|---:|']
    for method,f in family.items():lines.append(f"| {method} | {f['mean_net_utility']:.8f} | {f['mean_queries']:.3f} | {f['gain_capture']:.2%} |")
    lines+=['', 'ZERO_LOSS_ORACLE只缩短原路径且保持最终收益/动作；NET_ORACLE允许在原路径中用少量收益换费用，二者不混称质量上界。它们都受限于原FACTOR已经进行的搜索，不会找到原路径之外的新动作。','',
        '## 正向变化在哪里重复','',
        f"当前{len(same)}条配对实现同收益、同动作而少查；全部正向净变化涉及{key['positive_physical_groups']}个物理状态组，其中{key['recurring_positive_groups_atleast2seeds']}组至少两个种子重复出现、{key['recurring_positive_groups_all3']}组在三个种子重复出现。{key['mixed_sign_groups']}组同时出现正反变化。前缀和种子仍是相关记录，不能当独立场景。",'',
        f"全部{len(loss)}条净损失中，{key['loss_immediate']}条漏掉下一批就有的增益，{key['loss_delayed']}条漏掉后续才出现的增益；{key['loss_next_profitable']}条的下一批本身净收益已为正。完整物理组及每条机会记录均保留。",'',
        '## 对后续学习意味着什么','',
        '单向修正的长处是把查询支出控制住；在固定候选路径上，它只会保留或降低已获得的收益，不会额外创造更高收益。因此要放大现有小改善，首先应扩大正确识别无收益后缀的覆盖，同时避免取消原模型已经识别出的有利查询。',
        '', f"另外核查了{len(label_rows)}条原FACTOR继续状态：把下一批NET换成原FACTOR剩余路径NET，标签符号只在{key['baseline_tail_vs_immediate_label_sign_differences']}条改变；本轮净损失案例中有{key['loss_cases_with_changed_label_sign']}条因此改变符号。这用于判断改标签是否真正针对当前失误，不能假定换成整段目标就会解决问题。",
        '', '本轮把“有多少可省空间”“当前抓到了多少”“在哪些物理组漏掉有利机会”分开。任何后续标签方案都要先解释其与已有标签的实质差别、训练侧覆盖和错例对应，尚未开始新训练。已有加权分类、收益回归和校准门控的负面记录继续保留，避免换名字重复实验。',
        '', '全部四折均使用原固定费用和收益尺度，同网历史数据已暴露。没有新模型前向、拟合或物理回放，最终8未生成；本结果不证明整周质量或真实控制提速。']
    (O/'GRID172_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
    T.verify_trained();save('review.json',dict(passed=True,all7134_paths_checked=True,oracle_same_gain_action_verified=True,all_original_policy_queries_bound=True,weighted_query_fee_identity=True,all_unfavorable_pairs_retained=True,new_fits=0,new_forwards=0,new_physics=0,final8_absent=True))
    save('decision.json',dict(next='Use actual opportunity and label-sign audit to decide whether any target change addresses current errors before another fit; no hindsight-trigger deployment or immediate expansion.',new_learning_not_started=True));save('complete.json',dict(passed_audit=True,new_fits=0,new_forwards=0,new_physics=0))
    dest=A/'complete';dest.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
    collab=R/'论文协作资料/FRAME_无损停止研究空间_GRID172';collab.mkdir(exist_ok=False)
    for n in ['GRID172_report.md','key_changes.json','family_summary.json','physical_group_recurrence.json','review.json','complete.json']:shutil.copy2(O/n,collab/n);assert sha(collab/n)==sha(O/n)
    msg=f"GRID172全部7134折外配对核查原FACTOR路径内同收益动作可省查询上界{potential:.3f}/起点、净效用上界+{key['zero_loss_net_utility_upper_relative']:.2%}，当前同收益动作节省捕获{key['zero_loss_opportunity_capture']:.2%}；{len(same)}条正确省、{len(loss)}条净损失立即/延迟分列。事后不可部署，不新增模型/物理/最终8；保留正向复现并研究以原策略继续结果为参照的标签，不重复旧损失改造。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        path=REPO/'docs'/name;old=path.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID172无损停止机会\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        path.write_text(old,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — 已有节省与机会保护\n\n1. GRID170全四折结果、GRID171等价执行开销、GRID172事后空间共同判读；不挑日期/种子或重设门槛。\n2. 先核对相对原策略实际继续结果的停止标签：现NET下一批标签与原FACTOR后续收益的差别、12fit内部拟合偏差和日期泄漏风险；不是直接上新模型或重复旧完整补查标签。新训练前给具体方案和通知。\n3. 新数据/物理扩大另给范围等待用户决定，最终8不动，论文不改；小改善保留、收益损失同报。\n',encoding='utf8',newline='\n')
    index=REPO/'research_archive/power-grid-2026/README.md';index.write_bytes('# 最新：GRID172同收益动作停止空间\r\n\r\n[报告](protocols/GRID172_stop_opportunity/complete/GRID172_report.md)；先量化已有小改善的可扩空间与机会保护问题。\r\n\r\n'.encode('utf8')+index.read_bytes());print('GRID172_COMPLETE',json.dumps(key,ensure_ascii=False))
if __name__=='__main__':main()
