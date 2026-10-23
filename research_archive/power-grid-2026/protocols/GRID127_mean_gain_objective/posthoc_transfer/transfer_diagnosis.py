"""Post-hoc transfer accounting only,all dates/seeds/calls,no new fit or prediction."""
from common127 import *
import gzip

def main():
    T=O/'posthoc_transfer';assert not T.exists();T.mkdir();verify();assert read(O/'review.json')['passed'];Q=np.load(O/'predictions.npz');M=read(O/'state_metrics.json');D=read(O/'design.json');P=R/'outputs/grid125_value_underestimate';support={(r['part'],r['index']):r for r in read(P/'support_rows.json')};cases=src.pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cm={c['sample_id']:c for c in cases};meta={part:read(S/f'{part}_metadata.json') for part in ['train','validation']};lookup={part:{(m['sample_id'],m['position']):i for i,m in enumerate(mm)} for part,mm in meta.items()};prob=np.load(S/'train_data.npz')['probability'];risk=[]
    save(T/'scope.json',dict(posthoc=True,purpose='Separate avoidable unprofitable querying and foregone positive value in training versus held dates;reuse existing sourceGRID125support at every actual first divergence,no new trigger.',source_hashes={str(p.relative_to(R)):sha(p) for p in [W/'transfer_diagnosis.py',O/'predictions.npz',O/'state_metrics.json',O/'decision.json',O/'review.json',P/'support_rows.json']},new_fits=0,new_forwards=0,new_physical_calls=0,final_test_materialized=False))
    for split in ['full','f0','f1','f2','f3']:
        for part in (['train','validation'] if split=='full' else ['train']):
            x,y,G,c,ids,p=getdata(split,part);sets=[('train' if part=='train' else 'validation',np.arange(len(y)),p)] if split=='full' else [('fit',ids,p),('held',np.setdiff1d(np.arange(len(y)),ids),None)]
            for tag,ii,pp in sets:
                if split=='full' and part=='validation':
                    cc=collections.Counter(v['date'] for v in cases if v['partition']=='validation');pp=np.array([1/len(cc)/cc[m['date']]/len(cm[m['sample_id']]['states']) for m in meta[part]])
                if pp is None:pp=prob[ii]/prob[ii].sum()
                for arm in ['NET_ZERO','POS_KL']:
                    for seed in range(3):
                        name=f'{split}__{arm}_seed{seed}';go=Q[name+'__'+part][ii,1]>0;yy=y[ii];miss=float(pp@(np.maximum(yy,0)*~go));waste=float(pp@(np.maximum(-yy,0)*go));assert abs(miss+waste-M[name][tag]['sign_regret'])<1e-8
                        risk.append(dict(split=split,scope=tag,arm=arm,seed=seed,missed_positive_value=miss,wasted_query_value=waste,total=miss+waste))
    grouped={}
    for scope in ['full_train','validation','fold_fit','fold_held']:
        grouped[scope]={}
        for arm in ['NET_ZERO','POS_KL']:
            rr=[r for r in risk if r['arm']==arm and ((scope=='full_train' and r['split']=='full' and r['scope']=='train') or (scope=='validation' and r['scope']=='validation') or (scope=='fold_fit' and r['scope']=='fit') or (scope=='fold_held' and r['scope']=='held'))];grouped[scope][arm]={k:float(np.mean([r[k] for r in rr])) for k in ['missed_positive_value','wasted_query_value','total']}
    records=[];cohorts={};call_contributions={}
    for dom,fn,part in [('validation','full_validation_trajectories.json','validation'),('oof','oof_trajectories.json','train')]:
        rows=read(O/fn);look={(r['method'],r['sample_id']):r for r in rows};ids=[r['sample_id'] for r in rows if r['method']=='NET_ZERO_seed0'];cc=collections.Counter(cm[sid]['date'] for sid in ids);calls=collections.defaultdict(lambda:dict(contribution=0.,changed_records=0,profitable_false_stops=0))
        for sid in ids:
            case=cm[sid]
            for seed in range(3):
                a=look[f'POS_KL_seed{seed}',sid];b=look[f'NET_ZERO_seed{seed}',sid];weight=1/3/len(cc)/cc[case['date']];delta=a['net_utility']-b['net_utility'];call=calls[case['physical_state_group']];call['contribution']+=weight*delta
                j=next((j for j,(aa,bb) in enumerate(zip(a['actions'],b['actions'])) if aa!=bb),None)
                if j is None:assert a['actions']==b['actions'] and abs(delta)<1e-12;continue
                i=lookup[part][sid,j];s=support[part,i];label=(case['gains'][j]-D['cost_per_query']*case['queries'][j])/D['gain_scale'];group=('baseCONT_newSTOP' if b['actions'][j] else 'baseSTOP_newCONT')+('__nextPositive' if label>0 else '__nextNonpositive');call['changed_records']+=1;call['profitable_false_stops']+=int(group=='baseCONT_newSTOP__nextPositive');records.append(dict(domain=dom,seed=seed,sample_id=sid,position=j,index=i,part=part,physical_state_group=case['physical_state_group'],group=group,weighted_tail_difference=weight*delta,nearest_positive_fraction=s['nearest_positive_fraction'],beyond95=s['beyond_fit_callout95'],target_net=float(label),nearest_mean_net=s['nearest_mean_net']))
        expected=read(O/'decision.json')['family_utility'][dom]['POS_KL']-read(O/'decision.json')['family_utility'][dom]['NET_ZERO'];assert abs(sum(v['contribution'] for v in calls.values())-expected)<1e-12;call_contributions[dom]=[dict(call=k,**v) for k,v in sorted(calls.items(),key=lambda kv:kv[1]['contribution'])]
        cohorts[dom]={}
        for group in sorted({r['group'] for r in records if r['domain']==dom}):
            rr=[r for r in records if r['domain']==dom and r['group']==group];by=collections.defaultdict(list)
            for r in rr:by[r['physical_state_group']].append(r)
            cohorts[dom][group]=dict(records=len(rr),calls=len(by),call_balanced_nearest_positive_fraction=float(np.mean([np.mean([r['nearest_positive_fraction'] for r in vv]) for vv in by.values()])),call_balanced_beyond95=float(np.mean([np.mean([r['beyond95'] for r in vv]) for vv in by.values()])),tail_contribution=sum(r['weighted_tail_difference'] for r in rr))
    expected=read(O/'divergence_summary.json')
    for dom,groups in cohorts.items():
        for group,v in groups.items():assert v['records']==expected[dom]['POS_KL'][group]['records'] and abs(v['tail_contribution']-expected[dom]['POS_KL'][group]['utility_contribution'])<1e-12
    save(T/'risk_parts.json',dict(rows=risk,summary=grouped));save(T/'divergence_support.json',dict(cohorts=cohorts,records=records));save(T/'all_call_contributions.json',call_contributions);save(T/'review.json',dict(passed=True,all_risk_decompositions_match_saved_metrics=True,all_actual_divergences_match_saved_counts_and_tails=True,all_call_contributions_conserved=True,source_support_reused_without_refitting=True,new_fits=0,new_forwards=0,new_physical_calls=0))
    lines=['# GRID127补充：训练—留出差距落在哪里','',
      '此补充为事后只读诊断，不改变GRID127模型、主指标或判读条件。全状态风险与实际完整搜索尾部分开记账，不能相互替代。','',
      '## 单状态决策遗憾拆成两项','',
      '遗憾=停止时漏掉的正净收益+继续时支付的无利净查询代价。按与原报告相同的状态权重计算，三个种子平均；折内训练/留出再按四折平均。','']
    for scope,arms in grouped.items():
        for arm,v in arms.items():lines.append(f'- {scope}/{arm}：漏正收益{v["missed_positive_value"]:.8f}，无利继续{v["wasted_query_value"]:.8f}，合计{v["total"]:.8f}。')
    lines+=['','## 首次实际分歧处的训练支持','',
      '复用GRID125原41维最近训练状态与排除同次调用的95%距离参照，未改变距离、选k或重新拟合。组别由真实下一批标签事后定义，不能作为模型输入或新触发条件。','']
    for dom,gg in cohorts.items():
        for group,v in gg.items():lines.append(f'- {dom}/{group}：{v["records"]}条记录/{v["calls"]}次调用；按调用均衡的近邻正收益比例{100*v["call_balanced_nearest_positive_fraction"]:.2f}%，超覆盖参照比例{100*v["call_balanced_beyond95"]:.2f}%，尾部贡献{v["tail_contribution"]:+.8f}。')
    lines+=['','## 判断','',
      '训练侧与留出侧需要分开解释：少付无利查询费本身可能改善训练遗憾；真正需要迁移的是少数值得继续的机会。如果留出侧漏正收益的代价重新增加，就不能用训练损失或总体查询节省宣布机制成功。',
      '这些邻域统计不能证明特征充分或不足，也不能证明换成结构网络必然有效。它们只把下一问题收窄到有利机会的跨日期支持与识别，而不支持继续盲目调损失。',
      '每次调用的全部正负贡献保存在all_call_contributions.json，未筛除反例；前缀与训练种子相关，不计作独立新场景。没有新增权重更新、预测、电网回放或最终测试。']
    (T/'训练与留出差距.md').write_text('\n'.join(lines)+'\n',encoding='utf8');dest=A/'posthoc_transfer';dest.mkdir(exist_ok=False);shutil.copy2(W/'transfer_diagnosis.py',dest/'transfer_diagnosis.py')
    for f in T.iterdir():
        if f.is_file():shutil.copy2(f,dest/f.name)
    save(dest/'archive_manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()});c=R/'论文协作资料/FRAME_均值收益目标学习_GRID127/posthoc_transfer';shutil.copytree(T,c)
    docs('## 2026-10-23 — GRID127 posthoc training/held transfer decomposition\n\nReadonlyallstatesrisk=missedpositivenet+wastednonpositiveCONT,training/validation/fourfoldfit-heldkeptseparate. AllactualKLvsNETfirstdivergencesjoinsourceGRID125fit-onlysupport,allper-calltailcontributionsconserved,alloutcomeskept. Notnewfit/trigger/metricselection/independenttest. Researchfocusmovesfromrepeatlossreplacementtohighvalueopportunitytransfer; no physical/dataexpansion/test.',
      '# Next steps — 学习目标已验证，重点转向跨日期泛化\n\n1. 汇报GRID126/127共60次训练：KL修正部分训练机制，但未稳定超过原净收益回归。\n2. 依据逐调用与公开结构记录，研究高收益机会的跨日期可迁移信息；明确区分训练误差、代理查询权衡与物理质量。\n3. 保留原模型；不重复换损失、择种子、扩大数据或开启最终测试。若确需扩大独立日期覆盖，先给出固定规模方案。\n\n汇总：research_archive/power-grid-2026/protocols/GRID127_mean_gain_objective/complete/GRID126_127_阶段汇总.md\n补充：research_archive/power-grid-2026/protocols/GRID127_mean_gain_objective/posthoc_transfer/训练与留出差距.md\n')
    print('TRANSFER_DECOMPOSITION_PASSED',grouped,cohorts,flush=True)

if __name__=='__main__':main()
