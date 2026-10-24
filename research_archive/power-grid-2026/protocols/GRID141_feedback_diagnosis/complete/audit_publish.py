"""Independent checks and descriptive publication; no fitting or policy change."""
from run import *
def main():
    verify();assert read(O/'state.json')['stage']=='analysed_pending_review';assert not (O/'review.json').exists()
    fit=rowsfile('fit_rows.json.gz');cal=rowsfile('calibration_rows.json.gz');val=C.eligible_data('validation');rows=fit+cal+val;parts=['fit']*386+['calibration']*198+['validation']*142
    z=np.load(S/'train_scores.npz');scores=np.concatenate([z['fit'],z['calibration'],np.load(S/'validation_scores.npz')['scores']]);pre=read(S/'preprocessing.json')
    with gzip.open(O/'public_supplemental_inputs.jsonl.gz','rt',encoding='utf8') as f:supp=[json.loads(s) for s in f]
    assert len(supp)==726;fieldnames=read(O/'design.json')['supplemental_features'];assert len(fieldnames)==28
    source_checks=0
    for r,s in zip(rows,supp):
        c=C.call_at(r['meta']['source']);n=r['meta']['visible_count'];assert C.P.input_at(c,n)['features']==r['features'];assert s['sample_id']==r['meta']['sample_id']
        assert s['features']==extra(dict(order=c['order'],aliases=c['aliases'],rho_limit=c['rho_limit'],outcomes=c['outcomes'][:n]),n)
        seen=set(c['order'][:n]);best=C.P.best(c['outcomes'][:n]);initial=C.P.best(c['outcomes'][:128]);extraids=set()
        if best is not None and n<len(c['order']) and (initial is None or initial['rho']>=.9 or C.P.bid(initial)!=C.P.bid(best)):
            station=C.subids()[best['pool_id']];extraids={int(rep) for alias,rep in c['aliases'].items() if C.subids()[int(alias)]==station}-seen
        assert n+len(extraids)==s['features']['stop_total_queries_known_now'];source_checks+=1
    with (S/'model.pkl').open('rb') as f:est=pickle.load(f)
    xx=C.transform(rows,pre['features'],pre['medians']).astype(np.float32);weights=np.asarray([r['weight'] for r in read(S/'fit_sample_weights.json')]);labels=np.asarray([r['target']['stop_label'] for r in fit],bool)
    datecalls={d:len({r['meta']['call_id'] for r in fit if r['meta']['date']==d}) for d in {r['meta']['date'] for r in fit}};counts=collections.Counter(r['meta']['call_id'] for r in fit)
    expected=np.asarray([1/(12*datecalls[r['meta']['date']]*counts[r['meta']['call_id']]) for r in fit]);expected*=386/expected.sum();assert np.max(abs(expected-weights))<1e-12
    leafrecords={(r['tree'],r['node']):r for r in read(O/'bootstrap_leaf_supports.json')};nodechecks=0;manual=np.zeros(726);boot=est.estimators_samples_
    signatures=[[] for _ in rows]
    for ti,tree in enumerate(est.estimators_):
        t=tree.tree_;nodeids=[]
        for x in xx:
            k=0
            while t.children_left[k]!=t.children_right[k]:k=int(t.children_left[k] if x[t.feature[k]]<=t.threshold[k] else t.children_right[k])
            nodeids.append(k)
        for i,k in enumerate(nodeids):manual[i]+=t.value[k,0,1]/t.value[k,0,:].sum()/64;signatures[i].append(k)
        for k in set(nodeids[:386]):
            ids=[int(j) for j in boot[ti] if nodeids[int(j)]==k]
            if not ids:continue
            u=set(ids);rr=leafrecords[ti,k]
            assert rr['unique_rows']==len(u) and rr['bootstrap_rows']==len(ids)
            assert rr['dates']==len({fit[j]['meta']['date'] for j in u}) and rr['calls']==len({fit[j]['meta']['call_id'] for j in u})
            p=sum(weights[j] for j in ids if labels[j])/sum(weights[j] for j in ids);assert abs(p-rr['stop_fraction'])<1e-10;nodechecks+=1
    assert np.max(abs(manual-scores))<1e-10 and nodechecks==455
    groups=collections.defaultdict(set)
    for r,signature in zip(rows,signatures):groups[tuple(signature)].add(r['target']['stop_label'])
    assert sum(len(v)>1 for v in groups.values())==read(O/'analysis_summary.json')['collisions']['forest_leaf_signature']['mixed_groups']
    pp=read(O/'all_negative_matched_pairs.json');normal=read(O/'matching_normalization.json');nn=xx.astype(float).copy();nn[:,:18]=(nn[:,:18]-normal['medians'])/normal['scales'];byid={r['meta']['sample_id']:i for i,r in enumerate(rows)}
    for pair in pp:
        i=byid[pair['sample_id']];r=rows[i];eligible=[j for j,s in enumerate(fit) if s['target']['stop_label'] and s['meta']['date']!=r['meta']['date'] and s['meta']['visible_count']==r['meta']['visible_count'] and s['features']['prefix_safe']==r['features']['prefix_safe']]
        assert bool(eligible)==pair['matched']
        if eligible:
            j=min(eligible,key=lambda j:(float(np.mean((nn[i]-nn[j])**2)),fit[j]['meta']['sample_id']));assert fit[j]['meta']['sample_id']==pair['nearest_positive_sample_id']
    paired=[]
    for part in ['fit','calibration','validation']:
        pairs=[p for p in pp if p['partition']==part and p['matched']]
        for feature in fieldnames:
            vv=[(p['date'],float(p['supplemental_negative'][feature])-float(p['supplemental_positive'][feature])) for p in pairs if p['supplemental_negative'][feature] is not None and p['supplemental_positive'][feature] is not None]
            dates=sorted({d for d,v in vv});dd=[dict(date=d,median_negative_minus_positive=float(np.median([v for dd,v in vv if dd==d]))) for d in dates]
            paired.append(dict(partition=part,feature=feature,pairs=len(vv),negative_greater=sum(v>0 for d,v in vv),equal=sum(v==0 for d,v in vv),negative_less=sum(v<0 for d,v in vv),median_difference=float(np.median([v for d,v in vv])) if vv else None,by_date=dd))
    save('all_supplemental_paired_differences.json',paired)
    dateweights=[]
    for d in sorted(datecalls):
        ids=[i for i,r in enumerate(fit) if r['meta']['date']==d];dateweights.append(dict(date=d,rows=len(ids),calls=datecalls[d],raw_stop_fraction=float(np.mean(labels[ids])),weighted_stop_fraction=float(np.sum(weights[ids]*labels[ids])/weights[ids].sum())))
    save('fit_date_weighting_diagnosis.json',dict(raw_stop_fraction=float(labels.mean()),weighted_stop_fraction=float(np.sum(weights*labels)/weights.sum()),dates=dateweights,weights_unchanged=True))
    cases=read(O/'prespecified_cases.json');bad=next(r for r in cases if r['call_id']=='2037-10-05_100|call000');matched=bad['matched_pair'];a=matched['original_features_negative'];b=matched['original_features_positive'];sa=matched['supplemental_negative'];sb=matched['supplemental_positive']
    conclusions=dict(weighted_stop_prior=float(np.sum(weights*labels)/weights.sum()),raw_stop_prior=float(labels.mean()),high_confidence_error_score=bad['score'],high_confidence_error_pure_trees=bad['pure_stop_trees'],high_confidence_error_leaf_dates=bad['bootstrap_leaf_unique_dates'],new_features_not_validated=True,reliable_trigger_not_established=True,new_fit_not_authorized_by_this_analysis=True)
    save('decision.json',dict(decision='REPRESENTATION_AND_WEIGHTING_DIAGNOSIS_ONLY',**conclusions,ready_to_scale=False,adopted=False))
    review=dict(passed=True,source_public_rows_checked=source_checks,bootstrap_leaves_independently_checked=nodechecks,manual_scores_checked=726,max_manual_score_error=float(np.max(abs(manual-scores))),nearest_opposite_pairs_checked=len(pp),source_weights_reconstructed=True,public_suffix_checks=726,known_query_counts_checked=726,fixed_labels_and_model_unchanged=True,new_fits=0,new_predict_calls=0,new_physical=0,final_tests_absent=True,audit_code_sha256=sha(Path(__file__)))
    verify();save('review.json',review)
    lines=['# GRID141：为什么高分停止仍会判断错','',
        '本轮只分析GRID140已锁定模型与已有726个前缀。没有重新训练、调整阈值、运行物理仿真或生成测试场景。结果有助于定位下一步，不代表新增模型已经改善。','',
        '## 这轮进展在哪里','',
        'GRID140把任务固定为提前结束全局搜索，再保留原补查；在62个开发验证状态上，校准模型保持了全部原FRAME动作，查询宏平均减少4.14%。这证明本轮找到了一小部分可以使用的停止机会。不过带相同补查的CAP256也保持了全部动作，减少8.08%，当前学习没有建立额外优势。此前局部补查学习与本轮全局压缩的任务不同，不能把两个数字直接相减声称性能提升。','',
        '## 1. 类别权重进一步放大了停止倾向','',
        f"拟合前缀中331/386可以停止，原始比例{100*labels.mean():.2f}%；应用已冻结的日期、调用、前缀分层等权后，加权比例为{100*np.sum(weights*labels)/weights.sum():.2f}%。前缀数量与每调用权重并非相同口径。该权重处理用于防止单个日期或长调用占据多数，但它在这份数据上也进一步提高了停止标签的有效占比。",'',
        '默认0.5在全部三个分区都退化为有当前可用动作就停止，与CAP128逐调用动作和查询完全相同。较高多数类准确率不能作为学到停止机制的证据。权重和标签均未改动；本轮没有试新的类别权重。','',
        '## 2. 高分错误不是单一日期的小叶节点造成的','',
        f"2037-10-05_100的call000在192查询处停止分数为{bad['score']:.9f}，但会改变原FRAME动作。64棵树中{bad['pure_stop_trees']}棵树落入加权训练样本全为停止的叶节点；对应叶节点包含的独立拟合日期为最少{bad['bootstrap_leaf_unique_dates']['minimum']}、中位数{bad['bootstrap_leaf_unique_dates']['median']:.0f}、最多12。",'',
        '因此，“这个高分只记住了一两个日期”不符合该案例。树叶内训练样本几乎都能停止，不等于新日期同一叶内也能停止。完整重建64棵树的bootstrap采样、样本权重和455个叶节点，与原树分数一致；未调用新的模型预测。','',
        '原18维特征在726条记录中没有完全相同的两行，不能据此证明信息足够或不足。完整64棵树的叶节点组合有4组同时包含两类答案，共9条记录，说明这份浅层模型确实把部分需要区分的状态压到了相同表示；这只解释一部分案例，不是所有错误的完整归因。','',
        '## 3. 无改善摘要没有保留候选分布','',
        f"上述错误与其跨日期、同192查询、同初始安全分支的最近拟合正例{matched['nearest_positive_call_id']}相比，都有1个无改善批次，当前最佳候选未变；分数分别为{matched['score']:.6f}与{matched['nearest_positive_score']:.6f}。",'',
        f"但最新64个候选的最好奖励距当前最好奖励，错误例仅差{a['incumbent_reward']-sa['tail64_best_reward']:.6f}，正例差{b['incumbent_reward']-sb['tail64_best_reward']:.6f}。最新一批的负载率中位数分别为{sa['tail64_rho_q50']:.6f}与{sb['tail64_rho_q50']:.6f}；可接纳比例分别为{sa['tail64_admissible_fraction']:.6f}与{sb['tail64_admissible_fraction']:.6f}。这些都是停止前已有的信息，却没有完整保留在旧的最好值/是否改善摘要中。",'',
        f"两例候选变电站覆盖率则很接近：{sa['seen_station_fraction']:.4f}与{sb['seen_station_fraction']:.4f}；尚未触及的站为{sa['unseen_station_count']}与{sb['unseen_station_count']}个。这说明不能凭这一对案例把站点覆盖量当作已经有效的触发条件。",'',
        '这是一对按事前指定距离取得的反标签案例，不是新规则的验证。全部122个继续前缀均保留，121个能匹配到不同日期、同查询深度和初始分支的拟合正例，1个缺少这种匹配；全部28项新增统计的配对差异及逐日期方向都保存，未只筛选方向有利的特征。','',
        '## 4. 标签仍是动作保持，不是业务收益','']
    for r in read(O/'negative_label_summary.json'):
        lines.append(f"- {r['partition']}：{r['n']}个继续前缀，{r['reasons'].get('original_reward_higher',0)}个提前动作预测奖励更低，{r['reasons'].get('stop_reward_higher',0)}个提前动作预测奖励反而更高。")
    lines += ['',
        '后一类仍为继续标签，是因为本任务要求保持原FRAME动作，并非认为较高的一步奖励一定有害。全部8个此类前缀原样保留，没有为了学习效果修改标签。预测奖励的微小差别也不能直接判断整周经济成本。','',
        '## 下一步','',
        '优先研究最近批次的分布信息与当前最好候选之间的关系，并保留可预知的后续补查成本。28项公开统计已完成资格核对，不能直接视为有效新特征集；部分字段是原查询深度或固定候选顺序的派生量，可能没有额外信息。先研究哪些统计有跨日期稳定的区分，再决定一个小规模学习方案，不直接扩网络、转RL或扩大仿真。', '',
        '类别不平衡、概率失准与决策阈值是不同问题。未来若研究训练权重，需事前固定一套处理及评价规则，不能看到这轮验证结果后扫描最有利的权重或阈值。本轮不执行第二次训练。','',
        '所有新增统计仅从已查询前缀、公开候选顺序和别名/变电站归属计算；726项隐藏后缀扰动核对与726项停止时可知查询成本复算通过。未来反馈只用于标签和核验。所有日期均已暴露于开发，最终8个测试场景未生成；没有整周质量或实际时间的新结论。']
    (O/'GRID141_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save('complete.json',dict(passed=True,analysis_only=True,new_fits=0,new_predict_calls=0,new_physical=0,ready_to_scale=False,final_tests_absent=True));save('state.json',dict(stage='complete'));verify();archive('complete')
    auditcopy=A/'complete/audit_publish.py';shutil.copy2(Path(__file__),auditcopy);am=read(A/'complete/archive_manifest.json');am[auditcopy.name]=sha(auditcopy);(A/'complete/archive_manifest.json').write_text(json.dumps(am,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    collab=R/'论文协作资料/FRAME_全局反馈诊断_GRID141';collab.mkdir(exist_ok=False)
    names=['GRID141_report.md','design.json','analysis_summary.json','decision.json','review.json','complete.json','fit_date_weighting_diagnosis.json','prespecified_cases.json','all_supplemental_paired_differences.json']
    for n in names:shutil.copy2(O/n,collab/n)
    entry='\n\n## 2026-10-24 — GRID141全局反馈与高分错误诊断完成\n\n只读原726前缀/固定森林，零拟合、predict调用和物理。原停止331/386=85.75%，分层等权后91.42%，默认0.5退化CAP128。重建64树bootstrap与455叶权重，独立分数及查询/公开输入核验通过。10月192查询高分错误0.996307，37树纯停止叶，叶支持日期最少8/中位12，不是只记住1–2日期。原18维无完全相同行，完整叶组合4混合组9行，不能据此证明任务不可学。全部122负例121有跨日期同深度/初始分支正例匹配；重点案例旧无改善相同，但最近64候选最好差距和分布不同，站点覆盖接近，不宣称新特征有效。28公开统计726后缀扰动/可知停止成本核对通过，全部配对方向保留。114继续前缀原动作一步奖励更高、8个提前动作更高但原动作保持标签不变；非整周收益。后续先研究批次分布和输入表示，未授权自动新训练/扩规模，最终8未生成。报告research_archive/power-grid-2026/protocols/GRID141_feedback_diagnosis/complete/GRID141_report.md\n'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
        if n=='RESEARCH_STATE.md':title,rest=old.split('\n',1);p.write_text(title+'\n'+entry+'\n'+rest,encoding='utf8')
        else:p.write_text(old+entry,encoding='utf8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — 全局反馈诊断后的学习设计\n\n1. GRID141只读核对完成。优先研究最近批次的候选分布与当前最佳动作的关系，再确定少量表示；28候选统计尚未验证预测增量，不直接全部塞进模型。\n2. 训练有效停止先验91.42%，未来类别权重与概率校准要分开设计，先冻结一个方案，不扫描权重/阈值救GRID140结果。本轮没有新拟合。\n3. GRID140主模型62/62动作保持、少查4.14%，CAP256少查8.08%，原筛查失败不变；高置信度错误有跨12日期训练支持，不能归因于单一日期小样本记忆。\n4. 所有122负例/121匹配与全28特征方向已保存。最近邻仅诊断，未来答案不进特征；8个提前动作一步更优仍按原动作保持标签保存，不换目标。\n5. 已暴露开发数据不能再称新独立验证；最终8封存。先通知明确的小规模训练方案再拟合，不扩网络/RL、物理场景或论文。\n',encoding='utf8')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8')+'\n- [GRID141全局反馈与高分错误诊断](protocols/GRID141_feedback_diagnosis/complete/GRID141_report.md)：加权先验、455叶支持及726公开前缀核验；28候选统计未训练，保留全部负例和标签边界。\n',encoding='utf8')
    count=0
    for d in A.iterdir():
        if d.is_dir():
            for n,h in read(d/'archive_manifest.json').items():assert sha(d/n)==h;count+=1
    for n in names:assert sha(collab/n)==sha(O/n)
    verify();save('final_verification.json',dict(passed=True,archive_files_verified=count,collaboration_files_verified=len(names),source_and_model_unchanged=True,new_fits=0,new_physical=0,final_tests_absent=True))
    d=A/'verification';d.mkdir(exist_ok=False);shutil.copy2(O/'final_verification.json',d/'final_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'final_verification.json':sha(d/'final_verification.json')},indent=2)+'\n',encoding='utf8')
    print('GRID141_COMPLETE',json.dumps(conclusions,ensure_ascii=False))
if __name__=='__main__':main()
