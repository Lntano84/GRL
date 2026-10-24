"""Read-only explanation after the frozen one-fit screen; never change the policy."""
from pathlib import Path
import collections,json,pickle,shutil
import numpy as np
import run as G

def distribution(values):
    return dict(min=float(min(values)),median=float(np.median(values)),max=float(max(values)))

def main():
    G.verify_trained();assert G.read(G.O/'review.json')['passed']
    assert not (G.O/'post_diagnosis.json').exists()
    O,R,REPO=G.O,G.R,G.REPO
    rank=[];trees=[]
    entry={p:G.rowfile(O/f'{p}_rows.json.gz') for p in ['fit','calibration','validation']}
    nz=np.load(O/'train_scores.npz');oz=np.load(G.B.O/'train_scores.npz')
    oldfull={'fit':G.rowfile(G.B.O/'fit_rows.json.gz'),'calibration':G.rowfile(G.B.O/'calibration_rows.json.gz'),'validation':G.B.eligible_data('validation')}
    newscore={'fit':nz['fit'],'calibration':nz['calibration'],'validation':np.load(O/'validation_scores.npz')['scores']}
    oldscore={'fit':oz['fit'],'calibration':oz['calibration'],'validation':np.load(G.B.O/'validation_scores.npz')['scores']}
    for part in entry:
        ids={r['meta']['sample_id']:float(s) for r,s in zip(oldfull[part],oldscore[part])}
        oldp=np.asarray([ids[r['meta']['sample_id']] for r in entry[part]])
        for model,pp,tau in [('RF140',oldp,G.read(O/'cutoff.json')['old140_entry_control_threshold']),('RF146',newscore[part],G.read(O/'cutoff.json')['primary_threshold'])]:
            m=G.metrics(entry[part],pp,tau);assert abs(m['continue_ap']-G.manual_ap(entry[part],pp))<1e-12
            rank.append(dict(model=model,partition=part,score_range=distribution(pp),**m))
    for name,root in [('RF140',G.B.O),('RF146',O)]:
        with (root/'model.pkl').open('rb') as f:model=pickle.load(f)
        trees.append(dict(model=name,leaves=distribution([t.tree_.n_leaves for t in model.estimators_]),
            realized_depth=distribution([t.tree_.max_depth for t in model.estimators_]),
            distinct_bootstrap_rows=distribution([len(set(x)) for x in model.estimators_samples_]),
            minimum_leaf_size=model.min_samples_leaf,fit_rows=len(G.rowfile(root/'fit_rows.json.gz'))))
    names=G.read(O/'design.json')['features']
    constant=[n for n in names if len({r['features'][n] for r in entry['fit']})==1]
    records=G.read(O/'all_search_replays.json');val=[r for r in records if r['evaluation_partition']=='validation']
    newstops=[r for r in val if r['method']=='RF146_ENTRY128_CALIBRATED' and r['stop_kind']=='learned_early_stop']
    critical=[]
    refmap={r['call_id']:r for r in G.B.refs()}
    fit=entry['fit'];w=G.weights(fit)
    for r in newstops:
        ref=refmap[r['call_id']];c=G.B.call_at(ref['source']);before=G.B.P.best(c['outcomes'][:128]);rr=next(x for x in entry['validation'] if x['meta']['call_id']==r['call_id'])
        byid={x['pool_id']:x for x in c['outcomes']}
        early=byid[r['selected']];original=byid[r['original_selected']]
        critical.append(dict(call_id=r['call_id'],action_match=r['action_match'],score=r['stopped_score'],queries=r['queries'],original_queries=r['original_queries'],selected=r['selected'],original=r['original_selected'],
            public_features=rr['features'],early_rho=early['rho'],original_rho=original['rho'],reward_gap=float(original['reward']-early['reward']),
            scope='Saved one-step values only;not weekly failure or cost.'))
    oldstops=[dict(call_id=r['call_id'],n=r['global_queries'],action_match=r['action_match'],queries=r['queries'],original_queries=r['original_queries']) for r in val if r['method']=='RF140_CALIBRATED' and r['stop_kind']=='learned_early_stop']
    G.save('post_diagnosis.json',dict(fits=0,new_policy_or_cutoff=False,ranking=rank,forest_granularity=trees,
        fit_constant_features=constant,fit_effective_continue_mass_fraction=float(w[[not r['target']['stop_label'] for r in fit]].sum()/w.sum()),
        all_primary_validation_stops=critical,original_multiexit_validation_stops=oldstops,
        conclusion='Single entry removes nonmonotonic exit structure but did not fix held-out ranking. Reduced fit support makes unchanged leaf20 relatively coarser;this observation does not prove a smaller leaf or larger network would generalize. No automatic second fit.'))
    new=lambda p:next(x for x in rank if x['model']=='RF146' and x['partition']==p)
    old=lambda p:next(x for x in rank if x['model']=='RF140' and x['partition']==p)
    leaves=next(x for x in trees if x['model']=='RF146')
    bad=next(x for x in critical if not x['action_match'])
    lines=['# GRID146结果解释：结构变清楚了，预测没有变好','',
        '本轮假设没有得到性能支持：减少为一次学习决策，可以让完整搜索误停损失随阈值单调，但没有让当前输入和森林更可靠。不能因为单调性证明成立就继续推广单入口。主模型不采纳。','',
        '## 真正需要解释的是跨日期区分下降','',
        '| 相同128入口 | 拟合继续AP / AUC | 校准继续AP / AUC | 开发验证继续AP / AUC |','|---|---:|---:|---:|']
    for name in ['RF140','RF146']:
        mm=[next(x for x in rank if x['model']==name and x['partition']==p) for p in ['fit','calibration','validation']]
        lines.append(f"| {name} | {mm[0]['continue_ap']:.4f} / {mm[0]['auc']:.4f} | {mm[1]['continue_ap']:.4f} / {mm[1]['auc']:.4f} | {mm[2]['continue_ap']:.4f} / {mm[2]['auc']:.4f} |")
    lines += ['',
        f"新模型拟合AUC {new('fit')['auc']:.3f}，校准降到{new('calibration')['auc']:.3f}，开发验证{new('validation')['auc']:.3f}。验证继续类AP {new('validation')['continue_ap']:.3f}，继续比例为16/56=28.57%。这些数据不支持它已稳定学会跨日期继续风险；不作显著性检验或宣称任务不可学习。",'',
        '## 两个作用要分开看','',
        '只把旧RF140改成单入口、完全不重训，开发验证仍62/62保持原动作，但查询节省从原多出口4.14%降至0.31%。旧模型原本的4次有效学习停止都在192查询处；只保留128入口会失去这些后续选择机会。这个控制说明：结构更简单不等于更省查询。','',
        '新模型在单入口结构上重新拟合后有2次停止，一次正确、一次动作不同，查询节省宏平均0.75%含有错误停止节省的查询，不能当同质量提速。正确和错误两条均保留，不选择更有利的一条汇报。','',
        f"错误来自{bad['call_id']}，停止分数{bad['score']:.9f}，高于冻结阈值0.897774783；选候选{bad['selected']}，原FRAME选{bad['original']}，查询{bad['queries']}而原查询{bad['original_queries']}。保存的一步奖励差为{bad['reward_gap']:.9f}，不据此推断整周损失。",'',
        '## 原参数在较小拟合集上变得更粗','',
        f"保留最小叶20，拟合状态从386变为153。旧RF140每棵树叶数为5–9、中位7；新森林为3–4、中位{leaves['leaves']['median']:g}，实际树深2–3。每棵树bootstrap的不同拟合行中位数也从244降到96.5。相同参数并不代表相同有效容量。",'',
        f"128入口有{len(constant)}项原始特征在拟合集中恒定，详见JSON；它们多数是固定深度或当前入口定义带来的退化量。并未据此删特征、添特征或重训。日期等权后的继续质量占{100*G.read(O/'post_diagnosis.json')['fit_effective_continue_mass_fraction']:.2f}%，也不能把问题简单归为之前仅8.58%的少数类质量。",'',
        '这些现象支持检查数据规模与模型粒度是否匹配，但不能直接认定唯一原因是欠拟合。拟合排序明显好于校准/验证，也存在日期间分布差异或现有摘要不足的可能；加深树、减小叶或换神经网络均未被本轮证明有效。','',
        '## 当前判断','',
        '不把“单入口”升级为新主方法，也不据这次结果直接扩大采集或网络。原多出口RF140与固定256继续作为有效参照。下一次若继续研究容量，需要先固定一个由拟合规模决定的单因素方案，与本轮比较，而不是同时改权重、特征、入口和训练算法；训练前通知。当前不执行第二份模型。','',
        '本说明只用冻结后的现有分数、标签和模型结构；未新增预测、拟合、物理、阈值重放或策略选择。主报告、坏结果及原标签全部保持。']
    (O/'单入口结果解释.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    G.verify_trained()
    G.archive('diagnosis')
    c=R/'论文协作资料/FRAME_单入口学习_GRID146'
    for n in ['post_diagnosis.json','单入口结果解释.md']:shutil.copy2(O/n,c/n)
    addition='\n\nGRID146解释：单入口新模型拟合AUC0.769、校准0.519、开发0.469；同入口继续AP及原RF140控制全部另存。旧分数仅改结构查询节省4.14%→0.31%，原4次有效出口均在192；新拟合2次停止1错，0.75%含错误节省，不能作同质量优势。153状态配合原最小叶20，树仅3–4叶（旧5–9），提示有效容量变化，但不证明减叶或扩网络可解跨日期问题。保留原模型/强简单对照，不把单入口采纳或自动再训。\n'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+addition,encoding='utf8',newline='\n')
    p=REPO/'docs/NEXT_STEPS.md';p.write_text(p.read_text(encoding='utf8')+'\n5. 单入口未解决区分且损失192出口；不当新主方法。容量方案若继续，须先固定单因素、训练前通知，不自动减叶/加深/换网络或扩大。\n',encoding='utf8',newline='\n')
    total=0
    for phase in G.A.iterdir():
        for n,h in G.read(phase/'archive_manifest.json').items():assert G.sha(phase/n)==h;total+=1
    for p in c.iterdir():assert G.sha(p)==G.sha(O/p.name)
    G.save('final_verification.json',dict(passed=True,archive_files_verified=total,collaboration_files_verified=len(list(c.iterdir())),
        total_new_fits=1,additional_diagnostic_fits=0,scientific_models_cutoffs_labels_unchanged=True))
    d=G.A/'final_verification';d.mkdir();shutil.copy2(O/'final_verification.json',d/'final_verification.json')
    (d/'archive_manifest.json').write_text(json.dumps({'final_verification.json':G.sha(d/'final_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n')
    print('GRID146_DIAGNOSIS_COMPLETE',json.dumps(dict(ranking=rank,trees=trees,constant_features=constant),ensure_ascii=False))

if __name__=='__main__':main()
