"""Read-only explanation of GRID168 held-date first differences; no new policy."""
from pathlib import Path
import collections,hashlib,importlib.util,json,math,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid169_held_stop_mechanism'
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID169_held_stop_mechanism'
spec=importlib.util.spec_from_file_location('g169_source',R/'work/grid168_one_sided_transfer/run.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S);sha=S.sha;read=S.read
FIELDS={39:'log_next_batch_queries',37:'local_batches_done',38:'local_budget_used_fraction',40:'locked_remaining_fraction',16:'reward_gap',31:'last_batch_reward_gain'}

def save(name,obj):(O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def stats(values):
    a=np.asarray(list(values),float)
    return None if not len(a) else dict(n=len(a),minimum=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),maximum=float(a.max()))
def manifest(dest):
    (dest/'archive_manifest.json').write_text(json.dumps({p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')

def main():
    assert not O.exists();O.mkdir(parents=True);S.verify_trained();assert read(S.O/'complete.json')['passed_audit']
    design=dict(id='GRID169_HELD_STOP_MECHANISM',authorization='User sustained exploration of recurring small positive changes.',scope='Read-only all held-date paired records of prespecifiedGRID168fold0; no newfit/forward/threshold/policy.',
        questions=['How much net change is query-fee saving versus lost local gain?','Are wrong stops missing an immediately profitable next batch or a later gain after an unprofitable batch?','Do correct savings recur across seeds and physical groups, and what public evidence was visible?'],
        summaries='Allfourdates/threeseeds; outcomes for audit only. Fixed six public fields, descriptive distributions only, no fitted boundary or significance claim.',
        fields=FIELDS,prohibited=['newprediction/fits','rule/price/cutoff search','newdata/physics/final8','paper'])
    save('design.json',design)
    files=[S.O/n for n in ['design.json','metadata.json','fold_arrays.npz','held_paired_events.json','held_trajectories.json','paired_summary.json','family_summary.json','review.json','complete.json','source_hashes.json','code_freeze.json','trained_freeze.json']]
    files +=[R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid121_date_transfer/train_arrays.npz',Path(__file__)]
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in files})
    events=read(S.O/'held_paired_events.json');meta=read(S.O/'metadata.json');ix={(m['sample_id'],m['position']):i for i,m in enumerate(meta)}
    raw=np.load(R/'outputs/grid121_date_transfer/train_arrays.npz')['raw'];cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())}
    d=read(S.O/'design.json');fee=d['cost_per_query'];scale=d['gain_scale'];records=[]
    for e in events:
        if not e['saved_queries']:continue
        case=cases[e['sample_id']];j=e['new_batches'];k=e['base_batches'];i=ix[e['sample_id'],j]
        missed_gains=case['gains'][j:k];saved=case['queries'][j:k];assert k>j
        assert int(sum(saved))==e['saved_queries'] and abs(float(sum(missed_gains))+e['gain_delta'])<1e-12
        future=[int(n+j) for n,g in enumerate(missed_gains) if g>0]
        if not future:kind='NO_GAIN_IN_SKIPPED_SUFFIX'
        elif case['gains'][j]>0:kind='IMMEDIATE_GAIN_MISSED'
        else:kind='DELAYED_GAIN_MISSED'
        firstnet=(float(case['gains'][j])-fee*int(case['queries'][j]))/scale
        row=dict(e,first_missed_positive_positions=future,missed_gain_kind=kind,skipped_batches=k-j,
            normalized_query_saving=fee*e['saved_queries']/scale,normalized_gain_loss=e['gain_delta']/scale,
            next_net=firstnet,next_profitable=firstnet>0,public_fields={v:float(raw[i,n]) for n,v in FIELDS.items()},
            margin_factor=e['scores']['FACTOR'][1]-e['scores']['FACTOR'][0],margin_routed=e['scores']['ROUTED'][1]-e['scores']['ROUTED'][0],margin_small=e['scores']['COST42'][1]-e['scores']['COST42'][0])
        assert row['margin_factor']>0 and row['margin_routed']<=0 and row['margin_small']<=0
        assert abs(row['normalized_query_saving']+row['normalized_gain_loss']-row['net_delta'])<1e-12
        records.append(row)
    save('all_changed_first_differences.json',records);subgroups=[]
    for cat in sorted({r['category'] for r in records}):
        rr=[r for r in records if r['category']==cat]
        subgroups.append(dict(category=cat,pairs=len(rr),physical_groups=len({r['physical_state_group'] for r in rr}),dates=sorted({r['date'] for r in rr}),
            net_contribution=math.fsum(r['weight']*r['net_delta'] for r in rr),query_saving_contribution=math.fsum(r['weight']*r['normalized_query_saving'] for r in rr),gain_loss_contribution=math.fsum(r['weight']*r['normalized_gain_loss'] for r in rr),
            missed_gain_kinds=dict(collections.Counter(r['missed_gain_kind'] for r in rr)),next_profitable_pairs=sum(r['next_profitable'] for r in rr),
            next_queries=stats(r['next_queries'] for r in rr),saved_queries=stats(r['saved_queries'] for r in rr),
            public_field_distributions={name:stats(r['public_fields'][name] for r in rr) for name in FIELDS.values()},
            decision_margins={name:stats(r[name] for r in rr) for name in ['margin_factor','margin_routed','margin_small']}))
    save('mechanism_subgroups.json',subgroups)
    groups=[]
    for group in sorted({r['physical_state_group'] for r in records}):
        rr=[r for r in records if r['physical_state_group']==group];pos=[r for r in rr if r['net_delta']>0];neg=[r for r in rr if r['net_delta']<0]
        groups.append(dict(physical_state_group=group,dates=sorted({r['date'] for r in rr}),positive_seeds=sorted({r['seed'] for r in pos}),negative_seeds=sorted({r['seed'] for r in neg}),pairs=len(rr),positive_pairs=len(pos),negative_pairs=len(neg),starts=len({r['sample_id'] for r in rr}),net_contribution=math.fsum(r['weight']*r['net_delta'] for r in rr)))
    save('physical_group_recurrence.json',groups)
    counts=dict(changed_pairs=len(records),unchanged_pairs=len(events)-len(records),positive_pairs=sum(r['net_delta']>0 for r in records),negative_pairs=sum(r['net_delta']<0 for r in records),
        positive_physical_groups=sum(bool(g['positive_seeds']) for g in groups),positive_in_at_least2seeds=sum(len(g['positive_seeds'])>=2 for g in groups),positive_in_all3seeds=sum(len(g['positive_seeds'])==3 for g in groups),mixed_sign_groups=sum(bool(g['positive_seeds']) and bool(g['negative_seeds']) for g in groups),
        query_saving_contribution=math.fsum(r['weight']*r['normalized_query_saving'] for r in records),gain_loss_contribution=math.fsum(r['weight']*r['normalized_gain_loss'] for r in records),net_contribution=math.fsum(r['weight']*r['net_delta'] for r in records))
    assert abs(counts['net_contribution']-read(S.O/'paired_summary.json')['held']['mean_net_delta'])<1e-12
    save('mechanism_summary.json',counts)
    lines=['# GRID169：留出日期上的单向停止，哪里省对了、哪里停早了','',
        f"全部{len(events)}条种子×起点配对中，{counts['changed_pairs']}条提前停；其中{counts['positive_pairs']}条净效用提高、{counts['negative_pairs']}条降低，{counts['unchanged_pairs']}条不变。所有变化来自削减原本的继续查询，没有新增继续。",'',
        f"日期宏平均净变化{counts['net_contribution']:+.8f}，由查询费用节省{counts['query_saving_contribution']:+.8f}和局部收益损失{counts['gain_loss_contribution']:+.8f}组成。这解释的是同一保存起点上的局部搜索变化，不是整周收益。",'',
        '## 哪些变化可以重复看到','',
        f"正向变化涉及{counts['positive_physical_groups']}个物理状态组；{counts['positive_in_at_least2seeds']}个组在至少2个种子出现正向变化，其中{counts['positive_in_all3seeds']}个组在3个种子都出现。{counts['mixed_sign_groups']}个组同时有正反变化。同一物理组内多个前缀和种子不算独立场景。",'',
        '| 类型 | 配对数 | 物理组 | 下一批本身净正 | 无收益后缀 / 立即漏收益 / 延迟漏收益 |','|---|---:|---:|---:|---|']
    for g in subgroups:
        kinds=g['missed_gain_kinds'];lines.append(f"| {g['category']} | {g['pairs']} | {g['physical_groups']} | {g['next_profitable_pairs']} | {kinds.get('NO_GAIN_IN_SKIPPED_SUFFIX',0)} / {kinds.get('IMMEDIATE_GAIN_MISSED',0)} / {kinds.get('DELAYED_GAIN_MISSED',0)} |")
    lines+=['','## 如何解释','',
        '正确节省意味着原FACTOR计划继续评估的一段候选没有带来额外局部收益，而两个帮助模型都已建议停止。单向约束把这种共识用在提前停上，并保留原FACTOR已做出的停止决定。它把“减少无利评估”和“重新打开搜索”分开，后者不会抵消前者的节省。',
        '', '收益损失需要区分两种：下一批就有可获得增益，属于机会识别不足；下一批无利但更后面有收益，属于短视停止。这里仅描述实际记录中的情况，不能假定所有失误都来自同一个原因。未来后缀只用于核验，不作为模型特征。',
        '', '固定六项公开量及三模型决策分数的分布保存在mechanism_subgroups.json。它们用于检查模型在怎样的已知信息下改变决定，不从本轮留出结果反推新阈值、黑名单或触发规则。',
        '', '## 下一步','',
        '继续验证已固定的小改善：使用剩下三组既有日期折，保持同样的三个组成模型、训练预算和单向规则，所有日期折共同报告。这样能区分机制能否多日期迁移与某一折偶然合适；不根据这一折挑选下一折，也不因为存在损失就丢弃整个方向。',
        '', '目前尚未增加物理场景、最终测试或完整周回放。是否值得进入真实控制验证，还要结合各折收益损失、节省规模与三模型推理开销决定。']
    (O/'GRID169_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    for path,h in read(O/'source_hashes.json').items():assert sha(R/path)==h,path
    S.verify_trained();save('review.json',dict(passed=True,all_held_pairs=len(events),changed_first_differences_checked=len(records),all_public_fields_raw_saved_prefix=True,margin_direction_checked=True,weighted_net_decomposition_checked=True,new_fits=0,new_forwards=0,new_physics=0,final8_absent=True))
    save('decision.json',dict(next='Allremaining3 pre-existingfolds jointly fixed; no fold selection, rule tuning, new physics or final test.',new_training_not_started=True,adopted=False))
    save('complete.json',dict(passed_audit=True,new_fits=0,new_model_forwards=0,new_physics=0))
    dest=A/'complete';dest.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    manifest(dest);collab=R/'论文协作资料/FRAME_留出停止机制_GRID169';collab.mkdir(exist_ok=False)
    for n in ['GRID169_report.md','mechanism_summary.json','mechanism_subgroups.json','physical_group_recurrence.json','review.json','complete.json']:shutil.copy2(O/n,collab/n);assert sha(collab/n)==sha(O/n)
    msg=f"GRID169只读解释GRID168全部{len(events)}条留出配对：{counts['positive_pairs']}对净正、{counts['negative_pairs']}对净负；正向{counts['positive_physical_groups']}物理组中{counts['positive_in_at_least2seeds']}组至少两种子重现。费用节省{counts['query_saving_contribution']:+.8f}、收益损失{counts['gain_loss_contribution']:+.8f}分列，立即与延迟漏收益分列。六项公开量描述无阈值拟合。下一步剩余三固定日期折统一核查，不因局部损失一票否定、不挑下一折。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        path=REPO/'docs'/name;old=path.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID169留出停止机制\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        path.write_text(old,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID169固定机制跨折\n\n1. 保持单向停止规则，通知训练后统一完成剩余3组既有日期折，27个固定组成模型拟合；不挑折或调阈值，全部保留。\n2. 结合4折逐日期/种子效用、查询、收益损失与物理组复现解释，再研究推理成本；不以全赢门槛抹去小改善。\n3. 新数据/物理扩大先给具体范围，最终8不动，论文不改。\n',encoding='utf8',newline='\n')
    index=REPO/'research_archive/power-grid-2026/README.md';index.write_bytes('# 最新：GRID169留出停止机制\r\n\r\n[报告](protocols/GRID169_held_stop_mechanism/complete/GRID169_report.md)；正反变化、立即和延迟漏收益分列，固定规则继续核查其余日期折。\r\n\r\n'.encode('utf8')+index.read_bytes())
    for p,h in read(dest/'archive_manifest.json').items():assert sha(dest/p)==h
    print('GRID169_COMPLETE',json.dumps(counts,ensure_ascii=False))

if __name__=='__main__':main()
