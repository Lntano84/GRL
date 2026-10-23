"""Independent saved-data checks and static-plan coverage; no model execution."""
from run import *
def canonical_key(plan):
    # Same information as bus-label-invariant local partition, independent code.
    p=np.array(plan,dtype=int);active=p>0;first=p[np.flatnonzero(active)[0]] if active.any() else 0
    return tuple((int(live),int(live and v!=first)) for v,live in zip(p,active))
def main():
    verify();assert read(O/'review.json')['passed'];assert not (O/'support_detail.json').exists()
    X,ids,Q,fit,cal,labels,cases=load();records=read(O/'train_calibration_records.json');cohorts=read(O/'gain_scale_and_opportunities.json');portraits=read(O/'critical_public_portraits.json');idx={sid:i for i,sid in enumerate(ids)}
    plan_variants=collections.defaultdict(set);plan_calls=collections.defaultdict(set);uses=collections.Counter();station_counts={};contexts={}
    with gzip.open(R/'outputs/grid134_full_tail_inputs/train_calibration_public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line);sid=r['sample_id'];i=idx[sid]
            if Q[i]<=64:continue
            c=r['public_input'];contexts[sid]=c
            for action_id,plan in zip(c['next_ids'],c['next_plans']):
                plan_variants[action_id].add(canonical_key(plan));plan_calls[action_id].add(labels[i]['call']);uses[action_id]+=1
    for part,m in [('fit',fit&(Q>64)),('calibration',cal&(Q>64))]:
        station_counts[part]=dict(collections.Counter(str(records[i]['station']) for i in np.flatnonzero(m)))
    descriptive={}
    for part,m in [('fit',fit&(Q>64)),('calibration',cal&(Q>64))]:
        rr=[records[i] for i in np.flatnonzero(m)];pp=[r for r in rr if r['gain']>0];small=[r for r in pp if r['gain_ulps']<=4]
        descriptive[part]=dict(positive=len(pp),at_most4ulp=len(small),small_positive_fraction=len(small)/len(pp),small_gain_share=math.fsum(r['gain'] for r in small)/math.fsum(r['gain'] for r in pp),
            no_profitable_prefix_positive_fraction=sum(r['category']=='no_profitable_prefix' for r in pp)/len(pp),partial_only_positive_fraction=sum(r['category']=='partial_only_profitable' for r in pp)/len(pp))
    # Recompute old opportunity categories independently from direct cumulative arrays.
    fee=read(R/'outputs/grid102_search_rl/design.json')['cost_per_query'];checks=0
    for r in records:
        c=cases[r['sample_id']];G=float(np.sum(c['gains']));net=np.r_[0.,np.cumsum(c['gains'])-fee*np.cumsum(c['queries'])]
        cat='zero_gain' if G==0 else ('full_profitable' if net[-1]>0 else ('partial_only_profitable' if net.max()>0 else 'no_profitable_prefix'))
        assert cat==r['category'];checks+=1
    for part,mask in [('fit',fit),('calibration',cal)]:
        for scope,extra in [('all',np.ones(len(Q),bool)),('cheap',Q<=64),('expensive',Q>64)]:
            ii=np.flatnonzero(mask&extra);count=collections.Counter(records[i]['category'] for i in ii)
            assert all(count[k]==v['prefixes'] for k,v in cohorts[part+'_'+scope]['categories'].items())
    miss=[v for v in portraits.values() if v['role']=='known_validation_miss'];outside={}
    for sid,p in portraits.items():
        if p['role']=='known_validation_miss':outside[sid]=[n for n,v in p['all37_public_features']['fit'].items() if v and v['outside_observed_range']]
    facts=dict(passed=True,independent_opportunity_category_checks=checks,expensive_station_prefix_counts=station_counts,
        pending_action_ids=len(plan_variants),pending_plan_rows=sum(uses.values()),
        action_ids_with_one_canonical_plan=sum(len(v)==1 for v in plan_variants.values()),
        canonical_variant_count_distribution=dict(collections.Counter(str(len(v)) for v in plan_variants.values())),
        plan_row_fraction_from_single_variant_ids=sum(uses[k] for k,v in plan_variants.items() if len(v)==1)/sum(uses.values()),
        max_distinct_physical_calls_per_action=max(len(v) for v in plan_calls.values()),gain_scale_details=descriptive,
        known_misses_outside_fit_univariate_ranges=outside,
        caveat='Descriptions conditional on exposed data. Static plans can still interact with paid feedback;no proof structure has no predictive value or full observation would solve it.',
        new_fits=0,new_model_forwards=0,new_thresholds=0,new_physical_calls=0)
    save('support_detail.json',facts)
    text=['# GRID136补充：动作结构覆盖与目标取舍','',
        '以下是已冻结训练/校准数据的覆盖描述与算术复核，没有新训练或门槛评估。','',
        '## 不能再用“没见过这个站”解释','',
        f"大补查拟合站分布：{station_counts['fit']}；校准站分布：{station_counts['calibration']}。它们全部位于原编号48站，恰好也是7条新误停所在站。", 
        '七条误停的37维摘要在同站拟合数据的逐项范围内。由此可以排除“新站完全无训练样本”和“某一列简单越界”作为直接解释；仍不能排除联合条件稀疏、不同日期联合分布变化或未编码状态的影响。','',
        '## 接线计划有多少运行时变化','',
        f"638条大补查前缀累计包含{facts['pending_plan_rows']}条待查动作记录，涉及{facts['pending_action_ids']}个公共代表编号。其中{facts['action_ids_with_one_canonical_plan']}个编号在这些记录中只有一种母线重标不变的站内接线计划，占记录{100*facts['plan_row_fraction_from_single_variant_ids']:.2f}%。", 
        f"每动作的计划变体数分布：{facts['canonical_variant_count_distribution']}。统计按公开候选编号关联，不把编号作为模型输入。", 
        '这些计划主要说明“要做什么连接”。它们与当前最佳动作及已购反馈结合仍可能有价值，但补齐完整待查计划本身没有提供新的全网潮流空间分布。因此当前负结果不等于所有结构学习无效，也不支持继续只扩充动作计划。','',
        '## 微小正收益不是主要正例人群','']
    for part,v in descriptive.items():text.append(f"- {part}大组：≤4ULP正例{v['at_most4ulp']}/{v['positive']}，占正例{100*v['small_positive_fraction']:.2f}%，占总正收益{100*v['small_gain_share']:.6f}%；有收益但没有任何扣费有利前缀的正例占{100*v['no_profitable_prefix_positive_fraction']:.2f}%。")
    text+=['',
        '所以不能把问题说成“大量精度噪声污染了标签”；没有高精度重算，不能认定这些差值是噪声。能确认的是，保护所有正例的极端门槛受少数小增量见证影响，而相当一部分正例按旧代理费任务本来就应停止。','',
        '## 后续研究取舍','',
        '我的建议是保留论文原来的最终要求：首先保持整周完成与运行成本质量，再研究能省多少真实计算。最大负载率增益和代理查询费可以是局部学习信号，但不再把它们直接当成同一个验收目标。',
        '下一步在现有保存记录中先检查每种候选结果究竟存了哪些业务相关量、哪些必须靠新的对照分支才能回答。若只有最大负载率，就不能凭它给“整周可接受损失”贴可靠标签，更不能从当前7条验证错例反推容忍度。',
        '旧逐批任务能表示“查一批再停”，当前跳过/查完门控表示不了；如果回到逐批价值任务，需要继承并解释旧127不如固定一批的反例，不能直接重训原RL作为解决方案。',
        '当前保持原FRAME和GROUP参照；不选择新价格/标签/阈值，不扩大场景或跑物理回放。']
    (O/'动作结构覆盖与目标取舍.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    # Evidence inventory, not a claim these fields are counterfactual week labels.
    sample_meta=read(R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples'];path=next(m['source_call'] for m in sample_meta if m['sample_id'] in {sid for sid,p in portraits.items() if p['role']=='known_validation_miss'})
    call=read(R/path);row=call['outcomes'][0]
    save('saved_field_inventory.json',dict(source_call=path,top_level_keys=sorted(call),outcome_keys=sorted(row),
        purpose='Schema inspection only;one-step rho/reward do not identify whole-week alternative-policy cost.'))
    dest=A/'support_detail';dest.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),O/'support_detail.json',O/'动作结构覆盖与目标取舍.md',O/'saved_field_inventory.json']:shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8')
    verify();print(json.dumps(facts,ensure_ascii=False))
if __name__=='__main__':main()
