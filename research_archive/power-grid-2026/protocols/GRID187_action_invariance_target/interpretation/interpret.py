"""Post hoc descriptive decomposition, no scores changed or new model forwards."""
from pathlib import Path
import collections,json,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
O=C.O;R=C.R;read=C.read

def main():
    C.verify_trained();assert read(O/'complete.json')['passed']
    assert not (O/'interpretation.json').exists()
    C.save('interpretation_plan.json',dict(posthoc=True,questions=['Which dates improve across all fixed seeds?','Are saved queries on unchanged full-local choices?','Where are restoration and loss repeated across seeds and physical groups?','How do query and reward changes decompose?'],new_fits=0,new_model_forwards=0,new_physics=0,code_sha256=C.sha(Path(__file__))))
    tr=read(O/'oof_trajectories.json');by={}
    for x in tr:by.setdefault(x['method'],{})[x['sample_id']]=x
    metrics=read(O/'search_metrics.json');states=read(O/'state_metrics.json');fam=read(O/'family_summary.json')
    dates=sorted({x['date'] for x in tr});n={d:sum(x['date']==d for x in by['NET_seed0'].values()) for d in dates}
    categories=collections.defaultdict(lambda:dict(pairs=0,query_macro_delta=0.,gain_macro_delta=0.,ids=set(),groups=set()))
    per=[];direction={}; changed=read(O/'paired_action_audit.json')['all_changed_action_examples']
    for d in dates:
        seedrows=[]
        for seed in range(3):
            a=next(v for v in metrics[f'CHANGE_seed{seed}']['per_date'] if v['date']==d);b=next(v for v in metrics[f'NET_seed{seed}']['per_date'] if v['date']==d)
            seedrows.append(dict(seed=seed,action_match_delta_pp=100*(a['action_match_rate']-b['action_match_rate']),query_delta=a['mean_queries']-b['mean_queries'],gain_delta=a['mean_gain']-b['mean_gain'],NET_delta=a['mean_net_utility']-b['mean_net_utility']))
        per.append(dict(date=d,seed_results=seedrows,all_seeds_match_improved=all(v['action_match_delta_pp']>0 for v in seedrows),all_seeds_match_worse=all(v['action_match_delta_pp']<0 for v in seedrows),mean_match_delta_pp=float(np.mean([v['action_match_delta_pp'] for v in seedrows])),mean_query_delta=float(np.mean([v['query_delta'] for v in seedrows]))))
    for seed in range(3):
        for sid,old in by[f'NET_seed{seed}'].items():
            new=by[f'CHANGE_seed{seed}'][sid];same=old['chosen_pool_id']==new['chosen_pool_id'];delta=new['queries']-old['queries'];weight=1/(3*16*n[new['date']])
            key=('same_full_action' if new['action_matches_full_local'] else 'same_nonfull_action')+('_fewer' if delta<0 else '_more' if delta>0 else '_equal') if same else ('restored_full_action' if new['action_matches_full_local'] else 'lost_full_action' if old['action_matches_full_local'] else 'both_mismatch_different')
            c=categories[key];c['pairs']+=1;c['query_macro_delta']+=weight*delta;c['gain_macro_delta']+=weight*(new['gain']-old['gain']);c['ids'].add(sid);c['groups'].add(new['physical_state_group'])
            direction.setdefault(sid,dict(group=new['physical_state_group'],date=new['date'],restored=[],lost=[]))
            if key=='restored_full_action':direction[sid]['restored'].append(seed);assert delta>0
            if key=='lost_full_action':direction[sid]['lost'].append(seed);assert delta<0
    cats={k:dict(**{a:b for a,b in v.items() if a not in ['ids','groups']},unique_starts=len(v['ids']),physical_groups=len(v['groups'])) for k,v in categories.items()}
    allrestored=[dict(sample_id=k,**v) for k,v in direction.items() if len(v['restored'])==3];alllost=[dict(sample_id=k,**v) for k,v in direction.items() if len(v['lost'])==3]
    group=collections.defaultdict(lambda:dict(restored=0,lost=0,examples=[]))
    for x in changed:
        g=group[x['physical_state_group']]
        if x['category']=='restored_full_local_action':g['restored']+=1
        elif x['category']=='lost_full_local_action':g['lost']+=1
        g['examples'].append(x)
    assert abs(sum(v['query_macro_delta'] for v in cats.values())-(fam['CHANGE']['search_macro']['mean_queries']-fam['NET']['search_macro']['mean_queries']))<1e-10
    assert abs(sum(v['gain_macro_delta'] for v in cats.values())-(fam['CHANGE']['search_macro']['mean_gain']-fam['NET']['search_macro']['mean_gain']))<1e-10
    out=dict(posthoc=True,per_date=per,categories=cats,all3seed_restored_starts=allrestored,all3seed_lost_starts=alllost,group_examples=dict(group),no_new_fits_or_forwards_or_physics=True)
    C.save('interpretation.json',out)
    positive=[x for x in per if x['all_seeds_match_improved']];negative=[x for x in per if x['all_seeds_match_worse']]
    a=fam['NET'];b=fam['CHANGE'];counts_a=a['state_counts'];counts_b=b['state_counts'];total=counts_a['states']
    lines=['# GRID187结果解释：保留重复改善，也查清主要损失','',
        '这是固定训练、评分及主报告完成后的描述性分析。没有重新评分、筛日期、修改阈值或追加训练；全部16日期仍在总体结果中。主报告中的条件性讨论在此按本轮实际结果具体化。','',
        f"总体上，新目标的动作一致率低0.826个百分点，三种子均下降；奖励增量保留率由{100*a['search_macro']['gain_capture']:.2f}%变为{100*b['search_macro']['gain_capture']:.2f}%。查询平均少7.37%，但各种子分别+0.60%、−31.21%、+18.74%，不能称为稳定的总体省查询改进。原NET继续作为现有主模型。",'',
        '## 哪些改善值得继续看','',
        f"有{len(positive)}个日期在全部三个种子上提高动作一致率："+'、'.join(x['date'] for x in positive)+'。这些是重复的开发线索，不能因为总体下降就删掉；也不能事后只用这些日期重算总成绩。','',
        '|日期|三种子动作一致率变化（百分点）|平均每起点查询变化|','|---|---|---:|']
    for x in positive:lines.append(f"|{x['date']}|"+' / '.join(f"{v['action_match_delta_pp']:+.2f}" for v in x['seed_results'])+f"|{x['mean_query_delta']:+.3f}|")
    lines += ['',f"同时，有{len(negative)}个日期在全部三个种子上下降。按相同起点比较，全部种子都恢复完整同站动作的起点有{len(allrestored)}个，属于{len({v['group'] for v in allrestored})}个物理调用组；全部种子都丢失原NET正确动作的起点有{len(alllost)}个，属于{len({v['group'] for v in alllost})}个物理调用组。多个前缀共享同一物理组，不能当作独立复现。",'',
        '## 模型行为到底改变了什么','',
        f"遍历全部留出状态时，预测STOP的数量从{counts_a['predicted_STOP']}/{total}增至{counts_b['predicted_STOP']}/{total}；在会改变动作的状态上，错误STOP由{counts_a['wrong_STOP']}增至{counts_b['wrong_STOP']}。对本来不会改变动作的继续判断则由{counts_a['unnecessary_CONTINUE']}减至{counts_b['unnecessary_CONTINUE']}。",'',
        f"因此，总体准确率{100*a['state_macro']['accuracy']:.2f}%→{100*b['state_macro']['accuracy']:.2f}%的提高，主要伴随更多STOP和更少无效继续；它没有带来更好的会变状态召回。不能把准确率提高写成保持动作更好。类别占比、目标幅度变化和跨日期估计都可能参与，目前尚未通过单独干预区分原因。",'',
        '## 查询与收益变化分解','',
        '下表的查询和奖励变化均为日期等权、再对三种子平均的贡献，各行之和等于总体变化。完整同站动作与FULL参考控制器不是同一概念。','',
        '|配对类别|配对数|物理组数|查询变化贡献|奖励增量变化贡献|','|---|---:|---:|---:|---:|']
    labels=dict(same_full_action_fewer='完整同站动作相同、查询更少',same_full_action_more='完整同站动作相同、查询更多',same_full_action_equal='完整同站动作与查询相同',same_nonfull_action_fewer='同样偏离完整动作、查询更少',same_nonfull_action_more='同样偏离完整动作、查询更多',same_nonfull_action_equal='同样偏离完整动作且查询相同',restored_full_action='恢复完整同站动作',lost_full_action='丢失原NET的完整同站动作',both_mismatch_different='双方偏离且动作不同')
    for k,v in sorted(cats.items()):lines.append(f"|{labels[k]}|{v['pairs']}|{v['physical_groups']}|{v['query_macro_delta']:+.4f}|{v['gain_macro_delta']:+.9f}|")
    lines += ['',
        '1982个同动作省查询配对不全是保持完整同站动作，也包含两者选择同一个非完整动作。上表已拆开，避免把它们全部算成无质量取舍的节省。动作恢复有实际例子，但丢失动作也更多；二者要与收益幅度一起解释。','',
        '## 接下来先查什么','',
        '优先比较跨三个种子重复恢复与重复丢失的物理调用组：恢复发生在哪个搜索深度、已购反馈有什么变化、被错过的增量是否更集中，以及下一批计划能否表示决定性的剩余候选。使用这些问题形成可验证的解释，不先改阈值或加一组权重。','',
        '新的二元目标把不同幅度的动作改变等同处理，而原NET保留收益幅度和费用信息。这是有依据的候选解释；本轮同时改变了标签语义与幅度，尚不能把结果归因于类别不平衡或目标信息损失中的某一个因素。下一批显式输入至多64个动作、标签覆盖全后缀的范围差异也需保留。','',
        '不立即部署替换原NET，不以此否定动作保持目标或强化学习。先把有重复改善的小群体解释清楚，再决定最小干预；新物理范围另行明确，最终8、论文和旧成本反例保持不动。','',
        'interpretation.json保留全部日期、分类分解、三种子共同恢复/丢失起点及物理组案例；没有新增模型前向、电网预测或优化器更新。']
    (O/'结果解释.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.verify_trained()
    C.save('interpretation_review.json',dict(passed=True,additive_query_and_gain_decomposition=True,all16dates_retained=True,new_fits=0,new_model_forwards=0,new_physics=0))
    C.archive('interpretation')
    dest=R/'论文协作资料/FRAME_动作不变目标对照_GRID187'
    for name in ['结果解释.md','interpretation.json','interpretation_review.json']:shutil.copy2(O/name,dest/name);assert C.sha(O/name)==C.sha(dest/name)
    msg=f"GRID187解释：总体动作一致−.826pp、收益保留63.75%→43.83%，查询三种子+.60%/−31.21%/+18.74%不稳定，保留旧NET主模型。4日期所有种子动作一致提高，重复恢复{len(allrestored)}起点/{len({v['group'] for v in allrestored})}物理组；重复丢失{len(alllost)}起点/{len({v['group'] for v in alllost})}组。先解释小群体与主要损失，不因总体下降删除重复改善，也不自动调权/阈值/重训或物理。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/name;p.write_text(p.read_text(encoding='utf8')+'\n\n### GRID187结果解释\n\n'+msg+'\n',encoding='utf8',newline='\n')
    print('GRID187_INTERPRETED',json.dumps(dict(positive_dates=[x['date'] for x in positive],negative_dates=len(negative),all3_restored_starts=len(allrestored),all3_restored_groups=len({v['group'] for v in allrestored}),all3_lost_starts=len(alllost),all3_lost_groups=len({v['group'] for v in alllost}),categories=cats),ensure_ascii=False))

if __name__=='__main__':main()
