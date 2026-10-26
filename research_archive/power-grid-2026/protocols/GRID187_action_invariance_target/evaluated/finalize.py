"""Saved-record audit and publication; no fitting or model inference."""
from pathlib import Path
import collections, json, pickle, shutil, sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import run as C
R=C.R; O=C.O; W=C.W; A=C.A; REPO=C.REPO
read=C.read; sha=C.sha

def audit():
    C.verify_trained(); assert read(O/'review.json')['passed']
    assert not (O/'complete.json').exists()
    meta=read(R/'outputs/grid113_relational_learning/train_metadata.json')
    cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'}
    lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)}
    scores=np.load(O/'oof_scores.npz'); tr=read(O/'oof_trajectories.json'); sums=read(O/'search_metrics.json')
    assert len(tr)==9*2378
    by=collections.defaultdict(dict); checks=collections.Counter()
    d=read(O/'design.json'); fee=d['cost_per_query']; scale=d['gain_scale']
    for x in tr:
        c=cases[x['sample_id']]; method=x['method']; n=len(c['states']); p=0; actions=[]
        while p<n:
            go=(bool(scores[method][lookup[c['sample_id'],p],1]>scores[method][lookup[c['sample_id'],p],0]) if method in scores.files else method=='FULL_LOCAL' or (method=='BATCH1' and p==0))
            actions.append(int(go))
            if not go: break
            p+=1
        q=sum(int(v) for v in c['queries'][:p]); gain=sum(float(v) for v in c['gains'][:p]); chosen=int(c['chosen'][p]); match=chosen==int(c['chosen'][-1])
        assert x['batches']==p and x['queries']==q and x['actions']==actions and x['chosen_pool_id']==chosen
        assert abs(x['gain']-gain)<1e-12 and abs(x['net_utility']-(gain-fee*q)/scale)<1e-10
        assert x['action_matches_full_local']==match
        assert x['sample_id'] not in by[method]; by[method][x['sample_id']]=x; checks['independent_saved_searches']+=1
    for method, rr in by.items():
        assert len(rr)==2378
        for v in sums[method]['per_date']:
            sub=[x for x in rr.values() if x['date']==v['date']]
            assert abs(np.mean([x['queries'] for x in sub])-v['mean_queries'])<1e-12
            assert abs(np.mean([x['action_matches_full_local'] for x in sub])-v['action_match_rate'])<1e-12
            checks['independent_method_dates']+=1
    pairs=[]; examples=[]
    for seed in range(3):
        counts=collections.Counter(); date_counts=collections.defaultdict(collections.Counter)
        for sid, old in by[f'NET_seed{seed}'].items():
            new=by[f'CHANGE_seed{seed}'][sid]; same=old['chosen_pool_id']==new['chosen_pool_id']; delta=new['queries']-old['queries']
            key=('same_choice_fewer_queries' if delta<0 else 'same_choice_more_queries' if delta>0 else 'same_choice_same_queries') if same else ('restored_full_local_action' if new['action_matches_full_local'] else 'lost_full_local_action' if old['action_matches_full_local'] else 'both_mismatch_different_actions')
            counts[key]+=1;date_counts[new['date']][key]+=1
            if not same:
                examples.append(dict(seed=seed,sample_id=sid,date=new['date'],physical_state_group=new['physical_state_group'],category=key,old_chosen=old['chosen_pool_id'],new_chosen=new['chosen_pool_id'],full_chosen=int(cases[sid]['chosen'][-1]),old_queries=old['queries'],new_queries=new['queries'],old_gain=old['gain'],new_gain=new['gain'],old_NET=old['net_utility'],new_NET=new['net_utility']))
        assert sum(counts.values())==2378
        pairs.append(dict(seed=seed,counts=dict(counts),per_date={k:dict(v) for k,v in sorted(date_counts.items())}))
    C.save('paired_action_audit.json',dict(pairs=pairs,all_changed_action_examples=examples))
    C.save('publication_audit.json',dict(passed=True,checks=dict(checks),same_initialization_and_exposure=True,all12_final_weights_published_before_held=True,source_and_weight_hashes_unchanged=True,all3seeds_all16dates=True,new_physics=0,new_model_forwards=0,final8_absent=True))
    return pairs

def main():
    pairs=audit(); fam=read(O/'family_summary.json'); metrics=read(O/'search_metrics.json'); states=read(O/'state_metrics.json'); paired=read(O/'paired_summary.json')['pairs']; fits=read(O/'fits.json')
    old=fam['NET']['search_macro']; new=fam['CHANGE']['search_macro']
    match_delta=100*(new['action_match_rate']-old['action_match_rate']); query_change=100*(new['mean_queries']/old['mean_queries']-1)
    counts={k:sum(p['counts'].get(k,0) for p in pairs) for k in sorted({k for p in pairs for k in p['counts']})}
    decision=dict(complete_target_comparison=True,action_match_delta_pp=match_delta,date_macro_mean_query_change_percent=query_change,paired_action_counts=counts,automatic_adoption=False,automatic_physical_runs=False,arbitrary_allwin_gate=False,new_training_after_this=False,scope='Recorded FULL-teacher suffixes on historically exposed16trainingdates; OOF optimizer exclusion; not blind newphysical quality evidence')
    C.save('decision.json',decision)
    lines=['# GRID187：只改变学习目标，能否更好地保持动作','',
        '本轮保持原FACTOR网络、输入、日期划分、初始化、训练采样和预算，只把“下一批奖励增益减查询费”的目标，改成“查完剩余同站候选后，当前动作是否会改变”。原NET的12份权重全部复用。','',
        f"三种子、日期等权汇总：动作一致率从{100*old['action_match_rate']:.3f}%变为{100*new['action_match_rate']:.3f}%，变化{match_delta:+.3f}个百分点；每个起点购买的同站查询从{old['mean_queries']:.3f}变为{new['mean_queries']:.3f}，变化{query_change:+.2f}%。这是保持动作与计算量之间的实际变化，不能只凭预测准确率判断。",'',
        '## 固定条件与评价范围','',
        '四个原日期划分，每折12个日期拟合、4个日期留出；每折3个种子，共12次新拟合。每次7250次Adam更新、批量32、原采样流、均值Huber损失和严格q1>q0判断均不变。STOP目标为0，CONTINUE目标为动作会变时+1、不变时−1；没有类别重加权、门槛搜索、额外特征或共识模块。这些分数不是校准概率。','',
        f"全部拟合耗时{fits['total_wall_s']:.1f}秒。12份最终权重先上传，再计算留出日期输出；每个种子的3582个状态均由排除该日期的模型评分。对应2378个起点、222个物理调用组和16个旧训练日期，前缀并非独立场景。原4开发日期未新增评分，最终8未生成。",'',
        '## 保存搜索的结果','',
        '下表每个学习种子均汇总全部16个留出日期；参考策略只计算一次。动作一致指与同一起点的完整已记录同站搜索选择相同。收益保留率指一步奖励增量，不是完整周成本。','',
        '|方法|动作一致率|每起点查询|完整查询占比|奖励增量保留率|原NET效用|','|---|---:|---:|---:|---:|---:|']
    for name in [*[f'{m}_seed{s}' for m in ['NET','CHANGE'] for s in range(3)],'STOP_ALL','BATCH1','FULL_LOCAL']:
        z=metrics[name]['macro'];lines.append(f"|{name}|{100*z['action_match_rate']:.3f}%|{z['mean_queries']:.3f}|{100*z['query_fraction']:.3f}%|{100*z['gain_capture']:.3f}%|{z['mean_net_utility']:.6f}|")
    lines+=['','原NET效用沿用旧固定查询价格和归一化尺度，便于看清目标变化的代价；新模型没有优化这一效用，不把两种目标混称为同一损失。每个日期的完整结果保存在search_metrics.json与paired_summary.json。','',
        '## 错误停止与种子差异','',
        '|模型|会变状态中的错误STOP|错误STOP率（日期等权）|预测STOP的正确率（日期等权）|','|---|---:|---:|---:|']
    for name in [f'{m}_seed{s}' for m in ['NET','CHANGE'] for s in range(3)]:
        z=states[name];c=z['counts'];v=z['macro'];lines.append(f"|{name}|{c['wrong_STOP']}/{c['must_change']}|{100*v['wrong_STOP_rate']:.3f}%|{100*v['STOP_precision']:.3f}%|")
    lines+=['','这里遍历所有留出前缀衡量识别能力；上面的保存搜索则从起点运行停止策略，二者分母不同。多数状态本来不会改变动作，因此不把总体分类准确率作为主结论。','']
    for p in paired:
        dd=p['per_date'];lines.append(f"种子{p['seed']}：动作一致率{100*p['action_match_delta']:+.3f}个百分点，查询{100*p['query_change']:+.2f}%；16个日期中动作一致率提高/相同/降低为{sum(x['action_match_delta']>1e-12 for x in dd)}/{sum(abs(x['action_match_delta'])<=1e-12 for x in dd)}/{sum(x['action_match_delta']<-1e-12 for x in dd)}。")
    lines+=['','## 哪些动作被恢复，哪些仍被错过','',
        '按同一起点、同一种子配对比较两个策略，全部计数如下；三种子的同一物理调用不能当作独立样本。','']
    labels=dict(same_choice_fewer_queries='选择相同且查询更少',same_choice_more_queries='选择相同但查询更多',same_choice_same_queries='选择和查询都相同',restored_full_local_action='新目标恢复完整同站动作',lost_full_local_action='原NET一致而新目标偏离',both_mismatch_different_actions='两者均偏离且选择不同')
    lines += [f"- {labels[k]}：{v}个种子—起点配对。" for k,v in counts.items()]
    lines+=['','paired_action_audit.json保留所有动作变化起点及原/新动作、查询、奖励和NET，不只挑有利案例。小而重复的改善保留其研究价值，同时完整呈现额外计算和反向结果。','',
        '## 能说明什么，下一步看什么','',
        '这轮直接检验监督目标的影响。动作一致率提高说明模型更倾向保留完整同站搜索的控制选择；若伴随更多查询，应继续检查额外查询是否主要用在恢复动作的状态，还是已经不会变化的状态上。若仍有错误停止，则目标对齐并未自动解决估计和跨日期区分问题。','',
        '当前结果来自旧FULL教师路径的保存搜索，没有新的电网预测，也没有测量实际控制时间或整周成本。完整同站动作本身不保证周成本最优；原FRAME与FULL的历史成本反例不因本轮动作一致而消失。全部旧日期已被研究过程使用，这不是盲测泛化成绩。','',
        '为使目标对照可解释，输入也保留了原限制：显式待查计划只包含下一批至多64个动作，目标却涉及完整剩余同站后缀。因此不能预先假设当前输入已经充分表示新目标；这轮也没有补全尾部输入。±1编码同时改变了目标含义和数值范围，结论针对这份固定目标实现，不外推为所有分类损失的结论。','',
        '下一步先解释现有恢复/丢失动作的配对与查询位置，再提出最小真实闭环范围。没有自动扩数据、换电网、改门槛、重训或进入最终测试。','',
        '## 核验与材料','',
        '12组初始化和全部采样次数与原训练匹配；留出优化器抽样为0。旧NET保存输出及决定逐项重现；10746个新留出输出由独立NumPy前向核验。21402条保存搜索另按分数、原查询数组、动作序列和费用重算，全部通过。来源、冻结代码和权重哈希保持一致。','',
        '方案见design.json；权重和曝光记录见models及exposure_*.npz；state_metrics.json、search_metrics.json和paired_summary.json包含全部种子与日期；review.json、publication_audit.json和preheld_publication.json记录核验与评估前发布。']
    (O/'GRID187_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    C.save('publisher_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')})
    C.save('complete.json',dict(passed=True,new_fits=12,reused_old_models=12,all3seeds_all16dates=True,all_weights_published_before_held=True,saved_searches=21402,new_physics=0,old4dev_new_scores=0,final8_absent=True))
    C.archive('complete')
    dest=R/'论文协作资料/FRAME_动作不变目标对照_GRID187';dest.mkdir(exist_ok=False)
    for name in ['GRID187_report.md','review.json','publication_audit.json','decision.json','complete.json','family_summary.json','paired_summary.json']:
        shutil.copy2(O/name,dest/name);assert sha(O/name)==sha(dest/name)
    msg=f"GRID187同架构/输入/初值/采样/预算目标对照12拟合完成；旧NET12权重复用、先发布后评分，16日期OOF。动作一致{100*old['action_match_rate']:.3f}%→{100*new['action_match_rate']:.3f}%({match_delta:+.3f}pp)，查询{old['mean_queries']:.3f}→{new['mean_queries']:.3f}({query_change:+.2f}%)。全部种子/日期和反例保留，无全赢门槛；保存教师后缀非物理整周质量/实际时间，原4开发未新评分，最终8未生成。下一步只解释恢复/丢失动作与查询，再明确小闭环范围；不自动追加拟合/物理。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID187动作不变目标对照\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID187后\n\n1. 解释固定目标对照中各日期恢复/丢失动作、同动作查询增减与跨种子重复，先用已有记录，不按留出结果挑门槛或种子。\n2. 保留动作不变学习信号，区分更像完整同站搜索与真正整周低成本；不把更少查询直接说成实际提速。新闭环必须先明确比较臂、已暴露日期、计时和质量参照再确定范围。\n3. 不自动训练、扩数据、换电网、进入最终8或修改论文；小改善不设全赢或5%门槛。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID187动作不变目标对照\r\n\r\n[报告](protocols/GRID187_action_invariance_target/complete/GRID187_report.md)；12份配对训练、16日期OOF，动作与查询分列，未新物理。\r\n\r\n'.encode('utf8')+p.read_bytes())
    print('GRID187_COMPLETE',json.dumps(decision,ensure_ascii=False))

if __name__=='__main__':main()
