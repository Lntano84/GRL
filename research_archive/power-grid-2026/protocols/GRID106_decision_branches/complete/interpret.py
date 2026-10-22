"""Posthoc comparison with already completed learners; no new physical calls or fits."""
import gzip,json
import numpy as np
from common106 import ROOT,OUT,read,save,sha,digest
def rows(p):
    with gzip.open(p/'steps.jsonl.gz','rt',encoding='utf8') as f:return [json.loads(s) for s in f]
def compare(a,b):
    keys=['before_hash','action_hash','after_hash','raw_cost','done','complete']
    same=len(a)==len(b) and all(all(x[k]==y[k] for k in keys) for x,y in zip(a,b))
    return dict(exact_physical_and_cost_trace=same,
        first_action_difference=next((x['step'] for x,y in zip(a,b) if x['action_hash']!=y['action_hash']),None),
        first_state_difference=next((x['step'] for x,y in zip(a,b) if x['after_hash']!=y['after_hash']),None))
def call_at(folder,step):
    for p in sorted(folder.glob('call[0-9][0-9][0-9].json')):
        c=read(p)
        if c['step']==step:return p,c
    return None,None
def main():
    assert not (OUT/'posthoc_interpretation.json').exists()
    base=OUT/'runs';old=ROOT/'outputs/grid104_learning_closedloop/runs';week='2037-03-02_100'
    p108=base/f'r0__{week}__SKIP_108';p552=base/f'r0__{week}__SKIP_552'
    a108=rows(p108);a552=rows(p552);rf=old/f'r0__{week}__LOCAL_RF'
    comparisons={};used=[]
    for name,target,source in [('SKIP_108_vs_BATCH1',p108,old/f'r0__{week}__LOCAL_BATCH1'),
            ('SKIP_552_vs_DQN0',p552,old/f'r0__{week}__LOCAL_DQN0_0'),
            ('SKIP_552_vs_DQN1',p552,old/f'r0__{week}__LOCAL_DQN0_1'),('SKIP_108_vs_RF',p108,rf)]:
        comparisons[name]=compare(rows(target),rows(source))
        used.extend([target/'steps.jsonl.gz',source/'steps.jsonl.gz'])
    step=comparisons['SKIP_108_vs_RF']['first_action_difference'];detail=dict(step=step)
    paid=[]
    if step is not None:
        p,c=call_at(p108,step);q,b=call_at(rf,step)
        if p is not None and q is not None:
            with np.load(p.with_name(p.stem+'_state.npz')) as x,np.load(q.with_name(q.stem+'_state.npz')) as y:
                same=np.array_equal(x['observation'],y['observation']) and np.array_equal(x['base_action'],y['base_action'])
            fields=['pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible','phase']
            globalpart=lambda z:[{k:r[k] for k in fields} for r in z['outcomes'] if r['phase']!='local_completion']
            same=bool(same and globalpart(c)==globalpart(b) and c['order']==b['order'] and c['aliases']==b['aliases'])
            detail.update(same_observation_base_and_global_feedback=same,frame_local_queries=c['local_queries'],rf_local_queries=b['local_queries'],
                reference_chosen=c['chosen_pool_id'],rf_chosen=b['chosen_pool_id'],rf_decisions=b['learned_decisions'])
            if same:
                globalrows=[r for r in c['outcomes'] if r['phase']!='local_completion'];local=[r for r in c['outcomes'] if r['phase']=='local_completion']
                value=lambda r:float(np.float32(r['reward']))
                initial=max(value(r) for r in globalrows if r['strict_admissible']);best=initial
                fee=read(ROOT/'outputs/grid102_search_rl/design.json')['cost_per_query'];points=[0.];gains=[]
                for j in range(0,len(local),64):
                    for r in local[j:j+64]:
                        if r['strict_admissible']:best=max(best,value(r))
                    gains.append(best-initial);points.append(best-initial-fee*min(j+64,len(local)))
                detail.update(actual_next_gain=gains[0] if gains else 0.,actual_next_cost=fee*min(64,len(local)),
                    actual_next_net=points[1] if len(points)>1 else 0.,best_net_stopping_point=max(points),
                    full_local_gain=best-initial,
                    limitation='New saved single-omission branch supplies paid feedback at this common observed context. No third intervention; not a causal week label for this RF decision.')
                paid=[p,q,p.with_name(p.stem+'_state.npz'),q.with_name(q.stem+'_state.npz')]
    result=dict(posthoc=True,comparisons=comparisons,rf_after_beneficial_108_omission=detail,new_fits=0,new_forecasts=0,new_physical_steps=0,
        input_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paid+used+[OUT/'labels.json']},
        labels_not_added_to_training=True,original_objective_misalignment_not_proven_by_108_case=True)
    save(OUT/'posthoc_interpretation.json',result)
    labels=read(OUT/'labels.json');a=next(l for l in labels if l['case_id']=='MARCH_108' and l['horizon']=='remaining_week');b=next(l for l in labels if l['case_id']=='MARCH_552' and l['horizon']=='remaining_week')
    lines=['# 两次关键决策：核查结论与训练前通知','',
        '三条回放和保存数据审计已完成，训练尚未启动。下面区分已验证的单次效果和进一步的只读定位。','',
        '## 1. 第108步：取消补查在成本与查询上更好','',
        f"只取消这一次198候选的同站补查、后续恢复FRAME，仍完成2017步，整周成本减少{abs(a['delta_cost_skip_minus_frame']):.3f}（约0.451%），全程实际预测查询减少831次。实际过载步数则从39增到41，不能扩大为所有风险指标都更好。",
        '该分支与原固定一批策略的整周物理观测、执行动作和逐步成本是否完全一致：'+str(comparisons['SKIP_108_vs_BATCH1']['exact_physical_and_cost_trace'])+'。查询轨迹不要求相同，因而不据此声称两种搜索策略完全等价。',
        '必须更正此前的解释：第108步不是已证实的“即时目标正确、长期质量却变差”。本轮该次取消在即时净收益与整周成本方向上一致。原RF策略整周成本较差，不能归给这一处停止。一步最大负载率奖励与整周经济成本在定义上不同，但这次反例不足以证明“目标不一致导致了这一轮失败”。','',
        '## 2. 第552步：这一次漏查足以产生明显成本损失','',
        f"只取消这一次13候选的补查、后续恢复FRAME，同样完成2017步，但整周成本增加{b['delta_cost_skip_minus_frame']:.3f}（约1.758%），全程实际预测查询反而增加45次。该次补查在当前即时目标下也值得做：增益0.00709510减费用0.00016175，净收益0.00693335。",
        '取消分支与原DQN种子0/1整周物理观测、执行动作和逐步成本完全一致：'+str(comparisons['SKIP_552_vs_DQN0']['exact_physical_and_cost_trace'])+' / '+str(comparisons['SKIP_552_vs_DQN1']['exact_physical_and_cost_trace'])+'。因此，在这个固定场景和FRAME续行条件下，单独漏掉这一次补查已经重现它们的整周物理及成本损失；不只是看到两个整策略的相关差异。',
        '不过风险不是单一数字：本轮保留补查的剩余时域过载步数更少（25对27），峰值却更高（1.87291对1.33674）。不能把成本和查询的优势写成全面安全优势，也不外推其他日期。','',
        '## 3. RF后来在哪里又开始偏离','',
        f"与只取消108后恢复FRAME相比，原RF下一次执行动作差异在第{step}步。这是同一批已保存轨迹的进一步定位，没有新增第三条取消实验。"]
    if detail.get('same_observation_base_and_global_feedback'):
        lines += [f"该处分歧有相同的物理观测、基础动作和全局反馈。FRAME补查{detail['frame_local_queries']}个、RF补查{detail['rf_local_queries']}个；已支付反馈对应下一批净收益{detail.get('actual_next_net',0):+.8f}，完整补查增益{detail.get('full_local_gain',0):.8f}。这定位了下一处可核对的估值问题，但尚未给该次RF决定独立生成整周因果标签。"]
    else:lines += ['该处分歧尚不能凭共同上下文给出完整的补查反事实标签，保留未确认状态。']
    lines += ['', '## 4. 对下一轮训练的影响','',
        '当前更直接的修复重点是“少数关键正收益补查被漏掉”，而不是先修改奖励系数。先保持原数据、41维输入、64×64隐藏层和费用，准备直接净收益回归与现有DQN的受控对照，检查训练采样与估值方式。训练时同时保留误停、误查和净收益代价，不能用提高所有状态的补查概率来掩盖错误。',
        '长期目标仍要继续研究，但不应因一次未经验证的归因就推翻现有奖励。若以后学习整周价值，沿用单次干预和共同续行的标签流程，不把整个策略的整周差额分发到每个决策。',
        '这是开始训练前的结果通知。本次没有启动拟合，也没有把这两个已暴露验证案例加入训练。两个案例和新RF定位都只是开发解释，既不构成新独立测试，也不支持扩大规模。']
    (OUT/'两次决策结论_中文.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
