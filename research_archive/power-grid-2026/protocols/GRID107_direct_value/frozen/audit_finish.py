"""Audit exported networks independently, all trajectories, then archive every seed."""
import gzip,json,shutil
import numpy as np,torch
from stable_baselines3 import DQN
from common107 import ROOT,W,OUT,REPO,ARCHIVE,read,save,sha,verify,loadcases,normalization,normalized,weight_digest
def main():
    assert not (OUT/'review.json').exists();verify();D=read(OUT/'design.json');fits=read(OUT/'fits.json');assert fits['new_fits']==3
    cases=loadcases();mean,std=normalization();meta=[];X=[]
    for c in cases:
        for j,s in enumerate(c['states']):X.append(normalized(s,mean,std));meta.append((c,j))
    X=np.stack(X);checks=0
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    # Independent NumPy dense-layer forward avoids relying only on the training framework's predict().
    for seed in D['seeds']:
        model=DQN.load(OUT/f'models/direct_seed{seed}_final.zip',device='cpu');weights=model.q_net.state_dict();y=X.copy()
        for j in [0,2,4]:
            y=y@weights[f'q_net.{j}.weight'].numpy().T+weights[f'q_net.{j}.bias'].numpy()
            if j<4:y=np.maximum(y,0)
        with torch.no_grad():q=model.q_net(torch.as_tensor(X)).numpy()
        assert np.allclose(y,q,rtol=2e-5,atol=2e-5)
        assert np.array_equal(np.argmax(y,axis=1),np.argmax(q,axis=1))
        assert weight_digest(model.q_net)==fits['records'][seed]['final_q_hash']
        assert model._n_updates==7250 and model.num_timesteps==0
        with np.load(OUT/f'training_exposure_seed{seed}.npz') as z:
            assert int(z['state_draws'].sum())==7250*32 and len(z['minibatch_losses'])==7250
        checks+=len(X)
    with gzip.open(OUT/'state_predictions.jsonl.gz','rt',encoding='utf8') as f:states=[json.loads(s) for s in f]
    cm={(c['sample_id'],j):(c,j) for c,j in meta}
    for r in states:
        c,j=cm[r['sample_id'],r['position']]
        assert r['target_net']==float(c['gains'][j]-D['cost_per_query']*c['queries'][j])
        assert r['action']==int(r['q1']>r['q0'])
    trajectories=read(OUT/'validation_trajectories.json');byid={c['sample_id']:c for c in cases}
    for r in trajectories:
        c=byid[r['sample_id']];assert c['partition']=='validation';n=r['batches']
        assert r['queries']==sum(c['queries'][:n]) and r['gain']==sum(c['gains'][:n]) and r['chosen_pool_id']==c['chosen'][n]
        assert abs(r['net_utility']-(r['gain']-D['cost_per_query']*r['queries'])/D['gain_scale'])<1e-12
        assert r['actions']==([1]*n+([0] if n<len(c['states']) else []))
    results=read(OUT/'validation_results.json');metrics=read(OUT/'state_metrics.json');probes=read(OUT/'critical_probes.json')
    diff=[results[f'DIRECT{s}']['all']['macro']['mean_net_utility']-results[f'DQN0_{s}']['all']['macro']['mean_net_utility'] for s in D['seeds']]
    trainold=np.mean([metrics[f'DQN0_{s}']['train']['date_macro_realized_sign_regret'] for s in D['seeds']])
    trainnew=np.mean([metrics[f'DIRECT{s}']['train']['date_macro_realized_sign_regret'] for s in D['seeds']])
    agg={kind:{k:float(np.mean([results[(f'DIRECT{s}' if kind=='DIRECT' else f'DQN0_{s}')]['all']['macro'][k] for s in D['seeds']])) for k in ['mean_net_utility','query_fraction','gain_capture','physical_call_balanced_utility']} for kind in ['DIRECT','DQN0']}
    supported=bool(trainnew<trainold and sum(v>0 for v in diff)>=2)
    save(OUT/'review.json',dict(passed=True,numpy_forward_checks=checks,target_sign_checks=len(states),trajectory_checks=len(trajectories),
        model_fits=3,updates_per_seed=7250,scalar_targets_per_seed=464000,all_seeds_retained=True,new_physical_steps=0,final_test_materialized=False))
    save(OUT/'decision.json',dict(training_mechanism_evidence_at_fixed_config=supported,train_date_macro_regret_direct=float(trainnew),
        train_date_macro_regret_dqn=float(trainold),paired_validation_utility_deltas=diff,all_seed_means=agg,
        batch1_utility=results['BATCH1']['all']['macro']['mean_net_utility'],
        direct_mean_above_batch1=agg['DIRECT']['mean_net_utility']>results['BATCH1']['all']['macro']['mean_net_utility'],
        closed_loop_or_quality_advantage_established=False,ready_to_scale=False,automatic_followup=False,
        limits='Full paired supervision and sampling coverage changed together. Four exposed dates, no selected seed/checkpoint; no independent confirmation or whole-week claim.'))
    lines=['# GRID107：相同网络的直接收益回归对照','',
        '用户授权后完成3个固定种子的直接收益回归。沿用原数据、41维输入、64×64隐藏层及查询费用，没有增加物理回放或测试场景。',
        '', '## 这次改变了什么','',
        '保留原DQN的6978参数双输出网络、逐种子初始化、Adam学习率0.0003、Huber损失、梯度截断10和7250次更新。直接学习STOP=0、CONTINUE=下一批即时净收益；gamma0原本也不含下一状态价值，因此不改目标含义。',
        '原DQN每次更新使用64条已访问状态动作；本轮用32个状态的两种动作，共64个标量目标，均为每种子464000个标量目标。每个训练日期等概率、同日期初始前缀等概率、前缀内各可达批次等概率。完整已知监督与状态覆盖同时替代探索/回放，不能把差异单独归给优化器或某一项采样技巧。',
        '所有更新次数在训练前固定，只使用训练标签；日志只看训练损失。三个最终模型全部评价，没有根据验证结果挑检查点、模型或价格。原DQN与RF直接复用，不重训。',
        '', '## 训练拟合与验证误判','']
    for seed in D['seeds']:
        for part,zh in [('train','训练'),('validation','验证')]:
            a=metrics[f'DIRECT{seed}'][part];b=metrics[f'DQN0_{seed}'][part]
            lines.append(f"- 种子{seed}，{zh}：正净收益状态{a['positive']}个，直接回归/DQN漏查{a['false_stop']}/{b['false_stop']}个，非正收益多查{a['unprofitable_continue']}/{b['unprofitable_continue']}次；继续价值RMSE {a['continue_value_rmse']:.6f}/{b['continue_value_rmse']:.6f}。")
    lines+=['', '3582个训练状态来自2378个相关前缀，856个验证状态来自590个前缀；按日期及物理搜索调用另报指标，不当作数千个独立场景。RMSE是归一化价值误差，误判按固定价格下实际下一批净收益符号计数，不是已证明的整周损害次数。',
        '', '## 固定价格下的搜索策略结果','']
    for method in D['evaluation']['methods']:
        v=results[method]['all']['macro']
        lines.append(f"- {method}：4日期宏平均净效用{v['mean_net_utility']:.6f}，查询比例{v['query_fraction']:.2%}，即时增益覆盖{v['gain_capture']:.2%}，物理调用平衡净效用{v['physical_call_balanced_utility']:.6f}。")
    lines+=['',f"直接回归相对配对DQN的净效用差（种子0/1/2）为{diff}。三种子均值：直接回归{agg['DIRECT']['mean_net_utility']:.6f}，DQN{agg['DQN0']['mean_net_utility']:.6f}；固定一批{results['BATCH1']['all']['macro']['mean_net_utility']:.6f}。",'',
        '逐日期、单批与多批结果全部保留在validation_results.json。这里是同一教师反馈上局部查询策略的离线评价，不是新电网闭环；减少查询或提高净效用均不代表整周质量已经恢复，也不能据此声称在线控制提速。',
        '', '## 三个已知上下文的固定检查','']
    for p in probes:
        lines.append(f"- {p['context']}：实际下一批归一化净收益{p['actual_next_normalized_net']:+.6f}；直接回归种子0/1/2动作"+str([p['scores'][f'DIRECT{s}']['action'] for s in D['seeds']])+ '，原DQN动作'+str([p['scores'][f'DQN0_{s}']['action'] for s in D['seeds']])+'（0停，1继续）。')
    lines+=['', '三个上下文已经暴露，只作错误定位，未加入训练，也未用于调整阈值或选模型。某个上下文修正不代表所有后续学习决策会恢复整周轨迹。',
        '', '## 判读','',
        f"训练前预设的训练机制证据条件为{supported}：训练日期宏平均事后符号遗憾下降，且至少2/3配对种子验证净效用提高。该条件是开发诊断，不能代替强对照和独立测试。",
        f"直接回归三种子平均是否超过固定一批：{agg['DIRECT']['mean_net_utility']>results['BATCH1']['all']['macro']['mean_net_utility']}。扩大资格保持False；不自动追加模型、训练量或闭环。",
        '若训练误差降低而验证未改善，仍需区分数据覆盖、特征信息与目标条件期望的难度；若同预算直接监督也没有改善，不据此断言模型无法学习或DQN已经收敛。目标与整周质量的关系仍需独立核查，不能用一次离线指标替代。',
        '', '## 复核','',
        f"独立NumPy逐层前向核对{checks}个状态输出与动作；核对{len(states)}个目标/符号、{len(trajectories)}条搜索策略轨迹。已保存实际训练抽样次数、每步损失、全部种子和最终权重。没有新预测仿真或独立AC重求解，最终8测试场景未生成。"]
    (OUT/'GRID107_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    verify();dest=ARCHIVE/'complete';dest.mkdir(parents=True,exist_ok=False)
    for p in list(OUT.rglob('*'))+list(W.glob('*.py')):
        if p.is_file():
            rel=p.relative_to(OUT) if OUT in p.parents else __import__('pathlib').Path('source')/p.name
            q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    collab=ROOT/'论文协作资料/FRAME_直接收益回归_GRID107_20261022';collab.mkdir(exist_ok=False)
    for n in ['GRID107_report.md','decision.json','review.json','critical_probes.json','validation_results.json']:shutil.copy2(OUT/n,collab/n)
    save(OUT/'complete.json',dict(passed=True,model_fits=3,new_physical_steps=0,final_test_materialized=False,
        archive=str(dest),collaboration=str(collab),report_sha256=sha(OUT/'GRID107_report.md'),ready_to_scale=False))
    shutil.copy2(OUT/'complete.json',dest/'complete.json');shutil.copy2(OUT/'complete.json',collab/'complete.json')
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})
    note='\n\n## 2026-10-22 — GRID107 same-network direct regression completed\n\nExactly3 supervised fits, same41/64/64/2 network6978 parameters, paired original initializations, Adam/Huber and7250 updates.32 states with both actions gives same464000 scalar targets as DQN64 sampled state-actions. Full paid-feedback coverage and supervision changed together; not optimizer-only causality. Original data/normalization/price held, no positive oversampling/validation selection/new physical rollout/final-test access. Mean validation utilities direct/DQN/BATCH1: '+str(agg['DIRECT']['mean_net_utility'])+'/'+str(agg['DQN0']['mean_net_utility'])+'/'+str(results['BATCH1']['all']['macro']['mean_net_utility'])+'; paired differences '+str(diff)+'. All seeds, dates, errors and three exposed diagnostic contexts retained; no automatic scaling or closed-loop claim.\n'
    for n in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write(note)
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID107离线对照完成\n\n1. 阅读全3种子、4日期、固定一批强对照及3个已暴露上下文结果，区分拟合改善与实际策略收益。\n2. 当前不自动增加训练量/网络/日期或物理闭环；扩大资格False。最终8测试场景保持封存。\n3. 根据GRID107实际结果选择下一路线，不从这轮验证中选最好模型或修改费用救结果。\n\n报告：research_archive/power-grid-2026/protocols/GRID107_direct_value/complete/GRID107_report.md\n',encoding='utf8')
    print(json.dumps(read(OUT/'decision.json'),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
