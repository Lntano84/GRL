"""Validate the two-factor comparison and publish every endpoint and adverse result."""
import gzip,json,shutil
import numpy as np,torch
from stable_baselines3 import DQN
from common108 import ROOT,W,OUT,REPO,ARCHIVE,read,save,sha,verify,loadcases,normalization,normalized,weight_digest
def main():
    assert not (OUT/'review.json').exists();verify();D=read(OUT/'design.json');fits=read(OUT/'fits.json')
    assert fits['new_training_paths']==6 and fits['new_checkpoints']==9 and fits['matched_draw_sequences']
    cases=loadcases();mean,std=normalization();X=np.stack([normalized(x,mean,std) for c in cases for x in c['states']]);checks=0;near=0
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    checkpoints={r['method']:r for r in fits['checkpoints']}
    for entry in D['models']:
        m=DQN.load(ROOT/entry['path'],device='cpu');weights=m.q_net.state_dict();y=X.copy()
        for j in [0,2,4]:
            y=y@weights[f'q_net.{j}.weight'].numpy().T+weights[f'q_net.{j}.bias'].numpy()
            if j<4:y=np.maximum(y,0)
        with torch.no_grad():q=m.q_net(torch.as_tensor(X)).numpy()
        assert np.allclose(y,q,rtol=2e-5,atol=2e-5)
        error=np.max(np.abs(y-q),axis=1);margin=np.abs(q[:,1]-q[:,0]);robust=margin>2*error
        assert np.array_equal(y[robust].argmax(axis=1),q[robust].argmax(axis=1));near+=int(np.sum(~robust));checks+=len(X)
        assert m._n_updates==entry['updates'] and m.num_timesteps==0
        assert all(float(v['step'])==entry['updates'] for v in m.policy.optimizer.state.values())
        if not entry['reused']:assert weight_digest(m.q_net)==checkpoints[entry['method']]['q_hash']
    for seed in D['seeds']:
        a=checkpoints[f'HUBER_29000_{seed}'];b=checkpoints[f'MSE_29000_{seed}']
        assert a['stream_sha256']==b['stream_sha256']
        with np.load(OUT/f'draws_HUBER_29000_{seed}.npz') as z:ha=z['state_draws']
        with np.load(OUT/f'draws_MSE_29000_{seed}.npz') as z:mb=z['state_draws']
        assert np.array_equal(ha,mb) and ha.sum()==29000*32
        with np.load(ROOT/f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz') as z:old=z['state_draws']
        with np.load(OUT/f'draws_MSE_7250_{seed}.npz') as z:short=z['state_draws']
        assert np.array_equal(old,short)
    byid={c['sample_id']:c for c in cases};trajectories=read(OUT/'validation_trajectories.json')
    assert len(trajectories)==11210
    for r in trajectories:
        c=byid[r['sample_id']];n=r['batches'];assert c['partition']=='validation'
        assert r['actions']==[1]*n+([0] if n<len(c['states']) else [])
        assert r['queries']==sum(c['queries'][:n]) and r['gain']==sum(c['gains'][:n]) and r['chosen_pool_id']==c['chosen'][n]
        assert abs(r['net_utility']-(r['gain']-D['cost_per_query']*r['queries'])/D['gain_scale'])<1e-12
    old={(r['method'],r['sample_id']):r for r in read(ROOT/'outputs/grid107_direct_value/validation_trajectories.json')};anchorchecks=0
    for r in trajectories:
        if not r['method'].startswith('HUBER_7250_'):continue
        p=old[f'DIRECT{r["method"][-1]}',r['sample_id']]
        for k in ['actions','batches','gain','queries','chosen_pool_id','net_utility']:assert r[k]==p[k]
        anchorchecks+=1
    with gzip.open(OUT/'state_predictions.jsonl.gz','rt',encoding='utf8') as f:predictions=[json.loads(s) for s in f]
    for r in predictions:
        c=byid[r['sample_id']];j=r['position']
        assert r['target_net']==float(c['gains'][j]-D['cost_per_query']*c['queries'][j])
        assert r['action']==int(r['q1']>r['q0'])
    results=read(OUT/'validation_results.json');metrics=read(OUT/'state_metrics.json');probes=read(OUT/'critical_probes.json')
    families={}
    for loss in D['losses']:
        for budget in D['budgets']:
            names=[f'{loss}_{budget}_{s}' for s in D['seeds']]
            families[f'{loss}_{budget}']=dict(
                **{k:float(np.mean([results[n]['all']['macro'][k] for n in names])) for k in ['mean_net_utility','query_fraction','gain_capture','physical_call_balanced_utility']},
                train_false_stops=[metrics[n]['train']['false_stop'] for n in names],validation_false_stops=[metrics[n]['validation']['false_stop'] for n in names],
                critical_actions={p['context']:[p['scores'][n]['action'] for n in names] for p in probes})
    contrasts={}
    for label,left,right in [('loss_at7250','MSE_7250','HUBER_7250'),('loss_at29000','MSE_29000','HUBER_29000'),
            ('budget_underHuber','HUBER_29000','HUBER_7250'),('budget_underMSE','MSE_29000','MSE_7250')]:
        delta=[results[f'{left}_{s}']['all']['macro']['mean_net_utility']-results[f'{right}_{s}']['all']['macro']['mean_net_utility'] for s in D['seeds']]
        contrasts[label]=dict(left=left,right=right,paired_utility_deltas=delta,mean_utility_delta=float(np.mean(delta)),positive_seeds=sum(x>0 for x in delta),
            per_date_mean_delta={date:float(np.mean([results[f'{left}_{s}']['all']['per_date'][date]['mean_net_utility']-results[f'{right}_{s}']['all']['per_date'][date]['mean_net_utility'] for s in D['seeds']])) for date in results[f'{left}_0']['all']['per_date']})
    loss_curves={}
    for loss in D['losses']:
        key='train_weighted_huber' if loss=='HUBER' else 'train_weighted_mse'
        for seed in D['seeds']:
            t=read(OUT/f'trace_{loss}_{seed}.json');start=t[0];short=next((x for x in t if x['updates']==7250),start)
            mid=next(x for x in t if x['updates']==22000);last=t[-1]
            loss_curves[f'{loss}_{seed}']=dict(objective=key,value7250=short[key],value22000=mid[key],value29000=last[key],
                last7000_relative_drop=(mid[key]-last[key])/max(abs(mid[key]),1e-12),convergence_not_claimed=True)
    decision=dict(families=families,contrasts=contrasts,training_objective_curves=loss_curves,
        batch1_utility=results['BATCH1']['all']['macro']['mean_net_utility'],
        old_dqn_family_utility=float(np.mean([results[f'DQN0_{s}']['all']['macro']['mean_net_utility'] for s in D['seeds']])),
        new_training_paths=6,new_checkpoints=9,ready_to_scale=False,closed_loop_or_week_quality_evaluated=False,
        best_model_selected=False,no_automatic_followup=True,limits='Exposed development dates; all seeds/endpoints retained. No convergence proof from a finite budget or causal whole-week claim.')
    save(OUT/'decision.json',decision)
    save(OUT/'review.json',dict(passed=True,numpy_network_checks=checks,near_numeric_ties=near,trajectory_checks=len(trajectories),
        target_sign_checks=len(predictions),old_huber_trajectory_reproductions=anchorchecks,matched_minibatches_checked=True,optimizer_steps_checked=True,
        new_training_paths=6,new_checkpoints=9,new_physical_steps=0,final_test_materialized=False))
    lines=['# GRID108：区分损失函数与训练量的影响','',
        '完成固定的2×2离线对照：Huber与平方损失，各检查7250和29000次更新，全部保留种子0/1/2。Huber短训练直接复用GRID107，不重复拟合；新增3条Huber续训和3条MSE训练路径。没有新电网回放、数据、特征、网络规模或查询费用。','',
        '## 对照怎样保持可比','',
        '同一种子使用原41→64→64→2网络及相同初始化、Adam学习率0.0003、32状态×2动作目标、梯度截断10和逐批抽样顺序。同预算比较只改变损失；同损失比较只改变更新次数。Huber续训恢复原7250更新的权重、Adam动量与后续抽样位置；MSE前7250次抽样逐项重现原Huber样本次数，29000次完整抽样序列哈希也一致。',
        '较长预算为事前固定的4倍上限，依据上一轮训练损失尚在下降选定，不根据验证成绩决定何时停。只保存两个预定端点，不挑选中间检查点。',
        '平方损失对误差取平方；Huber在大误差区间改用线性项。比较针对少数高收益是否被弱化这一假设，不预设平方损失一定更好。[MSE官方公式](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MSELoss.html)，[SmoothL1官方公式](https://docs.pytorch.org/docs/2.8/generated/torch.nn.SmoothL1Loss.html)。','',
        '## 四组结果：全部种子的平均','']
    for name,v in families.items():
        lines.append(f"- {name}：验证日期宏平均净效用{v['mean_net_utility']:.8f}，查询比例{v['query_fraction']:.2%}，即时增益覆盖{v['gain_capture']:.2%}；训练正收益漏查{v['train_false_stops']}，验证漏查{v['validation_false_stops']}。")
    lines += ['',f"固定一批净效用{decision['batch1_utility']:.8f}；原DQN三种子平均{decision['old_dqn_family_utility']:.8f}。净效用使用原固定代理费用，是局部搜索权衡，不等于整周成本或物理控制时间。",'', '## 两个因素分别带来什么','']
    for name,c in contrasts.items():
        lines.append(f"- {name}（{c['left']}减{c['right']}）：配对种子净效用差{c['paired_utility_deltas']}，平均差{c['mean_utility_delta']:+.8f}，正向种子{c['positive_seeds']}/3；日期差{c['per_date_mean_delta']}。")
    lines += ['', '同一因素若在两个预算下方向相反，或种子/日期有反向结果，均按原值保留。不能仅选较好端点宣称损失或时长普遍有效。','', '## 三处已知上下文','']
    for name,v in families.items():lines.append(f"- {name}：{v['critical_actions']}（0停止，1继续）。")
    lines += ['', '这些上下文在此前已经暴露，仅用于定位，不加入训练、不据其选模型。第552步预测修正也不等于修改整套策略后必然恢复整周质量。','', '## 训练是否充分','']
    for name,c in loss_curves.items():lines.append(f"- {name}：本训练目标损失7250/22000/29000更新为{c['value7250']:.8f}/{c['value22000']:.8f}/{c['value29000']:.8f}，末7000次相对下降{c['last7000_relative_drop']:.2%}。")
    lines += ['', '固定上限不构成收敛证明。若训练继续改善而验证收益下降，应保留这种分离，不把继续降低训练误差直接当作部署收益。','', '## 复核与下一步边界','',
        f"独立NumPy前向核对{checks}个网络状态，核对{len(predictions)}个目标与动作、{len(trajectories)}条离线搜索轨迹，精确重现{anchorchecks}条旧Huber短训练轨迹；抽样序列、Adam步数和原冻结哈希均通过。数值近边界样本数{near}，按预定误差界核对。",
        '全部模型、端点、训练曲线、状态预测和不利结果已归档。最终8测试场景未生成，扩大资格仍False。不自动追加训练或启动新闭环，应依据上述对照选择下一步。']
    (OUT/'GRID108_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');verify()
    dest=ARCHIVE/'complete';dest.mkdir(parents=True,exist_ok=False)
    for p in list(OUT.rglob('*'))+list(W.glob('*.py')):
        if p.is_file():
            rel=p.relative_to(OUT) if OUT in p.parents else __import__('pathlib').Path('source')/p.name
            q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    collab=ROOT/'论文协作资料/FRAME_损失与训练量_GRID108_20261022';collab.mkdir(exist_ok=False)
    for n in ['GRID108_report.md','decision.json','review.json','critical_probes.json','validation_results.json']:shutil.copy2(OUT/n,collab/n)
    save(OUT/'complete.json',dict(passed=True,new_training_paths=6,new_checkpoints=9,new_physical_steps=0,
        final_test_materialized=False,ready_to_scale=False,archive=str(dest),collaboration=str(collab),report_sha256=sha(OUT/'GRID108_report.md')))
    shutil.copy2(OUT/'complete.json',dest/'complete.json');shutil.copy2(OUT/'complete.json',collab/'complete.json')
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})
    note='\n\n## 2026-10-22 — GRID108 matched loss x budget completed\n\n2losses(Huber/MSE) x2fixed budgets7250/29000 x3seeds. Old Huber short reused,3 exact Adam/RNG continuations and3 MSE paths; all endpoints kept, identical sample-stream hashes. No data/feature/price/network or physical changes. Family mean utilities '+str({k:v['mean_net_utility'] for k,v in families.items()})+'. All factor/seed/date reversals retained; no best configuration selected or automatic follow-up. Offline only, final8 test ungenerated, scale readinessFalse.\n'
    for n in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write(note)
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID108损失×训练预算对照完成\n\n1. 汇报全部损失/预算/种子、日期与关键552动作，依据配对因素效果判断，不选最优端点。\n2. 尚无物理整周或在线计算优势，不自动追训、加特征/数据、变费用或启动闭环。\n3. 最终8测试场景保持封存，不恢复旧自动化。\n\n报告：research_archive/power-grid-2026/protocols/GRID108_loss_budget/complete/GRID108_report.md\n',encoding='utf8')
    print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
