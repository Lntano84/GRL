"""Post-completion explanation from saved records only; no prediction/fit/forecast."""
from pathlib import Path
import collections,gzip,json,shutil,sys
import numpy as np
ROOT=Path('D:/Codex/GRL研究');sys.path.insert(0,str(ROOT/'work/grid104_learning_closedloop'))
import registry as R
assert R.read(R.OUT/'complete.json')['passed'] and not R.read(R.OUT/'decision.json')['ready_to_scale']
R.verify();review=R.read(R.OUT/'review.json');D=R.read(R.OUT/'design.json');rows=[r for r in review['descriptive_runs'] if r['repeat']==0]
def step_rows(folder):
    with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf8') as f:return [json.loads(s) for s in f]
def folder(week,method):return R.OUT/'runs'/f'r0__{week}__{method}'
def calls(f):return [R.read(p) for p in sorted(f.glob('call[0-9][0-9][0-9].json'))]
signatures={};forecast_sources={}
for m in R.read(ROOT/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples']:
    if m['partition']=='validation' and m['scenario'].endswith('_100'):
        signatures.setdefault(m['scenario'],set()).add(m['state_hash'])
        forecast_sources[m['source_call']]=m['source_sha256']
details=[];overlap={}
for selected in D['selected']:
    week=selected['scenario'];ff=folder(week,'FRAME_GUARD');fr=step_rows(ff);fc=calls(ff)
    for method in ['LOCAL_RF','LOCAL_DQN0_0','LOCAL_DQN0_1','LOCAL_DQN0_2']:
        cf=folder(week,method);cr=step_rows(cf);cc=calls(cf)
        eligible=[c for c in cc if c['completion_budget']>0]
        overlap.setdefault(method,[]).extend(dict(week=week,state=c['state_hash'],matches_full_teacher=c['state_hash'] in signatures[week]) for c in eligible)
        first=next((i for i in range(min(len(fr),len(cr))) if fr[i]['action_hash']!=cr[i]['action_hash']),None)
        entry=dict(week=week,method=method,first_delivered_action_divergence=None if first is None else fr[first]['step'])
        if first is not None:
            assert all(fr[i]['after_hash']==cr[i]['after_hash'] for i in range(first))
            entry['same_before_state']=fr[first]['before_hash']==cr[first]['before_hash']
            step=fr[first]['step'];a=[c for c in fc if c['step']==step];b=[c for c in cc if c['step']==step]
            entry['calls_at_step']=[len(a),len(b)]
            if len(a)==len(b)==1:
                a,b=a[0],b[0]
                ap=ff/f"call{fc.index(a):03d}_state.npz";bp=cf/f"call{cc.index(b):03d}_state.npz"
                with np.load(ap) as za,np.load(bp) as zb:
                    same=bool(np.array_equal(za['observation'],zb['observation']) and np.array_equal(za['base_action'],zb['base_action']))
                entry['same_observation_and_base']=same
                if same:
                    keys=['pool_id','rho','reward','done','strict_admissible','illegal','ambiguous','exceptions','phase']
                    ga=[{k:r[k] for k in keys} for r in a['outcomes'] if r['phase']!='local_completion']
                    gb=[{k:r[k] for k in keys} for r in b['outcomes'] if r['phase']!='local_completion']
                    entry['same_global_feedback']=ga==gb
                    av={r['pool_id']:r for r in a['outcomes']};bv={r['pool_id']:r for r in b['outcomes']}
                    if a['chosen_pool_id'] is not None and b['chosen_pool_id'] is not None:
                        reward=lambda r:float(np.float32(r['reward']))
                        entry.update(frame_choice=a['chosen_pool_id'],learner_choice=b['chosen_pool_id'],
                            frame_forecast_reward=reward(av[a['chosen_pool_id']]),learner_forecast_reward=reward(bv[b['chosen_pool_id']]),
                            reward_gap=reward(av[a['chosen_pool_id']])-reward(bv[b['chosen_pool_id']]),
                            frame_local_queries=a['local_queries'],learner_local_queries=b['local_queries'],
                            frame_choice_not_queried_by_learner=a['chosen_pool_id'] not in bv,
                            learner_stopped=any(d['action']==0 for d in b['learned_decisions']))
        details.append(entry)
means={}
for method in [m['method'] for m in D['methods']]:
    allruns=[r for r in review['descriptive_runs'] if r['method']==method]
    mean=lambda k:float(np.mean([r[k] for r in allruns]))
    means[method]={k:mean(k) for k in ['controller_s','optimizer_s','forecast_s','search_overhead_s','other_s','local_feature_s','local_inference_s']}
    means[method]['optimizer_fraction']=mean('optimizer_s')/mean('controller_s')
    means[method]['learning_overhead_fraction']=(mean('local_feature_s')+mean('local_inference_s'))/mean('controller_s')
profile={m:dict(completion_calls=len(rs),matching_full_teacher_states=sum(r['matches_full_teacher'] for r in rs),
    fraction_matching=sum(r['matches_full_teacher'] for r in rs)/len(rs) if rs else None) for m,rs in overlap.items()}
supplement=dict(passed_saved_record_checks=True,new_fits=0,new_forecasts=0,new_physical_steps=0,posthoc=True,
    timing_means=means,first_divergences=details,full_teacher_overlap=profile,
    limitations='Exact state-hash mismatch is descriptive distribution difference, not a causal explanation. One-step reward gap does not explain all downstream week costs. No counterfactual AC re-solve.')
R.save(R.OUT/'readonly_explanation.json',supplement)
base=means['FRAME_GUARD'];rf=means['LOCAL_RF']
lines=['# GRID104：为什么暂不扩大学习规模','',
 '48条正式回放及保存数据审计已完成，所有原始结果保留。本补充只读取已保存记录，没有训练、查询仿真或新物理步。',
 '', '## 已确认的结果','',
 '全部六种策略均完成3/4个完整周；5月日期全部在1390步提前结束。完整周中，3月RF成本比FRAME增加1.112%，DQN种子0/1增加1.758%，超过原0.5%质量容差；种子2保持了FRAME轨迹，但不能据此挑选最好种子。',
 '', 'RF相对FRAME的实际控制时间日期宏平均减少1.75%、候选查询减少9.38%；相对固定一批仅少0.53%时间、1.08%查询。三个DQN相对FRAME约少1.70%—1.99%时间，相对固定一批约少0.48%—0.77%；没有达到预设扩大条件。所有种子的质量/时间/查询边界以正式报告为准。',
 '', '## 不是模型推理吃掉了大部分节省','',
 f"FRAME平均每场景控制{base['controller_s']:.3f}秒，其中连续优化{base['optimizer_s']:.3f}秒，占{base['optimizer_fraction']:.2%}；全部预测{base['forecast_s']:.3f}秒，搜索非预测开销{base['search_overhead_s']:.3f}秒。",
 f"RF特征与推理合计平均{rf['local_feature_s']+rf['local_inference_s']:.6f}秒，只占其完整控制时间{rf['learning_overhead_fraction']:.3%}。因此，本轮整体收益有限的主要背景是补查只占整个控制流程的一部分，而不是小模型推理本身很慢。不同控制路径会同时改变优化与预测次数，不能把全部时间差都归给局部查询减少。",
 '', '## 第一处分歧的已揭示证据','',
 '| 日期 | 学习策略 | 首次实际动作分歧步 | 共享状态与基础动作 | FRAME补查/学习补查 | 当步预测奖励差（FRAME−学习） |',
 '|---|---|---:|---|---:|---:|']
for e in details:
    if e['first_delivered_action_divergence'] is None:
        lines.append(f"| {e['week']} | {e['method']} | 无 | 相同已执行轨迹 | — | — |")
    else:
        gap=e.get('reward_gap');count=f"{e['frame_local_queries']}/{e['learner_local_queries']}" if 'frame_local_queries' in e else '未定位'
        lines.append(f"| {e['week']} | {e['method']} | {e['first_delivered_action_divergence']} | {e.get('same_observation_and_base')} | {count} | {gap:.8f} |" if gap is not None else f"| {e['week']} | {e['method']} | {e['first_delivered_action_divergence']} | {e.get('same_observation_and_base')} | {count} | 未定位 |")
lines+=['','这里只比较双方在共同状态/基础动作下实际查询过的反馈，不重求解任何未观察动作。当步奖励差可以定位少查是否漏掉了更好预测动作，但不能单凭一次分歧解释整周成本涨幅。',
 '', '## FULL教师路径与FRAME运行路径','']
for m,p in profile.items():lines.append(f"- {m}：{p['completion_calls']}次有剩余候选的补查调用，其中{p['matching_full_teacher_states']}次物理状态哈希出现在该日期FULL教师的保存状态中。")
lines+=['','哈希不同说明精确运行状态不同，不能据此直接证明特征分布漂移导致失败；当前的即时奖励标签也没有学习整周成本。离线结果迁移到闭环时，至少同时面对路径变化与目标差异。',
 '', '## 当前建议','',
 '暂不扩大数据、网络或训练量。后续若继续学习，应先围绕既有闭环失败判断：究竟要预测下一批即时收益，还是需要与闭环代价更一致的决策目标，并把固定一批保留为强对照。当前只读证据还不足以决定新的训练标签或宣称某个修复方案有效；这属于下一路线选择，需要用户审阅后再安排。',
 '', '原审计在最终JSON落盘时遇到NumPy整数序列化错误，另存工程恢复收据。原冻结源码未改，48条回放及1436个结果文件哈希不变；恢复只增加无损标量序列化和NPZ一次读取，原全部核验断言保留。不是实验方法失败，也没有重跑物理回放。']
(R.OUT/'闭环失败与开销解释_中文.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
collab=ROOT/'论文协作资料/FRAME_学习补查闭环_GRID104_20261022'
for name in ['闭环失败与开销解释_中文.md','readonly_explanation.json','recovery_01_audit_receipt.json']:shutil.copy2(R.OUT/name,collab/name)
dest=R.ARCHIVE/'posthoc_explanation';dest.mkdir(exist_ok=False)
for name in ['闭环失败与开销解释_中文.md','readonly_explanation.json','recovery_01_audit_receipt.json','complete.json']:shutil.copy2(R.OUT/name,dest/name)
shutil.copy2(Path(__file__),dest/'readonly_explanation.py')
R.save(dest/'archive_manifest.json',{p.name:R.sha(p) for p in dest.iterdir() if p.is_file()})
note='\n\n## 2026-10-22 — GRID104 post-completion explanation\n\nRead-only first-delivered-action divergence, actual compute decomposition and FULL-teacher exact-state overlap added after full audit; no new fit/forecast/physical step. Learners did not pass scaling gate: March RF whole-week cost+1.112%, DQN seeds0/1+1.758% versus FRAME; all arms3/4 complete. RF macro actual controller reduction1.75% versus FRAME and0.53% versus one-batch; no best-seed rescue. Serializer recovery preserved all frozen sources and1436 original benchmark file hashes. Interpretation: distinguish limited whole-controller compute leverage and immediate-reward/closed-loop objective/path differences; no causal drift or independent AC claim. No scaling started.\n'
for name in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
    with (R.REPO/'docs'/name).open('a',encoding='utf8') as f:f.write(note)
print(json.dumps(dict(timing_means=means,first_divergences=details,full_teacher_overlap=profile),ensure_ascii=False,indent=2))
