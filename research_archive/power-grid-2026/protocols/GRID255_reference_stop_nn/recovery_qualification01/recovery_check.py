"""Read-only shutdown recovery qualification; no model build/forward/optimizer."""
from pathlib import Path
import os,sys,json,hashlib,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid255_reference_stop_nn';P=R/'outputs/grid254_reference_preflight';Q=O/'recovery_qualification01';W=R/'work/grid255_reference_stop_nn'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not Q.exists();Q.mkdir()
for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel
for rel,h in read(P/'source_freeze_recovery01.json').items():assert sha(R/rel)==h,rel
jobs=read(P/'jobs.json');snap=read(O/'shutdown_receipt.json');assert read(O/'reference_error.json')['all_workers_exited'];assert len(jobs)==len(snap['rows'])==18
import numpy as np,torch
rows=[]
for j in jobs:
 s=next(s for s in snap['rows'] if s['node']==j['node']);cp_path=O/s['checkpoint'];assert sha(cp_path)==s['sha256'];cp=torch.load(cp_path,map_location='cpu',weights_only=False);step=cp['updates'];assert cp['seed']==j['seed'] and cp['pair']==j['pair'] and 0<step<=7250
 opt=cp['optimizer'];g=opt['param_groups'];assert len(g)==1 and g[0]['lr']==.0003 and tuple(g[0]['betas'])==(.9,.999) and g[0]['eps']==1e-8 and g[0]['weight_decay']==0 and opt['state'];assert all(int(v['step'])==step for v in opt['state'].values())
 h=hashlib.sha256()
 for k,v in cp['state_dict'].items():assert torch.isfinite(v).all();h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 assert cp['torch_rng'].dtype==torch.uint8
 z=np.load(P/j['package']);ids=z['fit_ids'];prob=z['fit_probability'];rng=np.random.default_rng(20261022+j['seed']);count=np.zeros(3582,np.int64);stream=hashlib.sha256()
 for _ in range(step):
  ix=rng.choice(len(ids),32,replace=True,p=prob);globalix=ids[ix];np.add.at(count,globalix,1);stream.update(globalix.astype('<i8').tobytes())
 assert rng.bit_generator.state==cp['rng_state'];assert count.sum()==step*32 and count[z['score_ids']].sum()==0
 if s['complete']:
  assert step==7250;rec=read(O/'jobs'/j['node']/'complete.json');exp=np.load(P/j['expected_exposure']);ex=np.load(O/'jobs'/j['node']/'exposure.npz');assert np.array_equal(count,exp['state_draws']) and np.array_equal(count,ex['state_draws']);assert stream.hexdigest()==j['sampling_hash']==rec['sampling_hash'];assert h.hexdigest()==rec['final_hash'] and len(ex['losses'])==7250 and np.isfinite(ex['losses']).all()
 else:
  progress=read(O/'jobs'/j['node']/'progress.json');assert step>=progress['update'];assert not (O/'jobs'/j['node']/'complete.json').exists()
  # Only sampler replay: no network operation. Prefix losses were not atomically saved.
  np.savez_compressed(Q/(j['node']+'_prefix_exposure.npz'),state_draws=count)
 remain=7250-step;v=dict(node=j['node'],complete=s['complete'],checkpoint_sha256=s['sha256'],checkpoint_updates=step,progress_updates=s['updates'],remaining_updates=remain,optimizer_step_match=True,finite_weights=True,torch_rng_present=True,numpy_rng_exact_replay=True,replayed_draws=int(count.sum()),excluded_draws=0,final_weight_hash=h.hexdigest())
 if not s['complete']:
  progress=read(O/'jobs'/j['node']/'progress.json');v.update(historical_fit_s_per_update=progress['fit_wall_s']/progress['update'],estimated_remaining_fit_s=remain*progress['fit_wall_s']/progress['update'],prefix_losses_not_saved=True)
 rows.append(v)
partial=[r for r in rows if not r['complete']];used=snap['wall_used_s'];left=600-used;estimate=max(v['estimated_remaining_fit_s'] for v in partial)
q=dict(passed=True,qualification_only=True,new_trained_models=0,new_optimizer_instances=0,new_optimizer_steps=0,new_network_forward=0,complete_models=12,partial_models=6,rows=rows,remaining_updates_total=sum(v['remaining_updates'] for v in partial),original_window_remaining_s=left,historical_fit_only_critical_tail_estimate_s=estimate,cold_start_overhead_additional=True,remaining_window_not_qualified=estimate>left,checkpoint_resume_supported_numerically=True,old_complete_models_must_not_repeat=True,source_hashes_unchanged=True,partial_temp_files_ignored_and_preserved=True,prefix_losses_unavailable_no_replay_training=True)
save(Q/'qualification.json',q)
plan=dict(proposed=True,execution_authorized=False,operation='Resume six existing model trajectories only from qualified checkpoints; no new model or repeated completed fit.',max_concurrent_workers=6,proposed_resume_window_s=120,total_component_window_if_approved_s=used+120,extra_over_original_600_s=used+120-600,updates_final=7250,remaining_updates_total=q['remaining_updates_total'],same_original_model_data_sampler_optimizer=True,restore=['state_dict','Adam state','numpy rng_state','torch_rng'],reconstruct_sampler_prefix_only=True,prefix_loss_history_unavailable_disclosed=True,complete_models_reuse=12,source_and_authorization_publish_before_training=True,one_manager_one_resume_no_extension=True,new_policy_score_or_target_join=False,reason='Original 41.844s arithmetic remainder is below historical fit-only critical-tail estimate, before cold start. Need explicit compute-window decision, not method/label change.')
save(Q/'resume_proposal.json',plan)
report=f'''# GRID255关机恢复检查

18份已保存检查点均通过只读核验：12完整模型保持不动，6部分模型的权重/Adam状态、随机数状态、固定采样流前缀和日期排除均可恢复。本次0训练、0优化器、0网络前向；逐状态采样仅按保存随机流重建。

6个模型剩余更新合计{q['remaining_updates_total']}，最长单模型根据自身历史拟合速度估计还需{estimate:.2f}秒，仅含更新、不含启动和缓存重建。原600秒窗口已用{used:.3f}秒，算术剩余{left:.3f}秒，不足以据现有证据承诺完整恢复。

可审核的唯一建议：给剩余6个既有轨迹一次120秒恢复窗口，六单线程并行，只补至原7250更新；12完整项不重训。总组件允许窗口将变为{used+120:.3f}秒，比原600多{used+120-600:.3f}秒，属于计算窗口修订，需用户明确同意。不是新增模型、改种子、降更新数或调参；尚未执行。

恢复时重建采样前缀计数/哈希，不重放梯度。部分模型旧逐步loss未落盘，不能伪造完整loss历史；将保留旧进度摘要并另存续训段loss。中断产生的.tmp/空partial_exposure不用于恢复，全部保留。实际原子.pt检查点的步数以qualification.json为准，不能仅用progress.json推定。

未做新参考评分、标签连接或策略评估。原标签/费用/日期种子/部署控制器及最终8边界不变；祖先局限仍按253方案保留，不称全链独立或学习效果改善。
'''
(Q/'GRID255_recovery_report.md').write_text(report,encoding='utf8');print('RECOVERY_QUALIFIED_ZERO_FIT',json.dumps(dict(remaining_updates=q['remaining_updates_total'],remaining_window=left,fit_only_tail_estimate=estimate,partial=[(v['node'],v['checkpoint_updates']) for v in partial])))
