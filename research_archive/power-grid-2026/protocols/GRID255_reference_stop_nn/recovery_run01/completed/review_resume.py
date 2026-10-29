"""First recovery audit and delivery, no model construction or new forward."""
from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid255_reference_stop_nn';T=O/'recovery_run01';Q=O/'recovery_qualification01';P=R/'outputs/grid254_reference_preflight';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID255_reference_stop_nn/recovery_run01'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
 assert not (T/'review.json').exists();success=(T/'finished.json').exists();result=read(T/('finished.json' if success else 'error.json'));assert result['all_workers_exited']
 cmd=r"Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'grid255_reference_stop_nn[\\/]resume01\.py' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
 pr=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True);assert pr.returncode==0 and not pr.stdout.strip(),pr.stdout+pr.stderr
 for rel,h in read(T/'source_freeze.json').items():assert sha(R/rel)==h,rel
 ctx=read(T/'run_context.json');assert ctx['deadline']-ctx['started_monotonic']==120
 import numpy as np,torch
 jobs=read(P/'jobs.json');oldq=read(Q/'qualification.json');model_index=[];newupdates=0;completed=[];incomplete=[]
 for j in jobs:
  row=next(v for v in oldq['rows'] if v['node']==j['node']);oldcomplete=row['complete'];d=O/'jobs'/j['node'] if oldcomplete else T/'jobs'/j['node']
  if not (d/'complete.json').exists():incomplete.append(j['node']);continue
  rec=read(d/'complete.json');assert rec['passed'] and rec['updates']==7250
  if oldcomplete:
   assert sha(d/'model.pt')==row['checkpoint_sha256'];assert rec['final_hash']==row['final_weight_hash'];archive_path='user_shutdown01/jobs/'+j['node']+'/model.pt'
  else:
   restored=read(d/'restored.json');assert restored['passed'] and restored['restored_weight_hash']==row['final_weight_hash'] and restored['restored_updates']==row['checkpoint_updates'];assert restored['restored_adam_steps'] and restored['restored_numpy_torch_rng'] and restored['prefix_replay_was_sampler_only'];assert rec['start_updates']==row['checkpoint_updates'] and rec['new_updates']==7250-row['checkpoint_updates'];assert rec['finished_monotonic']<ctx['deadline'] and rec['resume_elapsed_s']<120;assert sha(d/'model.pt')==rec['checkpoint_sha256']
   cp=torch.load(d/'model.pt',map_location='cpu',weights_only=False);assert cp['updates']==7250 and cp['continued_from_updates']==row['checkpoint_updates'] and cp['continued_from_sha256']==row['checkpoint_sha256'];assert digest(cp['state_dict'])==rec['final_hash'];assert cp['optimizer']['state'] and all(int(v['step'])==7250 for v in cp['optimizer']['state'].values());g=cp['optimizer']['param_groups'];assert len(g)==1 and g[0]['lr']==.0003 and tuple(g[0]['betas'])==(.9,.999) and g[0]['eps']==1e-8 and g[0]['weight_decay']==0
   for value in cp['state_dict'].values():assert torch.isfinite(value).all()
   ex=np.load(d/'exposure.npz');expected=np.load(P/j['expected_exposure']);pkg=np.load(P/j['package']);assert np.array_equal(ex['state_draws'],expected['state_draws']) and int(ex['state_draws'].sum())==232000 and ex['state_draws'][pkg['score_ids']].sum()==0;assert len(ex['continuation_losses'])==rec['new_updates'] and np.isfinite(ex['continuation_losses']).all() and int(ex['loss_first_update'])==row['checkpoint_updates']+1;assert rec['sampling_hash']==j['sampling_hash'];newupdates+=rec['new_updates'];archive_path='recovery_run01/completed/jobs/'+j['node']+'/model.pt'
  completed.append(j['node']);model_index.append(dict(node=j['node'],seed=j['seed'],pair=j['pair'],model=str((d/'model.pt').relative_to(R).as_posix()),model_sha256=sha(d/'model.pt'),receipt=str((d/'complete.json').relative_to(R).as_posix()),exposure=str((d/'exposure.npz').relative_to(R).as_posix()),archive_model=archive_path,updates=7250,continued_from_checkpoint=not oldcomplete,old_partial_loss_history_missing=not oldcomplete))
 if success:assert len(completed)==18 and not incomplete and newupdates==8285 and result['resume_wall_s']<=120 and result['cumulative_actual_s']<=678.1560001
 review=dict(passed=success,receipt_checks_passed=True,complete_models=len(completed),incomplete=incomplete,new_optimizer_updates=newupdates,new_model_trajectories=0,reused_complete_models=12,continued_models=sum(v['continued_from_checkpoint'] for v in model_index),weight_sampler_optimizer_rng_restoration_checked=True,completed_models_not_retrained=True,source_and_original_checkpoints_unchanged=True,resume_window_s=120,resume_wall_s=result['resume_wall_s'],old_used_s=result['original_used_s'],cumulative_actual_s=result['cumulative_actual_s'],cumulative_authorized_cap_s=678.1559999999954,all_workers_exited=True,new_reference_forward=0,new_teacher_label_join=0,new_policy_scores=0,prefix_loss_history_not_fabricated=True,previously_verified_12_not_reaudited_only_hash_reused=True)
 save(T/'review.json',review);save(T/'model_index.json',model_index);save(T/'complete.json',dict(passed=success,models_complete=len(completed),expected=18,reference_scoring_not_started=True,teacher_join_not_started=True,new_policy_quality_unknown=True))
 if success:
  files=[T/'source_freeze.json',T/'authorization.json',T/'review.json',T/'model_index.json',T/'complete.json']
  for m in model_index:files += [R/m[k] for k in ['model','receipt','exposure']]
  save(T/'trained_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files})
 save(T/'decision.json',dict(passed=success,ready_for_next_public_data_generation=success,no_policy_improvement_claim=True,next='GRID256 public reference score and original-parent input generation after all18weights published; no new teacher join yet.' if success else 'Preserve complete and partial output; no automatic extension/retry.'))
 report=f'''# GRID255：18个参考模型恢复完成核验

{'18/18完成。' if success else str(len(completed))+'/18完成，组件未完整。'} 12个昨晚完整模型只复用，6个模型从5×6000步和1×5215步继续，共补{newupdates}次更新至原7250步。未新增模型轨迹、未重训已保存更新。

用户明确批准一次120秒恢复窗口，恢复实际{result['resume_wall_s']:.3f}秒；加昨晚已用558.156秒，实际累计{result['cumulative_actual_s']:.3f}秒，允许上限678.156秒。本次计算窗口修订已单列，不能称原600秒组件内完成。恢复源码b538de5c、哈希授权4b24a0fc先发后运行。

## 核验

原模型、数据、种子、Adam、学习率、损失、clip10、7250更新终点均保持；逐项恢复state_dict/Adam状态/numpy随机流/torch随机状态，并在首次更新前检查。采样前缀仅重建计数和哈希，没有重放梯度。全部完整模型每份232000次采样、排除日期优化采样0、固定流一致；恢复权重/Adam步数、检查点SHA、截止/进程退出均核验。旧12模型使用刚完成的恢复资格检查，只以哈希复用，未重复构造或前向。中断前部分模型逐步loss没有完整落盘，本轮只记录续训段loss，不伪造历史。

## 这轮能说明什么

仅完成参考模型数据基础。新参考评分、新停止点标签连接、新门控训练、策略评估均为0，不能据此说补查效果已经改善。

{'全部18权重和trained_freeze发布后，下一步才生成32238行公开参考分数/选择及原父输入。' if success else '未通过全18权重屏障，不能进入公开参考评分。'} 原父NN/24收益头保持，22653缓存字节复用、缺失至多9585行必须使用原owner网络隐藏表示，不能把新8天参考隐藏向量接旧头。公开新选择输入全部先发布，再连接原teacher NET；后续同250类40输入41参数，不加252qgap，至多11线性+11原正例保护步长，另一600秒。

本探针只修正nativeSTOP来源的记录日期参与及outer隔离。原父NN/收益头仍有own12标签祖先，内部cal4可能进入其他参考祖先，不称全链独立OOF/盲测。开发已暴露，相关前缀不独立，局部代理效用不等整周质量或真实控制时间。原费用/标签/所有日期种子/控制器/排序别名/锁站/128/64/停滞2/.9不变，最终8及新物理/电网/论文/FRAME未动。
'''
 (T/'GRID255_report.md').write_text(report,encoding='utf8')
 handoff=f'''# GRID255恢复后持续交接

先读integration/docs/RESEARCH_STATE/NEXT，以最新为准。用户明确同意120s恢复窗，源码b538de5c/授权4b24a0fc先发布后运行。恢复结果{'通过' if success else '未完整'}；{len(completed)}/18完整，12旧完整不重训，6旧轨迹补{newupdates}更新；恢复{result['resume_wall_s']:.3f}s、累计{result['cumulative_actual_s']:.3f}s/上限678.156，旧600窗修订明确，不冒充原窗。全部worker退出；review/complete/decision读取实际passed。恢复核验及旧12资格不重做，旧停止/错误/源和部分loss缺失保留，trained_freeze在recovery_run01。

{'下一GRID256仅公开参考评分和原父输入构造，先验证此18权重/全部model_index及trained_freeze已发布且Git blob/远端一致。不可再训练本参考模型，不能调用旧manager/audit(其原600s路径已不适用)。' if success else '组件未完整，不能跨18权重屏障，不自动追加窗口或重训；先只读分析，需改变协议再请用户决定。'} 参考公开32238行，全6无序对×3种子固定，记录日期与当前outer组排除。旧父缓存22653字节复用，缺失<=9585必须原ownerf NN隐藏，不能新参考8天隐藏进旧208头；24父头0拟合。全部公共新分数/选择/输入先发后再接原teacher NET；本256不提前读新STOP标签，不训练读出或评分新outer策略。

之后同250 canonical38+原父b+nativeSTOP=40输入41参，不加252qgap；fit8/cal4日期固定，f0_s1KEEP/空域或无fitSTOP正例KEEP。新来源改变样本、原两域半半mass*absNET风险数值与全部own12 selectedSTOP正例保护集合，不能再说85固定。后续<=11linear+11原221alpha/一次600s，全探针18模型轨迹+<=22读出依赖拟合，不隐藏6续训的优化器实例/更新与额外授权时间。

原父NN/头own12及内部cal4祖先局限仍在，不称全链OOF/盲测；开发已暴露/相关前缀不独立，代理不等整周/实时时延。保留230主候选、250两种子低query和252改善线索及全部反例，不择种子/按outer调阈值alpha正则费用，不设全赢5%门槛，不把额外查询质量提升当同预算。原费用标签/日期种子/部署控制器/排序别名/锁站/128/64/停滞2/.9不改，最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。

每实质阶段汇报真实报告，正常无变化安静。按AGENTS更新docs/归档首页，实际A归档/协作/hash核对/推送。Git helper严格串行，先检查待提交路径白名单，不夹带无关文件。用户授权自主推进训练至喊停，已恢复grl ACTIVE，不恢复其他旧自动化/不归档聊天。
'''
 (T/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8')
 D=A/'completed';D.mkdir(exist_ok=False)
 for p in T.iterdir():
  if p.is_file():shutil.copy2(p,D/p.name)
 shutil.copytree(T/'jobs',D/'jobs');shutil.copy2(Path(__file__),D/'review_resume.py');save(D/'archive_manifest.json',{p.relative_to(D).as_posix():sha(p) for p in D.rglob('*') if p.is_file()})
 C=R/'论文协作资料/FRAME_参考停止点数据基础_GRID255_恢复完成';C.mkdir(exist_ok=False)
 for name in ['GRID255_report.md','review.json','decision.json','complete.json','CONTINUOUS_HANDOFF.md','model_index.json']:shutil.copy2(T/name,C/name);assert sha(T/name)==sha(C/name)
 save(T/'delivery_check.json',dict(passed=True,collaboration=str(C),files={p.name:sha(p) for p in C.iterdir()},archive=str(A),reference_forward=0))
 DD=REPO/'docs';summary=f'GRID255恢复{len(completed)}/18完整；旧12复用，6旧轨迹补{newupdates}更新(6000×5/5215×1至7250)。用户批准120s恢复，实际{result["resume_wall_s"]:.3f}s，累计{result["cumulative_actual_s"]:.3f}/678.156s；源/授权先发，权重/Adam/RNG/采样/排除/截止核验，0新参考forward/标签连接/策略评估。'
 p=DD/'RESEARCH_STATE.md';s=p.read_text(encoding='utf8');title,rest=s.split('\n',1);p.write_text(title+'\n\n## 2026-10-29 — GRID255显式120秒恢复完成\n\n'+summary+' 下一256先核对全权重发布，再仅公共参考分数与原父输入。\n'+rest,encoding='utf8')
 (DD/'NEXT_STEPS.md').write_text('# Next steps — GRID255恢复后\n\n1. '+summary+' 不重训或重复完整审计/归档。\n2. 读取outputs/grid255_reference_stop_nn/recovery_run01/CONTINUOUS_HANDOFF；下一256先全18权重发布核验，再公开参考32238行/原父缓存22653+缺失<=9585，仅公共分数选择输入，不提前连新teacher标签/训练读出。原父隐藏/24头复用，不混新8天隐藏。\n3. 后续同250 40input41param/<=11linear+11alpha一次600s，不加252gap；重新样本/风险数值/全selectedSTOP正例集合而非85固定，原fit8cal4/KEEP规则保持。所有发布屏障/祖先及开发局限保留，最终8和新物理/电网/论文/FRAME不动。Git helper严格串行及路径白名单，持续自主至用户停。\n',encoding='utf8')
 for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
  p=DD/name;p.write_text(p.read_text(encoding='utf8')+'\n\n## 2026-10-29 — GRID255显式恢复\n\n'+summary+' 原600s不是完成窗口，用户显式增加恢复窗口，旧loss不伪造；下一数据公开阶段后才连接标签。\n',encoding='utf8')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID255参考模型恢复完成\n\n[报告](protocols/GRID255_reference_stop_nn/recovery_run01/completed/GRID255_report.md)；'+summary+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
 print('RESUME_REVIEW_DELIVERED',success,len(completed),result['resume_wall_s'],newupdates)
def digest(sd):
 h=hashlib.sha256()
 for k,v in sd.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
if __name__=='__main__':main()
