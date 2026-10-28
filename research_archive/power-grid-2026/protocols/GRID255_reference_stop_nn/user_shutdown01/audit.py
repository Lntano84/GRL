"""GRID255 first receipt/checkpoint audit; no NN construction or forward."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,shutil,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid255_reference_stop_nn';P=R/'outputs/grid254_reference_preflight';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID255_reference_stop_nn'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
 assert not (O/'audit.json').exists(), 'Completed audit must not be repeated'
 assert (O/'launcher_finished.json').exists(), 'Wait for original runner; never launch another'
 cmd=r"Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'grid254_reference_preflight[\\/](worker|manager)\.py' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
 x=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True);assert x.returncode==0 and not x.stdout.strip(),x.stdout+x.stderr
 for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel
 for rel,h in read(P/'source_freeze_recovery01.json').items():assert sha(R/rel)==h,rel
 ctx=read(O/'run_context.json');assert ctx['deadline']-ctx['started_monotonic']==600 and ctx['max_workers']==6
 jobs=read(P/'jobs.json');success=(O/'reference_finished.json').exists();result=read(O/('reference_finished.json' if success else 'reference_error.json'))
 assert result['all_workers_exited'] and len(set(result['completed']))==len(result['completed'])
 import numpy as np,torch
 completed=[];partial=[]
 for j in jobs:
  d=O/'jobs'/j['node']
  if not (d/'complete.json').exists():
   partial.append(dict(node=j['node'],started=(d/'started.json').exists(),last_progress=read(d/'progress.json') if (d/'progress.json').exists() else None,error=read(d/'error.json') if (d/'error.json').exists() else None));continue
  r=read(d/'complete.json');assert r['passed'] and r['updates']==r['optimizer_steps']==7250 and r['optimizer_instances']==1 and r['parameters']==12980
  assert r['initial_hash']==j['initial_hash'] and r['sampling_hash']==j['sampling_hash'] and sha(d/'model.pt')==r['checkpoint_sha256']
  assert r['global_elapsed_s']<600 and r['finished_monotonic']<ctx['deadline'] and r['excluded_optimizer_draws']==0
  z=np.load(d/'exposure.npz');e=np.load(P/j['expected_exposure']);p=np.load(P/j['package']);assert np.array_equal(z['state_draws'],e['state_draws']) and z['state_draws'].sum()==232000 and z['state_draws'][p['score_ids']].sum()==0
  assert len(z['losses'])==7250 and np.isfinite(z['losses']).all()
  cp=torch.load(d/'model.pt',map_location='cpu',weights_only=False);assert cp['updates']==7250 and cp['seed']==j['seed'] and cp['pair']==j['pair']
  h=hashlib.sha256()
  for k,v in cp['state_dict'].items():assert torch.isfinite(v).all();h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
  assert h.hexdigest()==r['final_hash']
  groups=cp['optimizer']['param_groups'];assert len(groups)==1 and groups[0]['lr']==.0003 and tuple(groups[0]['betas'])==(.9,.999) and groups[0]['eps']==1e-8 and groups[0]['weight_decay']==0
  assert cp['optimizer']['state'] and all(int(s['step'])==7250 for s in cp['optimizer']['state'].values())
  dummy=[v for k,v in cp['state_dict'].items() if k.endswith('q_net.0.weight') and v.ndim==2 and v.shape[1]==42];assert len(dummy)==1 and not torch.count_nonzero(dummy[0][:,41])
  completed.append(dict(node=j['node'],updates=7250,draws=232000,excluded_draws=0,checkpoint_sha256=sha(d/'model.pt'),final_hash=r['final_hash'],fit_wall_s=r['fit_wall_s'],global_elapsed_s=r['global_elapsed_s'],last_training_loss=float(z['losses'][-1]),mean_training_loss=float(z['losses'].mean())))
 if success:assert len(completed)==18 and not partial and len(result['completed'])==18 and read(O/'launcher_finished.json')['passed']
 audit=dict(passed=success,receipt_and_checkpoint_checks_passed=True,component_complete=success,complete_models=len(completed),partial_or_unstarted=len(partial),completed=completed,partial=partial,manager_wall_s=result['wall_s'],optimizer_fits_started=sum((O/'jobs'/j['node']/'started.json').exists() for j in jobs),model_forward=0,policy_scores=0,teacher_label_join=0,all_workers_exited=True,source_unchanged=True,deadline_enforced=True,no_retry=True)
 save(O/'audit.json',audit)
 if success:
  fs=[O/'source_freeze.json',O/'authorization.json',O/'run_context.json',O/'reference_finished.json',O/'launcher_finished.json',O/'audit.json']
  for j in jobs:fs += [O/'jobs'/j['node']/n for n in ['model.pt','complete.json','exposure.npz']]
  save(O/'trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in fs})
 save(O/'decision.json',dict(passed=success,reference_foundation_complete=success,scientific_quality_not_yet_measured=True,reference_weights_publish_before_scoring=True,next='GRID256 public reference scores and original-parent input generation only, no teacher target join until publication.' if success else 'Preserve complete/partial results; fixed window exhausted or engineering failure. No automatic resume or changed update count. Need explicit resource/protocol decision.',primary_control_remains='GRID250',no_NEW_reference_forward=True))
 report=f'''# GRID255：按日期隔离的参考网络数据基础

固定18模型组件{'完成' if success else '未完成'}：完整{len(completed)}/18，其余{len(partial)}；管理器实际窗口{result['wall_s']:.3f}秒。所有完整模型均7250更新、232000次采样，记录日期组/当前外层组的优化器采样均为0。未自动重试、续窗或更改更新数/种子。

本轮只训练离线参考网络，不替换部署控制器；模型前向评分、门控新标签连接和新策略比较均为0。训练损失只用于收据核验，不能说明补查质量已经改善。

## 核验范围

源码输入发布eb6fdb65，再发布具体哈希授权877fa65c，之后首次启动唯一管理器。复用GRID254已资格代码，6单线程worker、3种子、6个固定8天数据对、原FACTOR12980参数/Adam/SmoothL1/clip10。每个完整模型的初始/最终权重、检查点哈希、Adam步数、固定采样流/逐状态暴露、损失有限性、600秒截止和进程退出均已检查。审计只读权重和收据，不构造或前向新网络。

## 判读和下一步

{'全部18参考权重封存推送后，才能生成32238行公开参考分数和训练停止点；本阶段尚未执行。' if success else '组件未达到全18权重屏障，不能进入参考公开评分或新标签连接；完整项和部分输出全部保留。错误/截止原因见reference_error.json与各jobs日志，不能当作方法失败。'}

后续主控制固定GRID250，不增加252qgap。部署NN/旧24收益头仍原样，原父b只能使用原owner网络隐藏表示；22653缓存字节复用、缺失至多9585行原网络前向。新选择/输入必须先发布，再连接原teacher NET；随后最多11线性+11原保护步长/另一600秒，全单探针至多40拟合。

只修正参考停止点来源的记录日期参与及outer隔离；父NN/头仍own12含自身标签，内部cal4还可能进入其他参考祖先，不是全链独立OOF或盲测。旧开发已暴露，相关前缀不独立，代理效用不等整周质量/实际控制时延。原费用/标签/全部日期种子/控制器/排序别名/锁站/128/64/停滞2/.9保留，最终8/物理/电网/论文/FRAME不动。
'''
 (O/'GRID255_report.md').write_text(report,encoding='utf8')
 # Archive once, including complete and partial artifacts; no second training or scoring.
 D=A/'trained';D.mkdir(exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,D/p.name)
 shutil.copytree(O/'jobs',D/'jobs')
 save(D/'archive_manifest.json',{p.relative_to(D).as_posix():sha(p) for p in D.rglob('*') if p.is_file()})
 print('GRID255_AUDITED',success,len(completed),result['wall_s'])
if __name__=='__main__':main()
