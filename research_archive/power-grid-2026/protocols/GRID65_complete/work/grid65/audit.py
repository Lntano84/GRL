"""Saved-data audit after all timing runs; no independent AC solve."""
from pathlib import Path
import csv,gzip,hashlib,json,math,statistics,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid65'
sys.path.insert(0,str(Path(__file__).resolve().parent));import policies as P
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
mean=lambda a:statistics.mean(a) if a else None
def quantile(a,q):
 if not a:return None
 a=sorted(a);k=(len(a)-1)*q;i=int(k);j=min(i+1,len(a)-1)
 return a[i]+(a[j]-a[i])*(k-i)
def csvsave(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 D=read(OUT/'design.json');F=read(OUT/'finished.json')
 assert F['passed_engineering'] and len(F['summaries'])==70
 for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
 subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
 ix={};paths={};inputs={};latency=[];call_rows=[];physical=feedback=forecasts=equal_input_checks=0
 for s in F['summaries']:
  k=(s['repeat'],s['week'],s['method']);folder=OUT/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
  assert s==read(folder/'summary.json') and s['illegal']==s['ambiguous']==0
  with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(line) for line in f]
  assert len(steps)==s['steps'] and abs(math.fsum(r['raw_cost'] for r in steps)-s['cost'])<1e-6
  assert abs(math.fsum(r['decision_s'] for r in steps)-s['controller_s'])<1e-6
  assert sum(r['simulations'] for r in steps)==s['public_forecasts']
  with np.load(folder/'vectors.npz') as z:acts=z['action'];obs=z['observation']
  assert len(acts)==len(obs)==len(steps)
  for i,r in enumerate(steps):
   assert digest(acts[i])==r['action_hash'] and digest(obs[i])==r['after_hash']
   if r['ledger'] is not None:assert abs(r['ledger']['recomputed_raw_cost']-r['raw_cost'])<=r['ledger_tolerance']
  paths[k]=[(r['action_hash'],r['after_hash'],r['raw_cost']) for r in steps]
  cs=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]
  assert len(cs)==s['n1_calls'] and sum(c['public_queries'] for c in cs)==s['n1_forecasts']
  byinput={}
  for n,c in enumerate(cs):
   assert c['order']==sorted(c['representatives']) and c['prior_hash'] is None
   truth={r['pool_id']:r for r in c['outcomes']};assert len(truth)==c['public_queries']
   for r in c['outcomes']:
    valid=0<r['rho']<c['rho_limit'] and not r['done'] and not r['exceptions'] and not r['illegal'] and not r['ambiguous']
    assert valid==r['strict_admissible']
   seen=[]
   def reveal(i,phase):
    assert i in truth and i not in seen;seen.append(i);assert truth[i]['phase']==phase
    return {**truth[i],'phase':phase}
   r=P.run(c['order'],reveal,s['policy'],subs,{int(a):int(b) for a,b in c['aliases'].items()},limit=c['rho_limit'])
   for field in ['chosen_pool_id','public_queries','local_queries','global_extra_queries','completion_budget','completion_candidate_ids','extra_selected_ids','pre_extra_choice','stop_reason']:
    assert r[field]==c[field],(k,n,field)
   assert seen==[r['pool_id'] for r in c['outcomes']]
   if s['method']=='FULL':assert set(truth)==set(c['representatives'])
   if s['method']=='GLOBAL_MATCH':assert len(c['extra_selected_ids'])==len(c['completion_candidate_ids'])==c['global_extra_queries']
   if s['method']=='FRAME':assert c['extra_selected_ids']==c['completion_candidate_ids'] and c['completion_budget']==c['local_queries']
   with np.load(folder/f'call{n:03d}_state.npz') as z:state=(digest(z['observation']),digest(z['base_action']))
   assert state[0]==c['state_hash'] and state not in byinput;byinput[state]=c
   call_rows.append({'repeat':s['repeat'],'date':s['week'],'method':s['method'],'step':c['step'],'queries':c['public_queries'],'extra_queries':c['completion_budget'],'chosen':c['chosen_pool_id'],'pre_extra_choice':c['pre_extra_choice'],'extra_changes_choice':c['chosen_pool_id']!=c['pre_extra_choice']})
  profile=read(folder/'components.json');assert profile['prior_cache_stats']==[None]
  assert math.fsum(x['wall_s'] for x in profile['calls'] if x['in_control'])+s['all_forecast_s']<=s['controller_s']+1e-6
  active_steps={c['step'] for c in cs};alltimes=[r['decision_s'] for r in steps];active=[r['decision_s'] for r in steps if r['step'] in active_steps]
  latency.append({'repeat':s['repeat'],'date':s['week'],'method':s['method'],'all_steps':len(alltimes),'all_p95_s':quantile(alltimes,.95),'all_p99_s':quantile(alltimes,.99),'max_decision_s':max(alltimes),'search_steps':len(active),'search_p95_s':quantile(active,.95),'search_max_s':max(active) if active else None})
  ix[k]=s;inputs[k]=byinput;physical+=len(steps);forecasts+=s['public_forecasts']
 comparisons=[]
 for rep in range(2):
  for week in D['weeks']:
   f=ix[rep,week,'FULL']
   for method in P.POLICIES:
    s=ix[rep,week,method];survival=bool(s['complete'] if f['complete'] else s['steps']>=f['steps'])
    equal=s['steps']==f['steps'] and s['complete']==f['complete']
    delta=(s['cost']-f['cost'])/f['cost'] if s['complete'] and f['complete'] else None
    comparisons.append({'repeat':rep,'date':week,'method':method,'steps':s['steps'],'complete':s['complete'],'cost':s['cost'],'controller_s':s['controller_s'],'n1_queries':s['n1_forecasts'],'no_earlier_failure':survival,'equal_horizon':equal,'cost_delta_full':delta,'quality_pass':survival and (delta is None or delta<=D['quality_threshold'])})
    for state in set(inputs[rep,week,method])&set(inputs[rep,week,'FULL']):
     a,b=inputs[rep,week,method][state],inputs[rep,week,'FULL'][state];assert a['representatives']==b['representatives'] and a['aliases']==b['aliases']
     truth={r['pool_id']:r for r in b['outcomes']}
     for r in a['outcomes']:
      assert all(r[t]==truth[r['pool_id']][t] for t in ['rho','reward','done','strict_admissible','exceptions','illegal','ambiguous']);feedback+=1
   for state in set(inputs[rep,week,'FRAME'])&set(inputs[rep,week,'GLOBAL_MATCH']):
    a,b=inputs[rep,week,'FRAME'][state],inputs[rep,week,'GLOBAL_MATCH'][state]
    assert a['completion_budget']==b['completion_budget'] and a['public_queries']==b['public_queries']
    assert a['completion_candidate_ids']==b['completion_candidate_ids'];equal_input_checks+=1
 family=[]
 for week in D['weeks']:
  fm=[ix[rep,week,'FULL'] for rep in range(2)];fullcost=mean([r['cost'] for r in fm])
  for method in P.POLICIES:
   own=[ix[rep,week,method] for rep in range(2)];cs=[r for r in comparisons if r['date']==week and r['method']==method]
   ts=[r['controller_s'] for r in own];tfull=mean([r['controller_s'] for r in fm]);qfull=mean([r['n1_forecasts'] for r in fm])
   comparable=all(r['complete'] for r in own+fm)
   family.append({'date':week,'method':method,'complete_all_repeats':all(r['complete'] for r in own),'quality_all_repeats':all(r['quality_pass'] for r in cs),'cost_mean':mean([r['cost'] for r in own]),'cost_delta_full':(mean([r['cost'] for r in own])-fullcost)/fullcost if comparable else None,'controller_mean_s':mean(ts),'controller_std_s':statistics.stdev(ts),'controller_min_s':min(ts),'controller_max_s':max(ts),'time_reduction_full':1-mean(ts)/tfull if all(r['equal_horizon'] for r in cs) else None,'queries_mean':mean([r['n1_forecasts'] for r in own]),'query_reduction_full':1-mean([r['n1_forecasts'] for r in own])/qfull if qfull and all(r['equal_horizon'] for r in cs) else None,'physical_repeats_equal':paths[0,week,method]==paths[1,week,method]})
 # The old non-learned controls must preserve GRID63's physical/quality/query outcomes.
 historical=read(ROOT/'outputs/grid63/finished.json')['summaries'];history={(s['week'],s['method']):s for s in historical}
 reference_checks=0
 for s in F['summaries']:
  if s['method'] in ['FULL','FRAME','GLOBAL_MATCH']:
   old=history[s['week'],s['method']]
   assert all(s[k]==old[k] for k in ['steps','complete','cost','n1_forecasts','public_forecasts']);reference_checks+=1
 summaries=[]
 for method in P.POLICIES:
  own=[r for r in family if r['method']==method]
  summaries.append({'method':method,'quality_dates':sum(r['quality_all_repeats'] for r in own),'completed_dates':sum(r['complete_all_repeats'] for r in own),'macro_time_reduction_full':mean([r['time_reduction_full'] for r in own if r['time_reduction_full'] is not None]),'time_eligible_dates':sum(r['time_reduction_full'] is not None for r in own),'macro_query_reduction_full':mean([r['query_reduction_full'] for r in own if r['query_reduction_full'] is not None]),'query_eligible_dates':sum(r['query_reduction_full'] is not None for r in own),'max_complete_cost_delta_full':max((r['cost_delta_full'] for r in own if r['cost_delta_full'] is not None),default=None)})
 paired=[];seed_summary=[]
 frame_quality=all(r['quality_all_repeats'] for r in family if r['method']=='FRAME')
 for seed in [0,1]:
  method=f'META_DQN{seed}';seed_rows=[]
  for week in D['weeks']:
   a=next(r for r in family if r['date']==week and r['method']=='FRAME');b=next(r for r in family if r['date']==week and r['method']==method);f=next(r for r in family if r['date']==week and r['method']=='FULL')
   equal=all(ix[rep,week,'FRAME']['steps']==ix[rep,week,method]['steps'] and ix[rep,week,'FRAME']['complete']==ix[rep,week,method]['complete'] for rep in range(2))
   complete=a['complete_all_repeats'] and b['complete_all_repeats'] and f['complete_all_repeats']
   row={'seed':seed,'date':week,'frame_quality':a['quality_all_repeats'],'meta_quality':b['quality_all_repeats'],'equal_horizon':equal,'both_complete':complete,'frame_cost_gain_of_full':(b['cost_mean']-a['cost_mean'])/f['cost_mean'] if complete else None,'frame_query_reduction_meta':1-a['queries_mean']/b['queries_mean'] if equal and b['queries_mean'] else None,'frame_time_reduction_meta':1-a['controller_mean_s']/b['controller_mean_s'] if equal else None,'frame_queries':a['queries_mean'],'meta_queries':b['queries_mean']}
   paired.append(row);seed_rows.append(row)
  seed_summary.append({'seed':seed,'meta_quality_all_dates':all(r['meta_quality'] for r in seed_rows),'frame_macro_query_reduction_meta':mean([r['frame_query_reduction_meta'] for r in seed_rows if r['frame_query_reduction_meta'] is not None]),'frame_macro_time_reduction_meta':mean([r['frame_time_reduction_meta'] for r in seed_rows if r['frame_time_reduction_meta'] is not None]),'worst_frame_cost_harm_of_full':max((-r['frame_cost_gain_of_full'] for r in seed_rows if r['frame_cost_gain_of_full'] is not None),default=None)})
 # Match the frozen rule: the <=102% test uses mean raw amounts over the same eligible dates.
 competitive=[];frame_advantage=[]
 for s in seed_summary:
  method=f"META_DQN{s['seed']}";weeks=[r['date'] for r in paired if r['seed']==s['seed'] and r['equal_horizon']]
  a=[r for r in family if r['method']=='FRAME' and r['date'] in weeks];b=[r for r in family if r['method']==method and r['date'] in weeks]
  ratio_q=mean([r['queries_mean'] for r in b])/mean([r['queries_mean'] for r in a]) if a else None
  ratio_t=mean([r['controller_mean_s'] for r in b])/mean([r['controller_mean_s'] for r in a]) if a else None
  s.update(meta_to_frame_mean_query_ratio=ratio_q,meta_to_frame_mean_time_ratio=ratio_t,matched_horizon_dates=len(weeks))
  competitive.append(s['meta_quality_all_dates'] and len(weeks)==7 and ratio_q<=1.02 and ratio_t<=1.02)
  frame_advantage.append(not s['meta_quality_all_dates'] or (s['frame_macro_query_reduction_meta'] is not None and s['frame_macro_query_reduction_meta']>=.05 and s['frame_macro_time_reduction_meta']>=0 and (s['worst_frame_cost_harm_of_full'] is None or s['worst_frame_cost_harm_of_full']<.001)))
 verdict='FRAME_QUALITY_FAILED' if not frame_quality else 'META_COMPETITIVE' if all(competitive) else 'FRAME_ADVANTAGE_OVER_ADAPTED_STOPPING_IN_MATRIX' if all(frame_advantage) else 'MIXED'
 csvsave('per_run.csv',comparisons);csvsave('per_family.csv',family);csvsave('frame_vs_meta.csv',paired);csvsave('latency_per_run.csv',latency);csvsave('call_accounting.csv',call_rows)
 host='C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe'
 cmd="$s=[datetime]::Parse('"+F['started_utc']+"').ToLocalTime();$e=[datetime]::Parse('"+F['ended_utc']+"').ToLocalTime();$a=@(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-Power';Id=42,107,506,507;StartTime=$s;EndTime=$e} -ErrorAction SilentlyContinue|ForEach-Object {@{id=$_.Id;utc=$_.TimeCreated.ToUniversalTime().ToString('o')}});ConvertTo-Json -InputObject $a"
 proc=subprocess.run([host,'-NoProfile','-Command',cmd],capture_output=True,text=True);events=json.loads(proc.stdout or '[]')
 intervals=[read(OUT/f'interval_{i:02d}.json') for i in range(70)]
 import datetime as dt
 parse=lambda s:dt.datetime.fromisoformat(s.replace('Z','+00:00'))
 event_matches=[{**e,'worker_indices':[p['index'] for p in intervals if parse(p['started_utc'])<=parse(e['utc'])<=parse(p['ended_utc'])]} for e in events]
 timing=not any(e['worker_indices'] for e in event_matches)
 (OUT/'power_events.json').write_text(json.dumps(event_matches,indent=2),encoding='utf-8')
 training=read(ROOT/'outputs/grid64/training_finished.json')
 result={'passed_saved_data':True,'runs':70,'dates':7,'repetitions':2,'physical_rows':physical,'public_forecasts':forecasts,'matched_full_feedback':feedback,'equal_input_budget_checks':equal_input_checks,'repeat_path_equal_groups':sum(r['physical_repeats_equal'] for r in family),'historical_reference_checks':reference_checks,'summaries':summaries,'seed_comparison':seed_summary,'verdict':verdict,'timing_qualified':timing,'power_events':event_matches,'training_fits':6,'training_wall_s':training['training_wall_s'],'calibration_quality_matched':training['calibration_quality_matched'],'selected_calibration':training['selected'],'new_online_fits':0,'independent_ac_resolve':False,'scope':'Seven exposed dates; two DQN seeds, two serial timing repeats; literature stopping-only adaptation. Old control outcomes independently match prior GRID63 receipts; saved data, no independent AC re-solve.'}
 (OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 pct=lambda x:'不可比' if x is None else f'{100*x:+.3f}%'
 lines=['# GRID65：论文来源的自适应停止对照','',f'冻结判读：{verdict}。','',result['scope'],'','|方法|质量日期|完整周|查询减少/FULL|控制时间减少/FULL|最坏完整周成本增幅|','|---|---:|---:|---:|---:|---:|']
 for r in summaries:lines.append(f"|{r['method']}|{r['quality_dates']}/7|{r['completed_dates']}/7|{pct(r['macro_query_reduction_full'])}|{pct(r['macro_time_reduction_full'])}|{pct(r['max_complete_cost_delta_full'])}|")
 lines+=['','|META种子|META全部日期质量通过|FRAME查询减少/META|FRAME控制时间减少/META|等时域日期|','|---|---|---:|---:|---:|']
 for r in seed_summary:lines.append(f"|{r['seed']}|{r['meta_quality_all_dates']}|{pct(r['frame_macro_query_reduction_meta'])}|{pct(r['frame_macro_time_reduction_meta'])}|{r['matched_horizon_dates']}|")
 lines+=['',f"六次固定拟合耗时 {training['training_wall_s']:.2f}s；只用旧训练数据，三个时间价格在隔离校准日期上选择，共享选择索引 {training['selected']['costindex']}。校准一步质量达到FRAME门槛：{training['calibration_quality_matched']}。",'',f'物理路径重复一致 {result["repeat_path_equal_groups"]}/35；共有老方法与GRID63的质量/查询/步数复现检查 {reference_checks} 项；计时资格 {timing}。','',
 '此对照保留ICAPS2022官方Double DQN机制，限制为停止/继续动作，并用电网已观测一步N1奖励表达搜索质量。属于明确标注的领域适配版，不能称原论文原任务完整复现或AAAI2024完整方法。',
 '源方法包含参数调整；本次没有把这项能力偷换成FRAME的结构化补全。部署策略显式关闭探索，保留两个种子。原始未选中时间价格模型也全部保存。',
 '一步搜索质量的训练校准不是长期闭环质量保证。若校准本身未达到FRAME标准，FRAME的优势仅是对这个有限适配配置的结果，不据此宣称普遍超过元推理或SOTA。',
 '查询与时间只在相同完成状态和时域的日期比较，完整周成本仅对共同完成日期比较。失败和新存活差异在逐日期表中保留；两个计时重复不是新增独立日期。',
 '下一步只汇总本轮结果，不自动重新拟合、选择其他时间价格、日期或模型。']
 (OUT/'GRID65_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (OUT/'RUN_STATE.md').write_text('COMPLETE AND AUDITED:70 runs; '+verdict+';6 prior offline fits,0onlinefits.\n',encoding='utf-8')
 print(json.dumps({'verdict':verdict,'summaries':summaries,'calibration_quality_matched':training['calibration_quality_matched'],'timing_qualified':timing},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
