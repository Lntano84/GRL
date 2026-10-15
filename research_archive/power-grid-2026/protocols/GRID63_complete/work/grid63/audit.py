"""Saved-data audit after all timing runs; no independent AC solve."""
from pathlib import Path
import csv,gzip,hashlib,json,math,statistics,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid63'
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
 assert F['passed_engineering'] and len(F['summaries'])==63
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
   r=P.run(c['order'],reveal,s['policy'],subs,{int(a):int(b) for a,b in c['aliases'].items()})
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
 for rep in range(3):
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
  fm=[ix[rep,week,'FULL'] for rep in range(3)];fullcost=mean([r['cost'] for r in fm])
  for method in P.POLICIES:
   own=[ix[rep,week,method] for rep in range(3)];cs=[r for r in comparisons if r['date']==week and r['method']==method]
   ts=[r['controller_s'] for r in own];tfull=mean([r['controller_s'] for r in fm]);qfull=mean([r['n1_forecasts'] for r in fm])
   comparable=all(r['complete'] for r in own+fm)
   family.append({'date':week,'method':method,'complete_all_repeats':all(r['complete'] for r in own),'quality_all_repeats':all(r['quality_pass'] for r in cs),'cost_mean':mean([r['cost'] for r in own]),'cost_delta_full':(mean([r['cost'] for r in own])-fullcost)/fullcost if comparable else None,'controller_mean_s':mean(ts),'controller_std_s':statistics.stdev(ts),'controller_min_s':min(ts),'controller_max_s':max(ts),'time_reduction_full':1-mean(ts)/tfull if all(r['equal_horizon'] for r in cs) else None,'queries_mean':mean([r['n1_forecasts'] for r in own]),'query_reduction_full':1-mean([r['n1_forecasts'] for r in own])/qfull if qfull and all(r['equal_horizon'] for r in cs) else None,'physical_repeats_equal':paths[0,week,method]==paths[1,week,method]==paths[2,week,method]})
 pairs=[]
 for week in D['weeks']:
  a=next(r for r in family if r['date']==week and r['method']=='FRAME');b=next(r for r in family if r['date']==week and r['method']=='GLOBAL_MATCH');f=next(r for r in family if r['date']==week and r['method']=='FULL')
  pair={'date':week,'frame_quality':a['quality_all_repeats'],'global_quality':b['quality_all_repeats'],'both_complete':a['complete_all_repeats'] and b['complete_all_repeats'] and f['complete_all_repeats'],'cost_gain_of_full':(b['cost_mean']-a['cost_mean'])/f['cost_mean'] if a['complete_all_repeats'] and b['complete_all_repeats'] and f['complete_all_repeats'] else None,'frame_time_reduction_global':1-a['controller_mean_s']/b['controller_mean_s'] if all(ix[rep,week,'FRAME']['steps']==ix[rep,week,'GLOBAL_MATCH']['steps'] and ix[rep,week,'FRAME']['complete']==ix[rep,week,'GLOBAL_MATCH']['complete'] for rep in range(3)) else None,'frame_queries_mean':a['queries_mean'],'global_queries_mean':b['queries_mean']}
  pairs.append(pair)
 frame_quality=all(p['frame_quality'] for p in pairs);global_quality=all(p['global_quality'] for p in pairs)
 gains=sum(p['cost_gain_of_full'] is not None and p['cost_gain_of_full']>=D['material_cost_threshold'] for p in pairs)
 losses=sum(p['cost_gain_of_full'] is not None and p['cost_gain_of_full']<=-D['material_cost_threshold'] for p in pairs)
 if not frame_quality:verdict='FRAME_QUALITY_FAILED'
 elif (any(not p['global_quality'] for p in pairs) or gains>=2) and not losses:verdict='STRUCTURAL_INCREMENT_SUPPORTED_IN_MATRIX'
 elif global_quality and not gains:verdict='GLOBAL_COMPETITIVE_NO_MATERIAL_STRUCTURE_GAIN'
 else:verdict='MIXED_OR_SMALL'
 summaries=[]
 for method in P.POLICIES:
  a=[r for r in family if r['method']==method]
  summaries.append({'method':method,'quality_dates':sum(r['quality_all_repeats'] for r in a),'completed_dates':sum(r['complete_all_repeats'] for r in a),'macro_time_reduction_full':mean([r['time_reduction_full'] for r in a if r['time_reduction_full'] is not None]),'macro_query_reduction_full':mean([r['query_reduction_full'] for r in a if r['query_reduction_full'] is not None]),'max_complete_cost_delta_full':max((r['cost_delta_full'] for r in a if r['cost_delta_full'] is not None),default=None)})
 csvsave('per_run.csv',comparisons);csvsave('per_family.csv',family);csvsave('frame_vs_global.csv',pairs);csvsave('latency_per_run.csv',latency);csvsave('call_accounting.csv',call_rows)
 host='C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe'
 cmd="$startCheck=[datetime]::Parse('"+F['started_utc']+"').ToLocalTime();$endCheck=[datetime]::Parse('"+F['ended_utc']+"').ToLocalTime();$a=@(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-Power';Id=42,107,506,507;StartTime=$startCheck;EndTime=$endCheck} -ErrorAction SilentlyContinue|ForEach-Object {@{id=$_.Id;utc=$_.TimeCreated.ToUniversalTime().ToString('o')}});ConvertTo-Json -InputObject $a"
 proc=subprocess.run([host,'-NoProfile','-Command',cmd],capture_output=True,text=True)
 events=json.loads(proc.stdout or '[]');(OUT/'power_events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
 R={'passed_saved_data':True,'runs':63,'dates':7,'repetitions':3,'physical_rows':physical,'public_forecasts':forecasts,'matched_full_feedback':feedback,'equal_input_budget_checks':equal_input_checks,'repeat_path_equal_groups':sum(r['physical_repeats_equal'] for r in family),'summaries':summaries,'paired_dates':pairs,'verdict':verdict,'material_gain_dates':gains,'material_loss_dates':losses,'macro_frame_time_reduction_global':mean([p['frame_time_reduction_global'] for p in pairs if p['frame_time_reduction_global'] is not None]),'power_events':events,'timing_qualified':not events,'new_fits':0,'independent_ac_resolve':False,'scope':'Same seven exposed dates. Equal per-state extra budget, not guaranteed equal whole-week query totals. Saved-vector/cost/feedback audit, no independent AC solve. No GNN/RL.'}
 (OUT/'review.json').write_text(json.dumps(R,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 lines=['# GRID63 等预算全局扩展闭环比较','',f'冻结判读：{verdict}。','',R['scope'],'','|策略|质量日期|完整周日期|控制时间减少/FULL|查询减少/FULL|最大完整周成本增幅|','|---|---:|---:|---:|---:|---:|']
 for s in summaries:lines.append(f"|{s['method']}|{s['quality_dates']}/7|{s['completed_dates']}/7|{s['macro_time_reduction_full']:.3%}|{s['macro_query_reduction_full']:.3%}|{s['max_complete_cost_delta_full']:+.5%}|")
 lines+=['','|日期|FRAME质量|GLOBAL质量|GLOBAL减FRAME成本/FULL|FRAME时间减少/GLOBAL|两者平均查询|','|---|---|---|---:|---:|---|']
 for p in pairs:lines.append(f"|{p['date']}|{p['frame_quality']}|{p['global_quality']}|{'不可比' if p['cost_gain_of_full'] is None else format(p['cost_gain_of_full'],'+.5%')}|{'不可比' if p['frame_time_reduction_global'] is None else format(p['frame_time_reduction_global'],'.3%')}|{p['frame_queries_mean']:.1f} / {p['global_queries_mean']:.1f}|")
 lines+=['',f'三重复物理路径一致组：{R["repeat_path_equal_groups"]}/21；共同输入状态等查询预算核对{equal_input_checks}项。计时电源事件：{len(events)}，时间资格{R["timing_qualified"]}。',
 '', '63条回放来自7个已有日期，三次重复测量计时；不是63个独立场景。控制器各自形成后续轨迹，整周查询数可以不同。',
 '', '完整逐日期质量、查询和时间离散程度见per_family.csv；全决策与搜索发生步的P95/P99/最大值见latency_per_run.csv；补查触发和改变动作见call_accounting.csv。',
 '', '本次仅检验已冻结参数和共同频次排序下的结构补全增量，不追加参数、模型或日期。']
 (OUT/'GRID63_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (OUT/'RUN_STATE.md').write_text('COMPLETE AND AUDITED:63runs; '+verdict+'; no new fits.\n',encoding='utf-8')
 print(json.dumps({'verdict':verdict,'summaries':summaries,'timing_qualified':not events},ensure_ascii=False))
if __name__=='__main__':main()
