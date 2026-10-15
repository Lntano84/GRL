"""Vectors, public feedback, cost, causal search and timing accounting; no AC re-solve."""
from pathlib import Path
import argparse,csv,datetime as dt,gzip,hashlib,json,math,statistics,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));import policies as P
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
mean=lambda x:statistics.mean(x) if x else None
def quantile(a,p):
 if not a:return None
 a=sorted(a);k=(len(a)-1)*p;i=int(k);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(k-i)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,required=True);a=ap.parse_args();OUT=ROOT/f'outputs/grid{a.stage}'
 D=read(OUT/'design.json');F=read(OUT/'finished.json');assert F['passed_engineering'] and len(F['summaries'])==len(D['execution_order'])
 for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];ix={};paths={};inputs={};latency=[];components=[];count=feedback=0
 def csvsave(name,rows):
  with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 for s in F['summaries']:
  k=(s['repeat'],s['week'],s['method']);folder=OUT/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
  assert s==read(folder/'summary.json') and s['illegal']==s['ambiguous']==s['exception_steps']==0
  with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(l) for l in f]
  assert len(steps)==s['steps'] and abs(math.fsum(x['raw_cost'] for x in steps)-s['cost'])<1e-6
  assert abs(math.fsum(x['decision_s'] for x in steps)-s['controller_s'])<1e-6
  assert sum(x['simulations'] for x in steps)==s['public_forecasts']
  with np.load(folder/'vectors.npz') as z:
   for i,x in enumerate(steps):
    assert digest(z['action'][i])==x['action_hash'] and digest(z['observation'][i])==x['after_hash']
    if x['ledger'] is not None:assert abs(x['ledger']['recomputed_raw_cost']-x['raw_cost'])<=x['ledger_tolerance']
  paths[k]=[(x['action_hash'],x['after_hash'],x['raw_cost']) for x in steps];calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]
  assert len(calls)==s['n1_calls'] and sum(c['public_queries'] for c in calls)==s['n1_forecasts'];byinput={}
  for j,c in enumerate(calls):
   assert c['order']==sorted(c['representatives']) and c['prior_hash'] is None
   truth={r['pool_id']:r for r in c['outcomes']};assert len(truth)==c['public_queries'];seen=[]
   for r in c['outcomes']:assert r['strict_admissible']==bool(0<r['rho']<c['rho_limit'] and not r['done'] and not r['exceptions'] and not r['illegal'] and not r['ambiguous'])
   def reveal(i,phase):
    assert i in truth and i not in seen and truth[i]['phase']==phase;seen.append(i);return {**truth[i],'phase':phase}
   result=P.run(c['order'],reveal,s['policy'],subs,{int(i):int(v) for i,v in c['aliases'].items()},limit=c['rho_limit'])
   for field in ['chosen_pool_id','public_queries','local_queries','global_extra_queries','completion_budget','completion_candidate_ids','extra_selected_ids','pre_extra_choice','stop_reason']:assert result[field]==c[field],(k,j,field)
   assert seen==[x['pool_id'] for x in c['outcomes']]
   if s['policy']=='FULL':assert set(truth)==set(c['representatives'])
   with np.load(folder/f'call{j:03d}_state.npz') as z:state=(digest(z['observation']),digest(z['base_action']))
   assert state[0]==c['state_hash'] and state not in byinput;byinput[state]=c
  cp=read(folder/'components.json');outer=math.fsum(c['wall_s'] for c in cp['calls'] if c['in_control']);residual=s['controller_s']-outer-s['all_forecast_s'];assert residual>=-1e-6
  components.append({'repeat':s['repeat'],'date':s['week'],'method':s['method'],'controller_s':s['controller_s'],'all_forecast_s':s['all_forecast_s'],'n1_forecast_s':s['n1_forecast_s'],**{p:math.fsum(x['wall_s'] for x in cp['calls'] if x['in_control'] and x['phase']==p) for p in ['run_dc','compute_optimum_safe','compute_optimum_unsafe']},'other_controller_s':residual})
  active={c['step'] for c in calls};v=[x['decision_s'] for x in steps];w=[x['decision_s'] for x in steps if x['step'] in active]
  latency.append({'repeat':s['repeat'],'date':s['week'],'method':s['method'],'all_steps':len(v),'p95_s':quantile(v,.95),'p99_s':quantile(v,.99),'max_s':max(v),'search_steps':len(w),'search_p95_s':quantile(w,.95),'search_p99_s':quantile(w,.99)})
  ix[k]=s;inputs[k]=byinput;count+=len(steps)
 methods=[m['method'] for m in D['methods']];family=[];repeat_equal=0;budget_checks=0
 for week in D['weeks']:
  fs=[ix[r,week,'FULL'] for r in range(D['repeats'])]
  for method in methods:
   own=[ix[r,week,method] for r in range(D['repeats'])];equal=all(s['steps']==f['steps'] and s['complete']==f['complete'] for s,f in zip(own,fs));complete=all(x['complete'] for x in own+fs)
   survival=all(s['complete'] if f['complete'] else s['steps']>=f['steps'] for s,f in zip(own,fs));delta=(mean([x['cost'] for x in own])/mean([x['cost'] for x in fs])-1) if complete else None
   times=[x['controller_s'] for x in own];qfull=mean([x['n1_forecasts'] for x in fs]);paths_same=all(paths[r,week,method]==paths[0,week,method] for r in range(D['repeats']));repeat_equal+=int(paths_same)
   family.append({'date':week,'method':method,'steps':own[0]['steps'],'complete':all(x['complete'] for x in own),'quality_pass':survival and (delta is None or delta<=D['quality_threshold']),'cost':mean([x['cost'] for x in own]),'cost_delta_full':delta,'controller_mean_s':mean(times),'controller_std_s':statistics.stdev(times) if len(times)>1 else None,'controller_min_s':min(times),'controller_max_s':max(times),'time_reduction_full':1-mean(times)/mean([x['controller_s'] for x in fs]) if equal else None,'queries':mean([x['n1_forecasts'] for x in own]),'query_reduction_full':1-mean([x['n1_forecasts'] for x in own])/qfull if equal and qfull else None,'equal_horizon':equal,'repeat_paths_equal':paths_same})
   for rep in range(D['repeats']):
    for state in set(inputs[rep,week,method])&set(inputs[rep,week,'FULL']):
     c=inputs[rep,week,method][state];f=inputs[rep,week,'FULL'][state];truth={x['pool_id']:x for x in f['outcomes']}
     for row in c['outcomes']:
      assert all(row[t]==truth[row['pool_id']][t] for t in ['rho','reward','done','strict_admissible','exceptions','illegal','ambiguous']);feedback+=1
    if method=='GLOBAL_MATCH':
     for state in set(inputs[rep,week,'FRAME'])&set(inputs[rep,week,method]):
      c=inputs[rep,week,'FRAME'][state];g=inputs[rep,week,method][state];assert c['public_queries']==g['public_queries'] and c['completion_budget']==g['completion_budget'];budget_checks+=1
 summary=[]
 for method in methods:
  own=[x for x in family if x['method']==method]
  summary.append({'method':method,'quality_dates':sum(x['quality_pass'] for x in own),'complete_dates':sum(x['complete'] for x in own),'max_complete_cost_delta':max((x['cost_delta_full'] for x in own if x['cost_delta_full'] is not None),default=None),'macro_query_reduction':mean([x['query_reduction_full'] for x in own if x['query_reduction_full'] is not None]),'macro_time_reduction':mean([x['time_reduction_full'] for x in own if x['time_reduction_full'] is not None])})
 # Old-date default paths must match previous frozen outcomes exactly.
 historical_checks=0
 if a.stage==67:
  old={(s['week'],s['method']):s for s in read(ROOT/'outputs/grid65/finished.json')['summaries']}
  for s in F['summaries']:
   if s['method'] in ('FRAME','FULL'):
    assert all(s[t]==old[s['week'],s['method']][t] for t in ['steps','complete','cost','n1_forecasts']);historical_checks+=1
 host='C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe'
 cmd="$s=[datetime]::Parse('"+F['started_utc']+"').ToLocalTime();$e=[datetime]::Parse('"+F['ended_utc']+"').ToLocalTime();$a=@(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-Power';Id=42,107,506,507;StartTime=$s;EndTime=$e} -ErrorAction SilentlyContinue|ForEach-Object {@{id=$_.Id;utc=$_.TimeCreated.ToUniversalTime().ToString('o')}});ConvertTo-Json -InputObject $a"
 p=subprocess.run([host,'-NoProfile','-Command',cmd],capture_output=True,text=True);events=json.loads(p.stdout or '[]')
 parse=lambda x:dt.datetime.fromisoformat(x.replace('Z','+00:00'));intervals=[read(p) for p in sorted(OUT.glob('interval_*.json'))]
 overlapping=[{**e,'workers':[x['index'] for x in intervals if parse(x['started_utc'])<=parse(e['utc'])<=parse(x['ended_utc'])]} for e in events]
 timing=not any(x['workers'] for x in overlapping)
 frame=next(s for s in summary if s['method']=='FRAME')
 if a.stage==68:
  supported=frame['quality_dates']==len(D['weeks']) and frame['complete_dates']>=3 and frame['macro_query_reduction'] is not None and frame['macro_query_reduction']>=.5 and frame['macro_time_reduction'] is not None and frame['macro_time_reduction']>=.05
  verdict='FRESH_LOCAL_SCENES_SUPPORTED' if supported else 'QUALITY_FAILURE_OR_LIMITED_COMPUTE_GAIN'
 else:verdict='PARAMETER_SENSITIVITY_REPORTED_NO_SELECTION'
 result={'passed_saved_data':True,'runs':len(ix),'dates':len(D['weeks']),'physical_rows':count,'matched_full_feedback':feedback,'equal_input_budget_checks':budget_checks,'repeated_paths_equal':repeat_equal,'historical_reference_checks':historical_checks,'summaries':summary,'timing_qualified':timing,'power_events':overlapping,'verdict':verdict,'independent_ac_resolve':False,'new_fits':0}
 csvsave('per_family.csv',family);csvsave('latency.csv',latency);csvsave('components.csv',components)
 (OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 lines=[f'# GRID{a.stage} 实验结果','',D['scope'],'',f'裁决：{verdict}。','', '|方法|质量日期|完成日期|查询减少/FULL|控制时间减少/FULL|最坏完整周成本变化|','|---|---:|---:|---:|---:|---:|']
 fmt=lambda x:'不可比' if x is None else f'{x*100:+.3f}%'
 for x in summary:lines.append(f"|{x['method']}|{x['quality_dates']}/{len(D['weeks'])}|{x['complete_dates']}/{len(D['weeks'])}|{fmt(x['macro_query_reduction'])}|{fmt(x['macro_time_reduction'])}|{fmt(x['max_complete_cost_delta'])}|")
 lines+=['','逐日期成本、存活、每次原始计时、P95/P99、计算分解均保存。控制时间只比较等时域，完整周成本只比较共同完成日期。零查询日期不进入查询比例均值。','',f'保存数据核验通过；计时资格 {timing}。未独立重新求解AC潮流。参数研究单次计时只作描述，不作为稳定加速证据；不按结果更换默认参数。']
 (OUT/f'GRID{a.stage}_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(OUT/'RUN_STATE.md').write_text('COMPLETE AND AUDITED: '+verdict+'\n',encoding='utf-8')
 print(json.dumps({'stage':a.stage,'verdict':verdict,'summary':summary,'timing':timing},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
