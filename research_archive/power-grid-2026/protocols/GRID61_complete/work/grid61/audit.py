"""Saved vectors, causal policy replay, physics feedback and paired mechanisms."""
import csv,gzip,hashlib,importlib.util,json,math,shutil,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid61'
spec=importlib.util.spec_from_file_location('grid61_audit_policy',ROOT/'work/grid60/policies.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest();mean=lambda v:math.fsum(v)/len(v) if v else None
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(s) for s in f]
def main():
    D=read(OUT/'design.json');F=read(OUT/'finished.json');assert F['passed_engineering'] and len(F['summaries'])==32
    for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];ix={};calls={};paths={};total_rows=queries=feedback=0;local=0
    for s in F['summaries']:
        k=(s['repeat'],s['week'],s['method']);folder=OUT/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
        actual=rows(folder/'steps.jsonl.gz');assert len(actual)==s['steps'] and s['illegal']==s['ambiguous']==0
        assert abs(math.fsum(r['raw_cost'] for r in actual)-s['cost'])<1e-6
        assert abs(math.fsum(r['decision_s'] for r in actual)-s['controller_s'])<1e-6
        assert sum(r['simulations'] for r in actual)==s['public_forecasts']
        with np.load(folder/'vectors.npz') as z:actions=z['action'];obs=z['observation']
        assert len(actions)==len(obs)==len(actual)
        for i,r in enumerate(actual):
            assert digest(actions[i])==r['action_hash'] and digest(obs[i])==r['after_hash']
            if r['ledger'] is not None:assert abs(r['ledger']['recomputed_raw_cost']-r['raw_cost'])<=r['ledger_tolerance']
        paths[k]=[(r['action_hash'],r['after_hash'],r['raw_cost']) for r in actual]
        byinput={};cs=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]
        assert len(cs)==s['n1_calls'] and sum(c['public_queries'] for c in cs)==s['n1_forecasts']
        for i,c in enumerate(cs):
            assert c['order']==sorted(c['representatives']) and c['prior_hash'] is None
            truth={r['pool_id']:r for r in c['outcomes']};assert len(truth)==c['public_queries']
            assert set(truth)<=set(c['representatives'])
            for r in c['outcomes']:
                valid=0<r['rho']<c['rho_limit'] and not r['done'] and not r['exceptions'] and not r['illegal'] and not r['ambiguous']
                assert valid==r['strict_admissible']
            seen=[]
            def reveal(j,phase):
                assert j in truth and j not in seen;seen.append(j);assert truth[j]['phase']==phase;return {**truth[j],'phase':phase}
            result=P.run(c['order'],reveal,s['policy'],subs,{int(a):int(b) for a,b in c['aliases'].items()})
            assert result['chosen_pool_id']==c['chosen_pool_id'] and result['public_queries']==c['public_queries']
            assert result['local_queries']==c['local_queries'] and result['stop_reason']==c['stop_reason']
            assert seen==[r['pool_id'] for r in c['outcomes']]
            if s['method']=='FULL':assert set(truth)==set(c['representatives'])
            with np.load(folder/f'call{i:03d}_state.npz') as z:state=(digest(z['observation']),digest(z['base_action']))
            assert state[0]==c['state_hash'] and state not in byinput;byinput[state]=c
            local+=c['local_queries']
        profile=read(folder/'components.json');assert profile['prior_cache_stats']==[None]
        assert math.fsum(r['wall_s'] for r in profile['calls'] if r['in_control'])+s['all_forecast_s']<=s['controller_s']+1e-6
        ix[k]=s;calls[k]=byinput;total_rows+=s['steps'];queries+=s['public_forecasts']
    comparisons=[]
    for rep in (0,1):
        for week in D['weeks']:
            full=ix[rep,week,'FULL'];frame=ix[rep,week,'FRAME'];nol=ix[rep,week,'NO_LOCAL'];first=ix[rep,week,'FIRST_SAFE_BEST']
            for method in ('FULL','FIRST_SAFE_BEST','NO_LOCAL','FRAME'):
                s=ix[rep,week,method];survival=s['complete'] if full['complete'] else s['steps']>=full['steps']
                equal=s['steps']==full['steps'] and s['complete']==full['complete']
                cost=(s['cost']-full['cost'])/abs(full['cost']) if s['complete'] and full['complete'] else None
                comparisons.append({'repeat':rep,'week':week,'method':method,'complete':s['complete'],'steps':s['steps'],
                    'cost':s['cost'],'controller_s':s['controller_s'],'n1_queries':s['n1_forecasts'],'n1_source_s':s['n1_source_s'],
                    'no_earlier_failure':survival,'cost_delta_full':cost,'quality_pass':bool(survival and (cost is None or cost<=.005)),
                    'control_reduction_full':1-s['controller_s']/full['controller_s'] if equal else None,
                    'query_reduction_full':1-s['n1_forecasts']/full['n1_forecasts'] if equal and full['n1_forecasts'] else None,
                    'local_cost_gain_of_full':(nol['cost']-frame['cost'])/abs(full['cost']) if method=='FRAME' and nol['complete'] and frame['complete'] and full['complete'] else None})
                for state in set(calls[rep,week,method])&set(calls[rep,week,'FULL']):
                    a,b=calls[rep,week,method][state],calls[rep,week,'FULL'][state];assert a['representatives']==b['representatives'] and a['aliases']==b['aliases']
                    reference={r['pool_id']:r for r in b['outcomes']}
                    for r in a['outcomes']:
                        assert all(r[t]==reference[r['pool_id']][t] for t in ['rho','reward','done','strict_admissible','exceptions','illegal','ambiguous']);feedback+=1
    repeat_checks=[]
    for week in D['weeks']:
        for method in ('FULL','FIRST_SAFE_BEST','NO_LOCAL','FRAME'):
            assert paths[0,week,method]==paths[1,week,method];repeat_checks.append((week,method))
    summaries=[]
    for method in ('FULL','FIRST_SAFE_BEST','NO_LOCAL','FRAME'):
        a=[r for r in comparisons if r['method']==method]
        summaries.append({'method':method,'runs':len(a),'completed':sum(r['complete'] for r in a),'quality_pass':sum(r['quality_pass'] for r in a),
            'macro_control_reduction_full':mean([r['control_reduction_full'] for r in a if r['control_reduction_full'] is not None]),
            'macro_query_reduction_full':mean([r['query_reduction_full'] for r in a if r['query_reduction_full'] is not None]),
            'worst_complete_cost_delta_full':max((r['cost_delta_full'] for r in a if r['cost_delta_full'] is not None),default=None)})
    frame_rows=[r for r in comparisons if r['method']=='FRAME'];first_rows=[r for r in comparisons if r['method']=='FIRST_SAFE_BEST'];no_rows=[r for r in comparisons if r['method']=='NO_LOCAL']
    frame_quality=all(r['quality_pass'] for r in frame_rows)
    first_time_ratios=[ix[rep,w,'FIRST_SAFE_BEST']['controller_s']/ix[rep,w,'FRAME']['controller_s'] for rep in (0,1) for w in D['weeks'] if ix[rep,w,'FIRST_SAFE_BEST']['steps']==ix[rep,w,'FRAME']['steps'] and ix[rep,w,'FIRST_SAFE_BEST']['complete']==ix[rep,w,'FRAME']['complete']]
    first_competitive=bool(all(r['quality_pass'] for r in first_rows) and len(first_time_ratios)==8 and mean(first_time_ratios)<=1.02)
    local_help=bool(frame_quality and (any(not r['quality_pass'] for r in no_rows) or any((r['local_cost_gain_of_full'] or 0)>=.001 for r in frame_rows)))
    host=shutil.which('pwsh.exe');assert host
    command="$s=[datetime]::Parse('"+F['started_utc']+"').ToLocalTime();$e=[datetime]::Parse('"+F['ended_utc']+"').ToLocalTime();$a=@(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-Power';Id=42,107,506,507;StartTime=$s;EndTime=$e} -ErrorAction SilentlyContinue|ForEach-Object {@{id=$_.Id;utc=$_.TimeCreated.ToUniversalTime().ToString('o')}});ConvertTo-Json -InputObject $a"
    p=subprocess.run([host,'-NoProfile','-Command',command],capture_output=True,text=True);events=json.loads(p.stdout or '[]')
    (OUT/'power_events.json').write_text(json.dumps(events,indent=2),encoding='utf-8');assert not events,'Timing blocked by power event;retain allraw.'
    p=subprocess.run([host,'-NoProfile','-Command',"(Get-WinEvent -LogName System -MaxEvents 1 -ErrorAction Stop).RecordId"],capture_output=True,text=True);assert p.returncode==0 and p.stdout.strip().isdigit()
    verdict='FRAME_QUALITY_FAILED' if not frame_quality else 'FIRST_SAFE_COMPETITIVE' if first_competitive else 'LOCAL_COMPLETION_SUPPORTED_IN_MATRIX' if local_help else 'MECHANISM_SMALL_OR_MIXED'
    R={'passed_saved_data':True,'runs':32,'physical_rows':total_rows,'public_forecasts':queries,'matched_feedback':feedback,'exact_repeat_paths':len(repeat_checks),
        'summaries':summaries,'frame_quality_passed':frame_quality,'first_safe_competitive':first_competitive,
        'first_safe_mean_time_ratio_frame':mean(first_time_ratios),'local_completion_help_observed':local_help,'verdict':verdict,
        'local_query_total':local,'new_fits':0,'independent_ac_resolve':False,
        'warning':'Four olddevelopment families,two timing repeats;same frequency order/sharedcache/basecontrol. Mechanism ablation,not new independent generalization or global optimality. First-safe reference has stronger best-admissible exhausted fallback;not full SoftGNN reproduction.'}
    (OUT/'review.json').write_text(json.dumps(R,indent=2,allow_nan=False),encoding='utf-8')
    with (OUT/'per_run.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    lines=['# GRID61 同排序框架机制闭环比较','',f'裁决：{verdict}。','',R['warning'],'','|策略|完成|质量通过|控制减少/FULL|查询减少/FULL|最坏完整周成本变化/FULL|','|---|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"|{s['method']}|{s['completed']}/8|{s['quality_pass']}/8|{s['macro_control_reduction_full']:.3%}|{s['macro_query_reduction_full']:.3%}|{s['worst_complete_cost_delta_full']:+.5%}|")
    lines += ['',f"同方法两重复路径完全一致：{len(repeat_checks)}/16；保存数据核对{total_rows}步、{feedback}项匹配反馈。没有独立AC重求解。",'',
        f"局部补全在本矩阵有质量帮助：{local_help}；普通首个安全候选强参照具有竞争力：{first_competitive}。完整逐次成本、存活和查询见per_run.csv，不省略不利日期。",'',
        'GNN不是此矩阵的排序输入；所有策略用同一作者频次顺序。各臂共享DC/safeQP编译复用，不将共有工程算作GNN或此搜索策略独有贡献。']
    (OUT/'GRID61_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(OUT/'RUN_STATE.md').write_text('COMPLETE AND AUDITED:'+verdict+';no newtraining/tuning.\n',encoding='utf-8')
    print(json.dumps(R))
if __name__=='__main__':main()
