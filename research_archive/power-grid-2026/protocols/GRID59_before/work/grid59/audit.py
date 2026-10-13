"""Same-batch pilot saved-vector/feedback/cost/timer audit;no independent AC solve."""
import csv,gzip,hashlib,json,math,os,shutil,struct,subprocess
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid59'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
save=lambda n,x:(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
mean=lambda x:math.fsum(x)/len(x) if x else None
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(s) for s in f]
def main():
    D=read(OUT/'design.json');F=read(OUT/'finished.json');assert F['passed_engineering'] and len(F['summaries'])==16
    for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
    ix={};cs={};summaries=[];physical=forecasts=0
    for s in F['summaries']:
        folder=OUT/'runs'/f"r0__{s['week']}__{s['method']}";actual=rows(folder/'steps.jsonl.gz')
        assert len(actual)==s['steps'] and s['illegal']==s['ambiguous']==0
        assert abs(math.fsum(r['raw_cost'] for r in actual)-s['cost'])<1e-6
        assert abs(math.fsum(r['decision_s'] for r in actual)-s['controller_s'])<1e-6
        assert sum(r['simulations'] for r in actual)==s['public_forecasts']
        with np.load(folder/'vectors.npz') as z:actions=z['action'];observations=z['observation']
        assert len(actions)==len(observations)==len(actual)
        for i,r in enumerate(actual):
            assert digest(actions[i])==r['action_hash'] and digest(observations[i])==r['after_hash']
            if r['ledger'] is not None:assert abs(r['ledger']['recomputed_raw_cost']-r['raw_cost'])<=r['ledger_tolerance']
        calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))];assert len(calls)==s['n1_calls']
        assert sum(c['public_queries'] for c in calls)==s['n1_forecasts'];byinput={}
        for i,c in enumerate(calls):
            ids=[r['pool_id'] for r in c['outcomes']];assert len(set(ids))==c['public_queries'] and set(ids)<=set(c['representatives'])
            if s['method']=='FULL':assert set(ids)==set(c['representatives'])
            if s['method']=='STATIC_ID':assert c['order']==sorted(c['representatives']) and c['prior_hash'] is None
            if s['rule']=='FREQUENCYGNN':
                with np.load(folder/f'call{i:03d}_inputs.npz') as z:
                    prior=z['prior'].copy();prior_expected=np.full(909,-1e9,dtype=np.float32)
                    prior_expected[sorted(c['representatives'])]=-np.log1p(np.arange(len(c['representatives']),dtype=np.float32))
                    assert np.array_equal(prior,prior_expected) and digest(prior)==c['prior_hash']
            valid=[]
            for r in c['outcomes']:
                good=0<r['rho']<c['rho_limit'] and not r['done'] and not r['exceptions'] and not r['illegal'] and not r['ambiguous']
                assert good==r['strict_admissible']
                if good:valid.append(r)
            choice=max(valid,key=lambda r:(struct.unpack('f',struct.pack('f',r['reward']))[0],-r['pool_id']))['pool_id'] if valid else None
            assert choice==c['chosen_pool_id']
            with np.load(folder/f'call{i:03d}_state.npz') as z:key=(digest(z['observation']),digest(z['base_action']))
            assert key[0]==c['state_hash'] and key not in byinput;byinput[key]=c
        profile=read(folder/'components.json');assert profile['prior_cache_stats']==[None]
        active=[r for r in profile['calls'] if r['in_control']]
        component_total=math.fsum(r['wall_s'] for r in active)
        assert component_total+s['all_forecast_s']<=s['controller_s']+1e-6
        ix[(s['week'],s['method'])]=s;cs[(s['week'],s['method'])]=byinput
        physical+=s['steps'];forecasts+=s['public_forecasts']
    pairs=[];matched=0
    for week in D['weeks']:
        full=ix[(week,'FULL')];static=ix[(week,'STATIC_ID')]
        for method in ['STATIC_ID','FREQUENCY_GNN0','FREQUENCY_GNN1']:
            s=ix[(week,method)];survival=s['complete'] if full['complete'] else s['steps']>=full['steps']
            cost=(s['cost']-full['cost'])/abs(full['cost']) if full['complete'] and s['complete'] else None
            quality=bool(survival and (not full['complete'] or cost is not None and cost<=.005))
            row={'week':week,'method':method,'complete':s['complete'],'steps':s['steps'],'full_complete':full['complete'],'full_steps':full['steps'],
                'no_earlier_failure':survival,'cost_delta_full':cost,'quality_qualified':quality,'queries':s['n1_forecasts'],
                'control_s':s['controller_s'],'query_reduction_static':1-s['n1_forecasts']/static['n1_forecasts'] if s['steps']==static['steps'] and static['n1_forecasts'] else None,
                'control_reduction_static':1-s['controller_s']/static['controller_s'] if s['steps']==static['steps'] else None,
                'control_reduction_full':1-s['controller_s']/full['controller_s'] if s['steps']==full['steps'] else None,
                'cost_gain_static_of_full':(static['cost']-s['cost'])/abs(full['cost']) if full['complete'] and static['complete'] and s['complete'] else None}
            pairs.append(row)
            for key in set(cs[(week,method)])&set(cs[(week,'FULL')]):
                a,b=cs[(week,method)][key],cs[(week,'FULL')][key];assert a['representatives']==b['representatives'] and a['aliases']==b['aliases']
                other={r['pool_id']:r for r in b['outcomes']}
                for r in a['outcomes']:
                    if r['pool_id'] in other:assert all(r[k]==other[r['pool_id']][k] for k in ['rho','reward','done','strict_admissible','exceptions','illegal','ambiguous']);matched+=1
    for method in ['STATIC_ID','FREQUENCY_GNN0','FREQUENCY_GNN1']:
        a=[r for r in pairs if r['method']==method]
        summaries.append({'method':method,'completed':sum(r['complete'] for r in a),'quality_pass':sum(r['quality_qualified'] for r in a),
            'mean_control_reduction_full':mean([r['control_reduction_full'] for r in a if r['control_reduction_full'] is not None]),
            'mean_control_reduction_static':mean([r['control_reduction_static'] for r in a if r['control_reduction_static'] is not None]),
            'mean_query_reduction_static':mean([r['query_reduction_static'] for r in a if r['query_reduction_static'] is not None]),
            'positive_query_families':sum(r['query_reduction_static'] is not None and r['query_reduction_static']>0 for r in a),
            'max_complete_cost_delta_full':max([r['cost_delta_full'] for r in a if r['cost_delta_full'] is not None],default=None)})
    own=[s for s in summaries if s['method'].startswith('FREQUENCY')];both=all(s['quality_pass']==4 for s in own)
    query=mean([s['mean_query_reduction_static'] for s in own]);qsignal=query>=.05 and all(s['mean_query_reduction_static']>0 and s['positive_query_families']>=3 for s in own)
    verdict='QUALITY_AND_QUERY_DEVELOPMENT_SIGNAL' if both and qsignal else 'QUALITY_FAILED_STOP_REVISION' if not both else 'WEAK_OR_MIXED_STOP_REVISION'
    host=shutil.which('pwsh.exe');assert host
    command="$s=[datetime]::Parse('"+F['started_utc']+"').ToLocalTime();$e=[datetime]::Parse('"+F['ended_utc']+"').ToLocalTime();$a=@(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-Power';Id=42,107,506,507;StartTime=$s;EndTime=$e} -ErrorAction SilentlyContinue | ForEach-Object {@{id=$_.Id;utc=$_.TimeCreated.ToUniversalTime().ToString('o')}});ConvertTo-Json -InputObject $a"
    p=subprocess.run([host,'-NoProfile','-Command',command],capture_output=True,text=True);events=json.loads(p.stdout or '[]');save('power_events.json',events)
    assert p.returncode==0 or not p.stdout.strip();assert not events,'Timing interpretation blocked by power event.'
    p=subprocess.run([host,'-NoProfile','-Command',"(Get-WinEvent -LogName System -MaxEvents 1 -ErrorAction Stop).RecordId"],capture_output=True,text=True);assert p.returncode==0 and p.stdout.strip().isdigit()
    R={'passed_saved_data':True,'runs':16,'physical_rows':physical,'public_forecasts':forecasts,'matched_feedback':matched,'summaries':summaries,'pairs':pairs,
        'both_seeds_quality_passed':both,'seed_average_macro_query_reduction_static':query,'query_signal_passed':qsignal,'verdict':verdict,'power_events':0,
        'new_fits':0,'independent_ac_resolve':False,'warning':'Four olddevelopment dates;two overlapoldtraining families;single paired timing repeat;all seeds and common failure retained. Preliminary engineering screen,not significance/equivalence/unseenconfirmation or new RL.'}
    save('review.json',R)
    with (OUT/'per_family.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,list(pairs[0]));w.writeheader();w.writerows(pairs)
    lines=['# GRID59 频次先验GNN有限闭环试点','',f'裁决：{verdict}。','', '|方法|完成|质量通过|对FULL控制减少|对STATIC控制减少|对STATIC查询减少|查询正向日期|最大完整周成本变化/FULL|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"|{r['method']}|{r['completed']}/4|{r['quality_pass']}/4|{100*r['mean_control_reduction_full']:.3f}%|{100*r['mean_control_reduction_static']:.3f}%|{100*r['mean_query_reduction_static']:.3f}%|{r['positive_query_families']}/4|{100*r['max_complete_cost_delta_full']:.5f}%|")
    lines+=['',R['warning'],'','早失败不作低成本成功；不同存活长度不算提速。实际控制时间包括图构造/先验/前向/仿真/连续优化，不用局部组件替代。频次先验是现成公开作者信息，原PPO先验版与所有失败均保留。']
    (OUT/'GRID59_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(OUT/'RUN_STATE.md').write_text('COMPLETE AND AUDITED:'+verdict+';no new training/sweep.\n',encoding='utf-8')
    print(json.dumps({'passed':True,'verdict':verdict,'summaries':summaries,'query_gain':query}))
if __name__=='__main__':main()

