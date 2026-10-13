"""Saved-vector/feedback cost audit; primary is quality and queries, not batch time."""
import csv,gzip,hashlib,json,math,os,struct,subprocess,sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid57';REF=ROOT/'outputs/grid55b/qualified_view'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
mean=lambda x:math.fsum(x)/len(x)
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f]
def main():
    D=read(OUT/'design.json');F=read(OUT/'finished.json');assert F['passed_engineering'] and len(F['summaries'])==8
    for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
    old=read(REF/'finished.json');ix={(a['week'],a['repeat'],a['method']):a for a in old['summaries']}
    results=[];nphysical=nqueries=matched=0
    feedback_fields=['rho','reward','done','strict_admissible','exceptions','illegal','ambiguous']
    for s in F['summaries']:
        assert s['method']==s['rule']=='STATIC_ID' and s['illegal']==s['ambiguous']==0
        folder=OUT/'runs'/f"r0__{s['week']}__STATIC_ID";actual=rows(folder/'steps.jsonl.gz')
        assert len(actual)==s['steps'] and [r['step'] for r in actual]==list(range(1,len(actual)+1))
        assert abs(math.fsum(r['raw_cost'] for r in actual)-s['cost'])<1e-6
        assert abs(math.fsum(r['decision_s'] for r in actual)-s['controller_s'])<1e-6
        assert sum(r['simulations'] for r in actual)==s['public_forecasts']
        with np.load(folder/'vectors.npz') as z:
            assert len(z['action'])==len(z['observation'])==len(actual)
            for i,r in enumerate(actual):
                assert digest(z['action'][i])==r['action_hash'] and digest(z['observation'][i])==r['after_hash']
                if r['ledger'] is not None:assert abs(r['ledger']['recomputed_raw_cost']-r['raw_cost'])<=r['ledger_tolerance']
        calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]
        assert len(calls)==s['n1_calls'] and sum(c['public_queries'] for c in calls)==s['n1_forecasts']
        baseline_folder=Path(read(REF/'design.json')['run_paths'][f"0|{s['week']}|FULL"])
        fullcalls={c['state_hash']:c for c in [read(p) for p in baseline_folder.glob('call[0-9][0-9][0-9].json')]}
        for i,c in enumerate(calls):
            assert c['order']==sorted(c['representatives']) and c['prior_s']==c['feature_s']==0
            assert c['prior_hash'] is None and c['input_hash'] is None
            with np.load(folder/f'call{i:03d}_state.npz') as z:assert digest(z['observation'])==c['state_hash']
            ids=[r['pool_id'] for r in c['outcomes']];assert len(set(ids))==c['public_queries'] and set(ids)<=set(c['representatives'])
            valid=[]
            for a in c['outcomes']:
                good=0<a['rho']<c['rho_limit'] and not a['done'] and not a['exceptions'] and not a['illegal'] and not a['ambiguous']
                assert good==a['strict_admissible']
                if good:valid.append(a)
            chosen=max(valid,key=lambda r:(struct.unpack('f',struct.pack('f',r['reward']))[0],-r['pool_id']))['pool_id'] if valid else None
            assert chosen==c['chosen_pool_id']
            # Match only exact causal state AND base input, never future path.
            u=fullcalls.get(c['state_hash'])
            if u is not None:
                # Saved input states contain base hash for confirmation below.
                target=next(p for p in baseline_folder.glob('call[0-9][0-9][0-9].json') if read(p)['state_hash']==c['state_hash'])
                with np.load(target.with_name(target.stem+'_state.npz')) as a,np.load(folder/f'call{i:03d}_state.npz') as b:
                    equal_base=np.array_equal(a['base_action'],b['base_action'],equal_nan=True)
                if equal_base:
                    assert c['representatives']==u['representatives'] and c['aliases']==u['aliases']
                    theirs={r['pool_id']:r for r in u['outcomes']}
                    for r in c['outcomes']:
                        if r['pool_id'] in theirs:
                            assert all(r[k]==theirs[r['pool_id']][k] for k in feedback_fields);matched+=1
        profile=read(folder/'components.json');assert profile['prior_cache_stats']==[None]
        full=ix[(s['week'],0,'FULL')];gnns=[ix[(s['week'],0,f'COMPRESSED_GNN{seed}')] for seed in [0,1]]
        survival=s['complete'] if full['complete'] else s['steps']>=full['steps']
        comparable=s['complete'] and full['complete'];delta=(s['cost']-full['cost'])/abs(full['cost']) if comparable else None
        owncost=mean([g['cost'] for g in gnns]);ownqueries=mean([g['n1_forecasts'] for g in gnns])
        results.append({'week':s['week'],'steps':s['steps'],'complete':s['complete'],'no_earlier_failure':survival,
            'cost_delta_full':delta,'quality_qualified':bool(survival and comparable and delta<=.005),
            'static_queries':s['n1_forecasts'],'gnn_seed_average_queries':ownqueries,
            'gnn_query_reduction_vs_static':1-ownqueries/s['n1_forecasts'] if s['n1_forecasts'] else None,
            'gnn_cost_gain_of_full':(s['cost']-owncost)/abs(full['cost']) if comparable else None,
            'static_control_s_descriptive_only':s['controller_s'],'early_failure_cannot_be_speed_win':not comparable})
        nphysical+=s['steps'];nqueries+=s['n1_forecasts']
    gains=[r['gnn_query_reduction_vs_static'] for r in results if r['gnn_query_reduction_vs_static'] is not None]
    allq=all(r['quality_qualified'] for r in results);g=mean(gains);positive=sum(v>0 for v in gains)
    cheap_ratio=mean([r['static_queries']/r['gnn_seed_average_queries'] for r in results])
    if allq and cheap_ratio<=1.05:verdict='LEARNED_RANKING_NECESSITY_NOT_ESTABLISHED'
    elif not allq:verdict='STATIC_ORDER_QUALITY_TAIL_FAILS_WHILE_FINAL_GNNS_PASS'
    elif g>=.05 and positive>=6:verdict='EXTRA_QUERY_EVIDENCE_ONLY'
    else:verdict='MIXED'
    R={'passed_saved_data':True,'new_runs':8,'physical_rows':nphysical,'static_n1_queries':nqueries,
        'matched_public_feedback_entries':matched,'quality_pass':sum(r['quality_qualified'] for r in results),
        'all_static_quality_qualified':allq,'mean_gnn_query_reduction_vs_static':g,'positive_query_families':positive,
        'mean_static_over_gnn_query_ratio':cheap_ratio,
        'verdict':verdict,'results':results,'new_fits':0,'independent_ac_resolve':False,'cross_batch_timing_claim':False,
        'warning':'One fixed order,one timing repeat,eight already-exposed dates. Does not establish graph architecture necessity or new independent confirmation. Static batch wall time is descriptive only;no chosen seed/data/dropout.'}
    save(OUT/'review.json',R)
    with (OUT/'per_family.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(results[0]));w.writeheader();w.writerows(results)
    text=['# GRID57 最终配置的固定顺序消融','',f"裁决：{verdict}。STATIC_ID质量尺度通过{R['quality_pass']}/8；GNN种子平均相对STATIC_ID查询减少{100*g:.3f}%，正向{positive}/8。",'',
        '| 日期 | STATIC完成 | 成本变化/FULL | GNN成本增益/FULL | STATIC查询 | GNN平均查询 | GNN查询减少 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        cost='不可比' if r['cost_delta_full'] is None else f"{100*r['cost_delta_full']:+.4f}%"
        gain='不可比' if r['gnn_cost_gain_of_full'] is None else f"{100*r['gnn_cost_gain_of_full']:+.4f}%"
        text.append(f"|{r['week']}|{r['complete']}|{cost}|{gain}|{r['static_queries']}|{r['gnn_seed_average_queries']:.0f}|{100*r['gnn_query_reduction_vs_static']:.3f}%|")
    text+=['',R['warning'],'','不读取历史结果控制新轨迹；FULL和GNN记录仅在离线审计用于参照。固定顺序不使用先验，也不支付无用先验开销；DC与安全QP缓存仍共有。所有早失败和较差结果均保留。']
    (OUT/'GRID57_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('COMPLETED AND AUDITED: static-order quality/query ablation;no new fit or cross-batch speed claim.\n',encoding='utf-8')
    print(json.dumps({'passed':True,'quality_pass':R['quality_pass'],'verdict':verdict,'query_gain':g,'positive':positive}))
if __name__=='__main__':main()
