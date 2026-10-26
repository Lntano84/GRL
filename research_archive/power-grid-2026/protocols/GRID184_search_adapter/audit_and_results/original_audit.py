"""Audit saved observations/actions/costs and native choices; zero physical calls."""
import ast,collections,copy,gzip,hashlib,importlib.util,json,math,sys,types
import numpy as np
from run import R,W,O,REPO,read,save,verify,sha,extract
def main():
    verify();assert not (O/'review.json').exists();D=read(O/'design.json');summaries=read(O/'finished.json')['summaries'];assert len(summaries)==64
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sp=importlib.util.spec_from_file_location('g184_saved_audit',W/'adapter.py');B=importlib.util.module_from_spec(sp);sp.loader.exec_module(B);models=[B.M.FrozenModels(s) for s in [0,1,2]]
    sys.path.insert(0,str(R/'work/grid111_action_feedback'));from public import Decoder,av,digest
    decoder=Decoder();prepare=extract(R/'work/grid26/engine.py','prepare','Search',dict(np=np,action_vector=av));graph=extract(R/'work/grid08/common.py','graph_features',namespace=dict(np=np));checks=collections.Counter();runrows=[];rowmap={};callmap={}
    class Search:pass
    Search.prepare=prepare
    search=Search();search.env=decoder.ac;search.subids=decoder.subs;search.library=decoder.library
    for s in summaries:
        key=(s['repeat'],s['week'],s['method']);folder=O/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}";assert read(folder/'summary.json')==s
        with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf8') as f:rows=[json.loads(line) for line in f]
        rowmap[key]=rows;assert len(rows)==s['steps'];assert abs(math.fsum(r['raw_cost'] for r in rows)-s['cost'])<1e-6;assert abs(math.fsum(r['decision_s'] for r in rows)-s['controller_s'])<1e-6;assert sum(r['simulations'] for r in rows)==s['public_forecasts'];assert s['illegal']==s['ambiguous']==0
        with np.load(folder/'vectors.npz') as z:
            assert len(z['action'])==len(z['observation'])==len(rows)
            for j,r in enumerate(rows):
                assert digest(z['action'][j])==r['action_hash'] and digest(z['observation'][j])==r['after_hash']
                if j:assert rows[j-1]['after_hash']==r['before_hash']
                if r['ledger'] is not None:assert abs(r['ledger']['recomputed_raw_cost']-r['raw_cost'])<=r['ledger_tolerance']
                if r['exceptions']:assert r['done']
                checks['vectors']+=2
        mode=s['completion_mode'];model=None if s['model_seed'] is None else models[s['model_seed']];policy=B.N.CompletionPolicy(mode,model);calls=[]
        for path in sorted(folder.glob('call[0-9][0-9][0-9].json')):
            c=read(path);calls.append(c)
            with np.load(path.with_name(path.stem+'_state.npz')) as z:ob=decoder.oc.from_vect(z['observation']);base=decoder.ac.from_vect(z['base_action'])
            assert digest(ob.to_vect())==c['state_hash'];eligible,reps,aliases,actions,changed=search.prepare(ob,base);assert eligible==c['eligible'] and reps==c['representatives'] and sorted(reps)==c['order'] and aliases=={int(k):v for k,v in c['aliases'].items()} and changed==c['masked_ids']
            truth={r['pool_id']:r for r in c['outcomes']};assert len(truth)==len(c['outcomes'])==c['public_queries'];seen=[]
            for row in c['outcomes']:
                assert row['strict_admissible']==bool(0<row['rho']<c['rho_limit'] and not row['done'] and not row['illegal'] and not row['ambiguous'] and not row['exceptions'])
                assert digest(av(actions[row['pool_id']]+base))==row['combined_hash'];checks['legal_feedback']+=1
            def reveal(i,phase):assert i in truth and i not in seen and truth[i]['phase']==phase;seen.append(i);return {k:truth[i][k] for k in ['pool_id',*B.N.FEEDBACK]}
            result=policy.run(c['order'],reveal,search.subids,aliases,limit=c['rho_limit'],provider=B.N.PublicState(ob,base,actions,decoder.ac))
            assert seen==[r['pool_id'] for r in c['outcomes']]
            for field in ['chosen_pool_id','local_queries','public_queries','completion_budget','pre_extra_choice','extra_selected_ids','completion_candidate_ids','stop_reason','engineering_fallback','learned_decisions']:assert result[field]==c[field],(path,field)
            inc=B.N.best(c['outcomes']);assert c['chosen_pool_id']==(None if inc is None else inc['pool_id'])
            if inc is not None:assert digest(av(actions[inc['pool_id']]))==c['returned_action_hash']
            if mode=='LOCAL_BATCH1':assert c['local_queries']==min(64,c['completion_budget'])
            with np.load(path.with_name(path.stem+'_inputs.npz'),allow_pickle=True) as z:
                xx,ee,gg=graph(ob,decoder.ac,dict(offered=False,predictions=[None,None]));assert np.array_equal(xx,z['x']) and np.array_equal(ee,z['edges']) and np.array_equal(gg,z['glob'])
            assert c['prepare_s']+c['forecast_s']+c['feature_s']+c['local_inference_s']<=c['source_s']+1e-6;checks['native_call_reexecution']+=1
        assert len(calls)==s['n1_calls'] and sum(c['public_queries'] for c in calls)==s['n1_forecasts'];callmap[key]=calls
        components=read(folder/'components.json');optimizer=math.fsum(c['wall_s'] for c in components['calls'] if c['in_control']);overhead=s['n1_source_s']-s['n1_forecast_s'];other=s['controller_s']-optimizer-s['all_forecast_s']-overhead;assert min(overhead,other)>=-1e-5
        active={c['step'] for c in calls};lat=[r['decision_s'] for r in rows];slat=[r['decision_s'] for r in rows if r['step'] in active]
        full=read(R/D['full_references'][s['week']]);quality='PASS_COMPLETE' if s['complete'] and full['complete'] and s['cost']<=full['cost']*1.005 else 'FAIL_COST' if s['complete'] and full['complete'] else 'FAIL_COMPLETION' if full['complete'] else 'REFERENCE_LIMITED'
        if full['complete'] and s['complete']:cost_delta=s['cost']/full['cost']-1
        else:cost_delta=None
        runrows.append(dict(repeat=s['repeat'],week=s['week'],date=s['date'],method=s['method'],steps=s['steps'],complete=s['complete'],cost=s['cost'],quality=quality,cost_delta_full=cost_delta,controller_s=s['controller_s'],candidate_queries=s['n1_forecasts'],all_queries=s['public_forecasts'],local_queries=sum(c['local_queries'] for c in calls),fallback_calls=sum(c['engineering_fallback'] is not None for c in calls),local_feature_s=sum(c['local_feature_s'] for c in calls),local_inference_s=sum(c['local_inference_s'] for c in calls),optimizer_s=optimizer,forecast_s=s['all_forecast_s'],search_overhead_s=overhead,other_s=other,all_p95=float(np.percentile(lat,95)),all_p99=float(np.percentile(lat,99)),search_p95=float(np.percentile(slat,95)) if slat else None,search_p99=float(np.percentile(slat,99)) if slat else None,signature=hashlib.sha256(json.dumps([{k:r[k] for k in ['before_hash','action_hash','after_hash','raw_cost','done','complete']} for r in rows],sort_keys=True).encode()).hexdigest()))
    for a in runrows:
        if a['repeat']!=0:continue
        b=next(v for v in runrows if v['repeat']==1 and v['week']==a['week'] and v['method']==a['method']);assert a['signature']==b['signature'] and a['candidate_queries']==b['candidate_queries'];checks['repeat_pairs']+=1
    comparisons=[]
    for a in runrows:
        if not a['method'].startswith('CONSENSUS'):continue
        for reference in ['FRAME_GUARD','LOCAL_BATCH1',a['method'].replace('CONSENSUS','FACTOR')]:
            b=next(v for v in runrows if v['repeat']==a['repeat'] and v['week']==a['week'] and v['method']==reference);common=min(a['steps'],b['steps']);ka=(a['repeat'],a['week'],a['method']);kb=(b['repeat'],b['week'],b['method']);active={c['step'] for c in callmap[ka]+callmap[kb] if c['step']<=common};qa=sum(c['public_queries'] for c in callmap[ka] if c['step'] in active);qb=sum(c['public_queries'] for c in callmap[kb] if c['step'] in active);same=a['steps']==b['steps'] and a['complete']==b['complete']
            comparisons.append(dict(candidate=a['method'],reference=reference,week=a['week'],repeat=a['repeat'],candidate_quality=a['quality'],same_horizon=same,full_control_reduction=1-a['controller_s']/b['controller_s'] if same else None,matched_prefix_control_reduction=1-sum(v['decision_s'] for v in rowmap[ka][:common])/sum(v['decision_s'] for v in rowmap[kb][:common]),active_query_reduction=1-qa/qb if qb else None,whole_week_cost_delta=a['cost']/b['cost']-1 if a['complete'] and b['complete'] else None))
    save('decision.json',dict(descriptive_only=True,comparisons=comparisons,no_best_seed_selection=True,small_recurring_improvements_retained=True,no_automatic_scale=True))
    save('review.json',dict(passed_saved_data=True,runs=64,checks=dict(checks),runrows=runrows,new_fits=0,audit_physical_forecasts=0,independent_ac_resolve=False))
    lines=['# GRID184：固定共识真实开发闭环结果','', '全部64条固定计划已完成并通过保存向量、逐步成本、公共反馈、实际模型输入/决定及重复轨迹核验；没有审计物理预测或重训。', '', '逐条质量、整周成本、控制时间、候选/总查询、全步骤及搜索步骤P95/P99、显式工程回退见review.json；各日期/种子/重复与FRAME、BATCH1、同种子FACTOR比较见decision.json。完成回放不等于所有方法完成整周；质量结论按各条记录判读。', '', '本轮为同一电网原4已暴露开发日期、每日期第一份已有场景。最终8未生成，不作跨网或盲测主张；较差种子、提前终止和额外查询均保留，早停不解释为完整控制提速。','']
    for method in D['methods']:
        rs=[v for v in runrows if v['method']==method['method']];first=[v for v in rs if v['repeat']==0];lines.append(f"{method['method']}：质量达标{sum(v['quality']=='PASS_COMPLETE' for v in first)}/4，完整周{sum(v['complete'] for v in first)}/4；双重复总控制{sum(v['controller_s'] for v in rs):.3f}秒，总候选查询{sum(v['candidate_queries'] for v in rs)}；成本和时域逐日期保留。")
    (O/'GRID184_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');dest=R/'论文协作资料/FRAME_小闭环结果_GRID184';dest.mkdir(exist_ok=False)
    for n in ['GRID184_report.md','review.json','decision.json']:shutil.copy2(O/n,dest/n)
    verify();print('GRID184_SAVED_AUDIT_PASSED',json.dumps(dict(checks)))
if __name__=='__main__':main()
