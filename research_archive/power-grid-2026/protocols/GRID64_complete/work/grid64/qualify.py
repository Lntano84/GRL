"""All six fixed checkpoints, data separation, online/replay and hidden-feedback checks."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
import metacontrol as M
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid64'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(1);T=read(OUT/'training_finished.json');D=read(OUT/'design.json');rows=read(OUT/'records.json')
    assert read(OUT/'learning_preflight.json')['passed']
    for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
    excluded={w[:10] for w in D['evaluation_weeks_excluded']}
    assert all(r['family'] not in excluded for r in rows)
    assert set(D['training_families']).isdisjoint(D['calibration_families'])
    equivalence=permissions=0;latencies=[]
    for result in T['fits']:
        folder=OUT/'models'/f"cost{result['costindex']}_seed{result['seed']}"
        assert sha(folder/'final.pt')==result['final_sha256'] and sha(folder/'initial.pt')==result['initial_sha256']
        assert result['episodes']==15000 and result['updates']>0 and result['initial_sha256']!=result['final_sha256']
        model=M.QModel();model.load_state_dict(torch.load(folder/'final.pt',map_location='cpu',weights_only=True));model.eval()
        for rec in [r for r in rows if r['split']=='calibration']:
            truth={r['pool_id']:r for r in rec['rows']};order=[r['pool_id'] for r in rec['rows']]
            chosen=M.run(order,lambda i,p:{**truth[i],'phase':p},model,rec['rho_limit'])
            expected=M.replay(model,rec['rows'],rec['rho_limit'])
            assert chosen['chosen_pool_id']==expected['chosen'] and chosen['public_queries']==expected['end'];equivalence+=1
            unseen=set(order)-{r['pool_id'] for r in chosen['outcomes']}
            changed={i:dict(v) for i,v in truth.items()}
            for i in unseen:changed[i].update(reward=999.,rho=.00001,strict_admissible=True)
            assert M.run(order,lambda i,p:{**changed[i],'phase':p},model,rec['rho_limit'])==chosen;permissions+=1
            if chosen['chosen_pool_id'] is not None:assert truth[chosen['chosen_pool_id']]['strict_admissible']
            state=expected['state']
            for _ in range(5):
                start=time.perf_counter();M.greedy(model,state);latencies.append(time.perf_counter()-start)
    assert len(T['fits'])==6
    result={'passed':True,'fits':6,'evaluation_date_overlap':0,'prefix_online_equivalence':equivalence,'hidden_suffix_checks':permissions,'inference_p95_s':float(np.quantile(latencies,.95)),'selected_cost':T['selected'],'calibration_quality_matched':T['calibration_quality_matched'],'no_new_queries':True,'not_full_original_task_reproduction':True,'independent_closed_loop_value_learned':False}
    (OUT/'learning_qualification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('COMPLETE AND QUALIFIED:6 final fits retained; calibration selected without evaluation data; closed-loop comparison separate.\n',encoding='utf-8')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
