"""Recompute all fixed-model calibration outcomes and causal online equivalence."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid77'
sys.path.insert(0,str(Path(__file__).parent));import metacontrol as M
torch.set_num_threads(1)
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    fit=read(OUT/'training_finished.json');D=read(OUT/'design.json');records=read(OUT/'records.json');subs=read(OUT/'data_receipts.json')['subids']
    for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
    for r in records:
        r['subids']=subs;r['aliases']={int(k):int(v) for k,v in r['aliases'].items()}
        r['context']={k:np.asarray(v,dtype=np.int64 if k=='edges' else np.float32) for k,v in r['context'].items()};r['prefixes']=M.prefixes(r)
    assert len(records)==151 and len({r['family'] for r in records if r['split']=='train'})==14
    assert not {r['family'] for r in records}&{w[:10] for w in D['evaluation_weeks_excluded']}
    count=0;options=[];summaries=[]
    for index,cost in enumerate(D['cost_levels']):
        own=[]
        for seed in D['seeds']:
            folder=OUT/'models'/f'cost{index}_seed{seed}';stored=read(folder/'summary.json')
            assert sha(folder/'final.pt')==stored['final_sha256'] and sha(folder/'initial.pt')==stored['initial_sha256']
            assert stored['updates']>0 and stored['steps']>=D['min_explore_steps']
            assert read(folder/'curve.json')[-1]['episode']==D['episodes']
            model=M.QModel();model.load_state_dict(torch.load(folder/'final.pt',weights_only=True,map_location='cpu'));model.eval();per=[]
            for r in records:
                ref=M.replay(model,r);seen=[];truth={x['pool_id']:x for x in r['rows']}
                def observe(i,p):
                    assert i not in seen;seen.append(i);return {**truth[i],'phase':p}
                got=M.run(r['order'],observe,seed,subs,r['aliases'],r['rho_limit'],r['context'],model)
                assert got['public_queries']==ref['end'] and got['chosen_pool_id']==ref['chosen'],r['id']
                assert seen==[x['pool_id'] for x in got['outcomes']]
                count+=1
                if r['split']=='calibration':per.append(dict(id=r['id'],family=r['family'],queries=ref['end'],regret=max(0.,r['best_quality']-ref['quality']),guard_queries=r['guard_queries'],guard_regret=max(0.,r['best_quality']-r['guard_quality']),mode=ref['mode']))
            assert per==stored['rows']
            for key,field in [('calibration_queries','queries'),('calibration_mean_regret','regret'),('guard_mean_regret','guard_regret')]:assert abs(stored[key]-float(np.mean([x[field] for x in per])))<1e-12
            assert stored['calibration_max_regret']==max(x['regret'] for x in per)
            own.append(stored);summaries.append(stored)
        options.append(dict(costindex=index,cost_level=cost,feasible=all(r['calibration_mean_regret']<=r['guard_mean_regret']+1e-7 and r['calibration_max_regret']<=r['guard_max_regret']+1e-7 for r in own),mean_queries=float(np.mean([r['calibration_queries'] for r in own])),mean_regret=float(np.mean([r['calibration_mean_regret'] for r in own])),max_regret=max(r['calibration_max_regret'] for r in own)))
    assert options==fit['options'];feasible=[o for o in options if o['feasible']]
    chosen=min(feasible,key=lambda x:(x['mean_queries'],x['mean_regret'],x['costindex'])) if feasible else min(options,key=lambda x:(x['mean_regret'],x['max_regret'],x['mean_queries'],x['costindex']))
    assert chosen==fit['selected'] and read(OUT/'selected.json')==dict(chosen,calibration_quality_matched=bool(feasible))
    result=dict(passed=True,fits_checked=6,online_offline_causal_checks=count,calibration_rows_checked=132,shared_cost_selection_recomputed=True,selected=chosen,calibration_quality_matched=bool(feasible),new_physical_queries=0,new_fits=0,independent_refits=False)
    (OUT/'fit_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# GRID77 相同上下文与核查权限的会议方法适配','',
        '保留ICAPS2022式DoubleDQN训练内核，新增相同24维公开电网/搜索上下文、模式可用位，以及直接调用FRAME_GUARD的完整核查动作。不是原论文A*/RRT*任务的完整复现。','',
        '129训练状态、22校准状态，排除全部13评价家族；2种子×3既定成本，共6次拟合。使用末端模型，两个种子共同选择一个成本配置。无新教师查询。','',
        '| 费用 | 种子 | 校准查询 | 均值遗憾 | 最大遗憾 |','|---|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['cost_level']} | {r['seed']} | {r['calibration_queries']:.3f} | {r['calibration_mean_regret']:.6g} | {r['calibration_max_regret']:.6g} |")
    lines+=['',f"共享配置costindex={chosen['costindex']}，校准质量匹配={bool(feasible)}。训练总墙钟{fit['wall_s']:.2f}秒。",'',f'复算全部132行校准结果、共享配置选择和{count}次线上/离线合法动作一致性；没有独立重拟合，不读取测试成本选模型。','',
        '校准结果不是闭环成本保证。下一步在同一七个已暴露日期有限回放，保留两个种子及全部不利结果；未做新配对计时前不宣称实际控制提速。','']
    (OUT/'GRID77_report.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('COMPLETE:six fixed fits and calibration/causal-replay audits passed. No physical evaluation yet.\n',encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
