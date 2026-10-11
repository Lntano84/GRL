"""Post-run matched-state ranker diagnostics, no queries/fits/physics replay."""
import gzip,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'outputs/grid28';OUT=ROOT/'outputs/grid28_review'
assert json.loads((SRC/'audit.json').read_text())['passed']
OUT.mkdir(parents=True,exist_ok=False)
D=json.loads((SRC/'runtime_design.json').read_text());records=[];dates=[]
for week in D['weeks']:
    folders={k:SRC/'runs'/f'r0__{week}__{k}' for k in ['LOCAL_GNN0','LOCAL_PPO']}
    steps={}
    for method,folder in folders.items():
        with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:steps[method]=[json.loads(line) for line in f]
    keys=['step','before_hash','action_hash','after_hash','raw_cost','done','complete']
    exact=len(steps['LOCAL_GNN0'])==len(steps['LOCAL_PPO']) and all(all(a[k]==b[k] for k in keys) for a,b in zip(steps['LOCAL_GNN0'],steps['LOCAL_PPO']))
    dates.append({'week':week,'same_delivered_trajectory':exact,'physical_steps':len(steps['LOCAL_GNN0'])})
    calls={m:{(r['step'],r['state_hash']):r for file in sorted(folder.glob('call*.json')) for r in [json.loads(file.read_text())]} for m,folder in folders.items()}
    for key in sorted(set(calls['LOCAL_GNN0'])&set(calls['LOCAL_PPO'])):
        a=calls['LOCAL_GNN0'][key];b=calls['LOCAL_PPO'][key]
        x={r['pool_id']:r for r in a['outcomes']};y={r['pool_id']:r for r in b['outcomes']}
        common=set(x)&set(y)
        equal=sum(all(x[i][k]==y[i][k] for k in ['rho','reward','done','exceptions','illegal','ambiguous','strict_admissible','combined_hash']) for i in common)
        records.append({'week':week,'step':key[0],'same_state':True,
            'same_pool':a['representatives']==b['representatives'] and a['aliases']==b['aliases'] and a['masked_ids']==b['masked_ids'],
            'same_choice':a['chosen_pool_id']==b['chosen_pool_id'],
            'gnn_queries':a['public_queries'],'ppo_queries':b['public_queries'],
            'query_saving':b['public_queries']-a['public_queries'],
            'gnn_source_s':a['source_s'],'ppo_source_s':b['source_s'],
            'common_feedback':len(common),'equal_common_feedback':equal})
matched_g=sum(r['gnn_queries'] for r in records);matched_p=sum(r['ppo_queries'] for r in records)
result={'stage':'GRID28_MATCHED_RANKER_REVIEW','posthoc_diagnostic':True,'dates':dates,'matched_calls':len(records),
    'same_trajectory_dates':sum(d['same_delivered_trajectory'] for d in dates),
    'same_pool_calls':sum(r['same_pool'] for r in records),'same_choice_calls':sum(r['same_choice'] for r in records),
    'gnn_queries':matched_g,'ppo_queries':matched_p,'query_reduction_against_ppo':1-matched_g/matched_p,
    'query_direction':{key:sum((r['query_saving']>0 if key=='gnn_fewer' else r['query_saving']<0 if key=='ppo_fewer' else r['query_saving']==0) for r in records) for key in ['gnn_fewer','ppo_fewer','equal']},
    'common_feedback':sum(r['common_feedback'] for r in records),'equal_common_feedback':sum(r['equal_common_feedback'] for r in records),
    'records':records,'new_queries':0,'new_steps':0,'new_fits':0,'reserved_evaluations':0,
    'warning':'Posthoc exposed-date diagnostic. Matched query counts can show this frozen ranker configurations differ under the same stopping rule; not GNN architecture identification, independent generalization or stable whole-control timing.'}
(OUT/'paired.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
lines=['# GRID28 GNN 与 PPO 实际同状态诊断','',
       f"4个日期中，交付轨迹逐步完全相同的有{result['same_trajectory_dates']}个。匹配{len(records)}次辅助调用；相同候选池{result['same_pool_calls']}次，相同选择{result['same_choice_calls']}次。",'',
       f"在这些匹配调用中，GNN查询{matched_g}次、PPO查询{matched_p}次，少{matched_p-matched_g}次（{100*result['query_reduction_against_ppo']:.2f}%）。每调用查询方向：{result['query_direction']}。公共反馈{result['equal_common_feedback']}/{result['common_feedback']}逐项相同。",'',
       result['warning'],'',
       '不能将非搜索控制时间差解释为图模型效果；没有比较同信息MLP等强对照，也没有打开封存日期。查询节省是当前配置在已暴露状态上的读数，不升级为正式架构/泛化贡献。']
(OUT/'GRID28_ranker_review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['records','dates']},indent=2))
