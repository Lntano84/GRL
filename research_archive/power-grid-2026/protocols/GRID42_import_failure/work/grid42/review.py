"""One frozen observed-tie completion on saved development feedback; no new forecasts."""
import csv, hashlib, importlib.util, json, os, sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k] = '1'
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/grid42'
sys.path[:0]=[str(ROOT/'work/grid42'),str(ROOT/'work/grid08')]
from policy import tied_complete, hand_checks, key
from common import make_env, ResidualControl, action_vector, digest
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
OUT.mkdir(parents=True,exist_ok=False)
D={'stage':'GRID42_OBSERVED_TIE_COMPLETION','decision_basis':'GRID41 first SoftGNN physical divergence578 has exact fp32 reward and rho tie with FULL; safe-prefix branch has no semantic completion.',
   'models':'Original GNN0/1, existing PPO and GRID40B SoftGNN0/1, all unchanged.',
   'policy':'Same PATIENCE64 then complete every substation represented by observed exact fp32 maximum reward, including safe-prefix; repeat only if newly purchased feedback reveals another best-tied group. No rho tolerance or future read.',
   'scope':'26 existing FULL states only; no new teacher/physical queries, sealed final dates unchanged.',
   'screen':'Bounded diagnostic, not automatic quality proof. Retain only as candidate if both SoftGNN seeds recover578 and original known1733; report every query increase and unresolved choice.',
   'baseline_access':'Same policy offered to every ranked comparator; FULL stays exhaustive. A quality fix is not graph-specific novelty.'}
save(OUT/'design.json',D)
save(OUT/'code_freeze.json',{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [Path(__file__).resolve(),ROOT/'work/grid42/policy.py',ROOT/'work/grid27/policy.py']})
assert hand_checks()==5
E=make_env(); E.seed(0); initial=E.reset(); C=ResidualControl(E,initial)
library=C.base.topo_n1_unsafe.topo_act_list
subs=[int(a.as_dict()['set_bus_vect']['modif_subs_id'][0]) for a in library]
save(OUT/'library.json',{'subids':subs,'action_hashes':[digest(action_vector(a)) for a in library]})
E.close()
inputs={}; oldrows=read(ROOT/'outputs/grid33/review.json')['rows']
softrows=read(ROOT/'outputs/grid40b/offline_review.json')['rows']
rankers={'GNN0','GNN1','PPO','SOFTGNN0','SOFTGNN1'}
sources=[r for r in oldrows+softrows if r['ranker'] in rankers]; assert len(sources)==130
rows=[]
spec=importlib.util.spec_from_file_location('grid42_old_completion',ROOT/'work/grid28/policy.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
for r in sources:
    p=ROOT/'outputs/grid26/runs'/f"r0__{r['week']}__FULL"/r['call']
    full=read(p); inputs[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    truth={x['pool_id']:x for x in full['outcomes']}; aliases={int(k):int(v) for k,v in full['aliases'].items()}
    queried=[]
    def observe(i,phase):
        assert i not in queried
        queried.append(i); return {**truth[i],'phase':phase}
    base=previous.local_complete(r['order'],lambda i,p:{**truth[i],'phase':p},'PATIENCE64',subs,aliases,.9)
    assert base['chosen_pool_id']==r['choice'] and base['public_queries']==r['queries']
    z=tied_complete(r['order'],observe,'PATIENCE64',subs,aliases,.9)
    choices=[x for x in z['outcomes'] if x['strict_admissible']]
    assert z['chosen_pool_id']==(max(choices,key=key)['pool_id'] if choices else None)
    winner=full['chosen_pool_id']; chosen=z['chosen_pool_id']
    rows.append({'week':r['week'],'call':r['call'],'step':full['step'],'ranker':r['ranker'],
                 'old_queries':r['queries'],'queries':z['public_queries'],'old_choice':r['choice'],'choice':chosen,
                 'full_choice':winner,'same_full_choice':chosen==winner,
                 'rho_regret':None if winner is None or chosen is None else truth[chosen]['rho']-truth[winner]['rho'],
                 'tie_substations':z['tie_substations'],'observed_sequence':queried,'stop_reason':z['stop_reason']})
summary=[]
for name in sorted(rankers):
    g=[r for r in rows if r['ranker']==name]; assert len(g)==26
    def critical(step):return next(r for r in g if r['week']=='2035-03-05_0' and r['step']==step)['same_full_choice']
    summary.append({'ranker':name,'old_queries':sum(r['old_queries'] for r in g),'queries':sum(r['queries'] for r in g),
                    'same_full_choice':sum(r['same_full_choice'] for r in g),'max_rho_regret':max(r['rho_regret'] or 0 for r in g),
                    'critical578':critical(578),'critical1733':critical(1733)})
save(OUT/'review.json',{'passed':True,'hand_checks':5,'summaries':summary,'rows':rows,'source_hashes':inputs,
                       'new_fits':0,'new_forecasts':0,'reserved_evaluations':0})
with (OUT/'per_state.csv').open('w',newline='',encoding='utf-8') as f:
    fields=[k for k in rows[0] if k not in ['tie_substations','observed_sequence']]
    w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
lines=['# GRID42 已观测即时并列的局部补查','',
       '不新增训练或预测，仅重放26个已暴露完整搜索调用。同一补查给所有排序器；不是GNN独有改进或长期质量证书。','',
       '| 排序器 | 原查询 | 补查查询 | 与FULL选择相同 | 找回578 | 找回1733 |','|---|---:|---:|---:|---:|---:|']
for s in summary: lines.append(f"|{s['ranker']}|{s['old_queries']}|{s['queries']}|{s['same_full_choice']}/26|{s['critical578']}|{s['critical1733']}|")
lines+=['','即时奖励并列不保证长期成本并列；补查仍可能遗漏未观测变电站的更好动作。所有模型和旧策略保留。']
(OUT/'GRID42_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
