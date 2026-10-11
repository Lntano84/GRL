"""One common quality repair; six-arm bounded development control comparison."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent; OUT=ROOT/'outputs/grid43'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
assert read(ROOT/'outputs/grid42b/audit.json')['passed']
S=read(ROOT/'outputs/grid42b/review.json')['summaries']
assert all(s['critical578'] and s['critical1733'] for s in S)
previous=read(ROOT/'outputs/grid41/design.json')
methods=[dict(m) for m in previous['methods']]
methods.append(next(dict(m) for m in read(ROOT/'outputs/grid34/design.json')['methods'] if m['method']=='LOCAL_PPO'))
order=[]
for n,week in enumerate(previous['weeks']):
    rotation=n%len(methods)
    for m in methods[rotation:]+methods[:rotation]:
        order.append({**m,'week':week,'repeat':0,'stage_phase':'pilot' if n==0 else 'remaining'})
D={**previous,'stage':'GRID43_COMMON_OBSERVED_TIE_COMPLETION','methods':methods,'execution_order':order,
   'changes':'Only common stop completion is extended to substations of observed exact-fp32 best-reward ties, including safe prefix. All masks, canonical actions, reward, models, inputs, thresholds, caches and control branches unchanged; FULL stays exhaustive.',
   'scope':'4exposed development weeks; six implementations = own2seeds + one recent SoftGNN adaptation2seeds + authorPPO + matureFULL. Three main baseline method families, no new cheap-rule menu.',
   'pilot_gate':'March6runs first. Saved feasibility, no earlier failure and cost<=0.5%ofFULL for every ranked arm required for remaining18; no model selection or result-based retuning within this protocol.',
   'interpretation':'Same quality repair granted to every ranker, not graph-only novelty. Preserve prior GRID41 losses. Single timing repeat for development; no final8date access.',
   'caps':{'episode_s':1500,'wall_s':9000,'physical_steps':50000,'public_forecasts':450000,'new_fits':0,'reserved_evaluations':0}}
src=(ROOT/'work/grid41/run.py').read_text(encoding='utf-8')
changes={
 '"""GRID41 matched official-core SoftGNN and original GNN development runs."""':'"""GRID43 common observed-tie completion; all existing rankers, unchanged models."""',
 "OUT=ROOT/'outputs/grid41'":"OUT=ROOT/'outputs/grid43'",
 'all5arms shareDC+safe cache':'all6arms shareDC+safe cache and common tied completion',
 '    install_search_adapter(engine.Search)':'    install_search_adapter(engine.Search)\n    sys.path.insert(0,str(ROOT/\'work/grid42b\'))\n    from tied_policy import tied_complete\n    engine.P.local_complete=tied_complete'
}
for a,b in changes.items():assert src.count(a)==1,(a,src.count(a));src=src.replace(a,b)
assert not (HERE/'run.py').exists();(HERE/'run.py').write_text(src,encoding='utf-8')
a=(ROOT/'work/grid41/audit.py').read_text(encoding='utf-8')
old_pairs="[('LOCAL_GNN0','FULL'),('LOCAL_GNN1','FULL'),('SOFTGNN0','FULL'),('SOFTGNN1','FULL'),('LOCAL_GNN0','SOFTGNN0'),('LOCAL_GNN0','SOFTGNN1'),('LOCAL_GNN1','SOFTGNN0'),('LOCAL_GNN1','SOFTGNN1')]"
new_pairs=old_pairs[:-1]+",('LOCAL_PPO','FULL'),('LOCAL_GNN0','LOCAL_PPO'),('LOCAL_GNN1','LOCAL_PPO')]"
changes={"OUT=ROOT/'outputs/grid41'":"OUT=ROOT/'outputs/grid43'",
         "len(F['summaries'])==20":"len(F['summaries'])==24", "'runs':20":"'runs':24",
         old_pairs:new_pairs,
         '# GRID41 近期SoftGNN适配与原模型的同条件闭环比较':'# GRID43 共同并列补查的质量与速度',
         "f'20条；{physical}物理步；{forecasts}公开预测。'":"f'24条；{physical}物理步；{forecasts}公开预测。'",
         "OUT/'GRID41_report.md'":"OUT/'GRID43_report.md'",
         'COMPLETED AND AUDITED:20 matched SoftGNN/original model comparisons. No new fits/final dates.':'COMPLETED AND AUDITED:24 common tied-completion comparisons. No new fits/final dates.',
         'Fresh originalGNN and SoftGNN adaptation compared under common controller/caches/stop.':'Fresh originalGNN,SoftGNN adaptation and authorPPO with common observed-tie completion/controller/caches.'}
for x,y in changes.items():assert a.count(x)==1,(x,a.count(x));a=a.replace(x,y)
(HERE/'audit.py').write_text(a,encoding='utf-8')
OUT.mkdir(exist_ok=False)
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=[Path(__file__).resolve(),HERE/'run.py',HERE/'audit.py',HERE/'audit_pilot.py',ROOT/'work/grid42b/tied_policy.py',ROOT/'outputs/grid42b/design.json',ROOT/'outputs/grid42b/audit.json']
files += [ROOT/p for p in read(ROOT/'outputs/grid41/code_freeze.json')]
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in dict.fromkeys(files)},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:6March pilots,18remaining only after common quality gate. All original models unchanged; final8dates sealed.\n',encoding='utf-8')
print(json.dumps({'planned':24,'pilot':6,'new_models':0,'reserved_evaluations':0}))
