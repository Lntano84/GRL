"""One scoring variant versus the original fast controller;old development only."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid51'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
assert read(ROOT/'outputs/grid50/audit_fit.json')['passed']
assert all(s['known_critical_choice_match'] for s in read(ROOT/'outputs/grid50/review.json')['summaries'])
old=read(ROOT/'outputs/grid41/design.json');methods=[dict(m) for m in old['methods'] if m['method'] in ['FULL','LOCAL_GNN0','LOCAL_GNN1']]
for seed in [0,1]:methods.append({'method':f'COMPRESSED_GNN{seed}','rule':'COMPRESSEDGNN','budget':128,'checkpoint':f'outputs/grid50/models/gnn_seed{seed}/final.pt','policy':'PATIENCE64','cache_dc':True,'cache_safe':True})
order=[]
for n,week in enumerate(old['weeks']):
    shift=n%len(methods)
    for m in methods[shift:]+methods[:shift]:order.append({**m,'week':week,'repeat':0,'stage_phase':'pilot' if n==0 else 'remaining'})
D={**old,'stage':'GRID51_SOFT_COMPRESSION_ORIGINAL_FAST_CONTROL','methods':methods,'execution_order':order,
   'changes':'Only GNN scoring correction forward for COMPRESSED_GNN0/1. Original fast GRID28 PATIENCE64+local completion unchanged for ALL arms;strictGRID47guard andGRID43tie completion NOT used. Original/full/new have common cache,physicalbackend/control/masks/inputprior/teacherdata. No new threshold,sweep,portfolio or RL.',
   'scope':'5Marchpilot then15olddevelopment only when BOTH newseeds quality<=.5%/noearlierfailure. Five implementations are original2seeds,new2seeds andFULL;configuration ablation,not expanding formalbaseline families. AuthorPPO/SoftGNN strongest comparisons remain GRID41/44/47.',
   'pilot_gate':'Bothnewseeds complete whenFULLdoes,noearlyfailure,completed cost<=.5%FULL. A failing seed retained;do not run15remaining or pick otherseed alone.',
   'development_interpretation':'Assess paired new vs matched originalseed control/query/cost. Preserve mixed results;training objective decrease or cachedrank changes are not quality or speed. No automatic defaultmodel replacement.',
   'sealed_status':'GRID44all8families/variants remain excluded from training/new evaluation;no use here. Any future confirmation must freshly freeze datefamilies.'}
src=(ROOT/'work/grid41/run.py').read_text(encoding='utf-8')
changes={'"""GRID41 matched official-core SoftGNN and original GNN development runs."""':'"""GRID51 one soft-compressed residual scoring revision,original fast control retained."""',
 "OUT=ROOT/'outputs/grid41'":"OUT=ROOT/'outputs/grid51'",
 '    install_search_adapter(engine.Search)':'    install_search_adapter(engine.Search)\n    sys.path.insert(0,str(ROOT/\'work/grid50\'))\n    from online_adapter import install_free_adapter\n    install_free_adapter(engine.Search)',
 'all5arms shareDC+safe cache':'all5arms shareDC+safe cache and original fast semantic policy'}
for a,b in changes.items():assert src.count(a)==1,(a,src.count(a));src=src.replace(a,b)
(HERE/'run.py').write_text(src,encoding='utf-8')
a=(ROOT/'work/grid41/audit.py').read_text(encoding='utf-8')
oldpairs="[('LOCAL_GNN0','FULL'),('LOCAL_GNN1','FULL'),('SOFTGNN0','FULL'),('SOFTGNN1','FULL'),('LOCAL_GNN0','SOFTGNN0'),('LOCAL_GNN0','SOFTGNN1'),('LOCAL_GNN1','SOFTGNN0'),('LOCAL_GNN1','SOFTGNN1')]"
newpairs="[('LOCAL_GNN0','FULL'),('LOCAL_GNN1','FULL'),('COMPRESSED_GNN0','FULL'),('COMPRESSED_GNN1','FULL'),('COMPRESSED_GNN0','LOCAL_GNN0'),('COMPRESSED_GNN1','LOCAL_GNN1'),('COMPRESSED_GNN0','LOCAL_GNN1'),('COMPRESSED_GNN1','LOCAL_GNN0')]"
changes={"OUT=ROOT/'outputs/grid41'":"OUT=ROOT/'outputs/grid51'",oldpairs:newpairs,
 '# GRID41 近期SoftGNN适配与原模型的同条件闭环比较':'# GRID51 原快速控制下残差评分改动',"OUT/'GRID41_report.md'":"OUT/'GRID51_report.md'",
 'Fresh originalGNN and SoftGNN adaptation compared under common controller/caches/stop.':'Original/soft-compressed residualGNN compared under unchanged fast controller/caches/stop.',
 '20 matched SoftGNN/original model comparisons.':'20 matched original/soft-compressed-residual model comparisons.',
 "regret=(s['cost']-full['cost'])/max(1,abs(full['cost'])) if comparable else None":"regret=(s['cost']-full['cost'])/max(1,abs(full['cost'])) if full['complete'] and s['complete'] else None",
 "bool(survival and (regret is not None and regret<=.005))":"bool(survival and (not full['complete'] or (regret is not None and regret<=.005)))"}
for x,y in changes.items():assert a.count(x)==1,(x,a.count(x));a=a.replace(x,y)
(HERE/'audit.py').write_text(a,encoding='utf-8')
OUT.mkdir(exist_ok=False);(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=[Path(__file__).resolve(),HERE/'run.py',HERE/'audit.py',HERE/'audit_pilot.py',ROOT/'work/grid50/online_adapter.py',ROOT/'work/grid50/compressed_residual.py',ROOT/'outputs/grid50/audit_fit.json']
files += [ROOT/p for p in read(ROOT/'outputs/grid41/code_freeze.json')]
for seed in [0,1]:files.append(ROOT/f'outputs/grid50/models/gnn_seed{seed}/final.pt')
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in dict.fromkeys(files)},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:5oldMarchpilots;15remaining only if bothnewseeds qualify. Original fastpolicy/models unchanged;no GRID44reuse.\n',encoding='utf-8')
print(json.dumps({'planned':20,'pilot':5,'new_model_fits_here':0,'original_fast_guard_retained':True}))
