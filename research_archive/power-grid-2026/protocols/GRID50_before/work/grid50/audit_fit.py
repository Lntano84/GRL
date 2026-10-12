"""Postfit checkpoint/input reconciliation;not an independent retraining."""
import hashlib,json,os,sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid50'
sys.path[:0]=[str(HERE),str(ROOT/'work/grid21'),str(ROOT/'work/grid20')]
import numpy as np
import torch
from compressed_residual import CompressedResidualScorer
from prior import ResidualScorer
from components import tensors,fingerprint
torch.set_num_threads(1)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
D=read(OUT/'design.json');T=read(OUT/'training_finished.json');R=read(OUT/'review.json')
assert len(T['fits'])==2 and len(R['rows'])==52 and T['development_labels_opened']==0
for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
evaluated=[];checkpoints=[]
for seed in [0,1]:
    folder=OUT/'models'/f'gnn_seed{seed}';initial=ROOT/f'outputs/grid21/models/gnn_seed{seed}/initial.pt'
    assert sha(initial)==sha(folder/'initial.pt')
    state=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
    source=torch.load(initial,map_location='cpu',weights_only=True)
    p=ROOT/'outputs/grid21/inputs/train'/(read(ROOT/'outputs/grid21/training_finished.json')['training_input_ids'][0]+'.npz')
    with np.load(p) as z:nodes=len(z['x']);base=len(z['base']);plans=z['plans'].shape[1]
    model=CompressedResidualScorer('gnn',nodes,372,base,plans,state['inner.subids'].tolist());model.load_state_dict(state);model.eval()
    old=ResidualScorer('gnn',nodes,372,base,plans,state['inner.subids'].tolist());old.load_state_dict(torch.load(ROOT/f'outputs/grid21/models/gnn_seed{seed}/final.pt',map_location='cpu',weights_only=True));old.eval()
    assert set(state)==set(source)
    assert all(torch.equal(state[k],source[k]) for k in state if any(t in k for t in ['core.actor.','core.critic.','core.trunk.']))
    losses=[[],[]];maxraw=0.
    for identifier in read(ROOT/'outputs/grid21/training_finished.json')['training_input_ids']:
        p=ROOT/'outputs/grid21/inputs/train'/f'{identifier}.npz';meta=read(p.with_suffix('.json'))
        with np.load(p) as z:f={k:z[k].copy() for k in ['x','edges','edge_attr','glob','base','plans']};q=z['target'];mask=torch.as_tensor(z['mask']);prior=torch.as_tensor(z['prior'])
        assert fingerprint(f)==meta['input_hash']
        if not q.any():continue
        for n,m in enumerate([old,model]):
            with torch.no_grad():raw=m(**tensors(f))[0];logs=torch.nn.functional.log_softmax((prior+raw).masked_fill(~mask,-1e9),dim=0)
            assert torch.isfinite(raw).all();maxraw=max(maxraw,float(raw.abs().max()) if n else 0.)
            pl=torch.nn.functional.log_softmax(prior.masked_fill(~mask,-1e9),dim=0)
            loss=-(torch.as_tensor(q)*logs).sum()+D['prior_KL_weight']*(logs.exp()*(logs-pl)).sum();losses[n].append(float(loss))
    assert all(len(v)==148 for v in losses)
    evaluated.append({'seed':seed,'old_fixed_final_training_objective':sum(losses[0])/148,'new_fixed_final_training_objective':sum(losses[1])/148,'new_max_abs_training_residual':maxraw})
    for name in ['initial','final']:p=folder/f'{name}.pt';checkpoints.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
for p,h in R['input_hashes'].items():assert sha(ROOT/p)==h,p
A={'passed':True,'initial_checkpoint_byte_checks':2,'training_inputs_checked':334,'development_cached_checks':52,
   'fixed_training_evaluation':evaluated,'checkpoints':checkpoints,'new_forecasts':0,'GRID44_evaluations':0,
   'warning':'Recomputed fixed-final training objective/checkpoint consistency;not retraining independence or closed-loop quality. Higher representational capacity can overfit.'}
(OUT/'audit_fit.json').write_text(json.dumps(A,indent=2),encoding='utf-8')
lines=['# GRID50 原快速结构下的一次残差评分改动','', '仅解除末端硬上限；两种子同原初始化、原167状态/148目标、原64轮训练；全部原模型与曲线保留。','',
 '| 种子 | 原固定终点训练目标 | 新固定终点训练目标 | 开发原查询 | 新查询 | 同FULL选择 原→新 |','|---|---:|---:|---:|---:|---:|']
for e in evaluated:
    s=next(s for s in R['summaries'] if s['ranker']==f"GNN{e['seed']}")
    lines.append(f"| {e['seed']} | {e['old_fixed_final_training_objective']:.5f} | {e['new_fixed_final_training_objective']:.5f} | {s['old_queries']} | {s['queries']} | {s['old_same_full_choice']}→{s['same_full_choice']} /26 |")
lines+=['', '开发信号混合：种子0查询略增、选择恢复改善；种子1查询减少、某状态rho差增大。两者均找回既知1733；不挑种子、不由训练目标改善推断闭环收益。',
 '硬界限制不是Top128不可辨识证明。当前只说明一版合法评分配置已完成；新模型是否能保留原版质量和提速要由有限闭环试点决定。',
 'GRID44八个确认家族及其变体未用于训练或新模型评价；原14.6%确认提速与0.885%最坏成本差保持原样。']
(OUT/'GRID50_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'fixed_training_evaluation':evaluated},indent=2))
