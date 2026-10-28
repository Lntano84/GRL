"""Frozen original owner-NN hidden vectors; never accepts a new reference model."""
from pathlib import Path
import numpy as np
import runtime254 as C
SOURCES=[C.R/'outputs/grid168_one_sided_transfer']+[C.R/f'outputs/grid170_remaining_date_folds/fold{f}' for f in [1,2,3]]
def extract_original(f,seed,ids):
 import torch,pickle
 S=C.orig();root=SOURCES[f];a=np.load(root/'fold_arrays.npz');ids=np.asarray(ids,dtype=int);assert set(ids.tolist())<=set(a['fit_ids'].tolist()),'This data interface is restricted to original owner-fit dates.';raw=pickle.loads((C.R/'outputs/grid113_relational_learning/train_public.pkl').read_bytes());rows=[dict(raw[int(i)],global41=a['normalized'][int(i)]) for i in ids];path=root/'models'/f'FACTOR_seed{seed}.pt';before=C.sha(path);net=S.build(seed,'FACTOR');cp=torch.load(path,weights_only=False,map_location='cpu');net.load_state_dict(cp['state_dict']);net.eval();digest=S.digest(net);receipt=next(r for r in C.read(root/'fit_receipts.json') if r['arm']=='FACTOR' and r['seed']==seed);assert digest==receipt['final_hash'];capture={};hooks=[net.base.q_net[3].register_forward_hook(lambda m,i,o:capture.__setitem__('main',o.detach().numpy().copy())),net.delta.head[1].register_forward_hook(lambda m,i,o:capture.__setitem__('delta',o.detach().numpy().copy()))];out=[];maxerr=0.
 try:
  with torch.inference_mode():
   for start in range(0,len(rows),32):
    rr=rows[start:start+32];q=net(**S.G.old_api().collate(rr)).numpy();main,delta=capture['main'],capture['delta'];w=cp['state_dict'];manual=main.astype(float)@w['base.q_net.4.weight'].numpy().astype(float).T+w['base.q_net.4.bias'].numpy()+delta.astype(float)@w['delta.head.2.weight'].numpy().astype(float).T+w['delta.head.2.bias'].numpy();maxerr=max(maxerr,float(abs(manual-q).max()));assert np.allclose(manual,q,atol=3e-5,rtol=3e-5);out.append(np.c_[main,delta,q,np.stack([r['known_query_cost'] for r in rr])])
 finally:
  for h in hooks:h.remove()
 assert C.sha(path)==before and S.digest(net)==digest;result=np.concatenate(out).astype(float);ref=np.load(root/'state_scores.npz')[f'FACTOR_seed{seed}'][ids];assert result.shape==(len(ids),83) and np.allclose(result[:,80:82],ref,atol=3e-5,rtol=3e-5);return result,dict(owner_outer=f,seed=seed,rows=len(ids),model_path=path.relative_to(C.R).as_posix(),original_model_hash=before,original_model_unchanged=True,new_reference_hidden_never_used=True,max_linear_readout_error=maxerr,max_saved_q_error=float(abs(result[:,80:82]-ref).max()))
