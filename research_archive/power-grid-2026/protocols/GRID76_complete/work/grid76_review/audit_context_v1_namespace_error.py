"""Reconstruct every saved public gate input, without forecasts or AC solves."""
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid76'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def fingerprint(context):
    h=hashlib.sha256()
    for k in sorted(context):
        x=np.ascontiguousarray(context[k]);h.update(k.encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','all'],default='pilot');a=ap.parse_args()
    assert not (OUT/'RUN.lock').exists()
    sys.path[:0]=[str(ROOT/'work/grid08'),str(ROOT/'work/grid76')]
    from common import make_env,graph_features
    import policies as P
    env=make_env();obs0=env.reset();nb=int(env.nb_highres_called)
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];count=decisions=early=0;model=P.G.load('plain')
    try:
        for s in read(OUT/'manifest.json')['summaries']:
            folder=OUT/'runs'/f"r0__{s['week']}__{s['method']}"
            for j,p in enumerate(sorted(folder.glob('call[0-9][0-9][0-9].json'))):
                c=read(p)
                with np.load(folder/f'call{j:03d}_inputs.npz') as z:context={k:z[k] for k in ['x','edges','glob']}
                assert fingerprint(context)==c['input_hash']
                with np.load(folder/f'call{j:03d}_state.npz') as z:obs=obs0.copy();obs.from_vect(z['observation'])
                x,e,g=graph_features(obs,env,dict(offered=False,predictions=[None,None]))
                assert np.array_equal(x,context['x']) and np.array_equal(e,context['edges']) and np.array_equal(g,context['glob'])
                if c['gate_input'] is not None:
                    prefix=c['outcomes'][:min(128,len(c['order']))]
                    vector=P.G.features(prefix,c['order'],subs,{int(k):int(v) for k,v in c['aliases'].items()},c['rho_limit'],context)
                    assert np.array_equal(vector,np.asarray(c['gate_input'],dtype=np.float32))
                    defer,info=P.C.decide(model,vector);assert info==c['gate_decision']
                    assert (not defer)==(c['stop_reason']=='operating_context_early_return');decisions+=1;early+=int(not defer)
                count+=1
        assert int(env.nb_highres_called)==nb
    finally:env.close()
    result=dict(passed=True,phase=a.phase,public_contexts_reconstructed=count,gate_decisions_recomputed=decisions,early_returns=early,
        model_sha256=hashlib.sha256((ROOT/'outputs/grid75/models/plain.json').read_bytes()).hexdigest(),new_forecasts=0,independent_ac_resolve=False)
    (OUT/f'{a.phase}_context_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
