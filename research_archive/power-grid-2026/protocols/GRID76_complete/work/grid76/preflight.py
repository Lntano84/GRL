"""Legal context reconstruction and exact replay of fixed calibration gate."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));import policies as P
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    items={x['id']:x for x in read(ROOT/'outputs/grid75/input_receipts.json')['items']}
    records=read(ROOT/'outputs/grid64/records.json');subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
    checked=matched=0;examples=[]
    for r in records:
        if r['id'] not in items:continue
        raw=read(ROOT/r['source']);truth={x['pool_id']:x for x in r['rows']};order=sorted(truth)
        hashes={x['combined_action_hash']:x['pool_id'] for x in r['rows']}
        aliases={int(i):hashes[x['combined_action_hash']] for i,x in zip(raw['pool_ids'],raw['outcomes']) if x['combined_action_hash'] in hashes}
        with np.load(ROOT/items[r['id']]['context_source']) as z:context={k:z[k] for k in ['x','edges','glob']}
        seen=[]
        def observe(i,phase):
            assert i not in seen;seen.append(i);return {**truth[i],'phase':phase}
        got=P.run(order,observe,'CONTEXT_GATE',subs,aliases,limit=r['rho_limit'],context=context)
        assert seen==[x['pool_id'] for x in got['outcomes']] and len(seen)==len(set(seen))
        if r['split']=='calibration':assert got['chosen_pool_id']==items[r['id']]['teacher_action'];matched+=1
        checked+=1
        if len(examples)<2:examples.append((r,context,raw))
    sys.path[:0]=[str(ROOT/'work/grid08')]
    from common import make_env,graph_features
    env=make_env();obs0=env.reset();initial=int(env.nb_highres_called)
    try:
        for rec,cached,raw in examples:
            source=ROOT/rec['source']
            with np.load(source.with_name(source.stem+'_state.npz')) as z:obs=obs0.copy();obs.from_vect(z['observation'])
            x,e,g=graph_features(obs,env,dict(offered=False,predictions=[None,None]))
            assert np.array_equal(x,cached['x']) and np.array_equal(e,cached['edges']) and np.array_equal(g,cached['glob'])
        assert int(env.nb_highres_called)==initial
    finally:env.close()
    result=dict(passed=True,legal_trace_replays=checked,calibration_teacher_actions_matched=matched,
        reconstructed_public_contexts=2,new_forecasts=0,new_fits=0,portable_correction='work/grid75_review/runtime_corrected.py')
    OUT=ROOT/'outputs/grid76_preflight';OUT.mkdir(exist_ok=True)
    (OUT/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
