"""Common context and action-access source mapping, before six baseline fits."""
from pathlib import Path
import datetime as dt,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid77';sys.path.insert(0,str(Path(__file__).parent));import metacontrol as M
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not OUT.exists()
    assert read(ROOT/'outputs/grid76/review.json')['passed_saved_data']
    assert read(ROOT/'outputs/grid76/review.json')['quality_dates']==13
    assert read(ROOT/'outputs/grid76/all_context_review.json')['passed']
    source=read(ROOT/'outputs/grid64/design.json');D=dict(source)
    D.update(stage='GRID77_COMMON_CONTEXT_AND_MODE_DQN',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_mapping='Same ICAPS2022-style Double DQN core; adapted grid mode actions RETURN/GLOBAL64/DELEGATE_FRAME_GUARD(initial only). Both same public24 operating/search features; explicit initial-monitoring bit added to baseline. Not full original A*/RRT* reproduction.',
        changes_from_grid64=['input12→25 (24 shared context/search plus mode availability)','outputs2→3, macro verification action gives exact same complete control option as cheap context gate','episodes15000→30000 prospectively for enlarged controls','quality calibration reference is repaired FRAME_GUARD, not old FRAME'],
        episodes=30000,features=M.G.NAMES[:24]+['initial_monitoring'],actions=['RETURN','GLOBAL64','DELEGATE_FRAME_GUARD'],
        calibration='One shared cost level across both seeds; feasible if each seed calibration mean AND max one-step FULL-pool regret<=FRAME_GUARD+1e-7. Among feasible minimize mean queries, then regret/index; if none disclose and choose least mean/max regret. Final checkpoint only; no test feedback.',
        evaluation_weeks_excluded=read(ROOT/'outputs/grid70/design.json')['weeks'],
        double_q_target='Source softmax(online nextQ/.001) dot target nextQ, masked legal actions; selected-action target and MSE over all3 outputs; Adam and target retention unchanged.',
        caps=dict(training_wall_s=3600,fits=6,new_teacher_queries=0),training_teacher='Reuse fully-paid151 GRID64 traces and audited public graph cache. No GRID76 evaluation results are loaded by fit/calibration. No learner feature is a hidden suffix or a date ID.')
    records=[];receipts=[];excluded={w[:10] for w in read(ROOT/'outputs/grid70/design.json')['weeks']}
    assert not excluded&{r['family'] for r in read(ROOT/'outputs/grid64/records.json')}
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
    for r in read(ROOT/'outputs/grid64/records.json'):
        raw=read(ROOT/r['source']);rows=r['rows'];order=[x['pool_id'] for x in rows];hashes={x['combined_action_hash']:x['pool_id'] for x in rows}
        aliases={int(i):hashes[x['combined_action_hash']] for i,x in zip(raw['pool_ids'],raw['outcomes']) if x['combined_action_hash'] in hashes}
        cached=ROOT/'outputs/grid21/inputs/train'/f"{r['id']}.npz"
        with np.load(cached) as z:context={k:z[k].tolist() for k in ['x','edges','glob']}
        cc={k:np.asarray(v,dtype=np.int64 if k=='edges' else np.float32) for k,v in context.items()}
        truth={x['pool_id']:x for x in rows};guard=M.P.run(order,lambda i,p:{**truth[i],'phase':p},'FRAME_GUARD',subs,aliases,limit=r['rho_limit'])
        chosen=guard['chosen_pool_id'];records.append(dict(id=r['id'],family=r['family'],split=r['split'],rows=rows,order=order,aliases=aliases,context=context,rho_limit=r['rho_limit'],
            guard_queries=guard['public_queries'],guard_choice=chosen,guard_quality=M.quality(None if chosen is None else truth[chosen]),best_quality=r['best_quality']))
        receipts.append(dict(source=r['source'],source_sha256=sha(ROOT/r['source']),context_source=cached.relative_to(ROOT).as_posix(),context_sha256=sha(cached)))
    OUT.mkdir();save(OUT/'design.json',D);save(OUT/'records.json',records);save(OUT/'data_receipts.json',dict(receipts=receipts,subids=subs,excluded_families=sorted(excluded),new_queries=0))
    frozen=read(ROOT/'outputs/grid76/code_freeze.json')
    for p,h in frozen.items():assert sha(ROOT/p)==h,p
    for p in list(Path(__file__).parent.glob('*.py'))+list(OUT.glob('*')):
        if p.is_file():frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'code_freeze.json',frozen);save(OUT/'preflight.json',dict(passed=True,records=151,train=129,calibration=22,no_evaluation_overlap=True,actions=3,features=25,new_queries=0))
    (OUT/'RUN_STATE.md').write_text('FROZEN:six common-context/mode conference baseline fits; no physical queries.\n',encoding='utf-8')
    print(json.dumps(read(OUT/'preflight.json')))
if __name__=='__main__':main()
