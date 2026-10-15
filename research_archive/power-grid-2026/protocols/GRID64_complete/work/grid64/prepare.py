"""Qualify fully-paid historical training feedback, split date families, freeze before fits."""
from pathlib import Path
import hashlib,importlib.util,json,math,py_compile,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid64'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(HERE));import metacontrol as M
def main():
    assert not OUT.exists(),'Preserve existing run.'
    official=ROOT/'work/grid64_baseline_sources/Metareasoning.jl'
    source=read(ROOT/'outputs/grid64_baseline_qualification/source_qualification.json')
    for f in source['preferred_source']['files']:assert sha(official/f['path'])==f['sha256']
    evaluation=read(ROOT/'outputs/grid63/design.json')['weeks'];excluded={w[:10] for w in evaluation}
    ids=read(ROOT/'outputs/grid21/training_finished.json')['training_input_ids']
    metas=[(i,read(ROOT/'outputs/grid21/inputs/train'/f'{i}.json')) for i in ids]
    kept=[(i,m) for i,m in metas if m['week'][:10] not in excluded]
    families=sorted({m['week'][:10] for _,m in kept});assert len(families)==18 and len(kept)==151
    ranked=sorted(families,key=lambda f:hashlib.sha256(('GRID64_META_DQN_VAL_20261015|'+f).encode()).hexdigest())
    validation=set(ranked[:4]);training=set(families)-validation
    plan=read(ROOT/'outputs/review_budgetmatch_20261014/plan.json');subs=read(ROOT/plan['substation_metadata'])['public_subids']
    spec=importlib.util.spec_from_file_location('grid64_frame',ROOT/'work/grid63/policies.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
    records=[];receipts=[];permission=0
    for identifier,meta in kept:
        path=ROOT/meta['source'];assert sha(path)==meta['source_sha256']
        c=read(path);truth={int(i):{**v,'pool_id':int(i)} for i,v in zip(c['pool_ids'],c['outcomes'])}
        with np.load(ROOT/'outputs/grid21/inputs/train'/f'{identifier}.npz') as z:reps=sorted(int(x) for x in z['representatives'])
        assert len(reps)==meta['canonical_count'] and len(set(reps))==len(reps)
        rows=[truth[i] for i in reps]
        for row in rows:
            valid=0<row['rho']<c['rho_limit'] and not row['done'] and not row['exceptions'] and not row['illegal'] and not row['ambiguous']
            assert bool(row['strict_admissible'])==bool(valid)
            if valid:assert 0<=row['reward']<=2 and abs(row['reward']-(2-row['rho']))<2e-6
        # Reconstruct only canonical aliases represented by this observed state.
        hashes={truth[i]['combined_action_hash']:i for i in reps};assert len(hashes)==len(reps)
        aliases={i:hashes[v['combined_action_hash']] for i,v in truth.items() if v['combined_action_hash'] in hashes}
        frame=P.run(reps,lambda i,p:{**truth[i],'phase':p},'FRAME',subs,aliases)
        chosen=frame['chosen_pool_id'];fr=None if chosen is None else truth[chosen]
        ps=M.prefixes(rows,c['rho_limit'])
        # Any unrevealed suffix perturbation must leave a current decision feature unchanged.
        for p in ps[:-1]:
            altered=[dict(x) for x in rows]
            for x in altered[p['end']:]:x.update(reward=-999.,rho=999.,strict_admissible=False)
            z=M.prefixes(altered,c['rho_limit'])
            j=next(j for j,x in enumerate(ps) if x['end']==p['end'])
            assert np.array_equal(z[j]['state'],p['state']);permission+=1
        records.append({'id':identifier,'week':meta['week'],'family':meta['week'][:10],
            'split':'calibration' if meta['week'][:10] in validation else 'train','rho_limit':c['rho_limit'],
            'rows':rows,'frame_queries':frame['public_queries'],'frame_quality':M.quality(fr),
            'best_quality':ps[-1]['quality'],'source':meta['source']})
        receipts.append({'path':meta['source'],'sha256':sha(path),'canonical_count':len(rows),
            'paid_source_count':len(c['outcomes']),'paid_source_forecast_s':sum(r['forecast_s'] for r in c['outcomes'])})
    OUT.mkdir();(OUT/'records.json').write_text(json.dumps(records,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    D={'stage':'GRID64_ICAPS2022_STOPPING_ONLY_ADAPTATION','authorization':'User 2026-10-15: begin the published adaptive stopping baseline.',
       'source_commit':source['preferred_source']['commit'],'source':'Bhatia et al., ICAPS2022, Metareasoning.jl; Python/PyTorch algorithm port, stopping-only restriction, not full original A*/RRT* reproduction.',
       'evaluation_weeks_excluded':evaluation,'training_families':sorted(training),'calibration_families':sorted(validation),
       'monitoring':{'initial':128,'batch':64,'order':'same author frequency/representative ID','stop_mask':'No admissible incumbent -> continue, exhausted pool -> return.'},
       'quality':'q=clip(observed best strict-admissible N1 reward / 2,0,1); zero if none. Training valid rewards verified as 2-rho. This is known one-step search quality, NOT whole-week quality.',
       'utility':'U=q-expm1(beta*(queries/pool_size)); transition reward U_next-U_current; STOP zero; initial 128-query cost common and included online.',
       'cost_levels':[.25,.0025,0.],'beta':'log1p(cost_level); original example .25 and two predefined quality-oriented calibration alternatives; no test feedback.',
       'seeds':[0,1],'episodes':15000,'min_explore_steps':20000,'epsilon_anneal_steps':20000,'epsilon_final':.1,
       'hidden':[64,32],'activation':'relu','initialization':'Flux-style Glorot uniform, zero biases','lr':.0001,
       'gamma':.99,'batch_size':256,'replay_capacity':1000000,'target_retention':.995,'target_update_every':1,
       'double_q_target':'softmax(online next Q / .001) dot target next Q; mask invalid STOP; selected-action target, mean squared error over both outputs, Adam.',
       'deploy':'deterministic argmax with STOP/CONTINUE guard; epsilon disabled explicitly. This replaces the source soft sampler at inference for reproducibility.',
       'calibration':'Select one shared cost level across both seeds. Feasible if each seed calibration mean AND max full-pool one-step regret <= FRAME +1e-7. Among feasible minimize mean queries, then mean regret, then cost index. If none, minimize mean regret then max regret then queries and disclose no calibration quality match. All final 15000-episode checkpoints retained; no best-checkpoint selection.',
       'future_evaluation':'Only after code/data/learning qualification: separately freeze same seven exposed dates against FRAME/GLOBAL_MATCH/FULL and both selected seeds. No new-date generalization claim.',
       'scope':'One learned literature-based baseline family; six fits for prospective three-cost calibration across two fixed seeds. No GNN or new control RL; no new teacher simulation.',
       'caps':{'training_wall_s':3600,'new_teacher_queries':0,'fits':6},'features':M.NAMES}
    (OUT/'design.json').write_text(json.dumps(D,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'data_receipts.json').write_text(json.dumps({'sources':receipts,'records':151,'train_records':sum(r['split']=='train' for r in records),'calibration_records':sum(r['split']=='calibration' for r in records),'historical_forecast_s':sum(r['paid_source_forecast_s'] for r in receipts),'historical_queries':sum(r['paid_source_count'] for r in receipts),'excluded_date_families':sorted(excluded),'new_teacher_queries':0},indent=2),encoding='utf-8')
    frozen=read(ROOT/'outputs/grid63/code_freeze.json')
    for p,h in frozen.items():assert sha(ROOT/p)==h,p
    for p in list(HERE.glob('*.py'))+[OUT/'design.json',OUT/'records.json',OUT/'data_receipts.json']:
        if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
        frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    for f in source['preferred_source']['files']:frozen[(official/f['path']).relative_to(ROOT).as_posix()]=f['sha256']
    (OUT/'code_freeze.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    (OUT/'preflight.json').write_text(json.dumps({'passed':True,'records':151,'families':18,'calibration_families':4,'train_families':14,'suffix_permission_checks':permission,'new_queries':0,'new_fits':0},indent=2),encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('FROZEN:6 literature-baseline fits; no evaluation trajectories yet.\n',encoding='utf-8')
    print(json.dumps(read(OUT/'preflight.json')))
if __name__=='__main__':main()
