"""Independent scalar arithmetic and serialized-space probe reconstruction."""
import collections,copy,hashlib,json,pickle
import numpy as np
from extract import R,O,read,save,sha,av,digest,FIELDS
from structure import NAMES,compute

def scalar(c):
    def relation(v):return [int(v[i]>0 and v[j]>0 and v[i]==v[j]) for i in range(len(v)) for j in range(i+1,len(v))]
    def distance(a,b):return sum(x!=y for x,y in zip(a,b))/len(a)
    sig={i:relation(v) for i,v in c['plans'].items()};seen=c['seen_station_ids'];nxt=c['next_ids']
    b=[distance(sig[i],sig[c['incumbent_id']]) for i in nxt];s=[min(distance(sig[i],sig[j]) for j in seen) for i in nxt]
    current=relation(c['current_station_topology']);cs=[distance(sig[i],current) for i in nxt]
    live=[i for i,v in enumerate(c['line_status']) if v];rho=c['rho'];inc=[i for i in live if c['incident_lines'][i]];over=[i for i in live if rho[i]>1]
    return [min(b),sum(b)/len(b),max(b),sum(s)/len(s),max(s),sum(x>0 for x in s)/len(s),sum(cs)/len(cs),
        max([rho[i] for i in live],default=0),sum(max(rho[i]-1,0) for i in live)/len(live) if live else 0,
        max([rho[i] for i in inc],default=0),sum(i in inc for i in over)/len(over) if over else 0]

def main():
    assert not (O/'review.json').exists()
    assert read(O/'extraction_receipt.json')['passed'] and read(O/'comparison.json')['all_original41_neighbours_reproduced']
    from grid2op.Action import ActionSpace
    from grid2op.Observation import ObservationSpace,BaseObservation
    from grid2op.Environment import BaseEnv
    import grid2op
    physical=collections.Counter()
    def forbidden(*a,**kw):physical['calls']+=1;raise RuntimeError('No environment/physical call allowed')
    grid2op.make=BaseEnv.__init__=BaseEnv.reset=BaseEnv.step=BaseObservation.simulate=forbidden
    schema=R/'outputs/grid03/scoring_smoke/agent_episode';ac=ActionSpace.from_dict(read(schema/'dict_action_space.json'));oc=ObservationSpace.from_dict(read(schema/'dict_observation_space.json'))
    arrays=np.load(R/'work/grid02/ljn_nn/assets/action_N1_unsafe.npz')['action_space'];library=[ac.from_vect(v) for v in arrays]
    subs=read(R/'outputs/grid28/offline.json')['public_subids'];orpos=ac.line_or_pos_topo_vect;expos=ac.line_ex_pos_topo_vect;bounds=np.r_[0,np.cumsum(ac.sub_info)]
    probes=read(R/'outputs/grid108_loss_budget/probe_sources.json');critical=read(R/'outputs/grid108_loss_budget/critical_probes.json')
    supplemental=[R/p for p in probes.values()]+[(R/p).with_name(Path(p).stem+'_state.npz') for p in probes.values()]
    # Kept separately from the original extraction freeze; before probe arithmetic.
    save('verification_input_manifest.json',{p.relative_to(R).as_posix():sha(p) for p in supplemental+[__import__('pathlib').Path(__file__)]})
    cache={};hashchecks=0
    def context(path,visible_ids,remaining):
        nonlocal hashchecks
        if path not in cache:
            c=read(R/path)
            with np.load((R/path).with_name(Path(path).stem+'_state.npz')) as z:
                obs=oc.from_vect(z['observation']);base=ac.from_vect(z['base_action']);assert np.array_equal(z['observation'],obs.to_vect()) and np.array_equal(z['base_action'],av(base))
            assert digest(obs.to_vect())==c['state_hash'];cache[path]=(c,obs,base)
        c,obs,base=cache[path];outcomes={v['pool_id']:v for v in c['outcomes']}
        visible=[outcomes[i] for i in visible_ids];inc=max((v for v in visible if v['strict_admissible']),key=lambda v:(float(np.float32(v['reward'])),-v['pool_id']))['pool_id']
        station=subs[inc];aliases={int(k):v for k,v in c['aliases'].items()};targets={int(v) for k,v in aliases.items() if subs[k]==station}
        local=[i for i in visible_ids if i in targets];nxt=remaining[:64];assert set(nxt)<=targets
        pos=np.arange(bounds[station],bounds[station+1]);intent=(base.line_set_status>0)|base.line_change_status|(base.set_bus[orpos]>0)|(base.set_bus[expos]>0)
        mask=(~obs.line_status)&~intent;endpoints=np.r_[orpos[mask],expos[mask]];plans={}
        for i in set(local)|set(nxt):
            action=library[i].copy();v=action.set_bus.copy();v[endpoints]=0;action.set_bus=v;comb=action+base
            if i in outcomes:assert digest(av(comb))==outcomes[i]['combined_hash'];hashchecks+=1
            plan=obs.topo_vect.copy();cmd=comb.set_bus;plan[cmd!=0]=cmd[cmd!=0];toggle=comb.change_bus&(plan>0);assert not np.any(comb.change_bus&(cmd!=0));plan[toggle]=3-plan[toggle]
            off=(comb.line_set_status<0)|(comb.line_change_status&obs.line_status);on=(comb.line_set_status>0)|(comb.line_change_status&~obs.line_status)
            plan[orpos[off]]=-1;plan[expos[off]]=-1
            assert not np.any((plan[orpos[on]]<=0)) and not np.any((plan[expos[on]]<=0))
            plans[i]=plan[pos].tolist()
        return dict(plans=plans,incumbent_id=inc,seen_station_ids=local,next_ids=nxt,current_station_topology=obs.topo_vect[pos].tolist(),
            rho=obs.rho.astype(float).tolist(),line_status=obs.line_status.tolist(),incident_lines=((ac.line_or_to_subid==station)|(ac.line_ex_to_subid==station)).tolist())
    rows=read(O/'rows.json');X=np.load(O/'diagnostic_descriptors.npz')['X'];ix={(r['sample_id'],r['position']):i for i,r in enumerate(rows)}
    meta={r['sample_id']:r for r in read(R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples']}
    cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())}
    flips=read(R/'outputs/grid109_input_coverage/positive_flip_details.json');chosen={(r['sample_id'],r['position']) for r in flips}
    first={}
    for r in rows:first.setdefault(r['date'],(r['sample_id'],r['position']))
    chosen.update(first.values());audit=[]
    for sid,j in sorted(chosen):
        m=meta[sid];c=read(R/m['source_call']);n=m['visible_count'];truth={v['pool_id']:v for v in c['outcomes']};vis=c['order'][:n]
        inc=max((truth[i] for i in vis if truth[i]['strict_admissible']),key=lambda v:(float(np.float32(v['reward'])),-v['pool_id']))['pool_id']
        station=subs[inc];aliases={int(k):v for k,v in c['aliases'].items()};target={int(v) for k,v in aliases.items() if subs[k]==station};remaining=[i for i in c['order'][n:] if i in target]
        vis=vis+remaining[:j*64];remaining=remaining[j*64:]
        ctx=context(m['source_call'],vis,remaining);a=np.asarray(scalar(ctx));assert np.allclose(a,X[ix[sid,j]],rtol=0,atol=1e-14)
        audit.append(dict(sample_id=sid,position=j,independent_scalar_values=a.tolist(),matches=True))
    save('independent_arithmetic.json',audit)
    probe_rows=[];Z=[]
    for p in critical:
        c=read(R/p['source']);d=c['learned_decisions'][0];visible=[v['pool_id'] for v in c['outcomes'][:d['paid_queries']]]
        assert visible==c['order'][:d['paid_queries']]
        ctx=context(p['source'],visible,d['next_ids']);a=compute(ctx);b=np.asarray(scalar(ctx));assert np.allclose(a,b,rtol=0,atol=1e-14)
        Z.append(a);probe_rows.append(dict(context=p['context'],source=p['source'],state_hash=c['state_hash'],descriptor_values=dict(zip(NAMES,a.tolist())),normalized_target=p['actual_next_normalized_net'],independent_scalar_matches=True))
    save('critical_structure.json',probe_rows);np.savez_compressed(O/'critical_descriptors.npz',X=np.stack(Z))
    # All856 controls, all3 metrics: independent squared-distance formula with per-call minimum.
    C=list(cases.values());raw=np.concatenate([c['states'] for c in C]);Y=np.concatenate([c['gains']-read(R/'outputs/grid102_search_rl/design.json')['cost_per_query']*c['queries'] for c in C]);norm=np.load(R/'outputs/grid102_search_rl/normalization.npz')
    A=np.clip((raw-norm['mean'])/norm['scale'],-20,20).astype(np.float32).astype(float);nn=np.load(O/'diagnostic_normalization.npz');B=np.clip((X-nn['mean'])/nn['scale'],-20,20).astype(np.float32).astype(float)
    matrices={'original41':A,'structure11':B,'combined52':np.c_[A,B]};train=np.array([i for i,r in enumerate(rows) if r['partition']=='train']);gg=collections.defaultdict(list)
    for t,i in enumerate(train):gg[rows[i]['physical_state_group']].append(t)
    groups=[np.array(gg[g]) for g in sorted(gg)];checks=0
    for r in read(O/'validation_comparison.json'):
        i=ix[r['sample_id'],r['position']]
        for metric,Zmat in matrices.items():
            dz=Zmat[train]-Zmat[i];distance=np.sqrt(np.einsum('ij,ij->i',dz,dz)/Zmat.shape[1]);rep=np.array([g[np.argmin(distance[g])] for g in groups]);rep=rep[np.argsort(distance[rep],kind='stable')]
            for k in [1,5,20]:
                assert r['neighbours'][metric]['k'][str(k)]['positive_calls']==sum(Y[train[j]]>0 for j in rep[:k]);checks+=1
    scale=read(R/'outputs/grid102_search_rl/design.json')['gain_scale'];critical_old=read(R/'outputs/grid109_input_coverage/critical_coverage.json')
    for row,z,previous in zip(probe_rows,np.stack(Z) if False else np.load(O/'critical_descriptors.npz')['X'],critical_old):
        # Reuse recorded raw41 input, never new model inference.
        source=read(R/row['source'])['learned_decisions'][0];a=np.array(source['normalized_features'],dtype=np.float32).astype(float);b=np.clip((z-nn['mean'])/nn['scale'],-20,20).astype(np.float32).astype(float)
        row['neighbour_positive_counts']={}
        for metric,q in [('original41',a),('structure11',b),('combined52',np.r_[a,b])]:
            mat=matrices[metric][train];dist=np.linalg.norm(mat-q,axis=1)/np.sqrt(len(q));rep=np.array([g[np.argmin(dist[g])] for g in groups]);rep=rep[np.argsort(dist[rep],kind='stable')]
            row['neighbour_positive_counts'][metric]={str(k):int(sum(Y[train[j]]>0 for j in rep[:k])) for k in [1,5,20]}
            if metric=='original41':assert row['neighbour_positive_counts'][metric]['5']==previous['neighbours']['standardized41']['k']['5']['positive_calls']
    save('critical_structure.json',probe_rows)
    for n in ['input_manifest.json','verification_input_manifest.json']:
        assert all(sha(R/p)==h for p,h in read(O/n).items())
    base=read(R/'outputs/grid100_data_foundation/design.json');tests=[r for r in base['scenario_plan'] if r['partition']=='test'];assert len(tests)==8
    assert all(not (R/base['environment']/'chronics'/r['scenario']).exists() and not (R/'outputs/grid100_data_foundation/datasets'/r['scenario']).exists() for r in tests)
    assert physical['calls']==0
    save('review.json',dict(passed=True,independent_arithmetic_contexts=len(audit)+3,independent_values=(len(audit)+3)*11,
        all_validation_neighbour_checks=checks,combined_command_hashes_in_verification=hashchecks,
        sources_unchanged=True,final8_tests_ungenerated=True,environment_constructions=0,new_forecasts=0,new_fits=0,new_predictions=0,
        caveat='Action intentions, not solved future topology/flows; neighbourhood comparisons are descriptive and posthoc.'))
    print(json.dumps(probe_rows,ensure_ascii=False,indent=2));print(read(O/'review.json'))

if __name__=='__main__':
    from pathlib import Path
    main()
