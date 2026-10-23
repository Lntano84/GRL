"""Qualified serialized spaces and public commands only; no grid environment."""
from common111 import R,read
import collections,hashlib,numpy as np,warnings
def digest(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def av(a):return np.concatenate([a._get_array_from_attr_name(k).astype(np.float32) for k in type(a).attr_list_vect])
def best(v):return max((r for r in v if r['strict_admissible']),key=lambda r:(float(np.float32(r['reward'])),-r['pool_id']))
def legal(r):return not(r['done'] or r['illegal'] or r['ambiguous'] or r['exceptions']) and np.isfinite(r['rho']) and r['rho']>0 and np.isfinite(r['reward'])
def signature(v):
    a=np.asarray(v);i,j=np.triu_indices(len(a),1)
    return ((a[i]>0)&(a[j]>0)&(a[i]==a[j])).astype(np.int16)
class Decoder:
    def __init__(self):
        warnings.filterwarnings('ignore',message='Numba cannot be loaded.*')
        import grid2op
        from grid2op.Environment import BaseEnv
        from grid2op.Observation import ObservationSpace,BaseObservation
        from grid2op.Action import ActionSpace
        def forbidden(*a,**kw):raise RuntimeError('GRID111 forbids environment construction/reset/step/simulate')
        grid2op.make=BaseEnv.__init__=BaseEnv.reset=BaseEnv.step=BaseObservation.simulate=forbidden
        schema=R/'outputs/grid03/scoring_smoke/agent_episode'
        self.ac=ActionSpace.from_dict(read(schema/'dict_action_space.json'));self.oc=ObservationSpace.from_dict(read(schema/'dict_observation_space.json'))
        self.subs=read(R/'outputs/grid28/offline.json')['public_subids'];self.bounds=np.r_[0,np.cumsum(self.ac.sub_info)]
        vectors=np.load(R/'work/grid02/ljn_nn/assets/action_N1_unsafe.npz')['action_space'];self.library=[self.ac.from_vect(v) for v in vectors]
        self.hashes=[digest(av(a)) for a in self.library];assert self.hashes==read(R/'outputs/grid98_feature_enrichment/template_metadata.json')['library_hashes']
        self.checks=collections.Counter()
    def decode(self,path):
        p=R/path;c=read(p);assert read(p.parent/'metadata.json')['library_hashes']==self.hashes
        with np.load(p.with_name(p.stem+'_state.npz')) as z:
            ob=self.oc.from_vect(z['observation']);base=self.ac.from_vect(z['base_action']);assert np.array_equal(ob.to_vect(),z['observation']) and np.array_equal(av(base),z['base_action'])
        assert digest(ob.to_vect())==c['state_hash'];self.checks['vector_roundtrips']+=2
        op=self.ac.line_or_pos_topo_vect;ep=self.ac.line_ex_pos_topo_vect
        intent=(base.line_set_status>0)|base.line_change_status|(base.set_bus[op]>0)|(base.set_bus[ep]>0)
        disconnected=(~ob.line_status)&~intent;maskpos=np.r_[op[disconnected],ep[disconnected]]
        eligible=[i for i,s in enumerate(self.subs) if ob.time_before_cooldown_sub[s]==0]
        reps=[];aliases={};first={};masked=[];plans={};unknown={};truth={r['pool_id']:r for r in c['outcomes']}
        for i in eligible:
            a=self.library[i].copy();bus=a.set_bus.copy()
            if np.any(bus[maskpos]!=0):bus[maskpos]=0;a.set_bus=bus;masked.append(i)
            comb=a+base;key=av(comb).tobytes()
            if key not in first:
                first[key]=i;reps.append(i)
                if i in truth:assert digest(av(comb))==truth[i]['combined_hash'];self.checks['combined_hashes']+=1
                plan=ob.topo_vect.copy();cmd=comb.set_bus;plan[cmd!=0]=cmd[cmd!=0];sw=comb.change_bus
                assert not np.any(sw&(cmd!=0));live=sw&(plan>0);plan[live]=3-plan[live]
                off=(comb.line_set_status<0)|(comb.line_change_status&ob.line_status);on=(comb.line_set_status>0)|(comb.line_change_status&~ob.line_status)
                plan[op[off]]=-1;plan[ep[off]]=-1
                unknown[i]=set(np.r_[op[on&(plan[op]<=0)],ep[on&(plan[ep]<=0)]].tolist());plans[i]=plan
            aliases[i]=first[key]
        assert eligible==c['eligible'] and reps==c['representatives'] and aliases=={int(k):v for k,v in c['aliases'].items()} and masked==c['masked_ids'];self.checks['public_preparations']+=1
        return c,ob,plans,unknown,aliases
    def context(self,decoded,visible,nextids,station):
        c,ob,plans,unknown,aliases=decoded;target={int(v) for k,v in aliases.items() if self.subs[k]==station}
        local=[r for r in visible if r['pool_id'] in target];inc=best(visible);assert inc['pool_id'] in target
        ids=[r['pool_id'] for r in local]+nextids;pos=np.arange(self.bounds[station],self.bounds[station+1]);assert not any(unknown[i]&set(pos.tolist()) for i in ids)
        return dict(seen_ids=[r['pool_id'] for r in local],seen_plans=[plans[r['pool_id']][pos].tolist() for r in local],
            seen_feedback=[{k:r[k] for k in ['rho','reward','done','illegal','ambiguous','exceptions','strict_admissible']} for r in local],
            next_ids=list(nextids),next_plans=[plans[i][pos].tolist() for i in nextids],incumbent_id=inc['pool_id'],incumbent_reward=float(np.float32(inc['reward'])),
            incumbent_plan=plans[inc['pool_id']][pos].tolist())
