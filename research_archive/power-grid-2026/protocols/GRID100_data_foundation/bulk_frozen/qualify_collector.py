"""Replay the actual FULL adapter against stored paid feedback, no real forecast."""
from pathlib import Path
import json,sys,importlib.util,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid100_data_foundation'
sys.path.insert(0,str(ROOT/'work/grid08'))
import common
s=importlib.util.spec_from_file_location('grid100_saved_adapter',ROOT/'work/grid89_review/adapter.py');adapter=importlib.util.module_from_spec(s);s.loader.exec_module(adapter)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def main():
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];refs=read(ROOT/'outputs/grid91_mechanism/design.json')['references'];count=0;feedback=0;sources={}
    for cohort in ['old13','G1','G2']:
        ref=next(r for r in refs if r['cohort']==cohort);paths=sorted((ROOT/ref['full']).glob('call[0-9][0-9][0-9].json'))[:2]
        for path in paths:
            c=read(path);truth={r['pool_id']:r for r in c['outcomes']};seen=[]
            class Action:
                def __init__(self,i):self.i=i
                def __add__(self,base):return self.i
            class Observation:
                rho=common.np.array([c['rho_limit']])
                def simulate(self,i):
                    assert i not in seen;seen.append(i);r=truth[i]
                    return type('Future',(),dict(rho=common.np.array([r['rho']])))(),r['reward'],r['done'],dict(is_illegal=r['illegal'],is_ambiguous=r['ambiguous'],exception=r['exceptions'])
            class Search:
                rule='STATIC_ID';policy='FULL';budget=128;env=None
                def prepare(self,obs,base):return c['eligible'],c['representatives'],{int(k):v for k,v in c['aliases'].items()},{i:Action(i) for i in c['representatives']},c['masked_ids']
            Search.subids=subs;adapter.install(Search,dict(policy='FULL',order_mode='STATIC_ID'));selected,log,_,_=Search().choose(Observation(),None,0.)
            assert seen==c['order'] and log['chosen_pool_id']==c['chosen_pool_id']
            for r in log['outcomes']:
                assert all(r[k]==truth[r['pool_id']][k] for k in ['rho','reward','done','illegal','ambiguous','exceptions','strict_admissible'])
            assert (None if selected is None else selected.i)==c['chosen_pool_id'];count+=1;feedback+=len(seen);sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    receipt=dict(passed=True,adapter_saved_calls=count,feedback_reexecutions=feedback,new_physical_forecasts=0,new_physical_steps=0,sources=sources,scope='Actual existing FULL ranking/admissibility adapter with stubbed stored response API; does not certify new physical trajectories.')
    (OUT/'collector_qualification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
