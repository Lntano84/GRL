"""Hand cases plus immutable old-policy reproduction; no fitting or physics."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import policies as P
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    checks=[]
    def run(n=600, reward=None, valid=True, subs=None, aliases=None):
        seen=[]
        def observe(i,phase):
            assert i not in seen;seen.append(i)
            return dict(pool_id=i,reward=1.0 if reward is None else reward(i),rho=.5,
                        strict_admissible=valid,phase=phase)
        result=P.run(list(range(n)),observe,'FRAME_VERIFY',subs or list(range(n)),aliases or {i:i for i in range(n)})
        assert seen==[x['pool_id'] for x in result['outcomes']]
        return result
    r=run();assert r['public_queries']==256 and r['chosen_pool_id']==0;checks.append('Safe prefix cannot return at128')
    r=run(reward=lambda i:2. if i==192 else 1.);assert r['public_queries']==384 and r['chosen_pool_id']==192;checks.append('Improvement resets both stale verification batches')
    r=run(valid=False);assert r['public_queries']==600 and r['chosen_pool_id'] is None;checks.append('No admissible action exhausts')
    assert run(n=0)['public_queries']==0 and run(n=12)['public_queries']==12;checks.append('Empty and short pools')
    subs=list(range(600));subs[599]=subs[0]
    r=run(reward=lambda i:2. if i==599 else 1.,subs=subs)
    assert r['public_queries']==257 and r['chosen_pool_id']==599 and r['local_queries']==1;checks.append('Post-verification completion finds same-station late action')
    aliases={i:i for i in range(600)};aliases[600]=599;subs=list(range(601));subs[600]=subs[0]
    r=run(reward=lambda i:2. if i==599 else 1.,subs=subs,aliases=aliases)
    assert r['chosen_pool_id']==599;checks.append('Original alias membership retained')
    a=run(reward=lambda i:100. if i==500 else 1.)
    b=run(reward=lambda i:-100. if i==500 else 1.)
    assert a==b;checks.append('Hidden unrequested suffix changes no visible decision')
    assert a['chosen_pool_id']==0;checks.append('Plateau still deliberately misses remote late optimum; no safety certificate claimed')
    public_subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];reproduced=0
    for stage in [65,68]:
        for folder in sorted((ROOT/f'outputs/grid{stage}/runs').glob('r0__*')):
            s=read(folder/'summary.json')
            if s['policy'] not in ('FRAME','FULL','GLOBAL_MATCH'):continue
            for cp in sorted(folder.glob('call[0-9][0-9][0-9].json')):
                c=read(cp);truth={r['pool_id']:r for r in c['outcomes']};seen=[]
                def observe(i,phase):
                    seen.append(i);assert truth[i]['phase']==phase;return truth[i]
                r=P.run(c['order'],observe,s['policy'],public_subs,{int(k):int(v) for k,v in c['aliases'].items()},limit=c['rho_limit'])
                assert all(r[k]==c[k] for k in ['chosen_pool_id','public_queries','stop_reason','completion_budget','extra_selected_ids'])
                assert seen==[x['pool_id'] for x in c['outcomes']];reproduced+=1
    out=dict(passed=True,checks=checks,old_calls_reproduced=reproduced,new_fits=0,new_physical_steps=0)
    target=ROOT/'outputs/grid69';target.mkdir(parents=True,exist_ok=True)
    (target/'preflight.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out))
if __name__=='__main__':main()
