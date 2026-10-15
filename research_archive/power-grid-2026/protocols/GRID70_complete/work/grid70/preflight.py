"""Minimal guard qualification, including exact preservation of old fallback."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));import policies as P
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    checks=[]
    def run(n=600,reward=None,rho=.5,valid=True,subs=None,aliases=None,policy='FRAME_GUARD'):
        seen=[]
        def observe(i,phase):
            assert i not in seen;seen.append(i)
            return dict(pool_id=i,reward=1. if reward is None else reward(i),rho=rho,
                        strict_admissible=valid,phase=phase)
        result=P.run(list(range(n)),observe,policy,subs or list(range(n)),aliases or {i:i for i in range(n)})
        assert seen==[r['pool_id'] for r in result['outcomes']]
        return result
    r=run();assert r['public_queries']==256 and r['chosen_pool_id']==0 and r['verification_guard'];checks.append('Safe prefix verified through256')
    subs=list(range(600));subs[599]=subs[0]
    r=run(reward=lambda i:2. if i==599 else 1.,subs=subs)
    assert r['public_queries']==256 and r['local_queries']==0 and r['chosen_pool_id']==0;checks.append('Unchanged safe incumbent does not activate new local branch')
    subs=list(range(601));subs[192]=subs[0];subs[600]=subs[0];aliases={i:i for i in range(600)};aliases[600]=599
    r=run(reward=lambda i:3. if i==599 else 2. if i==192 else 1.,subs=subs,aliases=aliases)
    assert r['public_queries']==385 and r['chosen_pool_id']==599 and r['local_queries']==1;checks.append('Discovered incumbent receives alias-preserving completion')
    for kwargs in [dict(rho=1.),dict(rho=1.,reward=lambda i:2. if i==599 else 1.),dict(valid=False)]:
        new=run(**kwargs);old=run(**kwargs,policy='FRAME')
        assert not new.pop('verification_guard') and not new.pop('prefix_safe_at_entry') and new==old
    checks.append('Unsafe/invalid prefix exactly reproduces original fallback and no double dispatch')
    assert run(n=0)['public_queries']==0 and run(n=12)['public_queries']==12;checks.append('Empty and short pools')
    a=run(reward=lambda i:100. if i==500 else 1.)
    b=run(reward=lambda i:-100. if i==500 else 1.)
    assert a==b and a['chosen_pool_id']==0;checks.append('Hidden suffix perturbation has no online effect; no long-term guarantee claimed')
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
    target=ROOT/'outputs/grid70';target.mkdir(parents=True,exist_ok=True)
    (target/'preflight.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out))
if __name__=='__main__':main()
