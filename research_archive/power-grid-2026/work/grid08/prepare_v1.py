"""Create a distinct, preserved runner. Never alter live V0 source/results."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'work/grid08';OUT=ROOT/'outputs/grid08'
target=WORK/'pilot_v1.py'
assert not target.exists(),'Refuse to overwrite an existing V1 runner'
source=(WORK/'pilot.py').read_text(encoding='utf-8')
source=source.replace('from common import *','from common import *\nfrom networks_v1 import ActorCritic',1)
source=source.replace("D=json.loads((OUT/'design.json').read_text())","D=json.loads((OUT/'v1/design.json').read_text())",1)
source=source.replace("default='v0'","default='v1'",1)
start=source.index('    advantages=torch.from_numpy(')
end=source.index('    old_lp=',start)
source=source[:start]+'''    active=torch.tensor([bool(b['features'][2][6]>.5) for b in buffer],dtype=torch.bool)
    advantages=torch.zeros(n)
    if active.any():
        a=torch.from_numpy(adv)[active]
        advantages[active]=(a-a.mean())/(a.std(unbiased=False)+1e-8)
'''+source[end:]
source=source.replace("pg=-torch.minimum(ratio*advantages[ids],ratio.clamp(1-CFG['clip'],1+CFG['clip'])*advantages[ids]).mean()",'''local=active[ids]
            if local.any():
                pg=-torch.minimum(ratio[local]*advantages[ids][local],ratio[local].clamp(1-CFG['clip'],1+CFG['clip'])*advantages[ids][local]).mean()
                ent=dist.entropy()[local].mean()
            else:
                pg=logits.sum()*0.;ent=logits.sum()*0.''',1)
source=source.replace("vf=torch.nn.functional.smooth_l1_loss(v,ret[ids]);ent=dist.entropy().mean()","vf=torch.nn.functional.smooth_l1_loss(v,ret[ids])",1)
source=source.replace("torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()","torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5);torch.nn.utils.clip_grad_norm_(model.value.parameters(),.5);optimizer.step()",1)
assert 'from networks_v1 import ActorCritic' in source and 'local=active[ids]' in source
assert "D=json.loads((OUT/'v1/design.json').read_text())" in source
target.write_text(source,encoding='utf-8')
(OUT/'v1').mkdir(exist_ok=True)
D=json.loads((OUT/'design.json').read_text())
D['version']='v1'
D['revision']={'basis':'Training-only V0 diagnostics: offered fraction about 7%; deterministic actor increasingly constant.',
    'changes':['Actor advantages normalized only over offered decisions; actor/entropy losses evaluated only there.',
        'Independent, common pooled-state critic; no critic gradient to actor encoder.',
        'Critic ignores proposal/forecast fields, making bootstrap input consistent.',
        'Policy and critic gradients clipped separately.'],
    'unchanged':['Original controller and QP proposals','Legal information and screens','Reward and safety penalty',
        'Training/evaluation weeks','Fixed training schedule and PPO settings'],
    'evaluation_scope':'V0 validation outcomes may be observed during development; V1 is development, not independent confirmation. Final test remains sealed.'}
D['caps']['physical_steps']=150000
D['caps']['scope']='Cumulative V0 + V1 authorization budget; verify actual steps and bytes in final audit. Process wall cap remains enforced.'
(OUT/'v1/design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=[WORK/'pilot.py',WORK/'common.py',target,WORK/'networks_v1.py',OUT/'v1/design.json']
(OUT/'v1/code_freeze.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
print('Created distinct V1 runner and pre-evaluation design. No fit or simulation executed.')
