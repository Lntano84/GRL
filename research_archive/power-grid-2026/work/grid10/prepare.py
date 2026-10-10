"""Build a bounded paired critic-objective trial from immutable GRID09 code."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'work/grid10'
OUT = ROOT / 'outputs/grid10'
assert not OUT.exists(), 'Never overwrite an existing experiment'
assert json.loads((ROOT / 'outputs/grid09/finished.json').read_text())['passed']
assert (ROOT / 'outputs/grid09/delivery_manifest.json').exists()
OUT.mkdir()
design = json.loads((ROOT / 'outputs/grid09/design.json').read_text())
design['stage'] = 'GRID10'
design['task'] = 'Paired GNN PPO critic warm-fit + raw Huber vs fixed-normalized MSE'
design['versions'] = ['raw', 'normalized']
design.pop('adaptation', None)
design['screen'] = 'Completion first; final checkpoint vs raw control and ALWAYS_RESTORE. One seed, four development weeks; no final-test use or checkpoint selection.'
probes = ROOT / 'outputs/grid08/critic_capacity_probe'
design['critic_sources'] = {}
for version, name in [('raw', 'raw_huber'), ('normalized', 'static_normalized_mse')]:
    p = probes / f'{name}_critic_only.pt'
    design['critic_sources'][version] = {'path': str(p.relative_to(ROOT)),
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'objective': name}
design['revision'] = {
    'changed': 'Both arms use equal-budget archived critic-only fits. Raw arm continues raw Huber; normalized arm continues MSE in fixed training-return units. Native reward, gamma, bootstrap and actor advantages remain in original units.',
    'unchanged': ['V2 actor weights', 'Physical-step GAE', 'Actor architecture and features',
                  'Reward and terminal penalty', 'Optimizer reset and learning rate',
                  'Rollout, PPO epochs and minibatch', 'Candidates and safety screen',
                  'Training and development weeks'],
    'interpretation': 'Test a critic-training package; normalization and loss change together, so cannot attribute separately. No adaptive PopArt, no new method novelty claim.'}
design['critic_data'] = {'samples': 532, 'path': str((probes / 'training_data.npz').relative_to(ROOT)),
    'sha256': hashlib.sha256((probes / 'training_data.npz').read_bytes()).hexdigest(),
    'scope': 'Only last V2 training pass. Changing-policy realized returns, not unbiased final-policy labels.',
    'prior_fit': {'epochs': 64, 'minibatch': 64, 'learning_rate': .0003,
                  'seed': 20261009, 'raw_wall_s': .5357119, 'normalized_wall_s': .4465251},
    'reuse': 'Previously audited diagnostic weights promoted explicitly into two new controllers; no evaluation returns used.'}
design['caps']['scope'] = 'GRID10 only: two arms with eight training + four evaluation episodes each; at most 48,408 scheduled physical steps. Separate resource ledger; CPU runtime retained.'
(OUT / 'design.json').write_text(json.dumps(design, indent=2), encoding='utf-8')

code = (ROOT / 'work/grid09/pilot.py').read_text()
code = code.replace("OUT=ROOT/'outputs/grid09'", "OUT=ROOT/'outputs/grid10'")
code = code.replace('from networks_v1 import ActorCritic', 'from networks import ActorCritic, value_loss')
code = code.replace("choices=['physical','decision']", "choices=['raw','normalized']")
code = code.replace("CFG['gamma'],CFG['gae_lambda'],args.version)", "CFG['gamma'],CFG['gae_lambda'],'physical')")
code = code.replace('vf=torch.nn.functional.smooth_l1_loss(v,ret[ids])',
                    'vf=value_loss(v,ret[ids],model.value_mu,model.value_sigma,args.version)')
old = "        model.load_state_dict(saved['state'])\n        optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])"
new = """        missing=model.load_state_dict(saved['state'],strict=False)
        assert missing.missing_keys==['value_mu','value_sigma'] and not missing.unexpected_keys
        critic_meta=D['critic_sources'][args.version];critic_path=ROOT/critic_meta['path']
        assert hashlib.sha256(critic_path.read_bytes()).hexdigest()==critic_meta['sha256']
        critic_saved=torch.load(critic_path,weights_only=False)
        model.value.load_state_dict(critic_saved['state'])
        model.value_mu.fill_(critic_saved['mu']);model.value_sigma.fill_(critic_saved['sigma'])
        assert model.value_sigma>0
        write_json(RUN/'critic_warm_start.json',{**critic_meta,'mu':float(model.value_mu),'sigma':float(model.value_sigma)})
        optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])"""
assert old in code
code = code.replace(old, new)
(WORK / 'pilot.py').write_text(code, encoding='utf-8')
(WORK / 'trace_math.py').write_text((ROOT / 'work/grid09/trace_math.py').read_text(), encoding='utf-8')
for name in ['audit.py', 'finish.py']:
    code = (ROOT / 'work/grid09' / name).read_text().replace('grid09', 'grid10')
    code = code.replace("['physical','decision']", "['raw','normalized']")
    (WORK / name).write_text(code, encoding='utf-8')
freeze = [WORK / f for f in ['prepare.py', 'pilot.py', 'networks.py', 'trace_math.py', 'preflight.py']]
freeze += [ROOT / 'work/grid08/common.py', ROOT / 'work/grid08/networks_v1.py', OUT / 'design.json']
freeze += [ROOT / design['source_checkpoint']['path'], ROOT / design['critic_data']['path']]
freeze += [ROOT / r['path'] for r in design['critic_sources'].values()]
(OUT / 'code_freeze.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in freeze}, indent=2), encoding='utf-8')
(OUT / 'RUN_STATE.md').write_text('# GRID10\n\nDESIGN FROZEN; PRECHECK REQUIRED; NOT STARTED.\n', encoding='utf-8')
print(json.dumps({'stage': 'GRID10', 'design_frozen': True, 'source_files': len(freeze)}))
