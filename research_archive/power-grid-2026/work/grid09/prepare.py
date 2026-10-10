"""Freeze a single credit-trace modification with equal-budget PPO continuation."""
import json,hashlib,sys
from pathlib import Path
from trace_math import preflight
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid09'
assert not OUT.exists(),'Refuse overwrite'
checkpoint=ROOT/'outputs/grid08/v2/train/gnn/final.pt';assert checkpoint.exists()
design=json.loads((ROOT/'outputs/grid08/v2/design.json').read_text())
design.update(stage='GRID09',task='GNN PPO decision-linked eligibility trace vs physical-step continuation',
    training_epochs=2,versions=['physical','decision'],models=['gnn'])
design['source_checkpoint']={'path':str(checkpoint.relative_to(ROOT)),
    'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'warm_start':'Identical V2 GNN weights, no bias shift; Adam reset identically in both arms.'}
design['caps']={'process_wall_s':7200,'physical_steps':50000,'output_bytes':1073741824,
    'scope':'New GRID09 round: at most 48,408 scheduled physical steps (two arms, eight training and four evaluation weeks each). Separate from completed GRID08; all diagnostic/retry costs must also be reported.'}
design['revision']={'changed':'GAE lambda links: decision arm uses lambda=1 when the next physical state offers no choice, otherwise original lambda=0.95. Physical gamma=0.995 and all observed rewards retained.',
    'unchanged':['Source model weights and initialization RNG','Optimizer, rollout length, minibatch, epochs, reward, terminal penalty','Actor masks and critic architecture','Candidate actions and one-step screen','Training/evaluation weeks and order'],
    'interpretation':'Decision-linked trace with physical rollout truncation, not a full SMAAC or semi-Markov implementation. Ordinary PPO continuation controls for extra training. No final-test selection.'}
design['screen']='Completion first, then compare final checkpoint cost against physical continuation and ALWAYS_RESTORE. No guaranteed advantage; one-seed four-development-week comparison is not independent confirmation.'
design.pop('warm_start',None);design.pop('version',None)
source=(ROOT/'work/grid08/pilot_v2.py').read_text()
source=source.replace('from common import *',"from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parents[1]/'grid08'))\nfrom common import *\nfrom trace_math import advantages_trace\nOUT=ROOT/'outputs/grid09'")
source=source.replace("parser.add_argument('--kind',choices=['gnn','mlp']);parser.add_argument('--version',default='v2')", "parser.add_argument('--kind',choices=['gnn'],default='gnn');parser.add_argument('--version',choices=['physical','decision'],required=True)")
source=source.replace("OUT/'v2/design.json'","OUT/'design.json'")
start=source.index('    adv=np.zeros(n,dtype=np.float32);carry=0.')
end=source.index('    ret=torch.from_numpy(',start)
source=source[:start]+"""    adv=advantages_trace([b['reward'] for b in buffer],values,
        [b['terminal'] for b in buffer],[bool(b['features'][2][6]>.5) for b in buffer],
        CFG['gamma'],CFG['gae_lambda'],args.version)
"""+source[end:]
old="""        saved=torch.load(OUT/'v1/train'/args.kind/'final.pt',weights_only=False)
        model.load_state_dict(saved['state'])
        with torch.no_grad():model.policy.actor.bias[1]+=D['warm_start']['bias_shifts'][args.kind]
        optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])
        write_json(RUN/'warm_start.json',D['warm_start'])"""
new="""        checkpoint=ROOT/D['source_checkpoint']['path']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==D['source_checkpoint']['sha256']
        saved=torch.load(checkpoint,weights_only=False)
        model.load_state_dict(saved['state'])
        optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])
        write_json(RUN/'warm_start.json',D['source_checkpoint'])"""
assert old in source;source=source.replace(old,new)
runner=ROOT/'work/grid09/pilot.py';assert not runner.exists();runner.write_text(source,encoding='utf-8')
OUT.mkdir();(OUT/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
(OUT/'preflight.json').write_text(json.dumps(preflight(),indent=2),encoding='utf-8')
freeze={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [runner,Path(__file__),ROOT/'work/grid09/trace_math.py',ROOT/'work/grid08/common.py',ROOT/'work/grid08/networks_v1.py',OUT/'design.json',checkpoint]}
(OUT/'code_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('# GRID09\n\nDESIGN_FROZEN; PRECHECK_PASSED; NOT_STARTED.\n\nOnly decision-linked GAE changes; both arms continue the same V2 GNN with equal schedules. No final-test use.\n',encoding='utf-8')
print(json.dumps({'passed':True,'scheduled_physical_steps_max':48408,'source_checkpoint':design['source_checkpoint'],'preflight':preflight()},indent=2))
