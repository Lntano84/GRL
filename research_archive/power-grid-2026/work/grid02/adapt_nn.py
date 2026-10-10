import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'work/grid01/ljn_heuristic'
SRC = ROOT / 'work/grid02/original_nn'
DST = ROOT / 'work/grid02/ljn_nn'
OUT = ROOT / 'outputs/grid02'
if DST.exists():
    raise RuntimeError('Refuse overwrite of adapted package')
shutil.copytree(BASE, DST, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ['modules/topology_nn_policy.py', 'gym_assets/action_space.py', 'gym_assets/__init__.py',
             'models/RL_training_PPO.zip', 'assets/nn_act_space/action_12_unsafe_nn.npz']:
    (DST / name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC / name, DST / name)
source = (ROOT / 'work/grid01/original_ljn/LJNAgent.py').read_text(encoding='utf-8')
selected = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'LJNAgentTopoNN')
body = ''.join(source.splitlines(True)[selected.lineno-1:selected.end_lineno])
old = (DST / 'LJNAgent.py').read_text(encoding='utf-8')
new = old + '\nfrom .modules.topology_nn_policy import TopoNNTopKModule\n\n' + body + '\n'
(DST / 'LJNAgent.py').write_text(new, encoding='utf-8')
newclass = next(n for n in ast.parse(new).body if isinstance(n, ast.ClassDef) and n.name == 'LJNAgentTopoNN')
assert ast.dump(selected) == ast.dump(newclass)
factory = (ROOT / 'work/grid01/original_ljn/make_agent.py').read_text(encoding='utf-8')
adapted = factory.replace('from .LJNagent import', 'from .LJNAgent import')
assert adapted != factory
(DST / 'make_agent.py').write_text(adapted, encoding='utf-8')
assert ast.dump(next(n for n in ast.parse(factory).body if isinstance(n, ast.FunctionDef) and n.name == 'make_agent_topoNN')) == ast.dump(next(n for n in ast.parse(adapted).body if isinstance(n, ast.FunctionDef) and n.name == 'make_agent_topoNN'))
(DST / '__init__.py').write_text('from .make_agent import make_agent_challenge, make_agent_topoNN\n', encoding='utf-8')
(OUT / 'make_agent_import_case.patch').write_text(''.join(difflib.unified_diff(factory.splitlines(True), adapted.splitlines(True), fromfile='original/make_agent.py', tofile='adapted/make_agent.py')), encoding='utf-8')
manifest = {'NN_class_AST_identical': True, 'NN_factory_AST_identical': True,
            'NN_module_and_action_space_byte_identical': True,
            'changes': 'Copy GRID01 compatibility package; append exact LJNAgentTopoNN class; add unchanged NN module/action-space/assets; fix factory LJNagent filename case.',
            'hashes': {str(p.relative_to(DST)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DST.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}}
(OUT / 'adaptation_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Adapted NN class/factory AST identical; NN module/action space unchanged.')
