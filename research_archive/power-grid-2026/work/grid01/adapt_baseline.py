"""Produce a narrow import adaptation; verify selected decision class AST unchanged."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'work/grid01/original_ljn'
DST=ROOT/'work/grid01/ljn_heuristic'
OUT=ROOT/'outputs/grid01'
DST.mkdir(exist_ok=True)
(DST/'modules').mkdir(exist_ok=True)
(DST/'assets').mkdir(exist_ok=True)
manifest=[]
for name in ['LICENSE','AUTHORS.txt','README.MD','utils.py','modules/__init__.py','modules/utils.py',
             'modules/rewards.py','modules/base_module.py','modules/topology_heuristic.py','modules/convex_optim.py',
             'assets/action_12_unsafe.npz','assets/action_N1_unsafe.npz']:
    src=SRC/name
    dst=DST/name
    if name.endswith('.py'):
        original=src.read_text(encoding='utf-8')
        adapted=original.replace('np.NaN','np.nan')
        init_alias=False
        if name=='modules/convex_optim.py':
            assert 'from lightsim2grid.gridmodel import init' in adapted
            adapted=adapted.replace('from lightsim2grid.gridmodel import init',
                'from lightsim2grid.network import init_from_pandapower as init')
            init_alias=True
            # CVXPY 1.9 represents integer Parameter values with float dtype.
            # Convert only read-only indexing expressions; never alter in-place assignments.
            for array,params in [('theta',['or','ex']),('theta_is_zero',['or','ex','load','gen','storage']),('storage',['storage'])]:
                for param in params:
                    old=f'{array}[self.bus_{param}.value]'
                    adapted=adapted.replace(old,f'{array}[_integer_index(self.bus_{param}.value)]')
            for mask in ['gen_curt','gen_redi']:
                old=f'idx_gen = self.bus_gen.value[{mask}]'
                assert old in adapted
                adapted=adapted.replace(old,f'idx_gen = _integer_index(self.bus_gen.value[{mask}])')
            helper='''\n\n# GRID01 compatibility: preserve exact integer bus IDs; reject fractional IDs.\ndef _integer_index(value):\n    array = np.asarray(value)\n    if not np.isfinite(array).all() or not np.array_equal(array, np.rint(array)):\n        raise ValueError("Non-integral bus ID in integer CVXPY Parameter")\n    return array.astype(np.int64)\n\n'''
            adapted=adapted.replace('class OptimModule(BaseModule):',helper+'class OptimModule(BaseModule):')
        dst.write_text(adapted,encoding='utf-8')
        expected=original.replace('np.NaN','np.nan')
        if init_alias:
            expected=expected.replace('from lightsim2grid.gridmodel import init',
                'from lightsim2grid.network import init_from_pandapower as init')
            # Inverse normalization proves all original expressions remain after
            # removing only the explicitly enumerated indexing conversions/helper.
            normalized=adapted.replace(helper,'')
            for array,params in [('theta',['or','ex']),('theta_is_zero',['or','ex','load','gen','storage']),('storage',['storage'])]:
                for param in params:
                    normalized=normalized.replace(f'{array}[_integer_index(self.bus_{param}.value)]',f'{array}[self.bus_{param}.value]')
            for mask in ['gen_curt','gen_redi']:
                normalized=normalized.replace(f'idx_gen = _integer_index(self.bus_gen.value[{mask}])',f'idx_gen = self.bus_gen.value[{mask}]')
        else:
            normalized=adapted
        assert ast.dump(ast.parse(normalized))==ast.dump(ast.parse(expected))
        manifest.append({'path':name,'np_NaN_replacements':original.count('np.NaN'),'lightsim_init_import_alias':init_alias,
                         'ast_equal_after_named_compatibility_changes':True})
        if adapted!=original:
            (OUT/(name.replace('/','_')+'.patch')).write_text(''.join(difflib.unified_diff(original.splitlines(True),adapted.splitlines(True),fromfile='original/'+name,tofile='adapted/'+name)),encoding='utf-8')
    else:
        shutil.copyfile(src,dst)
original=(SRC/'LJNAgent.py').read_text(encoding='utf-8')
tree=ast.parse(original)
selected=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LJNAgent')
lines=original.splitlines(True)
header=''.join(lines[:selected.lineno-1])
for removed in ['from grid2op.gym_compat import BoxGymObsSpace\n','from stable_baselines3 import PPO\n',
                'from .modules.topology_nn_policy import TopoNNTopKModule\n']:
    assert removed in header
    header=header.replace(removed,'')
adapted=header+'\n# GRID01 import adaptation: non-NN class only; class body unchanged.\n'+''.join(lines[selected.lineno-1:selected.end_lineno])
newtree=ast.parse(adapted)
newclass=next(n for n in newtree.body if isinstance(n,ast.ClassDef))
assert ast.dump(selected)==ast.dump(newclass)
(DST/'LJNAgent.py').write_text(adapted,encoding='utf-8')
# Preserve the exact author factory body, while excluding unrelated NN imports/entrypoints.
factorysource=(SRC/'make_agent.py').read_text(encoding='utf-8')
factory=next(n for n in ast.parse(factorysource).body if isinstance(n,ast.FunctionDef) and n.name=='make_agent_challenge')
factorytext='from .LJNAgent import LJNAgent\n\n'+''.join(factorysource.splitlines(True)[factory.lineno-1:factory.end_lineno])+'\n'
assert ast.dump(factory)==ast.dump(next(n for n in ast.parse(factorytext).body if isinstance(n,ast.FunctionDef)))
(DST/'make_agent.py').write_text(factorytext,encoding='utf-8')
(DST/'__init__.py').write_text('from .make_agent import make_agent_challenge\n',encoding='utf-8')
(OUT/'LJNAgent_import_only.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),adapted.splitlines(True),fromfile='original/LJNAgent.py',tofile='adapted/LJNAgent.py')),encoding='utf-8')
save={'selected_class_AST_identical':True,'factory_function_AST_identical':True,
      'changes':'Separate heuristic-only import package; np.NaN -> np.nan; LightSim init import alias; exact-integral checked casts at bus indexing reads for CVXPY1.9 float representation. No LJNAgent decision class/factory changes; OptimModule inverse-normalized AST identical.',
      'files':manifest,
      'adapted_hashes':{str(p.relative_to(DST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DST.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}}
(OUT/'adaptation_manifest.json').write_text(json.dumps(save,indent=2),encoding='utf-8')
print('Selected class and factory AST identical; adaptation manifest saved')
