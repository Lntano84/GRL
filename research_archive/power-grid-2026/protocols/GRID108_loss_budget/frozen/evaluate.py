"""Reuse the qualified evaluator, replacing only model registry and output scope."""
from common108 import ROOT,OUT,read
source=ROOT/'work/grid107_direct_value/evaluate.py';code=source.read_text(encoding='utf8')
assert code.count('from common107 import ')==1;code=code.replace('from common107 import ','from common108 import ')
code=code.replace("assert read(OUT/'fits.json')['new_fits']==3","assert read(OUT/'fits.json')['new_training_paths']==6")
old="""    for seed in D['seeds']:
        models[f'DIRECT{seed}']=DQN.load(OUT/f'models/direct_seed{seed}_final.zip',device='cpu')
        models[f'DQN0_{seed}']=DQN.load(ROOT/f'outputs/grid103_learning_mechanism/models/dqn_gamma0_seed{seed}_final.zip',device='cpu')"""
new="""    for entry in D['models']:
        models[entry['method']]=DQN.load(ROOT/entry['path'],device='cpu')
    for seed in D['seeds']:
        models[f'DQN0_{seed}']=DQN.load(ROOT/f'outputs/grid103_learning_mechanism/models/dqn_gamma0_seed{seed}_final.zip',device='cpu')"""
assert code.count(old)==1;code=code.replace(old,new)
assert code.count("if r['method'].startswith('DIRECT'):continue")==1
code=code.replace("if r['method'].startswith('DIRECT'):continue","if r['method'].startswith(('HUBER_','MSE_')):continue")
code=code.replace('len(trajectories)==5900','len(trajectories)==11210').replace('trajectories=5900','trajectories=11210')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
