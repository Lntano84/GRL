"""Locally calibrated transparent prefix gate; only known support can stop."""
from pathlib import Path
import functools,importlib.util,json
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('grid73_features',ROOT/'work/grid72/feature_gate.py')
G=importlib.util.module_from_spec(sp);sp.loader.exec_module(G)
features=G.features;NAMES=G.NAMES
def leaf(model,x):
    t=model['trees'][0];node=0
    while t['left'][node]!=-1:
        node=t['left'][node] if x[t['feature'][node]]<=t['threshold'][node] else t['right'][node]
    return node
def decide(model,x):
    node=leaf(model,x);ood=G.out_of_support(model,x)
    eligible=node in model['stop_leaves']
    return bool(ood or not eligible),dict(leaf=node,eligible_leaf=eligible,outside_support=ood,
        probability=G.predict(model,x),support=model['leaf_support'].get(str(node)))
@functools.lru_cache(maxsize=4)
def load(name):
    assert name in ('plain','struct')
    return json.loads((ROOT/f'outputs/grid73/models/{name}.json').read_text(encoding='utf-8'))
