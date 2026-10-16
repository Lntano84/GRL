"""Final fixed verification vs two unchanged ICAPS2022-core adaptation seeds."""
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=module('grid82_fixed_source','work/grid76/policies.py')
B=module('grid82_conference_source','work/grid78/policies.py')
POLICIES=('FRAME_GUARD','META_CONTEXT0','META_CONTEXT1')
def run(order,observe,policy,subids,aliases,safe=.9,limit=None,context=None):
    assert policy in POLICIES
    return (C if policy=='FRAME_GUARD' else B).run(order,observe,policy,subids,aliases,safe,limit,context=context)
