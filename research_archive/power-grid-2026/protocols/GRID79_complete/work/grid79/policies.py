"""Unchanged four methods in one fresh timing batch."""
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=module('grid79_context_gate','work/grid76/policies.py')
B=module('grid79_conference_modes','work/grid78/policies.py')
POLICIES=('FULL','CONTEXT_GATE','META_CONTEXT0','META_CONTEXT1')
def run(order,observe,policy,subids,aliases,safe=.9,limit=None,context=None):
    assert policy in POLICIES
    target=C if policy in ('FULL','CONTEXT_GATE') else B
    return target.run(order,observe,policy,subids,aliases,safe,limit,context=context)
