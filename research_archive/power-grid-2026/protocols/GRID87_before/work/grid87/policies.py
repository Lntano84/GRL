"""Exact previously frozen fixed256 policy; no new rule."""
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('grid87_fixed_reference',ROOT/'work/grid86/policies.py')
REF=importlib.util.module_from_spec(s);s.loader.exec_module(REF)
POLICIES=('FIXED_256',)
key=REF.key
def run(order,observe,policy,subids,aliases,safe=.9,limit=None,context=None):
    assert policy in POLICIES
    return REF.run(order,observe,policy,subids,aliases,safe,limit,context)
