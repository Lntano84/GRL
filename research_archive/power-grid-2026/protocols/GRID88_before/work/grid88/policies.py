"""Delegate both arms to the unchanged GRID86 policy implementation."""
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('grid88_policy_reference',ROOT/'work/grid86/policies.py')
REF=importlib.util.module_from_spec(s);s.loader.exec_module(REF)
POLICIES=('FRAME_GUARD','FIXED_512')
key=REF.key
def run(order,observe,policy,subids,aliases,safe=.9,limit=None,context=None):
    assert policy in POLICIES
    return REF.run(order,observe,policy,subids,aliases,safe,limit,context)
