"""Same-context conference adaptation, both model seeds retained."""
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('grid78_common_context_meta',ROOT/'work/grid77/metacontrol.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
POLICIES=('META_CONTEXT0','META_CONTEXT1')
def run(order,observe,policy,subids,aliases,safe=.9,limit=None,context=None):
    assert policy in POLICIES and safe==.9 and limit is not None and context is not None
    return M.run(order,observe,int(policy[-1]),subids,aliases,limit,context)
