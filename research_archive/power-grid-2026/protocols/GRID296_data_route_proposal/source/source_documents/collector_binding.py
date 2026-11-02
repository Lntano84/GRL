"""Collect one fixed train/validation FULL teacher; controller unchanged."""
import argparse
import sys
from bulk_common import ROOT, OUT, RUNTIME, items, read, verify, test_sealed

ap = argparse.ArgumentParser()
ap.add_argument('--index', type=int, required=True)
args = ap.parse_args()
verify(read(OUT / 'bulk_code_freeze.json'))
test_sealed()
assert read(OUT / 'generation_trainval_finished.json')['passed']
assert read(OUT / 'pilot_collection_finished.json')['passed']
planned = items()
assert 1 <= args.index < len(planned), 'Pilot is reused, never replayed'
item = read(RUNTIME / 'design.json')['execution_order'][args.index]
assert item['week'] == planned[args.index]['scenario']
assert item['partition'] == planned[args.index]['partition']
verify(read(OUT / 'generation_receipts' / (item['week'] + '.json'))['files'])
source = ROOT / 'work/grid55/run.py'
code = source.read_text(encoding='utf8')
old_out = "OUT=ROOT/'outputs/grid55'"
assert code.count(old_out) == 1
code = code.replace(old_out, "OUT=ROOT/'outputs/grid100_data_foundation/bulk_runtime'")
marker = '    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose'
assert code.count(marker) == 1
patch = """    import importlib.util
    ss=importlib.util.spec_from_file_location('grid100_runtime_adapter',ROOT/'work/grid89_review/adapter.py')
    aa=importlib.util.module_from_spec(ss);ss.loader.exec_module(aa);aa.install(engine.Search,item)
    import common
    def learning_env():
        return common.grid2op.make(str(ROOT/'work/grid100_data_foundation/trainval_env'),test=True,backend=common.LightSimBackend(),reward_class=common.MaxRhoReward,other_rewards=common.operational_other_rewards())
    common.make_env=learning_env
"""
code = code.replace(marker, patch + marker)
sys.argv = [str(source), '--worker', str(args.index)]
exec(compile(code, str(source), 'exec'), {'__file__': str(source), '__name__': '__main__'})
