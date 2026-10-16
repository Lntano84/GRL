"""Use author generation API with all runtime settings recorded, never tune on control outcomes."""
from pathlib import Path
import hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid80'
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
sys.path.insert(0,str(ROOT/'work/grid80/generator_packages'))
os.environ['PATH']=str(ROOT/'work/grid80/generator_packages/pulp/solverdir/cbc/win/i64')+os.pathsep+os.environ['PATH']
def main():
    D=json.loads((OUT/'design.json').read_text(encoding='utf-8'));t=time.perf_counter()
    import grid2op
    from lightsim2grid import LightSimBackend
    import chronix2grid
    from chronix2grid.grid2op_utils import add_data
    env=grid2op.make(str(ROOT/D['environment']),test=True,backend=LightSimBackend())
    try:
        env.generate_data(nb_year=D['nb_scenario'],nb_core=1,seed=D['seed'],with_loss=True,day_lag=0)
        folders=sorted(p for p in (ROOT/D['environment']/'chronics').iterdir() if p.is_dir() and p.name.startswith('2036-'))
        assert len(folders)==6, [p.name for p in folders]
        receipt=dict(passed=True,wall_s=time.perf_counter()-t,scenarios=[p.name for p in folders],files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in folders for p in folder.iterdir() if p.is_file()})
        (OUT/'generation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    finally:env.close()
if __name__=='__main__':
    try:main()
    except BaseException:
        p=OUT/'engineering_errors';p.mkdir(exist_ok=True)
        (p/(str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8');raise
