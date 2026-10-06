"""Create a dedicated FA04 adapter without editing inherited code."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
src = (BASE / 'work/fa03b/common.py').read_text(encoding='utf8')
src = src.replace("OUT=ROOT/'outputs/fa03b'", "OUT=ROOT/'outputs/fa04'")
src = src.replace("ARMS=['LEX-PC-U-PD','RANDOM-ROW']", "ARMS=['COARSE-PC-100']")
src = src.replace('outputs/fa03b_design/FA03B_design_freeze.json', 'outputs/fa04_design/FA04_design_freeze.json')
src = src.replace("assert design['query_pair']==566 and design['query_row']==11", "assert design['row_batch']==11 and design['new_arm']=='COARSE-PC-100'")
assert not (HERE/'common.py').exists()
(HERE/'common.py').write_text(src, encoding='utf8')

src = (BASE / 'work/fa03b/run.py').read_text(encoding='utf8')
src = src.replace('Six frozen FA03B trajectories', 'Three frozen FA04 trajectories')
src = src.replace('from policy import LexSelector, lexical_indices, cheap_indices, row_batch', 'from coarse_policy import coarse_batch')
src = src.replace("cls=LexSelector if arm=='LEX-PC-U-PD' else Selector\n        selector=cls(", "selector=Selector(")
src = src.replace("arm=='RANDOM-ROW'", "arm=='COARSE-PC-100'")
start = src.index("                if arm=='LEX-PC-U-PD':")
end = src.index('                selector.selection_s+=time.perf_counter()-start', start)
src = src[:start] + '''                pool,mean_u,max_probs,missing,selected_rows,work=coarse_batch(selector,seed,round_no)
                if not len(pool):reason='candidate_pool_exhausted';phase='terminal';continue
                np.savez_compressed(dest/f'candidates_r{round_no:04d}.npz',pool=pool,
                    mean_uncertainty=mean_u,max_probabilities=max_probs,missing_models=missing,
                    selected=selected_rows,input_sha256=signature(selector))
                diag=dict(candidate_count=len(pool))
''' + src[end:]
src = src.replace('FA03B', 'FA04').replace('7200-previous', '5400-previous')
src = src.replace('len(complete)==6', 'len(complete)==3').replace('target=6', 'target=3').replace('len(complete)<6', 'len(complete)<3')
src = src.replace("args=parser.parse_args();verify_design()", "args=parser.parse_args();verify_design()\n    assert args.seed is None or args.arm is not None")
assert not (HERE/'run.py').exists()
(HERE/'run.py').write_text(src, encoding='utf8')
print('Created dedicated common.py and run.py; inherited files unchanged.')
