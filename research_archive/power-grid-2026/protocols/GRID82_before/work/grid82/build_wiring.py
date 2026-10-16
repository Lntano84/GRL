"""Reuse qualified serial control/timing and saved-vector audit, untouched source stages."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
def main():
    for name in ['environment.py','mechanism_adapter.py','run.py']:
        text=(ROOT/'work/grid81'/name).read_text(encoding='utf-8').replace('grid81','grid82');(HERE/name).write_text(text,encoding='utf-8')
    text=(ROOT/'work/grid81/audit.py').read_text(encoding='utf-8').replace('grid81','grid82')
    text=text.replace('assert len(S)==60','assert len(S)==36').replace('runs=60','runs=36').replace('COMPLETE:60 balanced runs;','COMPLETE:36 balanced runs;')
    # FULL is only an already frozen quality reference, never an input to actions or matched timing.
    anchor="    rows=[];checks=dict(physical_rows=0"
    text=text.replace(anchor,"    full_reference=read(ROOT/'outputs/grid81/finished.json')['summaries']\n"+anchor)
    text=text.replace("full=next(v for v in S if", "full=next(v for v in full_reference if")
    text=text.replace("for baseline in ['FULL','FRAME_GUARD','RANDOM_GATE0','RANDOM_GATE1']:","for baseline in ['META_CONTEXT0','META_CONTEXT1']:")
    text=text.replace("g['method']=='CONTEXT_GATE'","g['method']=='FRAME_GUARD'").replace("r['method']=='CONTEXT_GATE'","r['method']=='FRAME_GUARD'").replace("x['method']=='CONTEXT_GATE'","x['method']=='FRAME_GUARD'")
    text=text.replace('context_quality_dates=','guard_quality_dates=').replace('context_complete_dates=','guard_complete_dates=')
    text=text.replace("scope='Six prospectively generated 2036 weeks; same grid, two timing repeats are not independent dates; random rates fixed from original calibration only.'","scope='Same six now-exposed generated scenarios, final fixed-guard vs frozen ICAPS2022-core adaptation; no independent confirmation, original calibration limitation retained; no cross-batch timing claim.'")
    # Both repeats of final guard must reproduce its previously saved GRID81 trajectory.
    anchor="        calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]"
    added="        if s['method']=='FRAME_GUARD':\n            refpath=ROOT/'outputs/grid81/runs'/f\"r0__{s['week']}__FRAME_GUARD\"/'steps.jsonl.gz'\n            with gzip.open(refpath,'rt',encoding='utf-8') as ff: prior=[json.loads(l) for l in ff]\n            assert signature==[(x['action_hash'],x['after_hash'],x['raw_cost'],x['done'],x['complete']) for x in prior]\n"
    assert anchor in text;text=text.replace(anchor,added+anchor)
    (HERE/'audit.py').write_text(text,encoding='utf-8');print('GRID82 wiring built')
if __name__=='__main__':main()
