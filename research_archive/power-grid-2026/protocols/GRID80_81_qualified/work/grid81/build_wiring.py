"""Derive new runner/audit from qualified GRID79 without changing source files."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
def main():
    text=(ROOT/'work/grid79/mechanism_adapter.py').read_text(encoding='utf-8').replace('work/grid79/policies.py','work/grid81/policies.py')
    (HERE/'mechanism_adapter.py').write_text(text,encoding='utf-8')
    text=(ROOT/'work/grid79/run.py').read_text(encoding='utf-8').replace('grid79','grid81')
    marker='    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose'
    # This marker occurs inside the original worker source, inserted with its mechanism wiring.
    anchor='        text=text.replace(marker,patch+marker)'
    assert anchor in text
    replacement="        patch+=\"    from environment import make_env as fresh_make_env\\n    import common\\n    common.make_env=fresh_make_env\\n\"\n"+anchor
    text=text.replace(anchor,replacement)
    # Fail closed on generation/schema qualification, independently of action outcomes.
    anchor="    D=read(OUT/'design.json')"
    text=text.replace(anchor,anchor+"\n    assert read(ROOT/'outputs/grid80/generation.json')['passed']\n    assert read(ROOT/'outputs/grid80/qualification.json')['passed']\n    assert read(OUT/'preflight.json')['passed']\n    for file,digest in read(OUT/'data_freeze.json').items():\n        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,file")
    (HERE/'run.py').write_text(text,encoding='utf-8')
    text=(ROOT/'work/grid79/audit.py').read_text(encoding='utf-8').replace('grid79','grid81')
    text=text.replace("assert len(S)==56","assert len(S)==60")
    text=text.replace('    from common import make_env,graph_features','    from common import graph_features\n    from environment import make_env')
    a=text.index('    sources={};allsteps=');b=text.index('    try:\n',a)
    text=text[:a]+"    rows=[];checks=dict(physical_rows=0,causal_calls=0,public_contexts=0,terminal_exceptions=0,unchanged_repeat_steps=0);repeat_reference={}\n"+text[b:]
    a=text.index("        stage,ref=sources[");b=text.index('        for j,x in enumerate(steps):',a)
    text=text[:a]+text[b:]
    text=text.replace("            assert all(x[k]==old[j][k] for k in ['action_hash','after_hash','raw_cost','done','complete']), (s['week'],s['method'],j)\n",'')
    text=text.replace("checks['unchanged_reference_steps']+=len(steps);",'')
    a=text.index('    comparisons=[]');b=text.index('    result=dict(',a)
    text=text[:a]+'''    comparisons=[]
    for baseline in ['FULL','FRAME_GUARD','RANDOM_GATE0','RANDOM_GATE1']:
        paired=[];repeat=[]
        for w in D['weeks']:
            aa=next(g for g in groups if g['method']=='CONTEXT_GATE' and g['date']==w)
            bb=next(g for g in groups if g['method']==baseline and g['date']==w)
            ar=next(r for r in rows if r['method']=='CONTEXT_GATE' and r['date']==w)
            br=next(r for r in rows if r['method']==baseline and r['date']==w)
            equal=ar['steps']==br['steps'] and ar['complete']==br['complete']
            paired.append(dict(date=w,controller_reduction=1-aa['controller_s']/bb['controller_s'],p95_reduction=1-aa['decision_p95_s']/bb['decision_p95_s'],query_reduction=1-aa['queries']/bb['queries'] if equal and bb['queries'] else None,equal_horizon=equal,context_s=aa['controller_s'],baseline_s=bb['controller_s']))
        for rep in [0,1]:repeat.append(statistics.mean(1-next(x for x in rows if x['method']=='CONTEXT_GATE' and x['date']==w and x['repeat']==rep)['controller_s']/next(x for x in rows if x['method']==baseline and x['date']==w and x['repeat']==rep)['controller_s'] for w in D['weeks']))
        macro=statistics.mean(p['controller_reduction'] for p in paired);positive=sum(p['controller_reduction']>0 for p in paired)
        q=[p['query_reduction'] for p in paired if p['query_reduction'] is not None]
        comparisons.append(dict(baseline=baseline,macro_controller_reduction=macro,pooled_controller_reduction=1-sum(p['context_s'] for p in paired)/sum(p['baseline_s'] for p in paired),positive_dates=positive,repeat_macro_reductions=repeat,macro_p95_reduction=statistics.mean(p['p95_reduction'] for p in paired),macro_query_reduction_equal_horizons=statistics.mean(q) if q else None,quality_dates=sum(g['quality_pass'] for g in groups if g['method']==baseline),stable_5percent_screen=bool(macro>=.05 and positive>=4 and all(v>0 for v in repeat)),dates=paired))
''' +text[b:]
    text=text.replace('runs=56','runs=60').replace("scope='Balanced same-host exposed-date measurement; two timing repeats are not independent dates. Source-adapted conference models retain calibration-quality limitation.'","scope='Six prospectively generated 2036 weeks; same grid, two timing repeats are not independent dates; random rates fixed from original calibration only.'")
    text=text.replace("COMPLETE:56 balanced runs;","COMPLETE:60 balanced runs;")
    # Verify trajectory identity across timing repeats without accessing another method's hidden states.
    anchor="        calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]"
    prefix="        rk=(s['week'],s['method']); signature=[(x['action_hash'],x['after_hash'],x['raw_cost'],x['done'],x['complete']) for x in steps]\n        if rk in repeat_reference:\n            assert signature==repeat_reference[rk];checks['unchanged_repeat_steps']+=len(steps)\n        else:repeat_reference[rk]=signature\n"
    assert anchor in text;text=text.replace(anchor,prefix+anchor)
    (HERE/'audit.py').write_text(text,encoding='utf-8')
    print('WIRING BUILT; run preflight and freeze before closed loop')
if __name__=='__main__':main()
