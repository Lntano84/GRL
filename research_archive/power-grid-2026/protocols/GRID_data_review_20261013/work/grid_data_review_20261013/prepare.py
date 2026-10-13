"""Export frozen research data for user review, without reruns or fitting."""
import csv, hashlib, json, math, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/grid_data_review_20261013'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
def writecsv(name, records):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, list(records[0])); w.writeheader(); w.writerows(records)
def main():
    assert not OUT.exists(); OUT.mkdir()
    provenance=read(ROOT/'outputs/grid55b/qualified_view/provenance.json')
    quality=csvrows(ROOT/'outputs/grid55b/qualified_view/quality.csv')
    quality_ix={(r['repeat'],r['week'],r['method']):r for r in quality}
    formal=[]
    for p in provenance:
        rep,week,method=p['logical_slot'].split('|')
        path=ROOT/p['source_folder']/'summary.json'
        assert sha(path)==p['summary_sha256']
        s=read(path); q=quality_ix[(rep,week,method)]
        formal.append({'logical_repeat':int(rep),'physical_repeat':p['physical_repeat'],'week':week,'method':method,
            'complete':int(s['complete']),'steps':s['steps'],'cost':s['cost'],'control_s':s['controller_s'],
            'n1_queries':s['n1_forecasts'],'no_earlier_failure':int(q['no_earlier_failure']=='True'),
            'quality_pass':int(q['quality_qualified']=='True'),'n1_source_s':s['n1_source_s'],
            'source_folder':p['source_folder'],'source_sha256':p['summary_sha256']})
    assert len(formal)==128 and all(r['complete']==1 for r in formal)
    ix={(r['logical_repeat'],r['week'],r['method']):r for r in formal}
    summaries=read(ROOT/'outputs/grid55b/qualified_view/review.json')['summaries']
    for s in summaries:
        rows=[r for r in formal if r['method']==s['method']]
        ratio=math.fsum(1-r['control_s']/ix[(r['logical_repeat'],r['week'],'FULL')]['control_s'] for r in rows)/len(rows)
        if s['method']!='FULL': assert abs(ratio-s['macro_time_reduction_full'])<1e-12
        assert abs(math.fsum(r['control_s'] for r in rows)-s['control_s'])<1e-7
        assert sum(r['n1_queries'] for r in rows)==s['queries']
    pilot=read(ROOT/'outputs/grid59/finished.json')['summaries']
    assert len(pilot)==16
    static=csvrows(ROOT/'outputs/grid57/per_family.csv')
    diag=csvrows(ROOT/'outputs/grid59_closeout/paired_diagnostics.csv')
    payload={'formal':formal,'formal_summary':summaries,'pilot':pilot,'static':static,'diagnostics':diag}
    (OUT/'data.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False),encoding='utf-8')
    writecsv('GRID55B_128_runs.csv',formal)
    fields=['week','method','steps','complete','cost','controller_s','n1_forecasts','n1_source_s','prior_s','feature_s','forward_rank_s','public_forecasts']
    writecsv('GRID59_16_runs.csv',[{k:s[k] for k in fields} for s in pilot])
    originals=[
        'outputs/grid56/main_table.csv',
        'outputs/grid55b/qualified_view/review.json',
        'outputs/grid55b/qualified_view/quality.csv',
        'outputs/grid55b/qualified_view/family_comparisons.csv',
        'outputs/grid55b/qualified_view/provenance.json',
        'outputs/grid57/per_family.csv','outputs/grid57_prior_review/review.json',
        'outputs/grid59/review.json','outputs/grid59/per_family.csv',
        'outputs/grid59/design.json','outputs/grid59/finished.json',
        'outputs/grid59_closeout/paired_diagnostics.csv',
        'outputs/grid59_closeout/claim_evidence_current.json',
    ]
    receipts=[]
    for rel in originals:
        src=ROOT/rel; dest=OUT/'originals'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        assert sha(src)==sha(dest)
        receipts.append({'source':rel,'sha256':sha(src),'bytes':src.stat().st_size})
    (OUT/'sources.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    print(json.dumps({'formal_runs':len(formal),'pilot_runs':len(pilot),'source_files':len(receipts),'passed':True}))
if __name__=='__main__':main()
