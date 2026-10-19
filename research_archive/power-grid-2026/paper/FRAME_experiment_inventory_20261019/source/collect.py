from pathlib import Path
import json,re,hashlib
ROOT=Path('D:/Codex/GRL研究')
OUT=ROOT/'outputs/frame_experiment_inventory_20261019'
OUT.mkdir(parents=True,exist_ok=True)
extra={11:'grid11_failure_context',71:'grid71_contribution_design',83:'grid83_alignment',85:'grid85_mechanism',89:'grid89_review'}
folders=[ROOT/'outputs'/extra.get(i,f'grid{i:02}') for i in range(90)]
folders += [ROOT/'outputs'/x for x in ['grid29b','grid29c','grid35b','grid40b','grid42b','grid55b']]
rows=[]
for folder in folders:
    id=folder.name.upper().split('_')[0]
    files=list(folder.glob('*.md'))
    reports=[p for p in files if 'report' in p.name.lower() and 'initial' not in p.name.lower()]
    ideal=[p for p in reports if p.name.upper()==id+'_REPORT.MD']
    ordered=ideal+[p for p in reports if p not in ideal]
    if not ordered:
        ordered=[p for p in files if 'review_zh' in p.name.lower()]+[p for p in files if 'review_zh' not in p.name.lower() and p.name.upper()!='RUN_STATE.MD']+[p for p in files if p.name.upper()=='RUN_STATE.MD']
    entries=[]
    for p in ordered:
        t=p.read_text(encoding='utf-8-sig',errors='replace')
        entries.append({'path':str(p),'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':t})
    jsons={}
    for name in ['decision.json','review.json','audit.json','qualification.json','summary.json','results.json','interpretation.json','closed_after_pilot.json','pilot_audit.json','offline.json','complete.json','finished.json','benchmark_finished.json','RUN_STATE.md','design.json']:
        p=folder/name
        if p.exists():
            jsons[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    rows.append({'id':id,'folder':str(folder),'reports':entries,'structured_files':jsons,'exists':folder.exists()})
rows.sort(key=lambda r:(int(re.search(r'\d+',r['id']).group()),r['id']))
(OUT/'source_inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'{len(rows)} stages; {sum(bool(r["reports"]) for r in rows)} have root Markdown')
for row in []:
    n=int(re.search(r'\d+',row['id']).group())
    if 70<=n<=88:
        print('\n'+row['id']+' '+row['folder'])
        for e in row['reports'][:1]:
            print(e['text'][:850])
            if len(e['text'])>850:print('...END...\n'+e['text'][-950:])
        if not row['reports']:print('No root Markdown',list(row['structured_files']))
