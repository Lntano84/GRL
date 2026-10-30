from pathlib import Path
import re,json
R=Path('D:/Codex/GRL研究');W=R/'work/weekly_review_20261030';text=(R/'work/GRL-results-integration/docs/EXPERIMENT_LOG.md').read_text(encoding='utf8');entries=[]
for block in re.split(r'(?m)^## ',text)[1:]:
 head,_,body=block.partition('\n');date=re.search(r'2026-\d\d-\d\d',head);ids=re.findall(r'GRID(\d+)',head,re.I)
 if ids and date:entries.append(dict(grid=int(ids[0]),date=date[0],title=head,body=body.strip()))
by={}
for r in entries:by.setdefault(r['grid'],[]).append(r)
weekly=sorted({r['grid'] for r in entries if '2026-10-26'<=r['date']<='2026-10-30'})
reports={}
for p in (R/'outputs').rglob('*.md'):
 m=re.search(r'(?i)grid(\d+)',str(p.relative_to(R/'outputs')))
 if m and ('report' in p.name.lower() or p.name.lower() in ['summary.md','result.md']):reports.setdefault(int(m[1]),[]).append(p)
rows=[]
for i in sorted(set(by)|set(reports)):
 if i>284:continue
 candidates=reports.get(i,[]);preferred=sorted(candidates,key=lambda p:(('corrected' in p.name.lower()),bool(re.fullmatch(fr'GRID0*{i}_report\.md',p.name,re.I)),not any(x in p.name.lower() for x in ['initial','original']),len(p.parts),str(p)),reverse=True)
 rows.append(dict(grid=i,weekly=i in weekly,entries=by.get(i,[]),reports=[str(p) for p in preferred],canonical_report=str(preferred[0]) if preferred else None))
(W/'inventory_raw.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('count',len(rows),'week',len(weekly),'weekids',weekly,'missing_reports',[(r['grid'],r['entries'][-1]['title'] if r['entries'] else '') for r in rows if not r['reports']])
for r in rows:
 if r['weekly'] and r['grid']<=220:
  notes=[e for e in r['entries'] if '2026-10-26'<=e['date']<='2026-10-30'];print('\nGRID',r['grid'],notes[-1]['title']);print(notes[-1]['body'][:260])
  if r['canonical_report']:
   t=Path(r['canonical_report']).read_text(encoding='utf-8-sig');print('REPORT',re.sub(r'\s+',' ',t[:180]))
