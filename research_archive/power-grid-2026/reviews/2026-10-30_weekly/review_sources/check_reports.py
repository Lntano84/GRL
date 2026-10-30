from pathlib import Path
import json,re,hashlib,collections
R=Path('D:/Codex/GRL研究');P=R/'work/GRL-results-integration';W=R/'work/weekly_review_20261030';rows=json.loads((W/'inventory_raw.json').read_text());allmd=collections.defaultdict(list)
for p in (P/'research_archive/power-grid-2026').rglob('*.md'):allmd[p.name.lower()].append(p)
missing=[];mapping={}
for row in rows:
 if not row['canonical_report']:continue
 p=Path(row['canonical_report']);s=hashlib.sha256(p.read_bytes()).hexdigest();matches=[v for v in allmd[p.name.lower()] if hashlib.sha256(v.read_bytes()).hexdigest()==s]
 if matches:mapping[str(row['grid'])]=dict(sha256=s,source=str(p),git_path=matches[-1].relative_to(P).as_posix())
 else:missing.append(dict(grid=row['grid'],source=str(p),sha256=s,bytes=p.stat().st_size))
(W/'report_publication_mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n',encoding='utf8');(W/'missing_report_copies.json').write_text(json.dumps(missing,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('canonical',sum(bool(r['canonical_report']) for r in rows),'archived_identical',len(mapping),'notmatched',json.dumps(missing,ensure_ascii=False))
