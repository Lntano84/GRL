from pathlib import Path
import json,re,subprocess,collections
R=Path('D:/Codex/GRL研究');P=R/'work/GRL-results-integration';W=R/'work/weekly_review_20261030';tracked=set(subprocess.check_output(['git','-C',str(P),'ls-files','-z']).decode().split('\0'));missing=[];manifests=0;paths={}
for a in (P/'research_archive/power-grid-2026/protocols').iterdir():
 m=re.match(r'GRID(\d+)(?:_|\b)',a.name,re.I)
 if not m or not 184<=int(m[1])<=284:continue
 for manifest in a.rglob('*manifest*.json'):
  try:d=json.loads(manifest.read_text(encoding='utf-8-sig'))
  except Exception:continue
  if not isinstance(d,dict):continue
  # Only flat path->SHA256 manifests, not scientific outcome documents.
  if not d or not all(isinstance(k,str) and isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for k,v in d.items()):continue
  manifests+=1
  for rel,h in d.items():
   f=manifest.parent/rel
   if f.is_file():paths[str(f.relative_to(P)).replace('\\','/')]=h
for p,h in sorted(paths.items()):
 if p not in tracked:missing.append(dict(path=p,bytes=(P/p).stat().st_size,sha256=h))
(W/'missing_manifest_files.json').write_text(json.dumps(missing,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('weekly_manifest_count',manifests,'unique_manifest_files',len(paths),'untracked_manifest_files',len(missing),'bytes',sum(r['bytes'] for r in missing),'extensions',dict(collections.Counter(Path(r['path']).suffix for r in missing)))
print(json.dumps(missing[:12],ensure_ascii=False))
# Locate folders for all-history IDs without a direct report.
rows=json.loads((W/'inventory_raw.json').read_text());print('missing independently located reports',[(r['grid'],r['canonical_report']) for r in rows if not r['canonical_report']])
for i in [11,12,13,14,43,226]:print('folders',i,[p.name for p in (R/'outputs').iterdir() if re.match(fr'grid0*{i}(?:_|$)',p.name,re.I)])
