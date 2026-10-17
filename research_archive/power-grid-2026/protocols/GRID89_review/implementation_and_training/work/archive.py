"""Publish appropriate new records with hashes; keep raw vectors/data local."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid89_review';REPO=ROOT/'work/GRL-results-integration'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['frozen','complete'],required=True);a=ap.parse_args();dest=REPO/'research_archive/power-grid-2026/protocols/GRID89_review'/a.phase
 if (dest/'archive_receipt.json').exists():
  r=json.loads((dest/'archive_receipt.json').read_text(encoding='utf-8'))
  for x in r['archives']:assert hashlib.sha256((REPO/x['repository_path']).read_bytes()).hexdigest()==x['sha256']
  print('Existing archive verified; no duplicate');return
 dest.mkdir(parents=True,exist_ok=True);files=[p for p in W.glob('*.py') if p.is_file()];excluded=[]
 for p in OUT.rglob('*'):
  if not p.is_file():continue
  if any(part.startswith('qa') for part in p.relative_to(OUT).parts):continue
  if p.suffix.lower() in {'.json','.md','.csv','.log','.txt','.py','.pt','.docx','.bib'} or p.name.endswith('.jsonl.gz'):files.append(p)
  else:excluded.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 files=sorted(set(files));batches=[];current=[];size=0
 for p in files:
  if current and size+p.stat().st_size>32*1024*1024:batches.append(current);current=[];size=0
  current.append(p);size+=p.stat().st_size
 if current:batches.append(current)
 archives=[];index=[]
 for no,batch in enumerate(batches,1):
  zpath=dest/f'records-{no:03d}.zip'
  with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
   for p in batch:
    data=p.read_bytes();member=p.relative_to(ROOT).as_posix();z.writestr(member,data,compress_type=zipfile.ZIP_STORED if p.suffix in ('.gz','.docx') else zipfile.ZIP_DEFLATED);index.append(dict(source=member,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),archive=zpath.relative_to(REPO).as_posix()))
  assert zpath.stat().st_size<95*1024*1024
  with zipfile.ZipFile(zpath) as z:
   assert z.testzip() is None
   for r in index:
    if r['archive']==zpath.relative_to(REPO).as_posix():assert hashlib.sha256(z.read(r['source'])).hexdigest()==r['sha256']
  archives.append(dict(repository_path=zpath.relative_to(REPO).as_posix(),bytes=zpath.stat().st_size,sha256=hashlib.sha256(zpath.read_bytes()).hexdigest()))
 for name in ['design.json','evaluation_plan.json','preflight.json','qualification.json','GRID89_report.md','decision.json','per_date.csv','summary_by_method.csv']:
  if (OUT/name).exists():shutil.copy2(OUT/name,dest/name)
 (dest/'files.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
 receipt=dict(phase=a.phase,files=len(index),archives=archives,crc_and_member_sha256_verified=True,large_vectors_kept_local=excluded,scope='All new record text, per-query JSON, per-step gzip, audits, failures, models and manuscript; raw simulator vectors and generated input tables retained locally with hashes.')
 (dest/'archive_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(dict(phase=a.phase,files=len(index),archives=len(archives)),ensure_ascii=False))
if __name__=='__main__':main()
