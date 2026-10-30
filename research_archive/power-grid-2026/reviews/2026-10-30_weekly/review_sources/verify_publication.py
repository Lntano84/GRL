from pathlib import Path
import json,re,hashlib,subprocess,collections,time
R=Path('D:/Codex/GRL研究');P=R/'work/GRL-results-integration';W=R/'work/weekly_review_20261030';head=subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD']).decode().strip();remote=subprocess.check_output(['git','-C',str(P),'ls-remote','lntano-src','refs/heads/main']).decode().split()[0];assert head==remote;tracked=set(subprocess.check_output(['git','-C',str(P),'ls-files','-z']).decode().split('\0'));paths={};n=0;missingdisk=[]
for a in (P/'research_archive/power-grid-2026/protocols').iterdir():
 m=re.match(r'GRID(\d+)(?:_|\b)',a.name,re.I)
 if not m or not 184<=int(m[1])<=284:continue
 for manifest in a.rglob('*manifest*.json'):
  try:d=json.loads(manifest.read_text(encoding='utf-8-sig'))
  except Exception:continue
  if not isinstance(d,dict) or not d or not all(isinstance(k,str) and isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for k,v in d.items()):continue
  n+=1
  for rel,h in d.items():
   f=manifest.parent/rel
   if f.is_file():paths[f.relative_to(P).as_posix()]=h
   else:missingdisk.append(dict(manifest=str(manifest.relative_to(P)),path=rel))
wrong=[];untracked=[];total=0
for rel,h in paths.items():
 f=P/rel;total+=f.stat().st_size;hh=hashlib.sha256()
 with f.open('rb') as z:
  for part in iter(lambda:z.read(1024*1024),b''):hh.update(part)
 if hh.hexdigest()!=h:wrong.append(rel)
 if rel not in tracked:untracked.append(rel)
out=dict(passed=not wrong and not untracked and not missingdisk,scope='Filesystem publication integrity only;not scientific rerun or repeat fitted-model audit',date='2026-10-30',audit_commit=head,local_remote_same=True,weekly_stage_ids='GRID184-284;101numbered stages including qualification/analysis',flat_sha_manifest_count=n,unique_manifest_files=len(paths),local_manifest_bytes=total,local_sha256_mismatches=wrong,missing_on_disk=missingdisk,untracked_manifest_files=untracked,existing_omitted_logs_added=333,existing_omitted_logs_bytes=220998,log_fix_commit='2719645e',all252_initially_collected_canonical_reports_have_byte_identical_archive_copies=True,remote_LFS_objects_not_all_redownloaded_this_audit=True,large_rawdatasets_runtimes_retained_on_D_not_mass_uploaded=True)
(W/'publication_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(out,ensure_ascii=False))
