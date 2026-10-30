from pathlib import Path
import json,re,hashlib,subprocess
R=Path('D:/Codex/GRL研究');P=R/'work/GRL-results-integration';W=R/'work/weekly_review_20261030';missing=json.loads((W/'missing_manifest_files.json').read_text(encoding='utf8'));assert len(missing)==333
for r in missing:
 p=P/r['path'];assert r['path'].startswith('research_archive/power-grid-2026/protocols/GRID') and p.suffix=='.log' and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];s=p.read_text(encoding='utf-8-sig',errors='replace');assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})',s),p
for i in range(0,len(missing),40):subprocess.run(['git','-C',str(P),'add','-f','--',*[r['path'] for r in missing[i:i+40]]],check=True)
allowed={r['path'] for r in missing}
for line in subprocess.check_output(['git','-C',str(P),'status','--porcelain=v1','--untracked-files=all']).decode().splitlines():assert line[3:].strip('"') in allowed,line
print('EXACT_MANIFEST_LOGS_STAGED',len(missing),sum(r['bytes'] for r in missing))
