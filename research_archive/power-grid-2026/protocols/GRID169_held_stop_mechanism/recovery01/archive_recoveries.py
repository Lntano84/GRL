"""Preserve two engineering interruptions separately from frozen model results."""
from pathlib import Path
import hashlib,json,shutil
R=Path('D:/Codex/GRL研究');BASE=R/'work/GRL-results-integration/research_archive/power-grid-2026/protocols'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pack(dest,paths):
    dest.mkdir(parents=True,exist_ok=False)
    for path,name in paths:
        target=dest/name
        if path.is_dir():shutil.copytree(path,target)
        else:shutil.copy2(path,target)
    (dest/'archive_manifest.json').write_text(json.dumps({p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def main():
    pack(BASE/'GRID168_one_sided_transfer/recovery01',[(R/'outputs/grid168_one_sided_transfer/interrupted_attempts/audit_import_01','interrupted_attempt'),(R/'outputs/grid168_one_sided_transfer/recovery01.json','recovery01.json'),(R/'work/grid168_one_sided_transfer/evaluate_recovery01.py','evaluate_recovery01.py')])
    pack(BASE/'GRID169_held_stop_mechanism/recovery01',[(R/'outputs/grid169_held_stop_mechanism_interrupted_01','interrupted_attempt'),(Path(__file__),'archive_recoveries.py')])
    print('Both engineering interruptions and unchanged-result recoveries archived')
if __name__=='__main__':main()
