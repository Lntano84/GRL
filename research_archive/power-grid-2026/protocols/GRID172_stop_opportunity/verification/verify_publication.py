"""Cross-check new reports, collaboration copies, archived bytes and all36weights."""
from pathlib import Path
import datetime,hashlib,json,shutil
R=Path('D:/Codex/GRL研究');REPO=R/'work/GRL-results-integration';BASE=REPO/'research_archive/power-grid-2026/protocols'
TOPICS=['grid168_one_sided_transfer','grid169_held_stop_mechanism','grid170_remaining_date_folds','grid171_stop_execution_cost','grid172_stop_opportunity']
ARCH=['GRID168_one_sided_transfer','GRID169_held_stop_mechanism','GRID170_remaining_date_folds','GRID171_stop_execution_cost','GRID172_stop_opportunity']
COLLAB=['FRAME_单向停止日期迁移_GRID168','FRAME_留出停止机制_GRID169','FRAME_四折单向停止迁移_GRID170','FRAME_单向停止执行开销_GRID171','FRAME_无损停止研究空间_GRID172']
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    checks=[];archive_files=0;models=0
    for topic,arch,collab in zip(TOPICS,ARCH,COLLAB):
        output=R/'outputs'/topic;assert read(output/'complete.json')['passed_audit'];assert read(output/'review.json')['passed']
        for mf in (BASE/arch).glob('*/archive_manifest.json'):
            for name,h in read(mf).items():assert sha(mf.parent/name)==h,(mf,name);archive_files+=1
        for mf in ['source_hashes.json','code_freeze.json']:
            if (output/mf).exists():
                for name,h in read(output/mf).items():assert sha(R/name)==h,(mf,name)
        folder=R/'论文协作资料'/collab;count=0
        for p in folder.iterdir():
            if p.is_file():assert sha(p)==sha(output/p.name),p;count+=1
        checks.append(dict(topic=topic,collaboration_copies_checked=count,report_sha256=sha(output/(arch.split('_')[0]+'_report.md'))))
    for parent in [R/'outputs/grid168_one_sided_transfer',*[R/f'outputs/grid170_remaining_date_folds/fold{f}' for f in [1,2,3]]]:
        for rec in read(parent/'fits.json')['records']:
            path=parent/'models'/f"{rec['arm']}_seed{rec['seed']}.pt";assert sha(path)==rec['checkpoint_sha256'];assert rec['held_draws']==0;models+=1
    assert models==36
    result=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),topics=checks,archived_files_checked=archive_files,model_checkpoints_checked=models,interpretation='Artifact/hash verification; not another experiment or physical solver rerun.')
    output=R/'outputs/grid172_stop_opportunity';(output/'publication_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    dest=BASE/'GRID172_stop_opportunity/verification';dest.mkdir(exist_ok=False);shutil.copy2(Path(__file__),dest/'verify_publication.py');shutil.copy2(output/'publication_verification.json',dest/'publication_verification.json')
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
    print('ALL_FIVE_STAGES_VERIFIED',models,archive_files)
if __name__=='__main__':main()
