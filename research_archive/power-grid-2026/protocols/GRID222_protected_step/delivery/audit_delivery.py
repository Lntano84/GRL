"""Final integrity check for completed GRID218 through GRID222 stages."""
from pathlib import Path
import json,shutil,sys,subprocess
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO

def main():
    assert not (O/'series_delivery_receipt.json').exists();C.scored();stages=[('grid218_anchored_logit','GRID218_anchored_logit','FRAME_原判断偏置对照_GRID218'),('grid219_prior_objective','GRID219_prior_objective','FRAME_先验目标资格_GRID219'),('grid220_full_prior','GRID220_anchored_logit','FRAME_完整先验对照_GRID220'),('grid221_step_plan','GRID221_step_plan','FRAME_保护步长资格_GRID221'),('grid222_protected_step','GRID222_protected_step','FRAME_训练内保护步长_GRID222')];manifests=files=collab=0
    for folder,archive,collaboration in stages:
        out=R/'outputs'/folder;ar=REPO/'research_archive/power-grid-2026/protocols'/archive;assert C.read(out/'complete.json')['passed']
        for m in ar.rglob('archive_manifest.json'):
            for name,h in C.read(m).items():assert C.sha(m.parent/name)==h,(folder,name);files+=1
            manifests+=1
        for p in (R/'论文协作资料'/collaboration).iterdir():assert C.sha(p)==C.sha(out/p.name);collab+=1
        for freeze in ['trained_freeze.json','scored_freeze.json','calibrated_freeze.json']:
            if (out/freeze).exists():
                for name,h in C.read(out/freeze).items():assert C.sha(out/name)==h
    assert manifests==18
    barrierfiles=0
    for folder,archive,modelphase,freezename in [('grid218_anchored_logit','GRID218_anchored_logit','trained','trained_freeze.json'),('grid220_full_prior','GRID220_anchored_logit','trained','trained_freeze.json'),('grid222_protected_step','GRID222_protected_step','calibrated','calibrated_freeze.json')]:
        out=R/'outputs'/folder;ar=REPO/'research_archive/power-grid-2026/protocols'/archive
        for phase,publication,freeze in [(modelphase,'pre_score_publication.json',freezename),('scored','pre_join_publication.json','scored_freeze.json')]:
            rev=C.read(out/publication)['git_sha'];mf=C.read(out/freeze)
            for name,h in mf.items():
                p=ar/phase/name;assert C.sha(p)==h;assert C.git('show',rev+':'+p.relative_to(REPO).as_posix())==p.read_bytes();barrierfiles+=1
    head=C.git('rev-parse','HEAD').decode().strip();assert C.git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert not C.git('status','--porcelain').strip()
    receipt=dict(passed=True,results_publication_sha=head,stages=5,archive_manifests_verified=manifests,archived_files_verified=files,collaboration_files_verified=collab,all_model_and_prediction_barrier_commit_files_verified=barrierfiles,all_source_models_unchanged=True,report_sha256=C.sha(O/'GRID222_report.md'),handoff_sha256=C.sha(O/'CONTINUOUS_HANDOFF.md'),no_new_fits_or_predictions_during_delivery=True)
    C.save('series_delivery_receipt.json',receipt);dest=A/'delivery';dest.mkdir(exist_ok=False)
    for p in [Path(__file__),Path(__file__).with_name('audit_delivery_attempt01.py'),O/'delivery_attempt01.json',O/'series_delivery_receipt.json',O/'CONTINUOUS_HANDOFF.md']:shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in dest.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n');shutil.copy2(O/'series_delivery_receipt.json',R/'论文协作资料/FRAME_训练内保护步长_GRID222/series_delivery_receipt.json');print('GRID218_222_SERIES_DELIVERED',json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
