"""Final publication integrity for GRID215 and GRID216, with no model fitting."""
from pathlib import Path
import json,shutil,subprocess,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,A,R=C.O,C.A,C.R

def main():
    assert not (O/'delivery_receipt.json').exists();C.scored();assert C.read(O/'complete.json')['passed'] and C.read(O/'independent_audit.json')['passed'];assert C.read(C.D/'independent_audit.json')['passed'];count=0
    archive215=C.REPO/'research_archive/power-grid-2026/protocols/GRID215_cancellation_domain';dirs=[archive215,*[p for p in A.iterdir() if p.is_dir()]];assert len(dirs)==6
    for d in dirs:
        for p,h in C.read(d/'archive_manifest.json').items():assert C.sha(d/p)==h,(str(d),p);count+=1
    pub=C.read(O/'pre_score_publication.json');join=C.read(O/'pre_join_publication.json');verified=0
    for folder,phase,rev in [('models','trained',pub['git_sha']),('held_predictions','scored',join['git_sha'])]:
        for p in (O/folder).iterdir():
            a=A/phase/folder/p.name;assert C.sha(p)==C.sha(a);assert C.git('show',rev+':'+a.relative_to(C.REPO).as_posix())==p.read_bytes();verified+=1
    assert verified==24
    for stage,dest in [(C.D,R/'论文协作资料/FRAME_取消学习范围_GRID215'),(O,R/'论文协作资料/FRAME_取消分类器_GRID216')]:
        for p in dest.iterdir():assert C.sha(p)==C.sha(stage/p.name)
    for p,h in C.read(C.D/'target_freeze.json').items():assert C.sha(C.D/p)==h and C.sha(C.D/p)==C.sha(archive215/p)
    for p in [O/'GRID216_report.md',O/'complete.json',O/'independent_audit.json']:
        ar=A/'audited'/p.name;assert C.sha(p)==C.sha(ar);assert C.git('show','HEAD:'+ar.relative_to(C.REPO).as_posix())==p.read_bytes()
    head=C.git('rev-parse','HEAD').decode().strip();assert C.git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert C.git('status','--porcelain').strip()==b''
    receipt=dict(passed=True,result_publication_sha=head,verified_archive_files=count,all12_model_files_and12_scorefiles_match_barrier_git_commits=True,all12_GRID215_targets_match_original_and_archive=True,both_D_collaboration_folders_verified=True,source_and_trained_and_scored_hashes_unchanged=True,report_sha256=C.sha(O/'GRID216_report.md'),retain_GRID208=True,new_training_in_delivery=0)
    C.save('delivery_receipt.json',receipt);shutil.copy2(O/'delivery_receipt.json',R/'论文协作资料/FRAME_取消分类器_GRID216/delivery_receipt.json');C.archive('delivery');print('GRID216_DELIVERY_VERIFIED',json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
