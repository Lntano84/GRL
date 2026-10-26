"""Verify published GRID188 artifacts and saved-source matches."""
from pathlib import Path
import hashlib,subprocess,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
def main():
    C.verify();assert C.read(C.O/'complete.json')['passed'];assert C.read(C.O/'history_match_review.json')['passed'];assert C.read(C.O/'history_reconciliation.json')['passed']
    for p,h in C.read(C.O/'history_match_plan.json')['sources'].items():assert C.sha(C.R/p)==h
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=C.REPO,text=True).split()[0];assert local==remote
    checked=0;blobs=0
    for manifest in C.A.glob('*/archive_manifest.json'):
        for name,h in C.read(manifest).items():assert C.sha(manifest.parent/name)==h;checked+=1
    for phase in ['complete','history_reconciliation']:
        for p in (C.A/phase).iterdir():
            if not p.is_file():continue
            rel=p.relative_to(C.REPO).as_posix();actual=hashlib.sha256(subprocess.check_output(['git','show',f'{local}:{rel}'],cwd=C.REPO)).hexdigest();assert actual==C.sha(p);blobs+=1
    names=['GRID188_report.md','analysis.json','decision.json','review.json','native_feedback_audit.json','publication_review.json','complete.json','历史对照后的下一步.md','history_match.json','history_match_review.json','history_reconciliation.json']
    for n in names:assert C.sha(C.O/n)==C.sha(C.R/'论文协作资料/FRAME_重复改善机制_GRID188'/n)
    receipt=dict(passed=True,result_git_sha=local,remote_sha=remote,archive_files_checked=checked,complete_and_history_blobs_checked=blobs,collaboration_files_checked=len(names),old_sources_unchanged=True,new_fits=0,new_model_forwards=0,new_physics=0,final8_absent=True)
    C.save('delivery_receipt.json',receipt);dst=C.A/'delivery';dst.mkdir(exist_ok=False)
    for p in [Path(__file__),C.O/'delivery_receipt.json']:(dst/p.name).write_bytes(p.read_bytes())
    print('GRID188_DELIVERY_VERIFIED',C.json.dumps(receipt))
if __name__=='__main__':main()
