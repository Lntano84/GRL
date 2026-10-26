"""Read-back validation for GRID187 archives, collaboration copies and Git blobs."""
from pathlib import Path
import hashlib,json,subprocess,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C

def main():
    C.verify_trained();assert C.read(C.O/'complete.json')['passed'];assert C.read(C.O/'interpretation_review.json')['passed']
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.REPO,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=C.REPO,text=True).split()[0];assert local==remote
    subprocess.check_call(['git','merge-base','--is-ancestor',C.read(C.O/'preheld_publication.json')['git_sha'],local],cwd=C.REPO)
    checks=0;git_checks=0;files=[]
    for manifest in C.A.glob('*/archive_manifest.json'):
        for name,h in C.read(manifest).items():
            p=manifest.parent/name;assert C.sha(p)==h,str(p);checks+=1;files.append(p)
    for phase in ['complete','interpretation']:
        for p in (C.A/phase).rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(C.REPO).as_posix();blob=subprocess.check_output(['git','show',f'{local}:{rel}'],cwd=C.REPO)
            assert hashlib.sha256(blob).hexdigest()==C.sha(p),str(p);git_checks+=1
    assert C.sha(C.W/'finalize.py')==C.sha(C.A/'publisher/finalize.py')
    for phase,names in [('complete',['GRID187_report.md','review.json','publication_audit.json','decision.json','complete.json','family_summary.json','paired_summary.json']),('interpretation',['结果解释.md','interpretation.json','interpretation_review.json'])]:
        for name in names:
            assert C.sha(C.O/name)==C.sha(C.A/phase/name)==C.sha(C.R/'论文协作资料/FRAME_动作不变目标对照_GRID187'/name)
    receipt=dict(passed=True,result_git_sha=local,remote_sha=remote,archive_manifest_files_checked=checks,complete_and_interpretation_git_blobs_checked=git_checks,collaboration_files_checked=10,publisher_unchanged=True,models_complete=12,new_physics=0,final8_absent=True)
    C.save('delivery_receipt.json',receipt)
    p=C.A/'delivery';p.mkdir(exist_ok=False)
    for name in ['verify_delivery.py']: (p/name).write_bytes((C.W/name).read_bytes())
    (p/'delivery_receipt.json').write_bytes((C.O/'delivery_receipt.json').read_bytes())
    print('GRID187_DELIVERY_VERIFIED',json.dumps(receipt))

if __name__=='__main__':main()
