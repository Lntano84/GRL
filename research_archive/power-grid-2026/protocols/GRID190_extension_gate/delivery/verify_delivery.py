"""Read back all fixed training/evaluation artifacts before reporting completion."""
from pathlib import Path
import hashlib,subprocess,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
def main():
    C.trained();assert C.read(C.O/'complete.json')['passed'];assert C.read(C.O/'fits.json')['all_converged'];assert C.read(C.O/'publication_audit.json')['passed']
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=C.REPO,text=True).split()[0];assert local==remote
    subprocess.check_call(['git','merge-base','--is-ancestor',C.read(C.O/'preheld_publication.json')['git_sha'],local],cwd=C.REPO)
    checked=0;blobs=0
    for manifest in C.A.glob('*/archive_manifest.json'):
        for name,h in C.read(manifest).items():assert C.sha(manifest.parent/name)==h;checked+=1
    for phase in ['complete','motivating_cases']:
        for p in (C.A/phase).rglob('*'):
            if p.is_file():
                rel=p.relative_to(C.REPO).as_posix();assert hashlib.sha256(subprocess.check_output(['git','show',f'{local}:{rel}'],cwd=C.REPO)).hexdigest()==C.sha(p);blobs+=1
    names=['GRID190_report.md','family_summary.json','paired_summary.json','decision.json','review.json','publication_audit.json','complete.json','query_matched_random_expectation.json','motivating_cases.json','原改善与退化案例复核.md']
    for n in names:assert C.sha(C.O/n)==C.sha(C.R/'论文协作资料/FRAME_NET停止后学习追加_GRID190'/n)
    r=dict(passed=True,result_git_sha=local,remote_sha=remote,archive_files_checked=checked,complete_and_case_git_blobs_checked=blobs,collaboration_files_checked=10,gate_fits=12,all_gates_converged=True,NET_weights_unchanged=True,all12weights_before_held=True,new_physics=0,final8_absent=True)
    C.save('delivery_receipt.json',r);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),C.O/'delivery_receipt.json']:(d/p.name).write_bytes(p.read_bytes())
    print('GRID190_DELIVERY_VERIFIED',C.json.dumps(r))
if __name__=='__main__':main()
