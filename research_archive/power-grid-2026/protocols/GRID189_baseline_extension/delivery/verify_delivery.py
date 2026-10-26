"""Read-back delivery validation for preparation,not model acceptance."""
from pathlib import Path
import hashlib,subprocess,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
def main():
    C.verify();assert C.read(C.O/'complete.json')['passed'];assert not C.read(C.O/'complete.json')['gate_fitted']
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=C.REPO,text=True).split()[0];assert local==remote
    count=0;blobs=0
    for m in C.A.glob('*/archive_manifest.json'):
        for name,h in C.read(m).items():assert C.sha(m.parent/name)==h;count+=1
    for p in (C.A/'complete').iterdir():
        if p.is_file():
            rel=p.relative_to(C.REPO).as_posix();assert hashlib.sha256(subprocess.check_output(['git','show',f'{local}:{rel}'],cwd=C.REPO)).hexdigest()==C.sha(p);blobs+=1
    names=['GRID189_report.md','design.json','next_probe_spec.json','fit_support.json','review.json','audit.json','complete.json']
    for n in names:assert C.sha(C.O/n)==C.sha(C.R/'论文协作资料/FRAME_NET停止后单批追加方案_GRID189'/n)
    receipt=dict(passed=True,result_git_sha=local,remote_sha=remote,archive_files_checked=count,complete_git_blobs_checked=blobs,collaboration_files_checked=7,gate_fitted=False,new_model_forwards=0,new_physics=0,final8_absent=True)
    C.save('delivery_receipt.json',receipt);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),C.O/'delivery_receipt.json']:(d/p.name).write_bytes(p.read_bytes())
    print('GRID189_DELIVERY_VERIFIED',C.json.dumps(receipt))
if __name__=='__main__':main()
