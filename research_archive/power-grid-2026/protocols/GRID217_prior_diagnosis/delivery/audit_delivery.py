"""Verify saved GRID217 report, archive and collaboration publication."""
from pathlib import Path
import json,shutil,subprocess,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,A,R,REPO=C.O,C.A,C.R,C.REPO

def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args])
def main():
    assert not (O/'delivery_receipt.json').exists();C.verify();assert C.read(O/'independent_audit.json')['passed'] and C.read(O/'complete.json')['passed'];manifest=C.read(A/'archive_manifest.json')
    for name,h in manifest.items():
        p=A/name;assert C.sha(p)==h;assert git('show','HEAD:'+p.relative_to(REPO).as_posix())==p.read_bytes()
    dest=R/'论文协作资料/FRAME_判断继承诊断_GRID217';n=0
    for p in dest.iterdir():assert C.sha(p)==C.sha(O/p.name)==C.sha(A/p.name);n+=1
    assert n==8;head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('status','--porcelain').strip()==b''
    receipt=dict(passed=True,result_publication_sha=head,archived_files_verified=len(manifest),collaboration_files_verified=n,source_and216_214_model_score_label_freezes_unchanged=True,report_sha256=C.sha(O/'GRID217_report.md'),new_fits=0,new_NN_forwards=0,new_held_model_scores=0,new_policy_metrics=0)
    C.save('delivery_receipt.json',receipt);shutil.copy2(O/'delivery_receipt.json',dest/'delivery_receipt.json');d=A/'delivery';d.mkdir(exist_ok=False);shutil.copy2(Path(__file__),d/'audit_delivery.py');shutil.copy2(O/'delivery_receipt.json',d/'delivery_receipt.json');(d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID217_DELIVERY_VERIFIED',json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
