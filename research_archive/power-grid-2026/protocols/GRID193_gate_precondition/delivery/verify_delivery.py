"""Verify archived, committed and collaboration copies for GRID191--193."""
from pathlib import Path
import hashlib,json,subprocess,sys
R=Path('D:/Codex/GRL研究'); REPO=R/'work/GRL-results-integration'; O=R/'outputs/grid193_gate_precondition'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args])
def main():
    head=git('rev-parse','HEAD').decode().strip(); remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]
    assert head==remote
    spec=[('grid191_gate_diagnosis','GRID191_gate_diagnosis','FRAME_门控迁移与训练覆盖_GRID191','GRID191_report.md'),('grid192_gate_support','GRID192_gate_support','FRAME_门控覆盖对照_GRID192','GRID192_report.md'),('grid193_gate_precondition','GRID193_gate_precondition','FRAME_门控数值对照_GRID193','GRID193_report.md')]
    stages=[]
    for outname,archname,collabname,report in spec:
        src=R/'outputs'/outname; arch=REPO/'research_archive/power-grid-2026/protocols'/archname; dest=R/'论文协作资料'/collabname
        assert read(src/'complete.json')['passed']
        n=0
        for mf in arch.rglob('archive_manifest.json'):
            for name,h in read(mf).items():assert sha(mf.parent/name)==h,(mf,name); n+=1
        assert n>0
        collab=[]
        for p in dest.iterdir():
            assert p.is_file() and sha(p)==sha(src/p.name),p
            collab.append(p.name)
        assert len(collab)==8 and report in collab
        for name in collab:
            ap=arch/'complete'/name; assert sha(ap)==sha(src/name),ap
            data=git('show',f"HEAD:{ap.relative_to(REPO).as_posix()}")
            assert data==ap.read_bytes(),ap
        # Frozen input code/data receipts remain intact; recovery wrappers are separate.
        frozen=0
        for name in ['source_freeze.json','code_freeze.json']:
            p=src/name
            if p.exists():
                for rel,h in read(p).items():assert sha(R/rel)==h,(p,rel); frozen+=1
        stages.append(dict(stage=archname,archive_files_checked=n,collaboration_files_checked=len(collab),frozen_inputs_checked=frozen,report_sha256=sha(src/report),committed_complete_files_match=True))
    # This checker does not invoke a solver, fit, audit pipeline, or held feature extraction.
    receipt=dict(passed=True,verified_publication_head=head,remote_main=head,stages=stages,new_fits=0,new_physics=0,scope='Delivery hash and committed-byte verification; scientific evaluation receipts remain stage-specific')
    (O/'delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    d=REPO/'research_archive/power-grid-2026/protocols/GRID193_gate_precondition/delivery';d.mkdir(exist_ok=False)
    import shutil
    for p in [Path(__file__),O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
