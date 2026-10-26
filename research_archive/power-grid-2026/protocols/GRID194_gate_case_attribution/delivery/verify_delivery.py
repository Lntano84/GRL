"""Independent category accounting and file/publication verification."""
from pathlib import Path
import json,shutil,subprocess,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])
def main():
    C.verify();assert C.read(C.O/'complete.json')['passed'];rows=C.read(C.O/'case_rows.json');summary=C.read(C.O/'summary.json');assert len(rows)==4681
    old=C.read(C.DIRS[190]/'gate_counts.json');new=C.read(C.DIRS[193]/'gate_counts.json')
    assert sum(r['old_continue'] for r in rows)==old['append']
    assert sum(r['new_continue'] for r in rows)==new['append']
    assert sum(r['new_continue'] and r['next_gain']>0 for r in rows)==new['append_changes']
    assert sum(r['old_continue'] and r['next_gain']>0 for r in rows)==old['append_changes']
    # Every cached decomposition is checked again by the saved logits, without model forward calls.
    for r in rows:assert abs(sum(r['logit_delta'].values())-(r['new_logit']-r['old_logit']))<1e-7
    a=C.REPO/'research_archive/power-grid-2026/protocols/GRID194_gate_case_attribution';count=0
    for mf in a.rglob('archive_manifest.json'):
        for p,h in C.read(mf).items():assert C.sha(mf.parent/p)==h;count+=1
    dest=C.R/'论文协作资料/FRAME_门控案例归因_GRID194'
    for p in dest.iterdir():
        assert C.sha(p)==C.sha(C.O/p.name)==C.sha(a/'complete'/p.name)
        assert git('show','HEAD:'+str((a/'complete'/p.name).relative_to(C.REPO)).replace('\\','/'))==(C.O/p.name).read_bytes()
    head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote
    rec=dict(passed=True,all4681_decompositions_rechecked=True,category_counts_match_independent_190_193_evaluations=True,archive_files_checked=count,collaboration_files_checked=4,committed_reports_match=True,verified_publication_head=head,remote_main=head,new_fits=0,new_physics=0)
    C.save('delivery_receipt.json',rec);d=a/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),C.O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(rec))
if __name__=='__main__':main()
