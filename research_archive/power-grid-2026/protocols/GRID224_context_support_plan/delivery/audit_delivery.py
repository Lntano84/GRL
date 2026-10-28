"""Verify GRID223/224 source archives, revised risk arrays and collaboration copies."""
from pathlib import Path
import importlib.util,json,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO

def main():
    assert not (O/'delivery_receipt.json').exists();C.verify();assert C.read(O/'complete.json')['passed'] and C.read(O/'independent_audit.json')['passed'];rev=O/'revision02'
    for n,h in C.read(rev/'qualified_freeze.json').items():assert C.sha(rev/n)==h
    for r in C.read(rev/'qualification.json')['records']:
        x=np.load(rev/'qualified_inputs'/f"{r['pool']}.npz");y=np.load(O/'qualified_inputs'/f"{r['pool']}.npz")
        for k in x.files:
            if k!='weight':assert np.array_equal(x[k],y[k])
        s=x['domain_STOP'];assert abs(x['weight'][s].sum()/x['weight'].sum()-.5)<1e-12 and abs(x['weight'][~s].sum()/x['weight'].sum()-.5)<1e-12
    dirs=[(R/'outputs/grid223_remaining_support',REPO/'research_archive/power-grid-2026/protocols/GRID223_remaining_support',R/'论文协作资料/FRAME_剩余误停与支持_GRID223'),(O,A,R/'论文协作资料/FRAME_辅助经验方案_GRID224')];files=collaboration=0
    for out,ar,co in dirs:
        assert C.read(out/'complete.json')['passed']
        for p,h in C.read(ar/'archive_manifest.json').items():assert C.sha(ar/p)==h;files+=1
        for p in co.iterdir():assert C.sha(p)==C.sha(out/p.name);collaboration+=1
    head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();C.save('delivery_receipt.json',dict(passed=True,previous_published_stage_sha=head,archive_files_verified=files,collaboration_files_verified=collaboration,revision02_only_weights_changed=True,all_originalSTOP_constraints_unchanged=True,actual_fits=0,report223_sha256=C.sha(R/'outputs/grid223_remaining_support/GRID223_report.md'),plan224_sha256=C.sha(O/'GRID224_plan.md'),handoff_sha256=C.sha(O/'CONTINUOUS_HANDOFF.md')))
    d=A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),O/'delivery_receipt.json',O/'CONTINUOUS_HANDOFF.md']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n');shutil.copy2(O/'delivery_receipt.json',R/'论文协作资料/FRAME_辅助经验方案_GRID224/delivery_receipt.json');print('GRID223_224_DELIVERY_VERIFIED',files,collaboration)
if __name__=='__main__':main()
