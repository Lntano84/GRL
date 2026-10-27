"""Verify published GRID214 artifacts without fitting or policy evaluation."""
from pathlib import Path
import json,shutil,subprocess,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,A,R=C.O,C.A,C.R
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args]).strip()
def main():
    assert not (O/'delivery_receipt.json').exists()
    C.scored()
    for name in ['labelled_freeze.json']:
        for p,h in C.read(O/name).items():assert C.sha(O/p)==h,p
    assert C.read(O/'complete.json')['passed'] and C.read(O/'independent_audit.json')['passed']
    manifests=0;files=0
    for phase in sorted(p for p in A.iterdir() if p.is_dir()):
        manifest=C.read(phase/'archive_manifest.json')
        for name,h in manifest.items():assert C.sha(phase/name)==h,(phase.name,name)
        manifests+=1;files+=len(manifest)
    assert manifests==6,manifests
    pub=C.read(O/'pre_score_publication.json');join=C.read(O/'pre_join_publication.json')
    modelcount=predcount=labelcount=0
    for folder,phase,revision in [('models','trained',pub['git_sha']),('predictions','scored',join['git_sha']),('labelled_samples','labelled','HEAD')]:
        items=list((O/folder).iterdir());assert len(items)==(108 if folder=='models' else 36)
        for p in items:
            target=A/phase/folder/p.name;assert C.sha(p)==C.sha(target)
            rel=target.relative_to(C.REPO).as_posix()
            blob=subprocess.check_output(['git','-C',str(C.REPO),'show',revision+':'+rel]);assert blob==p.read_bytes(),rel
        if folder=='models':modelcount=len(items)
        elif folder=='predictions':predcount=len(items)
        else:labelcount=len(items)
    dest=R/'论文协作资料/FRAME_内层分歧样本_GRID214';collab=0
    for p in dest.iterdir():
        assert p.is_file() and C.sha(p)==C.sha(O/p.name)==C.sha(A/'audited'/p.name);collab+=1
    assert collab==8
    head=git('rev-parse','HEAD').decode();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote
    assert git('status','--porcelain')==b''
    processes=subprocess.check_output(['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?(.exe)?$' -and $_.CommandLine -match 'grid214_inner_head_generation' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],text=True)
    if processes.strip():
        rows=json.loads(processes);rows=rows if isinstance(rows,list) else [rows]
        assert all('audit_delivery.py' in r['CommandLine'] for r in rows),rows
    receipt=dict(passed=True,published_result_git_sha=head,source_protocol_and_all_freeze_hashes_verified=True,archive_phases=manifests,archived_files_checked=files,all108_model_bytes_verified_at_pre_score_commit=modelcount,all36_prediction_bytes_verified_at_pre_join_commit=predcount,all36_labelled_arrays_verified_at_published_result_commit=labelcount,collaboration_files_verified=collab,training_or_scoring_processes_running=False,router_fits=0,NN_fits=0,new_physics=0,report_sha256=C.sha(O/'GRID214_report.md'),audit_sha256=C.sha(O/'independent_audit.json'))
    C.save('delivery_receipt.json',receipt);shutil.copy2(O/'delivery_receipt.json',dest/'delivery_receipt.json');C.archive('delivery');print('GRID214_DELIVERY_VERIFIED',json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
