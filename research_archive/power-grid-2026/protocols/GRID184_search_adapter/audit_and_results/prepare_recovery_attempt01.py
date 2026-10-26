"""Qualify shutdown recovery and an assertion-identical I/O-only audit revision."""
from pathlib import Path
import ast,datetime,hashlib,json,shutil,sys
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid184_search_adapter';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID184_search_adapter'
sys.path.insert(0,str(R/'work/grid184_search_adapter'));import run as original
read=original.read;sha=original.sha
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def main():
    original.verify();assert not (O/'review.json').exists() and not (O/'decision.json').exists() and not (O/'complete.json').exists()
    assert not (O/'recovery_01.json').exists();d=read(O/'design.json');f=read(O/'finished.json');m=read(O/'manifest.json');assert len(f['summaries'])==len(m['summaries'])==64 and f['summaries']==m['summaries']
    assert {(v['repeat'],v['week'],v['method']) for v in m['summaries']}=={(v['repeat'],v['week'],v['method']) for v in d['execution_order']}
    receipts={};required=0
    for index,(s,item) in enumerate(zip(m['summaries'],d['execution_order'])):
        assert (s['repeat'],s['week'],s['method'])==(item['repeat'],item['week'],item['method'])
        folder=O/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
        assert read(folder/'summary.json')==s==read(O/f'worker_result_{index:02d}.json')
        for n in ['summary.json','metadata.json','components.json','steps.jsonl.gz','vectors.npz']:
            p=folder/n;assert p.is_file() and p.stat().st_size>0;required+=1
        receipts[str(folder.relative_to(R))]=sha(folder/'summary.json')
    old=R/'work/grid184_search_adapter/audit.py';cached=W/'audit_cached.py';plan=read(W/'preparation.json')
    assert sha(old)==plan['original_source_sha256'] and sha(cached)==plan['cached_source_sha256'] and sha(W/'original_audit.py')==sha(old)
    text=cached.read_text(encoding='utf8');marker='    for a in runrows:';assert text.count(marker)==1
    progress="        save('audit_recovery_progress.json',dict(completed_runs=len(runrows),planned=64,method=s['method'],week=s['week'],repeat=s['repeat'],checked_vectors=checks['vectors'],checked_calls=checks['native_call_reexecution']))\n        print('AUDIT_RUN_COMPLETE',len(runrows),64,s['method'],s['week'],flush=True)\n"
    new=text.replace(marker,progress+marker);path=W/'audit_cached_progress.py';assert not path.exists();path.write_text(new,encoding='utf8',newline='\n')
    def assertions(source):return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assert)]
    assert assertions(text)==assertions(new);assert new.replace(progress,'')==text
    backup=O/'interrupted_attempts/audit_shutdown01';backup.mkdir(parents=True,exist_ok=False)
    for n in ['pipeline_error.json','pipeline_state.json','active_worker.json','pipeline_stdout.log','pipeline_stderr.log','audit_pipeline.log','benchmark_started.json']:
        p=O/n
        if p.exists():shutil.copy2(p,backup/n)
    error=read(O/'pipeline_error.json');assert '1073807364' in error['traceback'] and 'audit.py' in error['traceback']
    (O/'pipeline_error.json').rename(O/'pipeline_error_shutdown01.json')
    receipt=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),user_instruction='继续;resume existing audit/publication only',all64complete_receipts=True,verified_required_files=required,
        code_freeze_verified=len(read(O/'code_freeze.json')),data_freeze_verified=len(read(O/'data_freeze.json')),original_shutdown_exit=1073807364,old_error='outputs/grid184_search_adapter/pipeline_error_shutdown01.json',
        original_audit_sha256=sha(old),cached_audit_sha256=sha(cached),progress_audit_sha256=sha(path),same_assertions=len(assertions(new)),assertions_and_policy_checks_unchanged=True,
        only_changes='ReadtwoNPZmembersonce/run,importoriginalregistry,writecompletedrunprogress;no frozen source/record/weight changes',
        estimated_member_load_operations_before=2*64+2*sum(v['steps'] for v in m['summaries']),member_load_operations_after=128,new_physical_runs=0,new_fits=0,benchmark_window_unchanged=True,source_run_summary_sha256=receipts)
    save(O/'recovery_01.json',receipt);save(W/'recovery_code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')})
    dest=A/'audit_recovery_frozen';dest.mkdir(exist_ok=False)
    for p in [*W.iterdir(),O/'recovery_01.json',O/'pipeline_error_shutdown01.json']:
        if p.is_file():shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
    with (O/'HANDOFF.md').open('a',encoding='utf8') as out:out.write('\n## 2026-10-26审计恢复01\n\n用户已回复继续，原审计随关机返回1073807364，无review/decision/complete落盘。64完整收据与320必需文件、全部冻结通过。原pipeline_error已保留为pipeline_error_shutdown01.json，并原样快照至interrupted_attempts/audit_shutdown01，不当当前错误。仅恢复audit_cached_progress.py审计，所有原assert一致、每run两成员各解压一次，逐条写audit_recovery_progress.json；禁止任何物理重放或启动原pipeline。当前日志audit_recovery01_stdout.log/stderr.log，恢复资格见recovery_01.json。\n')
    msg='GRID184关机仅中断原审计(exit1073807364)，64回放/320必需文件及全部冻结验证通过。恢复仅每run两NPZ成员解压一次并加入run进度，原assert/模型/质量标准不变；不重跑物理或全pipeline，不扩原benchmark窗口。旧错误与日志已保留，审计前冻结恢复代码。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID184仅恢复审计\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID184仅审计恢复\n\n1. 原64回放全部完成，禁止启动benchmark或原pipeline；旧关机错误已保留，不当方法失败。\n2. 唯一audit_cached_progress审计恢复，读取优化不减少核查；查看逐run进度，失败保留日志。\n3. 通过后整理全部种子/日期质量、成本、真实控制/查询/P95和反例，完成协作/归档/远端一致；不新增实验/最终8/论文。\n',encoding='utf8',newline='\n')
    print('GRID184_AUDIT_ONLY_RECOVERY_QUALIFIED',json.dumps({k:v for k,v in receipt.items() if k!='source_run_summary_sha256'},ensure_ascii=False))
if __name__=='__main__':main()
