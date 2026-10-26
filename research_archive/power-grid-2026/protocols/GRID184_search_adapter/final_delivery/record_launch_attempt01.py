"""Small startup receipt only; no audit/model work while benchmark is timed."""
from pathlib import Path
import datetime,hashlib,json,shutil
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid184_search_adapter';repo=R/'work/GRL-results-integration'
read=lambda p:json.loads(p.read_text(encoding='utf8-sig'))
state=read(O/'pipeline_state.json');active=read(O/'active_worker.json');launch=read(O/'launch_process.json')
assert not (O/'pipeline_error.json').exists();assert state['phase']=='benchmark';assert active['parent']==state['pid']
folder=O/'runs'/f"r{active['item']['repeat']}__{active['item']['week']}__{active['item']['method']}"
progress=read(folder/'progress.json');assert progress['step']>=512
receipt=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pipeline_launcher_pid=launch['pid'],actual_pipeline_pid=state['pid'],worker_launcher_pid=active['pid'],first_progress=progress,first_method=active['item']['method'],planned=64,error_absent=True,automation_id='grid184-consensus',monitoring='normalrunningquiet;notifyauditedcompletion_or_actionablefailure',note='Venv launcher and runtime child are one pipeline/worker chain,not duplicate runners.')
(O/'launch_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
d=repo/'research_archive/power-grid-2026/protocols/GRID184_search_adapter/active_launch';d.mkdir(exist_ok=False)
for n in ['launch_verification.json','launch_process.json','record_launch.py']:shutil.copy2(O/n,d/n)
(d/'archive_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
with (O/'HANDOFF.md').open('a',encoding='utf8') as f:f.write(f'\n实际启动已核实：唯一pipeline实际PID {state["pid"]}、worker launcher {active["pid"]}；首条{active["item"]["method"]}已前进到{progress["step"]}步，当前无pipeline_error。启动收据launch_verification.json；grid184-consensus跟进已启用。正常不启动其他运行器、不安装警告建议包。\n')
msg=f'GRID184唯一隐藏pipeline已运行，首条FRAME已前进到{progress["step"]}步，无当前工程错误。启动授权/冻结已推送，64条固定范围与4h总窗口保持；grid184-consensus负责安静跟进至审计发布，不重复运行器/拟合/安装或计时中重型审计。'
for n in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md']:
    p=repo/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID184启动核实\n\n'+msg+'\n'
    if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
    else:t+=entry
    p.write_text(t,encoding='utf8',newline='\n')
(repo/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID184固定闭环运行中\n\n1. 既有唯一pipeline已启动；按outputs/grid184_search_adapter/HANDOFF.md和状态/错误/进程核查，不再启动运行器。\n2. 全64条固定日期/种子/双反向，计时中不训练/安装或重型审计；旧资格警告不当当前故障。\n3. 全部结束后执行已有保存数据审计，核对质量/完成/成本/实际时间/查询/P95及反例、归档/协作副本/远端SHA；更新研究状态和最终记录后通知，不自动扩大/最终8/论文。\n',encoding='utf8',newline='\n')
print('GRID184_RUNNING_CONFIRMED',progress['step'])
