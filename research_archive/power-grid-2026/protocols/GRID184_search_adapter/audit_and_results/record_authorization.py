"""Record the human's explicit launch instruction; never launches a worker."""
from pathlib import Path
import datetime,json,shutil,sys
R=Path('D:/Codex/GRL研究');sys.path.insert(0,str(R/'work/grid184_search_adapter'))
from run import O,REPO,read,save,sha,verify,archive
verify();assert not (O/'physical_authorization.json').exists();assert not (O/'benchmark_started.json').exists();assert not (O/'runs').exists()
D=read(O/'design.json');review=read(O/'readiness_review.json');assert review['passed'] and review['current_design_sha256']==sha(O/'design.json')
first=D['execution_order'][:32];assert D['execution_order'][32:]==[dict(v,repeat=1) for v in reversed(first)]
assert len(D['execution_order'])==64 and len(D['methods'])==8 and len(D['selected'])==4
save('physical_authorization.json',dict(approved_by_user=True,user_instruction='启动',context='User approves the immediately preceding 64-run scope:4existingdevelopment scenarios,FRAME/BATCH1/FACTOR3seeds/CONSENSUS3seeds,two reverse repeats,4h total,no fits/newscenarios/final8.',design_sha256=sha(O/'design.json'),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
save('launch_preflight.json',dict(passed=True,code_files_verified=len(read(O/'code_freeze.json')),data_files_verified=len(read(O/'data_freeze.json')),readiness_passed=True,offline_rows=32238,offline_search_calls=1776,planned=64,new_fits=0,window_s=D['caps']['wall_s'],existing_completed_runs=0,partial_output_absent=True))
with (O/'HANDOFF.md').open('a',encoding='utf8') as f:f.write('\n## 2026-10-25 用户已确认启动\n\n用户直接回复“启动”，对应本聊天上一条64条范围。physical_authorization.json绑定最新design SHA，launch_preflight.json冻结核对通过。只启动唯一pipeline.py；当前原始pipeline日志pipeline_stdout.log/stderr.log。旧authorization_guard_stderr.log仍仅为有意阻止未授权worker的测试。首条FRAME实际运行须核对。\n')
msg='用户明确回复“启动”，授权GRID184已准备的64条同批闭环：原4开发日期scenario100，FRAME/BATCH1/F三种子/共识三种子，第二32条严格逆序；4小时窗口、零拟合/新场景/最终8。冻结哈希与离线资格通过，授权已与当前design SHA绑定；下一步启动唯一隐藏pipeline，首条FRAME检查真实接口。全部种子与不利结果保留，完成后零预测审计。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID184用户授权启动\n\n'+msg+'\n'
    if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
    else:t+=entry
    p.write_text(t,encoding='utf8',newline='\n')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID184已授权固定闭环\n\n1. 用户已批准64条范围；启动唯一隐藏pipeline，首条FRAME核对实际接口，不再请求同一授权。\n2. 按design完成全部种子/日期/双重复；计时期间不拟合、安装或重型审计，不重复完整项或自动扩大窗口。\n3. 全部结束后检查向量/成本/合法反馈/输入/查询/模型决定/重复轨迹和互斥计时，归档推送并通知；保存全部不利结果，小改善不设全赢门槛。最终8及论文不动。\n',encoding='utf8',newline='\n')
archive('authorized');dest=R/'论文协作资料/FRAME_小闭环准备_GRID184'
for n in ['physical_authorization.json','launch_preflight.json']:shutil.copy2(O/n,dest/n)
print('GRID184_AUTHORIZED_AND_HASH_VERIFIED')
