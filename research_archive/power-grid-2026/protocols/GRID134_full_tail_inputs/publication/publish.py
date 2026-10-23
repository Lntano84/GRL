"""Audit exported raw public contract and archive receipts; no model or simulator."""
from pathlib import Path
import gzip,hashlib,json,shutil
R=Path('D:/Codex/GRL研究'); O=R/'outputs/grid134_full_tail_inputs'; REPO=R/'work/GRL-results-integration'
A=REPO/'research_archive/power-grid-2026/protocols/GRID134_full_tail_inputs'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert read(O/'complete.json')['passed'] and read(O/'review.json')['passed']
recovery=read(O/'recovery_01.json'); assert sha(R/'work/grid134_full_tail_inputs/resume01.py')==recovery['recovery_source_sha256']
for name,h in recovery['preserved'].items():assert sha(O/'interrupted_attempts/pack_limit_01'/name)==h
for n in ['source_hashes.json','code_freeze.json']:
    for p,h in read(O/n).items():assert sha(R/p)==h
receipt=read(O/'dataset_receipt.json'); assert sha(O/receipt['path'])==receipt['sha256']
index=read(O/'input_index.json'); idx={m['sample_id']:m for m in index}; assert len(idx)==2378
fields={'seen_ids','seen_plans','seen_feedback','next_ids','next_plans','incumbent_id','incumbent_reward','incumbent_plan'}
feedback_fields={'rho','reward','done','illegal','ambiguous','exceptions','strict_admissible'}
found=set(); pending=0; contexts=0
with gzip.open(O/receipt['path'],'rt',encoding='utf8') as f:
    for line in f:
        r=json.loads(line); assert set(r)=={'sample_id','public_input'}; sid=r['sample_id']; assert sid not in found; found.add(sid)
        c=r['public_input']; assert set(c)==fields; m=idx[sid]
        assert len(c['next_ids'])==len(c['next_plans'])==m['pending']
        assert len(c['seen_ids'])==len(c['seen_plans'])==len(c['seen_feedback'])==m['seen']
        assert len(set(c['next_ids']))==len(c['next_ids']) and not set(c['seen_ids'])&set(c['next_ids'])
        assert all(set(row)==feedback_fields for row in c['seen_feedback'])
        assert hashlib.sha256(line.rstrip('\n').encode()).hexdigest()==m['public_sha256']
        pending+=len(c['next_ids']); contexts+=len(c['seen_ids'])
assert found==set(idx) and pending==122446 and contexts+pending==267311
check=dict(passed=True,records=len(found),all_pending_plan_rows=pending,already_paid_context_rows=contexts,strict_public_field_allowlists=True,
    unique_sample_ids_and_pending_representatives=True,original_frozen_source_and_recovery_sha_unchanged=True,all_old_failure_artifacts_retained=True,
    full_dataset_sha256=receipt['sha256'],archive_policy='Raw generated public context gzip stays at documented D: location; hash/index/reconstructing frozen code retained in Git; not a newly sampled dataset.',
    model_fits=0,validation_records_exported=0,new_physical_calls=0)
write(O/'additional_review.json',check)
collab=R/'论文协作资料/FRAME_完整尾部输入核查_GRID134';collab.mkdir(parents=True,exist_ok=False)
for name in ['GRID134_report.md','review.json','additional_review.json','dataset_receipt.json','complete.json']:
    shutil.copy2(O/name,collab/name);assert sha(O/name)==sha(collab/name)
write(O/'collaboration_receipt.json',dict(directory=collab.relative_to(R).as_posix(),hashes={p.name:sha(p) for p in collab.iterdir()}))
dest=A/'publication';dest.mkdir(parents=True,exist_ok=False)
for p in [O/'additional_review.json',O/'collaboration_receipt.json',O/'qualification_resume01.log',Path(__file__)]:shutil.copy2(p,dest/p.name)
write(dest/'archive_manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
for phase in ['frozen','recovery_01','complete','publication']:
    d=A/phase
    for n,h in read(d/'archive_manifest.json').items():assert sha(d/n)==h
entry='''

## 2026-10-23 — GRID134完整尾部公开输入资格通过

零拟合/模型推理/新标签/新仿真，仅原训练2378门控前缀(1637拟合/741校准)、222调用。完整剩余计划122446行，已购同站反馈144865行；大组首批以外55425候选重复行，单条最多237。每条初始已购上下文/首64/最佳动作与GRID111一致；别名/候选顺序、公开命令哈希、未购反馈毒化/字段拒绝、母线重标、配对行置换等变及序列化检查通过。原pack的<=64断言造成一次工程停止，错误/部分输出/原冻结源保留；外置恢复包装器仅分段调用原编码再按原序拼回全部计划，旧源未改，恢复哈希核查通过。267311计划/上下文行及2378导出白名单独立复核，模型0/验证调用解码0，最终8测试未生成。只是完整输入资格，不是预测价值证明；GRID98/99/110统计与113/116/119关系失败保留，不扩大或追加训练。报告：research_archive/power-grid-2026/protocols/GRID134_full_tail_inputs/complete/GRID134_report.md
'''
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+entry,encoding='utf8')
(REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps — 检验完整尾部公开信息的预测价值

1. GRID133固定森林没有超过GROUP；只说明此配置未改善，不证明所有非线性无用。保留统一4.81%/分组6.63%查询节省与旧7差异，不扫描树参数或门槛。
2. GRID134原2378训练/校准前缀的全尾公开输入已资格通过，原数据与最终测试不扩。新标签0、拟合0、验证解码0；历史64接口故障及恢复收据保留。
3. 下一拟合前先封存匹配方案：同完整补查目标、12拟合/4校准日期、相同便宜分支与校准；比较首批/全尾和配对/去配对，控制模型容量与训练预算。先通输入遮蔽/置换及同初始化检查，再按固定预算训练；训练前通知。
4. 旧结构摘要和关系模块失败是反证，不能因补全输入宣称结构有效。全部已暴露开发结果，局部增量/查询不当整周成本或速度；不自动扩大、做物理回放或修改论文。
''',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8')+'\n- [GRID134完整尾部公开输入资格](protocols/GRID134_full_tail_inputs/complete/GRID134_report.md)：2378原训练/校准门控前缀全候选结构还原，零拟合/新物理，历史64接口错误与恢复收据保留；尚无预测改进结论。\n',encoding='utf8')
print(json.dumps(check,ensure_ascii=False))
