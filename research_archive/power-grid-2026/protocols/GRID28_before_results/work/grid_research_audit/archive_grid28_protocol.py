"""Archive frozen staged-search designs before reading their rollout results."""
import hashlib
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT / "work/GRL-results-integration"
ARCH=REPO / "research_archive/power-grid-2026";DEST=ARCH / "protocols/GRID28_before_results"
assert not DEST.exists()
preserved=[]
for name in ["archive_manifest.json","grid24_archive_manifest.json","grid26_archive_manifest.json","grid26_review_archive_manifest.json"]:
    for r in json.loads((ARCH / name).read_text())["files"]:
        p=ARCH / r["path"];h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==r['sha256'];preserved.append((p,h))
files=[]
for stage in ["grid27","grid28"]:
    files+=list(sorted((ROOT / 'work' / stage).glob('*.py')))
    files+=[p for p in sorted((ROOT / 'outputs' / stage).glob('*')) if p.is_file() and p.suffix in ['.json','.csv','.md'] and p.name not in ['RUN_STATE.md','manifest.json','pilot_finished.json','finished.json']]
files+=[ROOT / 'outputs/grid27_profile/design.json',ROOT / 'outputs/grid27_profile/profile.json',Path(__file__).resolve()]
records=[]
for p in files:
    relative=p.relative_to(ROOT);target=DEST / relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==h
    records.append({'path':relative.as_posix(),'sha256':h,'bytes':p.stat().st_size})
(DEST / 'manifest.json').write_text(json.dumps({'files':records,'preserved_checks':len(preserved),
    'scope':'Frozen schedules/offline diagnostics/runtime preflight; GRID28 March pilot in progress, no closed-loop result copied. One completed protected profile included as diagnostic.'},indent=2),encoding='utf-8')
(DEST / 'README.md').write_text('''# GRID27/28 分批补查冻结与首轮诊断

GRID27 固定256/连续两批64无改善即停止，已暴露26状态上可省约54%–57%查询，但遗漏已知三月关键动作；按冻结门槛不启动其闭环。GRID28 只追加一个语义规则：查完当前最佳方案所在变电站的剩余候选。固定/自适应规则均保留，不扫描常数。PATIENCE64-local在GNN0/PPO上均找回关键动作，26状态上仍有2次选择不同；不能从微小rho差推断成本安全。

冻结三月闭环试点：当前受保护GNN0、局部补全GNN0、相同补全PPO；质量合格才继续其余3个已暴露日期。全部沿用V1/作者权重、掩码、别名、精确reward破并列、0.9阈值及连续控制/恢复周期。新规则改变unsafe补查结束条件，不声称保留旧全池保证或新安全保证。原模型/规则/失败不覆盖。

另完成一条受保护FULL三月剖析，2017步逐步同历史，优化求解约72.4%、N1搜索约15.0%的控制耗时。这是一条带剖析开销的开发轨迹，不能当速度比较。solver_records仅捕获34个以关键字传solver_type的求解调用；恢复器使用位置参数，细项未记录，缺失不可算0。分层总时间覆盖全部控制调用，但不能据这份不完整细项表给出整体编译/后端求解分解。

没有新拟合/RL，8个封存日期未打开。对应源码、设计、代码SHA、离线结果和实际接口预检见manifest。本快照不含尚在运行的闭环结果；不称独立确认。
''',encoding='utf-8')
docs=REPO / 'docs'
p=docs / 'RESEARCH_STATE.md';b=p.read_text(encoding='utf-8')
p.write_text(b.replace('# Research state — 2026-10-11','# Research state — 2026-10-11') .replace('\n\n- **GRID24', '\n\n- **新进度：GRID27/28分批补查**。GRID27简单停止规则在26个开发状态上省查询但遗漏已知三月关键动作，不进入闭环。GRID28增加当前最佳动作所在变电站的候选补全；离线找回关键动作、仍有2次选择变化，现正跑冻结三月试点及同规则PPO强对照。单条受保护FULL剖析确认大部分耗时在连续优化；不作为速度比较。原模型、守卫与失败均保留，尚无GRID28闭环裁决，新拟合/RL/封存评价0。\n\n- **GRID24',1),encoding='utf-8')
(docs / 'NEXT_STEPS.md').write_text('''# Next steps — 2026-10-11 / GRID28 pilot running

1. 完成并审计冻结三月试点：受保护GNN0、分批补查+同变电站补全GNN0、相同策略PPO。不能用离线rho差代替闭环成本/存活；成本恶化超过0.5%或新增失败先诊断，不扩大场景。
2. 三月质量合格才按固定规则运行其余3个已暴露日期；不改批量/耐心常数，不据现有离线结果换赢家模型。单次计时只作开发点估计，稳定提速需后续公平重复。
3. 整体提速线：依据一条受保护控制器剖析，优先查连续优化/重复DC求解的保真加速；所有强对照同样优化，不将共用工程加速归因于GNN。
4. 保留GRID27否定筛查、GRID28改进理由与全部版本。GNN/RL独有贡献仍未建立；当前没有新训练。8个封存日期仍未评价，LG01自动跟进仍暂停。
''',encoding='utf-8')
for name,body in [('EXPERIMENT_LOG.md','''\n## 2026-10-11 — GRID27 screen and GRID28 semantic completion qualification

156 legal feedback-replay rows per stage (26 exposed states x3 frozen rankers x2 fixed schedules).GRID27 plateau/fixed cutoff miss known March1733 action; no AC rollout.GRID28 adds only incumbent-substation completion,with aliases;GNN0/PPO PATIENCE64 now recover that action but still change2/26 decisions.78 real-runtime preflight checks and30074 fake feedback dispatches agree;no AC/refit in preflight.March protected/localGNN0/localPPO trial running under frozen design.One protectedFULL profile2017 physical steps exactly matches historical path;hierarchical optimizer solve share72.4%,N1search15.0% in that one instrumented March sample.Solver fine-statistics are partially captured,not zero for missing restoration calls.[Protocol snapshot](../research_archive/power-grid-2026/protocols/GRID28_before_results/README.md).\n'''),('DECISIONS.md','''\n## 2026-10-11 — Check closed-loop quality before interpreting staged-search savings

User authorizes both whole-controller and expensive-search acceleration experiments.GRID27 rho-small omissions include a known cost-critical action,so do not deploy that plateau alone.GRID28 semantic substation completion is a shared cheap rule,not graph-specific novelty or stopping proof.Keep all schedules/checkpoints;use matchingPPO,unchanged admission/continuous safeguards andpre-specified pilotgate.Exact-protected profiling is diagnostic;generic optimizer acceleration must be applied to all baselines.No immediate RL or hidden-date evaluation.\n''')]:
    p=docs / name;p.write_text(p.read_text(encoding='utf-8')+body,encoding='utf-8')
for p,h in preserved:assert hashlib.sha256(p.read_bytes()).hexdigest()==h
print(json.dumps({'archived_files':len(records),'preserved_checks':len(preserved),'passed':True}))
