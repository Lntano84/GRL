"""Preserve interpretation, check receipts and publish local research artifacts; no fitting."""
from pathlib import Path
import collections,gzip,hashlib,json,shutil
import numpy as np
R=Path('D:/Codex/GRL研究'); O=R/'outputs/grid133_nonlinear_gate'; REPO=R/'work/GRL-results-integration'
A=REPO/'research_archive/power-grid-2026/protocols/GRID133_nonlinear_gate'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert read(O/'complete.json')['passed'] and read(O/'review.json')['passed']
old={r['sample_id']:r for r in read(R/'outputs/grid132_conditional_gate/validation_trajectories.json') if r['method']=='GLOBAL'}
rows=read(O/'validation_trajectories.json'); assert len(rows)==590
for r in rows:
    b=old[r['sample_id']]
    assert all(r[k]==b[k] for k in ['queries','chosen_id','action_disagreement'])
    assert abs(r['gain_lost']-b['gain_lost'])<1e-12
with gzip.open(R/'outputs/grid129_baseline_risk/train_label_ledger.jsonl.gz','rt',encoding='utf8') as f: labels=[json.loads(l) for l in f]
with np.load(R/'outputs/grid129_baseline_risk/train_public_inputs.npz') as z: q=z['known_queries']
with np.load(R/'outputs/grid130_calibrated_gate/train_calibration_scores.npz') as z: fit=z['fitmask']; cal=z['calmask']; oldscore=z['score']
with np.load(O/'train_calibration_scores.npz') as z: rf=z['score']
with np.load(R/'outputs/grid132_conditional_gate/train_calibration_scores.npz') as z: lr=z['EXP_PUBLIC37']
from sklearn.metrics import roc_auc_score
y=np.array([r['full_baseline_gain']>0 for r in labels]); date_details={}
for d in sorted({r['date'] for r,m in zip(labels,cal) if m}):
    mask=np.array([r['date']==d for r in labels])&cal&(q>64)
    date_details[d]=dict(prefixes=int(mask.sum()),physical_calls=len({labels[i]['call'] for i in np.flatnonzero(mask)}),
        RF_auc=float(roc_auc_score(y[mask],1-rf[mask])),specialized_LR_auc=float(roc_auc_score(y[mask],1-lr[mask])),original_LR_auc=float(roc_auc_score(y[mask],1-oldscore[mask])))
with np.load(O/'validation_scores.npz') as z: score=z['score']
exp=np.array([r['expensive'] for r in rows]); threshold=read(O/'fits.json')['threshold']['expensive']
check=dict(passed=True,all590_decisions_queries_match_original_GLOBAL=True,calibration_date_diagnostics=date_details,
    expensive_validation_max_no_gain_score=float(score[exp].max()),expensive_frozen_threshold=threshold,
    expensive_validation_queries_all_fallback=bool((score[exp]<=threshold).all()),
    forest_fit_once=True,no_additional_models_or_thresholds=True)
write(O/'additional_review.json',check)
info=read(O/'information_audit.json')['cohorts']; d=read(O/'train_calibration_diagnostics.json')
text=['# GRID133结果解释与后续信息核查','',
    '固定随机森林没有改善决策：590条保存验证案例的最终动作与查询数均与原统一门控完全一致。新增训练不是造成7条旧动作差异的原因；这7条来自不变的小补查分支，也不能直接视为整周质量失败。',
    '关键是大补查一次也没有跳过，因此即使不采用严格的动作一致筛查，本轮也没有超过原分组校准的计算覆盖。','',
    f"拟合AUC：森林{d['fit']['RF_auc']:.4f}，匹配专门LR{d['fit']['LR_auc']:.4f}；校准AUC：森林{d['calibration']['RF_auc']:.4f}，匹配专门LR{d['calibration']['LR_auc']:.4f}。森林校准虽略好于这份专门LR，但仍较弱，且不能混淆为超过原全局LR。",'',
    '拟合AUC高而校准明显下降，说明当前森林的拟合区分没有稳定转移到保留日期。它不能证明所有非线性模型都无效，也不能证明摘要一定不够；单份浅树与LR的参数量不同。本轮不追加树深、叶大小、种子或门槛搜索。','',
    '## 四个校准日期全部保留','']
for date,v in date_details.items(): text.append(f"- {date}：{v['prefixes']}条相关前缀/{v['physical_calls']}个物理调用；森林AUC {v['RF_auc']:.4f}，专门LR {v['specialized_LR_auc']:.4f}，原全局LR {v['original_LR_auc']:.4f}。")
text+=['','AUC按前缀计算，前缀相互相关；不把它们当独立日期或独立场景，不作显著性推断。','', '## 范围差异是下一条可检验问题','']
for name,v in info.items(): text.append(f"- {name}：完整剩余候选合计{v['all_pending_queries']}个，旧关系输入只覆盖首批{v['old_next_batch_candidates']}个；其后{v['beyond_old_next_batch']}个，占{100*v['beyond_old_next_batch']/v['all_pending_queries']:.2f}%。这是相关前缀重复计数的候选总量，不是独立候选样本数。")
text+=['',
    '旧关系模型并未编码首批以外候选的接线结构。那些动作在当前时刻是公开的，可以补齐输入；其仿真反馈尚未购买，仍须严格遮蔽。当前37维也没有逐项结构与已购反馈的对应。',
    '但“有信息没输入”不等于“该信息一定有预测价值”。旧配对网络未形成稳定优势、反馈错位很少改变正例决策，这些反证全部保留。',
    '下一步先做训练/校准数据上的输入资格核查：还原锁定站的全部剩余接线结构；首64条与旧输入逐项一致；首批以外不读未购奖励/负载率；检验动作别名、母线标号互换和已购上下文顺序；之后再决定是否训练完整尾部模型。',
    '若进入学习，应固定完整补查目标和同一12/4日期划分，设同容量配对/去配对控制，显式区分增加候选范围与改变模型。不能仅换更大的网络后把变化都归于结构信息。',
    '无需为这一输入核查新增场景或扩大最终测试。当前仍保留原FRAME和分组校准开发参照，不更新论文、不部署新门控。']
(O/'结果解释与后续核查.md').write_text('\n'.join(text)+'\n',encoding='utf8')
collab=R/'论文协作资料/FRAME_同信息非线性核查_GRID133'; collab.mkdir(parents=True,exist_ok=False)
for name in ['GRID133_report.md','结果解释与后续核查.md','review.json','additional_review.json','decision.json','information_audit.json','train_calibration_diagnostics.json']:
    shutil.copy2(O/name,collab/name); assert sha(O/name)==sha(collab/name)
write(O/'collaboration_receipt.json',dict(directory=collab.relative_to(R).as_posix(),hashes={p.name:sha(p) for p in collab.iterdir()}))
dest=A/'interpretation'; dest.mkdir(parents=True,exist_ok=False)
for p in [O/'结果解释与后续核查.md',O/'additional_review.json',O/'collaboration_receipt.json',Path(__file__)]: shutil.copy2(p,dest/p.name)
write(dest/'archive_manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
for phase in ['frozen','complete','interpretation']:
    root=A/phase
    for n,h in read(root/'archive_manifest.json').items(): assert sha(root/n)==h
for name in ['source_hashes.json','code_freeze.json']:
    for n,h in read(O/name).items(): assert sha(R/n)==h
entry='''

## 2026-10-23 — GRID133同信息浅森林完成，未改善大补查

仅一份固定64树/深4/叶20 RF分类器，445/193原大组拟合/校准、原37维、同ID/权重/目标/校准，小分支不变。拟合AUC .9573对专门LR .8449，校准 .5953对 .5071（原全局LR另保留，不能混称全面改善）。全部权重和门槛在验证前封存；验证大组跳过0，590条动作/查询逐项等于原GLOBAL：增量保留99.98586%、省查询4.81142%、7旧差异；不及GROUP省6.62758%，即使放宽动作一致筛查也无增量覆盖。不能据一份森林断言非线性无用或信息不可学。完整输入核查显示旧关系网络仅下一批64计划，在638条训练/校准大组前缀中尾部候选合计55425/96257未编码；范围不同于整段门控，但不证明补齐有效。旧关系失败/扰动弱响应保留。独立逐树前向、590保存动作/查询/增量、小分支重现、来源/归档/协作哈希通过。零新数据/物理/最终测试，不采用/不扩大。下一步先补齐完整公开尾部并做无泄漏/别名/不变性资格检查，不直接换大网络。报告：research_archive/power-grid-2026/protocols/GRID133_nonlinear_gate/complete/GRID133_report.md
'''
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name; p.write_text(p.read_text(encoding='utf8')+entry,encoding='utf8')
(REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps — 完整剩余候选的信息资格

1. GRID133一份固定RF未改善大组：拟合AUC高、校准下降，大组验证全部回退，查询省4.81%，低于GROUP6.63%。保留所有旧差异，不能把RF新增训练当成其原因。不继续树参数/门槛搜索。
2. 下一步先仅训练/校准范围还原完整尾部公开接线计划，已购反馈按初始门控时刻固定；首64与旧输入一致，验证无未来反馈/别名/母线重标及上下文顺序不变性，不拟合新模型。
3. 之后才冻结全尾目标、候选范围与配对/去配对同容量控制；旧113/116/119的失败和扰动弱响应仍完整保留，不能宣称结构有效或旧模型从未试过。
4. 原四验证日期均已暴露；局部奖励/查询不等于整周质量或速度，不扩数据或物理回放。最终8测试仍封存，不修改论文，准备扩大前通知。
''',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md'; p.write_text(p.read_text(encoding='utf8')+'\n- [GRID133同信息浅森林与输入范围核查](protocols/GRID133_nonlinear_gate/complete/GRID133_report.md)：一份固定非线性分类未改善大组跳过；完整尾部输入与旧下一批关系任务区分，不把范围缺口当预测价值证明。\n',encoding='utf8')
print(json.dumps(check,ensure_ascii=False))
