"""Archive already completed GRID103, preserve all outcomes and prefit engineering failure."""
from pathlib import Path
import json,shutil,sys
root=Path('D:/Codex/GRL研究');sys.path.insert(0,str(root/'work/grid103_learning_mechanism'))
from common import OUT,W,REPO,ARCHIVE,read,save,sha,verify
verify();assert read(OUT/'review.json')['passed'] and read(OUT/'additional_checks.json')['passed']
assert read(OUT/'fits.json')['model_fits']==4
dest=ARCHIVE/'complete';dest.mkdir(parents=True,exist_ok=False)
for p in OUT.rglob('*'):
    if p.is_file():
        q=dest/p.relative_to(OUT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for p in W.glob('*.py'):shutil.copy2(p,dest/p.name)
shutil.copy2(root/'tmp/grid103_final_checks.py',dest/'post_evaluation_additional_checks.py')
shutil.copy2(Path(__file__),dest/'publication.py')
save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='archive_manifest.json'})
collab=root/'论文协作资料/FRAME_学习补查机制_GRID103_20261022';collab.mkdir(exist_ok=False)
for name in ['GRID103_report.md','学习机制结论_中文.md','design.json','review.json','decision.json','additional_checks.json','validation_results.json','validation_trajectories.csv']:
    shutil.copy2(OUT/name,collab/name)
summary='''

## 2026-10-22 — GRID103 learning mechanism comparisons completed

Four fixed fits completed: next-batch gain RF64 and three matched gamma0 DQN seeds; existing gamma1 models unchanged. Four-date macro net utility / query fraction / gain capture: FULL .137051/100%/100%; BATCH1 .162869/56.13%/98.17%; RF .160852/28.42%/70.77%; gamma1 DQN all-seed mean .157728/34.41%/65.94%; gamma0 mean .163198/31.13%/67.80%. On142 multi-batch prefixes gamma1 .351612 versus gamma0 .361643, only1/3 paired seeds favorable: no established future-reward RL advantage. Fixed one-batch is a competitive mandatory comparator; RF saves queries but sacrifices gain and is not a universal winner. October RF utility+.012315 but73/107 prefixes lose queried cost; all original DQN seeds remain negative. Validation has no hindsight delayed-payoff opportunities; post-evaluation training diagnosis finds40 states in just5 physical calls/3 dates (correlated), so do not claim such opportunities never exist. RF RMSE improves over zero, MAE worsens. 4438 reconstructed/hidden-suffix/label checks,7670 trajectory/78 summary/5992 prediction checks and3540 old-trajectory reproductions pass;5485 visited-action comparisons pass. Existing4 validation dates exposed, fixed proxy price, no physical control-time/week claim. No installations, extra data, cost tuning or final-test generation. All adverse results and prefit archive-path incident preserved; no automatic closed-loop experiment.
'''
for name in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
    with (REPO/'docs'/name).open('a',encoding='utf8') as f:f.write(summary)
(REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps — GRID103学习机制完成

1. 收益预测与gamma0对照、全部种子、单批/多批、10月失败及保存数据核验均完成；先交用户查看报告与机制结论。
2. 当前没有建立考虑后续奖励的DQN优势；优先研究下一批收益预测，保留gamma0 DQN和固定一批作为竞争对照，不宣布任何学习策略已部署或全面胜出。
3. 后续若由用户启动开发闭环，应冻结候选策略，比较原FRAME、固定一批与学习策略的存活/整周成本和完整实际时间；不得只比较查询数或挑最优种子。
4. 本轮到此结束，不自动扩网络/数据/训练步数、扫描价格、启动物理回放、触碰最终8测试场景或恢复旧自动化。
''',encoding='utf8')
with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:
    f.write('\n- [GRID103学习补查机制](protocols/GRID103_learning_mechanism/complete/学习机制结论_中文.md)：收益预测与匹配gamma0对照；未建立后续规划优势，固定一批仍具竞争力，所有种子与反例保留。\n')
manifest=read(dest/'archive_manifest.json');assert all(sha(dest/p)==h for p,h in manifest.items())
assert all(sha(OUT/n)==sha(collab/n) for n in ['GRID103_report.md','学习机制结论_中文.md','decision.json'])
print(json.dumps(dict(archive_files=len(manifest),archive_hashes_verified=True,collaboration_copies_verified=True),ensure_ascii=False))
