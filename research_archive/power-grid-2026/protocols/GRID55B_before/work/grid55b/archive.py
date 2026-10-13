"""Unique before/complete archive; never overwrite prior GRID55 evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid55b'
REPO = ROOT / 'work/GRL-results-integration'
ARCH = REPO / 'research_archive/power-grid-2026'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['before', 'complete'], required=True)
args = parser.parse_args()
D = read(OUT / 'design.json')
if args.phase == 'before':
    assert read(OUT / 'preflight.json')['passed'] and not (OUT / 'manifest.json').exists()
else:
    assert read(OUT / 'review.json')['passed']
old = {}
for m in list(ARCH.glob('*manifest.json')) + list((ARCH / 'protocols').glob('*/manifest.json')):
    for r in read(m).get('files', []):
        p = m.parent / r['path']
        if not p.exists(): p = ARCH / r['path']
        assert sha(p) == r['sha256'], p
        old[str(p)] = r['sha256']
dest = ARCH / 'protocols' / ('GRID55B_' + args.phase)
assert not dest.exists()
dest.mkdir()
files = list((ROOT / 'work/grid55b').glob('*.py'))
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json', '.md', '.csv']]
if args.phase == 'complete':
    files += [p for p in (OUT / 'qualified_view').iterdir() if p.is_file() and p.suffix in ['.json', '.md', '.csv']]
    for folder in (OUT / 'runs').iterdir():
        files += [folder / 'summary.json', folder / 'metadata.json']
    receipts = [{'local_path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
        for p in sorted((OUT / 'runs').rglob('*')) if p.is_file()]
    path = OUT / 'local_trajectory_receipts.json'
    path.write_text(json.dumps(receipts, indent=2), encoding='utf-8')
    files.append(path)
entries = []
for p in sorted(set(files)):
    q = dest / p.relative_to(ROOT); q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, q); assert sha(p) == sha(q)
    entries.append({'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size})
(dest / 'manifest.json').write_text(json.dumps({'files': entries, 'historical_preserved': len(old),
    'scope': 'One complete eight-arm replacement timing repetition; original128/raw reports immutable; local trajectory receipts only, no large vectors/checkpoints.'}, indent=2), encoding='utf-8')
if args.phase == 'before':
    note = '用户2026-10-13继续后，按关机前具体方案冻结GRID55B：仅待机受影响日期2035-11-12_9全部八方法补新重复2，2400秒墙钟上限；原128条不覆盖，不改模型/控制/阈值/先验缓存，不新增训练。运行期临时防空闲待机，不修改永久电源计划。结束后核对电源日志、原两次物理路径/成本/合法反馈/查询，对全部八方法一致构建原重复0/新重复2计时视图，沿用原对称选型。不是新独立确认。'
    nxt = '1. 执行GRID55B八臂有限计时补测；只一个串行运行器，计时期间不运行训练或重型审计。\n2. 核对电源日志与原路径/成本/查询，随后构建明确来源的128槽位修复视图；不减待机时间、不择种子。\n3. 保留RAW128和原选型，全部自动跟进仍暂停；本次异常不自动重试。\n'
else:
    R = read(OUT / 'qualified_view/review.json'); V = R['version_selection']
    note = 'GRID55B八条补测与保存数据核验通过：16/16与原两重复的路径、向量、成本、反馈和查询一致，运行期无电源待机事件。原RAW128保留，七日期原0/1加受影响日期原0/新2组成明确来源的128槽位计时视图。' + '；'.join(f"{r['method']}质量{r['quality_qualified_pairs']}/{r['runs']}、控制{r['control_s']:.2f}s" for r in R['summaries']) + f"。原对称选型={V['engineering_preference']}；这是质量—时间投资筛查，不是等价或独立确认；公共缓存不归于GNN。"
    nxt = '1. 按GRID55B质量—时间证据固定工程候选，保留原版作速度/质量消融；不新增模型菜单或择种子。\n2. 整理最小图评分/先验归因、训练教师成本、质量—速度总表及可运行复现入口。\n3. 继续论文方法和结果材料；LG01/GRID52/GRID55自动跟进仍暂停。\n'
p = REPO / 'docs/RESEARCH_STATE.md'
p.write_text('# Research state — matched version timing repair\n\n- **最新（2026-10-13）：** ' + note + '\n\n' + p.read_text(encoding='utf-8').split('\n', 1)[1], encoding='utf-8')
(REPO / 'docs/NEXT_STEPS.md').write_text('# Next steps\n\n' + nxt, encoding='utf-8')
for name in ['DECISIONS.md', 'EXPERIMENT_LOG.md']:
    p = REPO / 'docs' / name
    p.write_text(p.read_text(encoding='utf-8') + '\n## 2026-10-13 — GRID55B ' + args.phase + '\n\n' + note + '\n', encoding='utf-8')
p = ARCH / 'README.md'
p.write_text(p.read_text(encoding='utf-8') + '\n## GRID55B ' + args.phase + '\n\n' + note + '\n', encoding='utf-8')
for p, h in old.items(): assert sha(Path(p)) == h
print(json.dumps({'archived': args.phase, 'files': len(entries), 'historical_preserved': len(old)}, ensure_ascii=True))
