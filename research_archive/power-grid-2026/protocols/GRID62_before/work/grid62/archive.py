"""Preserve small receipts; retain heavy vectors locally with SHA256 receipts."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid62'
REPO = ROOT / 'work/GRL-results-integration'
ARCH = REPO / 'research_archive/power-grid-2026'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['before', 'complete'], required=True)
    phase = parser.parse_args().phase
    if phase == 'before':
        assert read(OUT / 'preflight.json')['passed'] and not (OUT / 'manifest.json').exists()
    else:
        assert read(OUT / 'review.json')['passed_saved_data']
    dest = ARCH / 'protocols' / ('GRID62_' + phase)
    assert not dest.exists(), 'Do not archive a completed directory twice.'
    dest.mkdir()
    receipts = []
    files = list((ROOT / 'work/grid62').glob('*.py')) + [p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json', '.csv', '.md')]
    if phase == 'complete':
        for folder in (OUT / 'runs').iterdir():
            files.extend([folder / 'summary.json', folder / 'metadata.json'])
    for src in files:
        dst = dest / src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert sha(src) == sha(dst)
        receipts.append({'path': src.relative_to(ROOT).as_posix(), 'sha256': sha(src), 'bytes': src.stat().st_size})
    (dest / 'manifest.json').write_text(json.dumps(receipts, indent=2), encoding='utf-8')
    note = 'GRID62冻结全部剩余三个未用日期、四既有同频次策略、两平衡计时重复，共24条。主方法FRAME、对照FULL/加强首个安全候选，NO_LOCAL只作已有内部消融。原控制/缓存/参数均不变，无训练；三个日期不能包装成八日期或跨电网确认。'
    if phase == 'complete':
        r = read(OUT / 'review.json')
        note += '保存数据审计通过，裁决' + r['verdict'] + '。全部失败/较慢/较差结果保留；非独立AC重求解。'
    for name in ('DECISIONS.md', 'EXPERIMENT_LOG.md'):
        p = REPO / 'docs' / name
        p.write_text(p.read_text(encoding='utf-8') + '\n## 2026-10-13 — GRID62 ' + phase + '\n\n' + note + '\n', encoding='utf-8')
    p = REPO / 'docs/RESEARCH_STATE.md'
    p.write_text('# Research state — final framework confirmation\n\n- **当前：** ' + note + '\n\n' + p.read_text(encoding='utf-8').split('\n', 1)[1], encoding='utf-8')
    p = ARCH / 'README.md'
    p.write_text(p.read_text(encoding='utf-8') + '\n## GRID62 ' + phase + '\n\n' + note + '\n', encoding='utf-8')
    (REPO / 'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. 完成GRID62已冻结24条与保存数据审计，不调模型/日程，不替换任何较差日期。\n2. 并行形成完整框架论文初稿；既有四日期机制与旧八日期模型表分开，GNN只作辅助。\n3. 冻结确认结束后按原质量/实际时间判据汇总，并决定主张强度；不追加模型/对照寻找赢家。\n', encoding='utf-8')
    print(json.dumps({'phase': phase, 'files': len(receipts)}))


if __name__ == '__main__':
    main()
