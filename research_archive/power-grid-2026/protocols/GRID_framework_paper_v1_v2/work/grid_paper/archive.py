"""Publish complete drafts and small reproducibility receipts, no heavy traces."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'work/GRL-results-integration'
DEST = REPO / 'research_archive/power-grid-2026/protocols/GRID_framework_paper_v1_v2'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not DEST.exists()
    v2 = ROOT / 'outputs/grid_paper_v2'
    r = json.loads((v2 / 'paper_status.json').read_text(encoding='utf-8'))
    assert r['new_fits'] == 0
    text = (v2 / 'manuscript.md').read_text(encoding='utf-8')
    assert 'PENDING IN V1' not in text
    assert all(f'## {i}.' in text for i in range(1, 9))
    assert r['verdict'] in text and '76.28 versus 44.73' in text
    assert 'reference also satisfies' in text and '83.543%' in text
    checks = {'passed': True, 'sections': 8, 'algorithm_present': '```text' in text,
        'fresh_result_present': r['verdict'], 'timing_variation_and_strong_reference_visible': True,
        'source_hashes_checked': 0, 'figure_visual_review': 'PNG manually inspected: labels, repeat dots, zero-query May and cost-screen line legible.',
        'word_count': len(text.split()), 'native_typesetting_or_submission_review': False}
    for x in json.loads((v2/'source_receipts.json').read_text()):
        assert sha(ROOT/x['path']) == x['sha256'], x['path']
        checks['source_hashes_checked'] += 1
    for name in ('REPRODUCE.md',):
        shutil.copy2(ROOT/'outputs/grid_paper_v1'/name, v2/name)
    (v2/'manuscript_validation.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    folders = ['outputs/grid_paper_v1', 'outputs/grid_paper_v2', 'outputs/grid62_timing_review',
               'outputs/grid_paper_table_reproduction', 'work/grid_paper']
    files = []
    for folder in folders:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    DEST.mkdir()
    receipts = []
    for src in files:
        dst = DEST/src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert sha(src) == sha(dst)
        receipts.append({'path': src.relative_to(ROOT).as_posix(), 'sha256': sha(src), 'bytes': src.stat().st_size})
    (DEST/'manifest.json').write_text(json.dumps(receipts, indent=2), encoding='utf-8')
    package = ROOT/'outputs/grid_paper_v2/框架论文与确认数据.zip'
    assert not package.exists()
    with zipfile.ZipFile(package, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for src in files:
            archive.write(src, src.relative_to(ROOT).as_posix())
        for name in ('design.json', 'reservation.json', 'review.json', 'per_run.csv', 'per_family.csv', 'GRID62_report.md', 'finished.json', 'payload_manifest.json'):
            src = ROOT/'outputs/grid62'/name
            archive.write(src, src.relative_to(ROOT).as_posix())
    note = '完整框架英文论文V1/V2与来源表、伪代码、单步性质、三日期封存主表、计时反例、科学图、表格独立复算和轻量复现入口完成。V1保留。原冻结确认通过，但FIRST_SAFE_BEST新日期也质量3/3，FRAME对它稳定优势未建立；主张限定为相对FULL有限质量—计算改善与已有具体机制，不扩大为普遍补全必要或GNN创新。尚需作者/目标会议格式和最终文献定位；不是可直接投稿或一键完整仿真复现的认证。'
    for name in ('DECISIONS.md', 'EXPERIMENT_LOG.md'):
        p = REPO/'docs'/name
        p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-13 — framework manuscript V1/V2\n\n'+note+'\n', encoding='utf-8')
    p = REPO/'docs/RESEARCH_STATE.md'
    p.write_text('# Research state — framework manuscript drafted\n\n- **当前交付：** '+note+'\n\n'+p.read_text(encoding='utf-8').split('\n',1)[1], encoding='utf-8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. GRID62全部24条与保存数据审计/计时分解完成，论文完整V2初稿已写入归档。无未结回放、训练或自动跟进。\n2. 当前需要根据目标会议/track精修贡献定位与篇幅；强首个安全候选参照的竞争力和计时变异必须保留，不写全面领先。\n3. 原三封存日期不再调参/选型，不扩大模型或基线菜单。署名、投稿方向与稿件格式需要用户决定；轻量复算可运行，完整仿真还须说明依赖与大文件获取。\n', encoding='utf-8')
    p = REPO/'docs/GRID_PAPER_EVIDENCE.md'
    p.write_text('# Grid paper evidence — framework manuscript\n\n'+note+'\n\n[完整英文初稿](../research_archive/power-grid-2026/protocols/GRID_framework_paper_v1_v2/outputs/grid_paper_v2/manuscript.md) · [新日期确认](../research_archive/power-grid-2026/protocols/GRID62_complete/outputs/grid62/GRID62_report.md) · [计时与强参照边界](../research_archive/power-grid-2026/protocols/GRID_framework_paper_v1_v2/outputs/grid62_timing_review/TIMING_AND_CLAIMS.md)\n\n'+p.read_text(encoding='utf-8').split('\n',1)[1], encoding='utf-8')
    print(json.dumps({'archived_files': len(receipts), 'draft_words': checks['word_count'],
        'package_bytes': package.stat().st_size, 'validation_passed': True}))


if __name__ == '__main__':
    main()
