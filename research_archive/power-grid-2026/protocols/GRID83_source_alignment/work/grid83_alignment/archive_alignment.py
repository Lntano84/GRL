"""Append-only alignment packet; excludes third-party PDF and raw traces."""
from pathlib import Path
import hashlib, json, shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid83_alignment'
REPO=ROOT/'work/GRL-results-integration'
DEST=REPO/'research_archive/power-grid-2026/protocols/GRID83_source_alignment'
COLLAB=ROOT/'论文协作资料/FRAME_ICAPS2022对标与论证_20261016'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not DEST.exists()
    receipt=json.loads((OUT/'source_receipt.json').read_text(encoding='utf-8'))
    for r in receipt['sources']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    assert receipt['new_training']==receipt['new_simulation']==0
    COLLAB.mkdir(parents=True,exist_ok=False)
    files=[p for p in OUT.iterdir() if p.is_file()]+list(Path(__file__).parent.glob('*.py'))
    entries=[]
    for p in files:
        target=DEST/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        assert sha(target)==sha(p)
        entries.append(dict(source=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
        if p.parent==OUT:shutil.copy2(p,COLLAB/p.name)
    (DEST/'archive_receipt.json').write_text(json.dumps(dict(files=entries,excluded=['Downloaded original paper PDF and raster reading previews; retained locally with URL/hash receipt.','Raw historical call logs already retained in workspace.']),ensure_ascii=False,indent=2),encoding='utf-8')
    note='GRID83原文/作者代码对标与历史行为归因完成，无新训练/回放。META_CONTEXT1在重复0的30次搜索调用中27次首决策委托FRAME_GUARD；META_CONTEXT0为4/32。适配含完整FRAME宏动作，原共同校准质量未匹配、所选费用0如实保留。贡献主线收敛为无需额外元策略训练的分阶段预测核查与实际完整控制提速，GNN/学习门控辅助；中文替换段落独立保存，不覆盖用户稿。新增种子尚未启动，种子研究只能检验该电网适配的训练波动。'
    state=REPO/'docs/RESEARCH_STATE.md'
    text=state.read_text(encoding='utf-8');lines=text.splitlines(keepends=True)
    state.write_text(lines[0]+'\n- **最新对标：** '+note+'\n'+''.join(lines[1:]),encoding='utf-8')
    for name in ('DECISIONS.md','EXPERIMENT_LOG.md'):
        p=REPO/'docs'/name
        p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-16 — GRID83 source alignment and seed attribution\n\n'+note+'\n',encoding='utf-8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. 使用GRID83_source_alignment中的原文对标和中文替换段落更新下一版主稿；原手写稿不覆盖。\n2. 主方法FRAME_GUARD；GRID81支持同电网新生成时序上的效果，GRID82支持两冻结种子同批比较，不混算计时。\n3. 会议对照标为ICAPS2022内核电网增强适配，披露FRAME宏动作使用、原校准质量匹配False及费用0。\n4. 若补种子，仅在运行前冻结有限方案后检验现有适配波动；不得择种子、删委托动作来削弱对照或在暴露日期挑费用。本次没有启动新拟合/回放。\n',encoding='utf-8')
    p=REPO/'research_archive/power-grid-2026/README.md'
    p.write_text(p.read_text(encoding='utf-8')+'\n## GRID83 原文对标与贡献重写\n\n'+note+'\n\n[对标方案](protocols/GRID83_source_alignment/outputs/grid83_alignment/ICAPS2022对标与贡献重写方案.md) · [中文替换段落](protocols/GRID83_source_alignment/outputs/grid83_alignment/中文论证替换段落_供用户审改.md)\n',encoding='utf-8')
    print(json.dumps(dict(files=len(entries),collaboration=str(COLLAB),archive=str(DEST)),ensure_ascii=False))

if __name__=='__main__':main()
