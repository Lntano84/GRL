"""Publish source-backed mechanism evidence without altering historical trials."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid85_mechanism';REPO=ROOT/'work/GRL-results-integration'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert read(OUT/'analysis_finished.json')['passed'] and read(OUT/'verification.json')['passed'] and read(OUT/'package_receipt.json')['passed']
    for r in read(OUT/'source_receipt.json')['files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    for r in read(OUT/'additional_details.json')['extra_sources']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    assert sha(OUT/'mechanism_feedback.json.gz')==read(OUT/'package_receipt.json')['sha256']
    # Preserve the policy import chain omitted from the top-level source receipt.
    deps=['work/grid27/policy.py']
    files=list(OUT.iterdir())+list(Path(__file__).parent.glob('*.py'))
    final=dict(files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in files if p.is_file()],
               additional_policy_dependencies=[dict(path=p,sha256=sha(ROOT/p)) for p in deps],new_fits=0,new_simulation=0)
    (OUT/'final_receipt.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8');files.append(OUT/'final_receipt.json')
    dest=REPO/'research_archive/power-grid-2026/protocols/GRID85_mechanism';assert not dest.exists()
    for p in files:
        if not p.is_file():continue
        q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p)
    for p in deps:
        target=dest/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,target)
    collab=ROOT/'论文协作资料/FRAME_机制分析_20261016';collab.mkdir(exist_ok=False)
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,collab/p.name)
    note='GRID85已有日志机制分析完成，无新训练/物理回放。旧13+新6日期，131实际调用/47140查询，137完整反馈状态；1096状态方法组合与131调用经独立无策略导入逻辑重算。初始rho<0.9后仍改变动作12/39、2/12；7个旧日期首次分歧均源于该分支核查，完整周变化3低2高并有1恢复完成。同站等预算增量旧14胜0负、新3胜1负，后者存在更大反向奖励差，不宣称稳定结构优势。固定512在新状态一步一致率26/28高于FRAME23/28；下一步关键为固定预算与反馈核查的闭环比较，尚未启动。GRID81完整控制均值差约81.3%来自预测时间差。原裁决、稿件和模型保留。'
    for n in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-16 — GRID85 mechanism study\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');lines=s.splitlines(keepends=True);p.write_text(lines[0]+'\n- **最新机制证据：** '+note+'\n'+''.join(lines[1:]),encoding='utf-8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. 阅读GRID85机制报告：核查的必要性有具体证据，结构补查增量跨日期不稳定；一步奖励与闭环成本分开。\n2. 下一次有限试验优先比较最终FRAME_GUARD与固定256/512查询的闭环质量和同批控制时间；先单独冻结，不择预算/日期/模型。本轮未启动新回放。\n3. 暂停GNN/RL常规调参。保留GRID84五种子结果、所有不利日期和原稿；按系统/机制实证贡献收敛，尚不宣称普遍结构搜索优势。\n',encoding='utf-8')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID85 机制分析\n\n'+note+'\n\n[完整报告](protocols/GRID85_mechanism/outputs/grid85_mechanism/GRID85_report.md) · [可复算反馈](protocols/GRID85_mechanism/outputs/grid85_mechanism/mechanism_feedback.json.gz)\n',encoding='utf-8')
    print(json.dumps(dict(archive=str(dest),collaboration=str(collab),files=len(final['files'])+1),ensure_ascii=False))
if __name__=='__main__':main()
