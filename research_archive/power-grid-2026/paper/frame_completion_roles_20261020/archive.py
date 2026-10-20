"""Archive the bounded saved-record analysis, without touching historical protocols."""
from pathlib import Path
import hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[2]
W=Path(__file__).parent
OUT=ROOT/'outputs/frame_completion_roles_20261020'
REPO=ROOT/'work/GRL-results-integration'
KEY='FRAME_COMPLETION_ROLES_20261020'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def append(p,key,note):
    s=p.read_text(encoding='utf-8')
    if key not in s:p.write_text(s.rstrip()+'\n\n'+note+'\n',encoding='utf-8')


def main():
    review=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    receipt=json.loads((OUT/'source_receipt.json').read_text(encoding='utf-8'))
    assert review['passed'] and receipt['all_source_hashes_unchanged']
    assert all(sha(ROOT/p)==h for p,h in receipt['sources'].items())
    dest=REPO/'research_archive/power-grid-2026/paper/frame_completion_roles_20261020'
    dest.mkdir(exist_ok=False)
    files=[p for p in OUT.iterdir() if p.suffix in ('.json','.csv','.md') and p.name!='analyze_stdout.json']
    copied=[]
    for p in files+[W/'analyze.py',W/'archive.py']:
        shutil.copy2(p,dest/p.name);assert sha(p)==sha(dest/p.name)
        copied.append(dict(name=p.name,sha256=sha(p)))
    collab=ROOT/'论文协作资料/FRAME_补查目的分解_20261020';collab.mkdir(exist_ok=True)
    for p in files:
        shutil.copy2(p,collab/p.name);assert sha(p)==sha(collab/p.name)
    note='仅保存数据分析：179次实际FRAME调用中109次补查，4次仍过载、83次0.9到1、22次低于0.9；对应预测奖励改善1/4、21/83、10/22。删除补查后4个失去质量的场景，首次实际动作分歧3个已低于0.9，不能据不过载取消补查。低ρ小改进对应的长期成本有正反例；触发、查询位置与动作采用分开。没有新预测、规则试验或训练。'
    for name in ['docs/DECISIONS.md','docs/EXPERIMENT_LOG.md']:
        append(REPO/name,KEY,'## 2026-10-20 — '+KEY+'\n\n'+note)
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8')
    if KEY not in s:
        lines=s.splitlines();s=lines[0]+'\n\n- **'+KEY+'：** '+note+'\n\n'+'\n'.join(lines[1:]).lstrip()+'\n';p.write_text(s,encoding='utf-8')
    p=REPO/'docs/NEXT_STEPS.md';s=p.read_text(encoding='utf-8')
    s=s.replace('1. GRID93首条触发假设已完成并否决：站内候选成对平均支持不足不能可靠否决补查。原13质量13→12，2035-07-09_2于1906步提前结束；G1/G2质量保持。先向用户汇报尾部关键候选风险、闭环查询反增及触发/动作采用的区别；尚未找到可替代原规则的新触发，不自动追加规则或扫描阈值。',
        '1. 补查目的分解已检查：不能把“已有0.9余量”直接作为取消补查条件；已有质量正例与成本反例均包含低ρ小改进。下一步优先对照既有风险缓解、低ρ有益与低ρ不利的保存状态，查完整预测负载分布、经济分量和共同控制器作用。尚未执行新增预测或改变动作采用规则；不扫阈值、不增模型。')
    p.write_text(s,encoding='utf-8')
    append(REPO/'research_archive/power-grid-2026/README.md',KEY,'## '+KEY+'\n\n'+note+'\n\n[探索报告](paper/frame_completion_roles_20261020/补查目的分解_探索报告.md)。')
    (OUT/'complete.json').write_text(json.dumps(dict(completed=True,analysis_only=True,
        new_rule_validated=False,archive=str(dest),collaboration_copy=str(collab),files=copied),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(OUT/'complete.json',dest/'complete.json')
    shutil.copy2(OUT/'complete.json',collab/'complete.json')
    print(json.dumps(dict(archive=str(dest),files=len(copied),report_sha256=sha(OUT/'补查目的分解_探索报告.md')),ensure_ascii=False))


if __name__=='__main__':main()
