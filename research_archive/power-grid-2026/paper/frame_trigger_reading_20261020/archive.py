"""Archive the reading record and own notes; original papers stay on D:."""
from pathlib import Path
import hashlib,json,shutil
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/frame_trigger_reading_20261020';REPO=ROOT/'work/GRL-results-integration'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def append(path,key,note):
    old=path.read_text(encoding='utf-8')
    if key not in old:path.write_text(old.rstrip()+'\n\n'+note.strip()+'\n',encoding='utf-8')


def main():
    specs=[
      ('ExpertOp_MEDPOWER2018.pdf','Expert System for topological remedial action discovery in smart grids','MEDPOWER 2018','10.1049/cp.2018.1875',[2,3,4,6,7],['https://github.com/ainetus/ExpertOp4Grid','https://hal.science/hal-01897931/document']),
      ('Evolving_EnergyAI2026.pdf','Evolving power system operator rules for real-time congestion management','Energy and AI 23 (2026), 100672','10.1016/j.egyai.2025.100672',[4,5,6,7,8,9,10,11,12],['https://ai4realnet.eu/wp-content/uploads/2026/01/1-s2.0-S2666546825002046-main.pdf']),
      ('CCMA_eEnergy2025.pdf','Centrally Coordinated Multi-Agent Reinforcement Learning for Power Grid Topology Control','ACM e-Energy 2025, formal 17-page version','10.1145/3679240.3734602',[5,6,7,8,11,17],['https://doi.org/10.1145/3679240.3734602','https://arxiv.org/html/2502.08681v2']),
      ('Bhatia_ICAPS2022.pdf','Tuning the Hyperparameters of Anytime Planning: A Metareasoning Approach with Deep Reinforcement Learning','ICAPS 2022','10.1609/icaps.v32i1.19842',[2,3,4,6,7,8],['https://abhinavbhatia.me/files/BSNZicaps22.pdf','https://ojs.aaai.org/index.php/ICAPS/article/view/19842']),
      ('StopPlanner_AAAI2024.pdf','Stop! Planner Time: Metareasoning for Probabilistic Planning Using Learned Performance Profiles','AAAI 2024','10.1609/aaai.v38i18.29983',[3,4,6,7],['https://ojs.aaai.org/index.php/AAAI/article/download/29983/31725'])]
    items=[]
    for file,title,venue,doi,pages,urls in specs:
        p=OUT/'papers'/file
        items.append(dict(title=title,venue=venue,doi=doi,primary_urls=urls,
                          local_pdf=p.relative_to(ROOT).as_posix(),sha256=sha(p),page_count=len(PdfReader(p).pages),
                          key_pdf_pages_opened_or_text_reviewed=pages,
                          scope='Methods, relevant results and conclusions; full text extracted, not a claim of line-by-line reading of every page or appendix'))
    images=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),actually_viewed=True) for p in sorted((OUT/'qa').glob('*.png'))]
    assert len(images)==6
    receipt=dict(checked_on='2026-10-20',papers=items,visual_pages=images,
                 fresh_web_primary_source_checks=True,
                 unavailable_endpoints=['HAL PDF through web tool','ExpertOp ReadTheDocs through web tool','Elsevier DOI redirect through web tool'],
                 fallbacks='Author repository PDF, project-hosted publisher PDF and local verified formal CCMA PDF',
                 notes='Prior literature notes are retrieval aids only. This record checks primary methods/results and distinguishes single-state from closed-loop claims.',
                 new_training=0,new_simulations=0,new_rules_tested=0)
    (OUT/'source_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    dest=REPO/'research_archive/power-grid-2026/paper/frame_trigger_reading_20261020';dest.mkdir(exist_ok=False)
    for p in [OUT/'相近机制论文_方法结论与FRAME启示.md',OUT/'source_receipt.json',Path(__file__)]:
        shutil.copy2(p,dest/p.name);assert sha(p)==sha(dest/p.name)
    (dest/'README.md').write_text('# 触发机制近邻论文阅读\n\n[逐篇方法、结论与研究判断](相近机制论文_方法结论与FRAME启示.md)。\n\n原PDF与抽取全文保存在D盘outputs/frame_trigger_reading_20261020，不在Git中复制全文；作者/出版链接及本地哈希见source_receipt.json。没有新增训练或仿真。\n',encoding='utf-8')
    collab=ROOT/'论文协作资料/FRAME_补查机制论文阅读_20261020';collab.mkdir(exist_ok=True)
    for p in [OUT/'相近机制论文_方法结论与FRAME启示.md',OUT/'source_receipt.json']:
        shutil.copy2(p,collab/p.name);assert sha(p)==sha(collab/p.name)
    key='FRAME_TRIGGER_READING_20261020'
    note='按用户要求先回读ExpertOp2018、Evolving EnergyAI2026、CCMA2025及ICAPS2022/AAAI2024元推理原文。核对停止/层级排序/追加动作、物理影响图、区域提案与长期协调、计算加执行成本目标及关键表格。区别控制干预门控与搜索补查门控，DQN原文与本地DoubleDQN适配、静态拥塞成功与整周质量、CCMA最佳验证种子测试与全种子均值。不新增规则/训练/回放；未来方向为风险与额外计算价值分开定义，非照搬阈值。'
    for n in ['docs/DECISIONS.md','docs/EXPERIMENT_LOG.md']:
        append(REPO/n,key,'## 2026-10-20 — '+key+'\n\n'+note)
    append(REPO/'docs/NEXT_STEPS.md',key,'## '+key+'\n\n原文方法与结论已核对，材料paper/frame_trigger_reading_20261020。新触发设计先区分任务风险、候选代理奖励及计算价值；当前没有启动下一条规则试验。')
    append(REPO/'research_archive/power-grid-2026/README.md',key,'## '+key+'\n\n'+note+'\n\n[阅读笔记](paper/frame_trigger_reading_20261020/相近机制论文_方法结论与FRAME启示.md)')
    print(json.dumps(dict(papers=len(items),visual_pages=len(images),archive=str(dest),notes_sha256=sha(OUT/'相近机制论文_方法结论与FRAME启示.md')),ensure_ascii=False))


if __name__=='__main__':main()
