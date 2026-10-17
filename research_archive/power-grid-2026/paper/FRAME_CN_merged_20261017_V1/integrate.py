from pathlib import Path
import shutil,json,hashlib
root=Path('D:/Codex/GRL研究');repo=root/'work/GRL-results-integration'
source=root/'论文协作资料/FRAME_中文合稿_20261017_V1'
dest=repo/'research_archive/power-grid-2026/paper/FRAME_CN_merged_20261017_V1'
assert not dest.exists(),'Preserve existing delivery'
shutil.copytree(source,dest)
for f in ['finalize.py','integrate.py']:
    shutil.copy2(root/'work/frame_manuscript_merge_20261017'/f,dest/f)
index=repo/'docs/PAPER_MATERIALS_INDEX.md'
index.write_text('''# FRAME论文材料入口

## 当前中文全文

[中文合稿V1 Word](../research_archive/power-grid-2026/paper/FRAME_CN_merged_20261017_V1/FRAME_ICAPS2027_中文合稿_V1.docx) · [在线阅读全文](../research_archive/power-grid-2026/paper/FRAME_CN_merged_20261017_V1/FRAME_ICAPS2027_中文合稿_V1.md) · [阅读说明](../research_archive/power-grid-2026/paper/FRAME_CN_merged_20261017_V1/阅读说明.md) · [BibTeX](../research_archive/power-grid-2026/paper/FRAME_CN_merged_20261017_V1/references.bib)

采用用户已确认摘要、引言表达修订V3、方法文字V2、实验文字V1、相关工作引文修订V2和参考文献核定V1；补齐结论与局限，统一章节及正文引文。章节顺序为摘要、引言、方法、实验、结论、相关工作、局限、参考文献。保留全部原稿和用户编辑版本。

当前Word共12页，52个原生可编辑公式对象，无图表。已逐页检查渲染和公式显示；未套用正式英文投稿模板。实验数值与裁决沿用原结果，无新增采集、回放、训练或参数调整。

## 研究记录与历史稿件

[研究记录总索引](RECORDS_INDEX.md)覆盖实验、审计、冻结清单、失败恢复和历史稿件；[逐文件清单](../research_archive/record_sync_20261017/records_index.jsonl.gz)提供原始路径、SHA256与仓库位置。2026年10月17日补齐快照早于当前合稿最后的公式显示和分页调整，以上当前全文目录保存最终版本及校验值。
''',encoding='utf-8')
p=repo/'README.md';s=p.read_text(encoding='utf-8')
s=s.replace('[研究记录总索引](docs/RECORDS_INDEX.md) · [当前研究状态](docs/RESEARCH_STATE.md)','[中文论文全文与材料](docs/PAPER_MATERIALS_INDEX.md) · [研究记录总索引](docs/RECORDS_INDEX.md) · [当前研究状态](docs/RESEARCH_STATE.md)',1)
p.write_text(s,encoding='utf-8')
p=repo/'docs/NEXT_STEPS.md'
p.write_text('''# Next steps

1. 用户通读中文全文合稿V1，优先检查新增结论、局限及章节衔接。正文采用已确认摘要与最新逐节文字；入口见PAPER_MATERIALS_INDEX。
2. 按用户修改意见逐项修订，保留用户稿、逐节稿和当前交付版本，不覆盖历史内容。
3. 当前仅文字合稿，无图表或新实验。英文翻译、正式ICAPS模板和后续配图等待用户确认后推进。
4. 全部应归档研究记录已补齐索引与校验；环境、第三方下载及大型原始二进制留本地，范围见RECORDS_INDEX。
''',encoding='utf-8')
p=repo/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8')
first,rest=s.split('\n',1)
p.write_text(first+'\n\n- **2026-10-17写作与资料：** 中文全文合稿V1已完成，按摘要、引言、方法、实验、结论、相关工作、局限、参考文献组织；52个原生公式、12页，无图表，逐页渲染检查完成。实验数值和既有裁决不改，原稿全部保留，入口docs/PAPER_MATERIALS_INDEX.md。研究记录核对23885项，7765项已有相同归档，新增16120项存入199个校验ZIP，入口docs/RECORDS_INDEX.md。本次没有新训练或科学实验。\n'+rest,encoding='utf-8')
with (repo/'docs/EXPERIMENT_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n## 2026-10-17 — 中文全文合稿V1\n\n合并最新已确认摘要与逐节文字，采用相关工作引文V2和9项核定参考文献，补写结论与局限，统一章节、引文和公式。147段、52个原生OMML公式、无图表；12页逐页渲染检查通过，A*及下标显示已检查，源文件和最终文件SHA256留存。当前只供中文通读修改，不称正式ICAPS模板稿。原用户编辑和历史文件保留；未新增实验或改变数值裁决。工作区、协作资料和仓库交付文件逐字节一致。\n')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
assert all(hashlib.sha256((dest/n).read_bytes()).hexdigest()==v for n,v in hashes.items())
(root/'outputs/frame_cn_merged_20261017_v1/归档交付一致性.json').write_text(json.dumps({'collaboration':str(source),'repository':str(dest),'byte_identical':True,'sha256':hashes},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'repository_delivery':str(dest),'all_copies_identical':True},ensure_ascii=False))
