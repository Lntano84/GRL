from pathlib import Path
import json,hashlib,shutil,re
from docx import Document
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/grid_icaps_cn_v2'
rec=json.loads((OUT/'source_receipt.json').read_text(encoding='utf-8'))
assert hashlib.sha256(Path(rec['user_docx']).read_bytes()).hexdigest()==rec['user_docx_sha256']
doc=Document(OUT/'ICAPS2027_中文重写稿_v2.docx')
heads=[p.text for p in doc.paragraphs if p.style.name=='Heading 1']
assert heads==['摘要','1 引言','2 方法','3 实验','4 结论','5 相关工作','6 局限','参考文献']
assert len(doc.tables)==3
body='\n'.join(p.text for p in doc.paragraphs)
assert not any(s in body for s in ['不是','而不是','不意味着','不能声称','尚不能','不代表'])
pdf=OUT/'qa/ICAPS2027_中文重写稿_v2.pdf';pages=len(PdfReader(pdf).pages)
assert pages==8
check=json.loads((OUT/'rewrite_check.json').read_text(encoding='utf-8'))
for name,sha in check['original_receipts'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
validation={'passed':True,'user_document_unchanged':True,'user_document_sha256':rec['user_docx_sha256'],'sections':heads,'tables':3,'native_equations':5,'reading_pages':pages,'visual_review':'All final pages 1-8 inspected at original rendered size. No clipping or table corruption observed.','main_numerical_tables_unchanged':True,'new_simulations':0,'new_fits':0,'scope':'Editorial rewrite and document verification; does not change experiment verdicts.'}
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
notes='''# 中文重写稿 V2

按作者已修改的前四页调整写法，重新组织完整正文。作者修改文件保持原路径与内容，V2为独立稿。

章节顺序：摘要、引言、方法、实验、结论、相关工作、局限、参考文献。

本轮主要改动：
- 摘要直接交代结构问题、方法思想与量化收益。
- 引言按背景、相关进展、具体缺口、本文方法与贡献推进。
- 方法采用概述、定义、模块、性质的顺序，先讲模块用途再给公式。
- 实验合并设置与结果，围绕整体收益、补全消融和停止策略分析。
- 删除散布在正文中的重复免责句与工程流水账，重要适用范围集中于局限。
- 三张表的数值和全部五个公式保持原有定义；FIRST的竞争结果和计时波动继续如实呈现。

## 实际阅读的指南正文

通过浏览器访客页面打开并阅读了以下正文与相关分节，范围为本次重写所需的写作说明；未声称通读仓库全部科研与投稿子章。

- [总纲](https://dy8q0bnq8y.feishu.cn/wiki/NsK3wnWeiivKNukNJOqcd985nTY)：围绕问题、已有工作、本文方法与证据组织。
- [Abstract](https://dy8q0bnq8y.feishu.cn/wiki/Tz2pwvxTyiBK0zkCVszcZmJ3nwd)：突出问题和思想，用量化结果回应问题。
- [Introduction](https://dy8q0bnq8y.feishu.cn/wiki/ULATwBu2MiLqqRkXxV6c2S7AnPd)：背景到缺口再到工作，每段围绕一个主题。
- [Methods](https://dy8q0bnq8y.feishu.cn/wiki/CtiAwh1eOiqOfCkPKVLc0WsxnZg)：先总览再定义，各模块解释目的、输入、流程、输出与设计理由。
- [Experiments](https://dy8q0bnq8y.feishu.cn/wiki/KHgSw2MkrikxlskGe5qclkdbn2c)：主结果对应核心结论，消融解释设计，分析解释表现差异。
- [Related Works](https://dy8q0bnq8y.feishu.cn/wiki/CLSAwWm6qifjwQkTd82czgMNnYf)：按研究路线分类，并说明本文与既有工作的联系。
- [Reference](https://dy8q0bnq8y.feishu.cn/wiki/A4q2wSnmRi91rykHke1cQIhnnAh)：检查条目类型、名称、作者、会议与期卷页等信息的一致性。

网页总纲示例将相关工作放在引言之后；本稿采用作者指定的章节顺序，将相关工作置于结论之后。正文按作者的直述风格改写，数据和比较结论保持一致。Word为作者编辑稿，8页为中文阅读布局。

输出：ICAPS2027_中文重写稿_v2.docx；同文Markdown。内部PDF和页面PNG仅作排版核验。
'''
(OUT/'README.md').write_text(notes,encoding='utf-8')
repro=OUT/'reproducibility';repro.mkdir(exist_ok=True)
for name in ['prepare.py','build.py','render.py','finalize.py','manuscript_zh.md','manuscript_ready.md','user_edits.diff']:
    p=HERE/name
    if p.exists():shutil.copy2(p,repro/name)
(repro/'engineering_notes.md').write_text('首次文风检查捕获引言中的一处“不意味着”，改为直接描述未查询配置。初次排版末页仅余两条参考文献，删除多余副标题并微调行距后为8页。用户原稿与旧实验文件未变。未新安装或训练。',encoding='utf-8')
files=[p for p in OUT.rglob('*') if p.is_file() and 'qa' not in p.parts and p.name!='manifest.json']
(OUT/'manifest.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},ensure_ascii=False,indent=2),encoding='utf-8')
repo=ROOT/'work/GRL-results-integration'
dest=repo/'research_archive/power-grid-2026/protocols/GRID_ICAPS_cn_v2'
assert not dest.exists()
dest.mkdir(parents=True)
for p in files+[OUT/'manifest.json']:
    q=dest/p.relative_to(OUT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
entry='''\n\n## 2026-10-13 — Chinese manuscript V2 following author edits

按用户前四页删改重写完整中文论文，改用直接阐述问题、方法和收益的语气。按用户顺序设置摘要/引言/方法/实验/结论/相关工作/局限/参考文献；重读Research-Starter-Kit七项相关正文。删除重复防御性对比句和审计流水账，保留三表数据、失败与竞争结果及必要计量定义。用户编辑DOCX哈希不变，新V2独立保存；5项可编辑公式，8页阅读排版全部检查。无新实验或训练，科学裁决不变。\n'''
for f in ['docs/DECISIONS.md','docs/EXPERIMENT_LOG.md','research_archive/power-grid-2026/README.md']:
    p=repo/f
    with p.open('a',encoding='utf-8') as out:out.write(entry)
p=repo/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');line='**最新交付：中文重写稿V2已按作者删改另存，正文以问题、设计与收益为主线，用户原稿保持不变；章节按作者指定顺序，三表数据不变。**\n\n';s=s.replace('\n\n','\n\n'+line,1);p.write_text(s,encoding='utf-8')
p=repo/'docs/NEXT_STEPS.md';s=p.read_text(encoding='utf-8');s=s.replace('先由作者阅读并自主重写完整中文版','先由作者继续修改中文重写稿V2（保留其手改V1）');p.write_text(s,encoding='utf-8')
print(json.dumps({'validation':validation,'archive':str(dest),'files':len(files)+1},ensure_ascii=False))
