"""Close and archive a new author draft without overwriting completed predecessors."""
from pathlib import Path
import hashlib,json,shutil,zipfile,re
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid_icaps_draft_v1'
REPO=ROOT/'work/GRL-results-integration'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if (OUT/'validation.json').exists():raise SystemExit('Completed version exists.')
check=json.loads((OUT/'numeric_check.json').read_text())
assert check['passed'] and check['previous_artifacts_unchanged']
receipts=json.loads((OUT/'source_receipts.json').read_text())
for rel,h in receipts['previous_files'].items():assert sha(ROOT/rel)==h
v=json.loads((OUT/'pdf_reading_layout_check.json').read_text())
v.update(visual_review_pending=False,visual_review='All 9 pages reviewed as contact sheet; pages 4, 6 and 9 inspected at full size. No clipping or table/diagram overlap observed.')
dump(OUT/'pdf_reading_layout_check.json',v)
dump(OUT/'typesetting_status.json',dict(official_style_unmodified=True,official_kit_year=2027,official_kit_source='https://aaai.org/authorkit27/',working_mode='preprint',built_in_compiler='XeTeX',formal_compile_success=False,diagnostic='pdfTeX is required to compile this document.',formal_aaai_pdf_available=False,formal_page_count_verified=False,reading_pdf_available=True,reading_pdf_pages=9,reading_pdf_is_submission_format=False,source_static_review='Numbered main/date tables match manuscript Table 1/2; shared-policy table unnumbered; author-year references alphabetized; no prohibited geometry/font tweaks added. Not a full compiler qualification.'))
dump(OUT/'validation.json',dict(passed_material_checks=True,words_in_english_source=4340,main_macro_rows=8,date_rows=28,previous_artifacts_unchanged=True,new_simulations=0,new_fits=0,reading_pdf_pages=9,reading_pdf_visual_check=True,formal_compile_success=False,target='ICAPS 2027 long paper',target_confirmed_by_user=True,scope='Guide-based working draft and offline material preparation; not submission-ready certification or a new research verdict.'))
text=(OUT/'manuscript_en.md').read_text(encoding='utf-8')
assert 'Figure 1' in text and '69.37%' in text and '83.54%' in text
assert not re.search(r'turn\d+(?:search|view)\d+',text)
# Keep compact scripts and retrieval receipt; do not copy third-party full archive or browser pages.
(OUT/'reproducibility').mkdir(exist_ok=True)
for f in ['fetch_sources.py','build.py','render_pdf.py','finalize.py']:
    shutil.copy2(ROOT/'work/grid_icaps_draft'/f,OUT/'reproducibility'/f)
shutil.copy2(ROOT/'work/grid_icaps_draft/reference_cache/retrieval.json',OUT/'reproducibility/retrieval.json')
dump(OUT/'reproducibility/engineering_notes.json',dict(csv_bom='Initial builder stopped before writing manuscript because CSV headers contain a BOM. Retained build_v0_csv_bom.py; reader changed to utf-8-sig. No experiment records or numerical inputs changed.',compile='Official 2027 style rejects built-in XeTeX. Style not patched; reading PDF is an explicitly separate layout. No false formal-compilation success claim.'))
files=[p for p in OUT.rglob('*') if p.is_file() and 'qa' not in p.parts and p.suffix.lower()!='.zip']
dump(OUT/'manifest.json',dict(files=[dict(path=str(p.relative_to(OUT)).replace('\\','/'),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
zpath=OUT/'ICAPS2027_初稿与改写材料.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files+[OUT/'manifest.json']:z.write(p,str(p.relative_to(OUT)))
with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
archive=REPO/'research_archive/power-grid-2026/protocols/GRID_ICAPS_draft_v1'
if archive.exists():raise SystemExit('Archive already exists; do not duplicate.')
archive.mkdir(parents=True)
for p in files+[OUT/'manifest.json']:
    dest=archive/p.relative_to(OUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
dump(archive/'archive_receipt.json',dict(source=str(OUT),files=len(files)+1,full_data_not_repeated=True,zip_sha256=sha(zpath),formal_compile_success=False))
entry='''\n## 2026-10-13 — ICAPS 2027 guide-based first draft\n\n用户明确确认目标ICAPS2027，按其提供的LAMDA-NeSy Research-Starter-Kit公开写作章节完成4340词英文初稿、中文导读、写作要求映射和两个月安排。正文以渐进搜索/变电站补全为主线，GNN辅助；保留FIRST的新日期竞争力、无搜索日期/共同失败和计时变异。八行统一主表复算通过，旧V1/V2与GRID61/62原材料不变；无新仿真或训练。官方AAAI2027样式原样保留，但应用内XeTeX被其PDFLaTeX要求拒绝，未宣称正式格式/8页认证；另交付9页可视核对的单栏PDF阅读版。当前是AI辅助作者重写工作稿，非直接提交或匿名认证。摘要/全文截止12/7与12/14 UTC-12，拟在最终政策和作者文本确认后精修。\n'''
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8').rstrip()+'\n'+entry,encoding='utf-8')
p=REPO/'docs/RESEARCH_STATE.md';old=p.read_text(encoding='utf-8');ls=old.splitlines();ls.insert(2,'**最新：用户确认ICAPS2027为暂定目标，指南式完整初稿、中文导读和两个月准备计划已完成；无新实验/训练。PDF为9页阅读布局，正式AAAI2027 PDFLaTeX格式尚未编译认证。GNN仍辅助，FIRST竞争力和计时变异保留。**');p.write_text('\n'.join(ls)+'\n',encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps\n\n1. 用户已确认暂定ICAPS2027长文；先由作者阅读并自主重写新初稿，提供署名/单位标准。材料在 `research_archive/power-grid-2026/protocols/GRID_ICAPS_draft_v1`，旧稿保留。\n2. 精修具体搜索贡献、最邻近结构补全方法定位与FIRST竞争解释；时间点估计不得超出重复变异的支持。没有自动新实验/训练任务。\n3. 正式AAAI2027样式需要PDFLaTeX；当前应用内XeTeX不支持该模板，阅读PDF不算8页/匿名/AI政策认证。最终准备真实匿名稿、第三方资产依赖与当届政策核对。\n4. 内部摘要12/05、全文12/10目标；官方截止分别12/07、12/14 23:59 UTC-12。自动跟进仍暂停。\n''',encoding='utf-8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf-8').rstrip()+'\n'+entry+'\n[ICAPS2027初稿与导读](protocols/GRID_ICAPS_draft_v1/README.md)\n',encoding='utf-8')
print(json.dumps(dict(archived=str(archive),small_files=len(files)+1,zip_bytes=zpath.stat().st_size,zip_crc_passed=True,numeric_and_visual_checks=True,formal_pdf_qualification=False)))
