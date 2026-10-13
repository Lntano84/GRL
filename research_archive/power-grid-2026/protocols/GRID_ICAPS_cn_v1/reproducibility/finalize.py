"""Archive only the verified bilingual paper material, not rendering runtimes or stale QA pages."""
from pathlib import Path
import hashlib,json,shutil
from zipfile import ZipFile
from pypdf import PdfReader
from lxml import etree
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid_icaps_cn_v1';HERE=Path(__file__).resolve().parent
REPO=ROOT/'work/GRL-results-integration'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if (OUT/'validation.json').exists():raise SystemExit('Completed version exists; do not repeat.')
check=json.loads((OUT/'translation_check.json').read_text())
for rel,h in check['original_receipts'].items():assert sha(ROOT/rel)==h
pdf=OUT/'qa/ICAPS2027_中文完整初稿.pdf';reader=PdfReader(pdf)
alltext='\n'.join(p.extract_text() or '' for p in reader.pages)
for value in ['69.37%','13.11%','83.54%','6.09%','FIRST_SAFE_BEST','198','1733','0.00782%']:
    assert value in alltext,value
assert len(reader.pages)==10
with ZipFile(OUT/'ICAPS2027_中文完整初稿.docx') as z:
    xml=etree.fromstring(z.read('word/document.xml'));styles=etree.fromstring(z.read('word/styles.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    assert len(xml.findall('.//m:oMathPara',ns))==5
    assert len(xml.findall('.//w:tbl',ns))==3
    assert not styles.findall('.//w:pBdr',ns)
    assert z.testzip() is None
dest=OUT/'ICAPS2027_中文通读版.pdf';shutil.copy2(pdf,dest)
(OUT/'reproducibility').mkdir(exist_ok=True)
for f in ['build.py','render.py','prepare_renderer.py','finalize.py','manuscript_zh.md']:
    shutil.copy2(HERE/f,OUT/'reproducibility'/f)
shutil.copy2(HERE/'render_dependency/receipt.json',OUT/'reproducibility/render_dependency_receipt.json')
dump(OUT/'reproducibility/engineering_notes.json',dict(renderer_initial_error='First log decoder assumed UTF-8 while child used GBK. PDF/PNG output existed. Original log and DOCX retained in work/grid_cn_draft/revisions/v0_render. Child PYTHONUTF8/PYTHONIOENCODING explicitly set; final render completed.',layout_repairs=['Remove blue Title border inherited from base template','Use valid page-number fields','Keep short policy table and algorithm block together','Set sufficient minimum spacing for native fractions/sum equations','Restrict caption styling to exact caption lines'],no_scientific_changes=True))
dump(OUT/'validation.json',dict(passed=True,complete_translation_sections=11,tables=3,display_equations=5,native_editable_word_equations=True,main_and_confirmation_numbers_equal=True,chinese_characters=check['chinese_characters'],reading_pages=len(reader.pages),visual_review='All final page PNGs 1-10 reviewed at original size. No missing glyphs, overlap, clipped equations or broken table rows found.',originals_unchanged=True,new_simulations=0,new_fits=0,scope='Full Chinese manuscript and document-format QA, not a new experimental verdict or independent scientific peer review.'))
files=[p for p in OUT.rglob('*') if p.is_file() and 'qa' not in p.parts]
dump(OUT/'manifest.json',dict(files=[dict(path=str(p.relative_to(OUT)).replace('\\','/'),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
archive=REPO/'research_archive/power-grid-2026/protocols/GRID_ICAPS_cn_v1'
if archive.exists():raise SystemExit('Archive already exists.')
archive.mkdir(parents=True)
for p in files+[OUT/'manifest.json']:
    q=archive/p.relative_to(OUT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
entry='''\n## 2026-10-13 — complete Chinese author manuscript\n\n用户要求先用中文通读重写，已制作完整中文正文、可编辑Word/五项原生OMML公式、中文框架图、10页同版PDF与Markdown。11个一级章节/全部二级章节、三表数值对应英文初稿；全部页面原尺寸视觉检查通过，旧英文和GRID61/62结果哈希不变。无新运行或拟合。Windows运行包缺转换器，使用官方校验后的任务目录行政解包作隔离排版依赖，未安装桌面软件或更改系统关联；大依赖不入Git。首轮日志GBK/UTF8解码错误和排版修订保留，最终渲染正常。下一步由作者中文重写，再同步英文；当前版仍为AI辅助工作稿，不是正式AAAI版或独立科学审计。\n'''
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8').rstrip()+'\n'+entry,encoding='utf-8')
p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');lines=s.splitlines();lines.insert(2,'**最新交付：完整中文作者稿与可编辑Word/同版PDF已完成，作者先中文通读重写，再整理英文。三表数字、五公式和原实验材料核验通过；不新增实验。**');p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
p=REPO/'docs/NEXT_STEPS.md';s=p.read_text(encoding='utf-8').replace('先由作者阅读并自主重写新初稿，提供署名/单位标准。','先由作者阅读并自主重写完整中文版，提供署名/单位标准，再以确认中文稿整理英文。').replace('GRID_ICAPS_draft_v1`，旧稿保留。','GRID_ICAPS_draft_v1` 与 `GRID_ICAPS_cn_v1`，旧稿保留。');p.write_text(s,encoding='utf-8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf-8').rstrip()+'\n'+entry+'\n[完整中文作者稿](protocols/GRID_ICAPS_cn_v1/README.md)\n',encoding='utf-8')
print(json.dumps(dict(output=str(OUT),archived_files=len(files)+1,reading_pages=len(reader.pages),old_data_unchanged=True),ensure_ascii=False))
