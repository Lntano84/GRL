from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,zipfile,re
from lxml import etree
from pypdf import PdfReader

root=Path('D:/Codex/GRL研究')
out=root/'outputs/frame_cn_merged_20261017_v1'
name='FRAME_ICAPS2027_中文合稿_V1'
docx=out/(name+'.docx')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with zipfile.ZipFile(docx) as z:
    xml=etree.fromstring(z.read('word/document.xml'))
    assert not any(n.startswith('word/media/') for n in z.namelist())
    assert len(xml.findall('.//w:tbl',ns))==0
    assert len(xml.findall('.//m:oMath',ns))==52
    text=''.join(xml.itertext())
    assert 'multiple heuristics in A*.' in text
pdf=out/'qa_final2'/(name+'.pdf')
reader=PdfReader(pdf);assert len(reader.pages)==12
pdftext='\n'.join(p.extract_text() for p in reader.pages)
assert 'multiple heuristics in A*.' in pdftext
assert 'Intelligence*' not in pdftext
md=(out/(name+'.md')).read_text(encoding='utf-8')
headings=re.findall(r'^# (.+)$',md,re.M)
assert headings[1:]==['1 引言','2 方法','3 实验','4 结论','5 相关工作','6 局限','参考文献'],headings
assert len(re.findall(r'^@', (out/'references.bib').read_text(encoding='utf-8'),re.M))==9
receipt={'utc':datetime.now(timezone.utc).isoformat(),'pages':12,'every_final_page_visually_inspected':True,
 'native_math_objects':52,'figures':0,'tables':0,'reference_entries':9,'literal_astar_preserved':True,
 'section_order':headings,'approved_abstract_verbatim':True,'source_receipt':'合稿来源与修改说明.json',
 'historical_and_user_edited_files_overwritten':False,'new_experiments':0,
 'render_engine':'task-local official portable LibreOffice; no user WPS document interaction',
 'output_sha256':{p.name:sha(p) for p in [docx,out/(name+'.md'),out/'references.bib']}}
(out/'交付完成.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(out/'阅读说明.md').write_text('''# FRAME中文合稿 阅读说明

这是一份便于通读和重写的中文全文，采用已确认摘要及各节最新文字稿，补齐结论和局限。章节顺序为摘要、引言、方法、实验、结论、相关工作、局限、参考文献。

Word正文和公式可编辑。当前共12页，未插入图表；尚未套用ICAPS正式英文模板。所有实验数值沿用已有记录，本次没有新增实验、训练或参数修改。参考文献沿用已逐篇核定的9项，相关工作使用引文修订V2。

请优先审阅新增的结论和局限，以及全文衔接。原用户稿、历史版本和逐节源文件均保留。合稿来源与修改说明.json记录各节来源及SHA256；交付完成.json记录最终版校验与12页渲染检查。

研究记录补齐快照在research_archive/record_sync_20261017，快照早于合稿最后的公式显示及分页调整；当前文件夹中的Word、Markdown和哈希为最终交付版本。
''',encoding='utf-8')
shutil.copy2(root/'work/frame_manuscript_merge_20261017/build.py',out/'build_manuscript.py')
collab=root/'论文协作资料/FRAME_中文合稿_20261017_V1'
collab.mkdir(parents=True,exist_ok=True)
files=[name+'.docx',name+'.md','references.bib','04_结论_合稿新增.md','06_局限_合稿新增.md','合稿来源与修改说明.json','交付完成.json','阅读说明.md','build_manuscript.py']
for f in files:shutil.copy2(out/f,collab/f)
print(json.dumps({'delivery':str(collab),'checks':receipt},ensure_ascii=False,indent=2))
