from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from docx import Document
import json,hashlib,shutil,csv,unicodedata
HERE=Path(__file__).resolve().parent
OUT=Path('D:/Codex/GRL研究/outputs/grid_icaps_en_from_cn_v2')
CN=Path('D:/Codex/GRL研究/outputs/grid_icaps_cn_v2/ICAPS2027_中文重写稿_v2.docx')
EN=OUT/'ICAPS2027_English_from_author_CN_V2.docx'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
check=json.loads((OUT/'translation_check.json').read_text(encoding='utf-8'))
assert sha(CN)==check['source_sha256']
c,e=Document(CN),Document(EN)
assert len(c.tables)==len(e.tables)==3
for i in range(1,3):
 for row in e.tables[i].rows[1:]:
  for cell in row.cells:
   assert cell.text
ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
xt=lambda p:E.fromstring(ZipFile(p).read('word/document.xml'))
ct,et=xt(CN),xt(EN)
norm=lambda t:[unicodedata.normalize('NFKC',s) for s in t.xpath('//m:oMathPara//m:t/text()',namespaces=ns)]
assert norm(ct)==norm(et)
assert len(et.xpath('//m:oMathPara',namespaces=ns))==5
main=[['development',69.37,13.11,.372,4,4],['confirmation',83.54,6.09,.014,3,3]]
for scope,q,t,cost,passed,total in main:
 row=next(r for r in e.tables[1].rows if r.cells[0].text.lower()==scope and r.cells[1].text=='FRAME')
 assert float(row.cells[4].text.strip('%'))==t
 assert float(row.cells[5].text.strip('%'))==q
 assert float(row.cells[6].text.strip('%'))==cost
 assert row.cells[2].text==f'{passed}/{total}'
assert all(x in e.paragraphs[74].text for x in ['1733','256','26','198','309','909','454'])
with (OUT/'figures/figure_data.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.writer(f);w.writerow(['scope','query_savings_pct','controller_savings_pct','max_full_week_cost_increase_pct','quality_passed','quality_dates']);w.writerows(main)
figures={name:{'svg_sha256':sha(OUT/'figures'/f'{name}.svg'),'pdf_sha256':sha(OUT/'figures'/f'{name}.pdf'),'png_sha256':sha(OUT/'figures'/f'{name}.png')} for name in ['framework_en','results_en','completion_case_en']}
old=HERE/'final_render_6';new=HERE/'final_render_7'
unchanged=[i for i in range(1,10) if sha(old/f'page-{i}.png')==sha(new/f'page-{i}.png')]
assert unchanged==[1,2,3,4,5,7,8,9],unchanged
receipt={**check,'output_sha256':sha(EN),'rendered_pages':9,'all_pages_visually_inspected':True,'final_revision_identical_visual_pages':unchanged,'final_changed_page_inspected':[6],'guide_url':'https://dy8q0bnq8y.feishu.cn/wiki/OGpcw6zaRiQ6OwkNxfYcl7IKnBh','guide_sections_read':['overview','teaser','method workflow and examples','result tables','analysis curves','palette','shadow and shape','vector icons','workflow tools','plotting libraries','AI draft','vector format and licensing'],'figure_data_checked_against_author_source':True,'publication_figure_width_inches':7.2,'figures':figures,'source_unchanged':True}
(OUT/'validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
dest=OUT/'scripts';dest.mkdir(exist_ok=True)
for name in ['build.py','translations.py','figure_draw.py','finalize.py']:
 shutil.copy2(HERE/name,dest/name)
shutil.copy2(HERE/'artifact.md',dest/'artifact.md')
print(json.dumps({'source_unchanged':True,'pages':9,'figures':len(figures),'visual_identical_pages':unchanged},ensure_ascii=False))
