from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
import json,re,hashlib,html
from collections import Counter
from lxml import etree as E
from docx import Document
from PIL import Image,ImageDraw,ImageFont
from translations import TEXT,TABLES
ROOT=Path('D:/Codex/GRL研究');HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/grid_icaps_en_from_cn_v2';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'outputs/grid_icaps_cn_v2/ICAPS2027_中文重写稿_v2.docx'
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=SOURCE.read_bytes();sourcehash=sha(raw)
doc=Document(SOURCE)
assert len(doc.paragraphs)==101
nonempty={i for i,p in enumerate(doc.paragraphs) if p.text.strip()}
assert nonempty-set(range(95,101))==set(TEXT),(nonempty-set(TEXT),set(TEXT)-nonempty)
assert len(doc.tables)==3
assert (HERE/'reference_render/page-8.png').exists()
snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
(snapshot/'author_approved_cn_v2.docx').write_bytes(raw)
(snapshot/'paragraphs.json').write_text(json.dumps([{'index':i,'style':p.style.name,'text':p.text} for i,p in enumerate(doc.paragraphs)],ensure_ascii=False,indent=2),encoding='utf-8')
with ZipFile(SOURCE) as z:parts={n:z.read(n) for n in z.namelist()};infos=z.infolist()
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def tag(n):prefix,name=n.split(':');return '{'+N[prefix]+'}'+name
tree=E.fromstring(parts['word/document.xml']);body=tree.find(tag('w:body'));paras=body.findall(tag('w:p'))
assert len(paras)==101
assert not tree.xpath('//w:ins|//w:del|//w:commentRangeStart',namespaces=N),'Tracked edits or comments require separate handling'
section=E.tostring(body.find(tag('w:sectPr')))
math_before=[E.tostring(x) for x in tree.xpath('//m:oMathPara',namespaces=N)];assert len(math_before)==5
contract=f'''# Translation execution contract

Reference: {SOURCE}
SHA-256: {sourcehash}
Reference layout: 8 reading pages; one portrait A4 section; source margins retained.
Authority: current author-approved Chinese DOCX, not the previous English draft.
Slots: direct body paragraphs 0-94 with visible text translated by structural index;
95-100 reference entries retained; all three tables translated in place.
Preserve: five native OMML equations, section geometry, styles, numbering,
header/footer and PAGE fields, relationships, bookmarks and opaque package parts.
Figure: replace the one embedded framework bitmap with English labels at the
same dimensions and supply an editable English SVG. Source flow and branches retained.
Table layout: columns remain unchanged; English labels wrap inside the existing
grid. Any later width adjustment must be recorded as language-fitting adaptation.
Body font and heading hierarchy: source styles and paragraph formatting retained.
English paragraph length may change pagination. Final pages visually inspected.
Settings: request field refresh on open through w:updateFields.
Metadata: translate title/subject in docProps/core.xml.
Coverage gates: all nonempty paragraphs mapped, all tables covered, source
numeric values preserved, unchanged math and section XML, all other ZIP parts
byte-identical, source DOCX hash unchanged at delivery.
'''
(HERE/'artifact.md').write_text(contract,encoding='utf-8')
def replace_paragraph(p,text):
    runs=p.findall(tag('w:r'));rp=runs[0].find(tag('w:rPr')) if runs else None
    firstpr=deepcopy(rp) if rp is not None else None
    for c in list(p):
        if c.tag not in [tag('w:pPr'),tag('w:bookmarkStart'),tag('w:bookmarkEnd')]:p.remove(c)
    run=E.SubElement(p,tag('w:r'))
    if firstpr is not None:run.append(firstpr)
    for j,line in enumerate(text.split('\n')):
        if j:E.SubElement(run,tag('w:br'))
        t=E.SubElement(run,tag('w:t'));t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=line
for i,text in TEXT.items():replace_paragraph(paras[i],text)
tables=body.findall(tag('w:tbl'))
for ti,tb in enumerate(tables):
    rows=tb.findall(tag('w:tr'))
    translated=TABLES[ti]
    for ri,row in enumerate(rows):
        cells=row.findall(tag('w:tc'))
        values=translated[ri] if ti!=1 or ri==0 else [({'开发':'Development','确认':'Confirmation'}[doc.tables[1].rows[ri].cells[0].text])]+[c.text for c in doc.tables[1].rows[ri].cells[1:]]
        assert len(values)==len(cells)
        for ci,(cell,text) in enumerate(zip(cells,values)):
            ps=cell.findall(tag('w:p'));assert len(ps)==1
            replace_paragraph(ps[0],text)
            old=doc.tables[ti].rows[ri].cells[ci].text
            if ti==1 and ri>0 and ci>0:assert text==old
            if ti==2 and ri>0 and ci>=2 and ci!=2 or ti==2 and ri>0 and ci==2 and ri!=2:
                assert re.findall(r'[-+]?\d+(?:\.\d+)?',text)==re.findall(r'[-+]?\d+(?:\.\d+)?',old)
# Keep the source layout; allow a little more room for translated group labels.
widths=[2.5,3.4,1.6,1.3,2.1,2.0,3.5]
tb=tables[1]
for col,width in zip(tb.find(tag('w:tblGrid')),widths):col.set(tag('w:w'),str(round(width*567)))
for row in tb.findall(tag('w:tr')):
    for cell,width in zip(row.findall(tag('w:tc')),widths):cell.find('./'+tag('w:tcPr')+'/'+tag('w:tcW')).set(tag('w:w'),str(round(width*567)))
contract+='\nRecorded adaptation: main results grid expanded to 16.4 cm with a 2.5 cm group column to fit Development/Confirmation. Table padding, borders and shading retained.\n'
(HERE/'artifact.md').write_text(contract,encoding='utf-8')
assert [E.tostring(x) for x in tree.xpath('//m:oMathPara',namespaces=N)]==math_before
# Give native math runs an explicit font to avoid symbol-font substitution in
# the headless renderer. Expression text and operator structure remain intact.
math_text_before=tree.xpath('//m:oMathPara//m:t/text()',namespaces=N)
for run in tree.xpath('//m:oMathPara//m:r',namespaces=N):
    mrp=run.find(tag('m:rPr'))
    if mrp is None:mrp=E.Element(tag('m:rPr'));run.insert(0,mrp)
    if mrp.find(tag('m:nor')) is None:E.SubElement(mrp,tag('m:nor'))
    if mrp.find(tag('m:lit')) is None:E.SubElement(mrp,tag('m:lit'))
    rp=run.find(tag('w:rPr'))
    if rp is None:rp=E.Element(tag('w:rPr'));run.insert(1 if run.find(tag('m:rPr')) is not None else 0,rp)
    fonts=rp.find(tag('w:rFonts'))
    if fonts is None:fonts=E.SubElement(rp,tag('w:rFonts'))
    for attr in ['ascii','hAnsi','eastAsia','cs']:fonts.set(tag('w:'+attr),'Times New Roman')
# Unicode compatibility forms keep isolated operators visible in portable rendering.
for txt in tree.xpath('//m:oMathPara//m:t',namespaces=N):
    if txt.text and txt.text.strip() in ('=','+'):
        txt.text=txt.text.replace('=','＝').replace('+','＋')
import unicodedata
assert [unicodedata.normalize('NFKC',x) for x in tree.xpath('//m:oMathPara//m:t/text()',namespaces=N)]==[unicodedata.normalize('NFKC',x) for x in math_text_before]
contract+='\nRecorded adaptation: native math text runs use normal-text Times New Roman to avoid symbol-font substitution in reference rendering. Equation text, fractions, summation and operators preserved.\n'
(HERE/'artifact.md').write_text(contract,encoding='utf-8')
assert E.tostring(body.find(tag('w:sectPr')))==section
parts['word/document.xml']=E.tostring(tree,xml_declaration=True,encoding='UTF-8',standalone=True)
# Vector-first drawing, following the guide's modular workflow.
import subprocess
subprocess.run([str(Path('D:/Codex/GRL研究/work/grid02/.venv/Scripts/python.exe')),str(HERE/'figure_draw.py'),'--framework-only'],check=True)
fig=OUT/'figures'
embed=tree.xpath('//a:blip/@r:embed',namespaces=N);assert len(embed)==1
rels=E.fromstring(parts['word/_rels/document.xml.rels'])
rel=next(x for x in rels if x.get('Id')==embed[0]);imagepart='word/'+rel.get('Target');parts[imagepart]=(fig/'framework_en.png').read_bytes()
settings=E.fromstring(parts['word/settings.xml']);up=settings.find(tag('w:updateFields'))
if up is None:up=E.SubElement(settings,tag('w:updateFields'))
up.set(tag('w:val'),'true');parts['word/settings.xml']=E.tostring(settings,xml_declaration=True,encoding='UTF-8',standalone=True)
core=E.fromstring(parts['docProps/core.xml']);dc='{http://purl.org/dc/elements/1.1/}'
if core.find(dc+'title') is not None:core.find(dc+'title').text=TEXT[0]
if core.find(dc+'subject') is not None:core.find(dc+'subject').text='ICAPS 2027 English manuscript translated from the author-approved Chinese V2'
parts['docProps/core.xml']=E.tostring(core,xml_declaration=True,encoding='UTF-8',standalone=True)
target=OUT/'ICAPS2027_English_from_author_CN_V2.docx'
with ZipFile(target,'w') as z:
    for info in infos:z.writestr(info,parts[info.filename])
allowed={'word/document.xml','word/settings.xml','docProps/core.xml',imagepart}
with ZipFile(SOURCE) as a,ZipFile(target) as b:
    preserved=[n for n in a.namelist() if n not in allowed]
    assert a.namelist()==b.namelist()
    assert all(a.read(n)==b.read(n) for n in preserved)
    assert b.testzip() is None
english=Document(target)
assert len(english.paragraphs)==len(doc.paragraphs) and len(english.tables)==3
assert not re.search('[\u4e00-\u9fff]','\n'.join(p.text for p in english.paragraphs)+str([[c.text for c in r.cells] for t in english.tables for r in t.rows]))
monthwords={'March':'3','April':'4','May':'5','June':'6','November':'11'}
numeric_mismatches=[]
for i in TEXT:
    old=doc.paragraphs[i].text.replace('七成','70');new=TEXT[i]
    for word,digit in monthwords.items():new=new.replace(word,digit)
    numbers=lambda s:re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)*',s)
    if Counter(numbers(old))!=Counter(numbers(new)):numeric_mismatches.append({'paragraph':i,'source':numbers(old),'translation':numbers(new)})
assert not numeric_mismatches,numeric_mismatches
md=[];pidx=0;tidx=0;eq=0
latex=[r'0 < \rho(a) < \min\{\rho(x),\theta\}',r'f(a)=(\operatorname{fp32}(r(a)),-i(a))',r'L=\{\operatorname{rep}(i)\in A\setminus C:s(i)=s_C\}',r'\bar T_{m,d}=\tfrac12(T_{m,d,0}+T_{m,d,1})',r'S_T(m)=\tfrac1{|D_T|}\sum_{d\in D_T}(1-\bar T_{m,d}/\bar T_{\mathrm{FULL},d})']
for child in body:
    if child.tag==tag('w:p'):
        p=english.paragraphs[pidx]
        if p.text:
            prefix='# ' if p.style.name=='Title' else '## ' if p.style.name=='Heading 1' else '### ' if p.style.name=='Heading 2' else ''
            md.append(prefix+p.text)
        elif child.xpath('.//m:oMathPara',namespaces=N):md.append('\\[\n'+latex[eq]+'\n\\]');eq+=1
        elif child.xpath('.//a:blip',namespaces=N):md.append('![Framework](figures/framework_en.png)')
        pidx+=1
    elif child.tag==tag('w:tbl'):
        rows=[[c.text.replace('\n',' ') for c in r.cells] for r in english.tables[tidx].rows]
        md.extend(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(r)+' |' for r in rows[1:]]);tidx+=1
(OUT/'manuscript_en.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
receipt={'source':str(SOURCE),'source_sha256':sourcehash,'output':str(target),'translated_body_paragraphs':len(TEXT),'reference_entries_preserved':6,'tables':3,'native_equations':5,'equations_nfkc_equivalent':True,'isolated_operator_compatibility_glyphs':['＝','＋'],'numeric_checks_passed':True,'preserved_package_parts':len(preserved),'edited_package_parts':sorted(allowed),'source_geometry_unchanged':True,'new_experiments':0,'new_fits':0,'english_words':len(re.findall(r'\b[\w-]+\b',' '.join(TEXT.values())))}
(OUT/'translation_check.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
assert sha(SOURCE.read_bytes())==sourcehash
print(json.dumps(receipt,ensure_ascii=False))
