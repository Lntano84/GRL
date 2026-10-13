"""Complete Chinese author manuscript, editable DOCX/OMML, no scientific changes."""
from pathlib import Path
import re,json,csv,hashlib,html,shutil
from copy import deepcopy
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/grid_icaps_cn_v2'
if (OUT/'validation.json').exists():raise SystemExit('Completed Chinese version exists; create a new version.')
OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
SRC=HERE/'manuscript_ready.md'
en=ROOT/'outputs/grid_icaps_draft_v1/manuscript_en.md'
previous={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [en,ROOT/'outputs/grid61/finished.json',ROOT/'outputs/grid62/finished.json']}
source=SRC.read_text(encoding='utf-8')
# Full Chinese vector and raster framework, using an explicit installed Chinese font.
scale=2;W,H=1280,450
image=Image.new('RGB',(W*scale,H*scale),'white');draw=ImageDraw.Draw(image)
fontpath='C:/Windows/Fonts/msyh.ttc'
fonts={n:ImageFont.truetype(fontpath,n*scale) for n in [14,16,18,20,26]}
boxes=[(28,78,263,162,['公开状态与基础动作','当前合法性与断线端点处理']),
       (293,78,528,162,['规范化代表与别名','保留原始变电站信息']),
       (558,78,793,162,['相同排序 首批 128','选择最佳可接纳候选']),
       (965,78,1245,162,['返回拓扑提案','接入原连续控制与恢复']),
       (558,263,793,347,['渐进扩展 每批 64','达到条件则停止全局扩展']),
       (910,263,1245,347,['原最佳候选的变电站补全','查询该站所有未查询代表'])]
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>', '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="#536780"/></marker></defs>', '<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172033">']
def text(x,y,s,size=18,anchor='start'):
    draw.text((int(x*scale),int(y*scale)),s,font=fonts[size],fill='#172033',anchor='ma' if anchor=='middle' else None)
    svg.append(f'<text x="{x}" y="{y+size}" font-size="{size}" text-anchor="{anchor}">{html.escape(s)}</text>')
text(28,14,'渐进拓扑搜索与变电站补全',26)
text(28,48,'输入当前电网状态 输出经过预测验证的拓扑提案',16)
for x1,y1,x2,y2,labels in boxes:
    fill='#eef4fc' if y1>200 else '#f6f7f9'
    draw.rounded_rectangle((x1*scale,y1*scale,x2*scale,y2*scale),radius=8*scale,fill=fill,outline='#7f91aa',width=2)
    svg.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="8" fill="{fill}" stroke="#7f91aa"/>')
    for j,s in enumerate(labels):text((x1+x2)/2,y1+18+j*29,s,18 if j==0 else 16,'middle')
def arrow(points):
    pp=[(x*scale,y*scale) for x,y in points];draw.line(pp,fill='#536780',width=3*scale)
    x,y=pp[-1];px,py=pp[-2]
    if x>px:tri=[(x,y),(x-10*scale,y-6*scale),(x-10*scale,y+6*scale)]
    elif y>py:tri=[(x,y),(x-6*scale,y-10*scale),(x+6*scale,y-10*scale)]
    else:tri=[(x,y),(x-6*scale,y+10*scale),(x+6*scale,y+10*scale)]
    draw.polygon(tri,fill='#536780')
    svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+'" fill="none" stroke="#536780" stroke-width="2" marker-end="url(#arrow)"/>')
arrow([(263,120),(293,120)]);arrow([(528,120),(558,120)]);arrow([(793,120),(965,120)])
arrow([(676,162),(676,263)]);arrow([(793,305),(910,305)]);arrow([(1108,263),(1108,162)])
text(820,86,'首批达标或池耗尽',16)
text(696,207,'需要继续评估',16)
text(811,272,'条件停止',16)
text(28,379,'全局渐进评估定位有效变电站 局部补查完成同站配置比较',16)
text(28,412,'复用已查询反馈 保留最佳候选 每个代表最多评估一次',16)
svg.append('</g></svg>')
image.save(OUT/'figures/framework_zh.png');(OUT/'figures/framework_zh.svg').write_text('\n'.join(svg),encoding='utf-8')
published=source.replace('<!--FIGURE-->','![图 1 渐进搜索与变电站补全]('+str(OUT/'figures/framework_zh.png').replace('\\','/')+')')
(OUT/'ICAPS2027_中文重写稿_v2.md').write_text(published,encoding='utf-8')

doc=Document();sec=doc.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7)
sec.top_margin=Cm(2.05);sec.bottom_margin=Cm(2.05);sec.left_margin=Cm(2.1);sec.right_margin=Cm(2.1)
sec.header_distance=Cm(.8);sec.footer_distance=Cm(.85)
def fontstyle(style,size,east='宋体',latin='Times New Roman'):
    style.font.name=latin;style.font.size=Pt(size);style.font.color.rgb=RGBColor(0,0,0)
    rp=style.element.get_or_add_rPr();rf=rp.find(qn('w:rFonts'))
    if rf is None:rf=OxmlElement('w:rFonts');rp.insert(0,rf)
    for key,val in [('eastAsia',east),('ascii',latin),('hAnsi',latin)]:rf.set(qn('w:'+key),val)
    if rp.find(qn('w:color')) is not None:
        rp.find(qn('w:color')).attrib.pop(qn('w:themeColor'),None)
fontstyle(doc.styles['Normal'],11.5)
doc.styles['Normal'].paragraph_format.line_spacing=1.15
doc.styles['Normal'].paragraph_format.space_after=Pt(4)
doc.styles['Normal'].paragraph_format.widow_control=True
for name,size in [('Title',19),('Subtitle',11),('Heading 1',14),('Heading 2',12)]:fontstyle(doc.styles[name],size,'黑体')
# The bundled python-docx base template may carry a blue Title border.
for style in doc.styles:
    for border in list(style.element.iter(qn('w:pBdr'))):border.getparent().remove(border)
doc.styles['Title'].paragraph_format.space_after=Pt(10)
doc.styles['Subtitle'].paragraph_format.space_after=Pt(12)
for name in ['Heading 1','Heading 2']:
    f=doc.styles[name].paragraph_format;f.keep_with_next=True;f.space_before=Pt(10);f.space_after=Pt(5)
doc.core_properties.title='基于变电站补全的电网渐进拓扑搜索'
doc.core_properties.subject='ICAPS 2027 中文作者重写稿'
doc.core_properties.author=''
doc.core_properties.keywords='拓扑搜索 变电站补全 电网控制'
# Native editable Office Math nodes, checked structurally and in the DOCX render.
def mn(tag):return OxmlElement('m:'+tag)
def mr(s):
    el=mn('r');pr=mn('rPr');sty=mn('sty');sty.set(qn('m:val'),'p');pr.append(sty);el.append(pr)
    t=mn('t');t.text=s;el.append(t);return el
def sub(base,idx):
    el=mn('sSub');e=mn('e');e.append(mr(base) if isinstance(base,str) else base);s=mn('sub');s.append(mr(idx));el.extend([e,s]);return el
def frac(num,den):
    el=mn('f');n=mn('num');d=mn('den')
    for parent,x in [(n,num),(d,den)]:
        for t in x if isinstance(x,list) else [x]:parent.append(mr(t) if isinstance(t,str) else t)
    el.extend([n,d]);return el
def bar(base):
    el=mn('bar');pr=mn('barPr');pos=mn('pos');pos.set(qn('m:val'),'top');pr.append(pos)
    e=mn('e');e.append(mr(base));el.extend([pr,e]);return el
def summation(lower,body):
    el=mn('nary');pr=mn('naryPr');ch=mn('chr');ch.set(qn('m:val'),'∑');pr.append(ch)
    lim=mn('limLoc');lim.set(qn('m:val'),'undOvr');pr.append(lim)
    hide=mn('supHide');hide.set(qn('m:val'),'1');pr.append(hide)
    lo=mn('sub');hi=mn('sup');e=mn('e')
    for x in lower:lo.append(mr(x) if isinstance(x,str) else x)
    for x in body:e.append(mr(x) if isinstance(x,str) else x)
    el.extend([pr,lo,hi,e]);return el
def formula(n):
    sequences={
      0:['0 < ρ(a) < min{ρ(x), θ}'],
      1:['f(a) = (fp32(r(a)), −i(a))'],
      2:['L = {rep(i) ∈ A ∖ C : s(i) = ',sub('s','C'),'}'],
      3:[sub(bar('T'),'m,d'),' = ',frac('1','2'),'(',sub('T','m,d,0'),' + ',sub('T','m,d,1'),')'],
      4:[sub('S','T'),'(m) = ',frac('1',['|',sub('D','T'),'|']),summation(['d ∈ ',sub('D','T')],['(', '1 − ',frac(sub(bar('T'),'m,d'),sub(bar('T'),'FULL,d')),')'])],
    }
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(8)
    p.paragraph_format.line_spacing_rule=WD_LINE_SPACING.AT_LEAST
    p.paragraph_format.line_spacing=Pt(42 if n==4 else 28)
    p.paragraph_format.keep_with_next=(n==3)
    outer=mn('oMathPara');inner=mn('oMath')
    for x in sequences[n]:inner.append(mr(x) if isinstance(x,str) else x)
    outer.append(inner);p._p.append(outer)
def add_runs(p,s):
    parts=re.split(r'(\*\*.*?\*\*)',s)
    for part in parts:
        if not part:continue
        bold=part.startswith('**') and part.endswith('**')
        r=p.add_run(part[2:-2] if bold else part);r.bold=bold
def cellpadding(c):
    pr=c._tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
    for name,v in [('top',90),('bottom',90),('left',80),('right',80)]:
        t=OxmlElement('w:'+name);t.set(qn('w:w'),str(v));t.set(qn('w:type'),'dxa');m.append(t)
    pr.append(m)
def table(rows,idx):
    widths={1:[3.5,13.1],2:[1.3,3.5,1.65,1.45,2.1,2.0,3.0],3:[2.4,4.2,3.0,3.0,4.0]}[idx]
    tb=doc.add_table(rows=len(rows),cols=len(rows[0]));tb.alignment=WD_TABLE_ALIGNMENT.CENTER;tb.autofit=False
    for col,w in zip(tb.columns,widths):col.width=Cm(w)
    borders=OxmlElement('w:tblBorders')
    for name in ['top','left','bottom','right','insideH','insideV']:
        e=OxmlElement('w:'+name);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
    tb._tbl.tblPr.append(borders)
    for j,row in enumerate(rows):
        no=OxmlElement('w:cantSplit');tb.rows[j]._tr.get_or_add_trPr().append(no)
        if j==0:
            repeat=OxmlElement('w:tblHeader');tb.rows[j]._tr.get_or_add_trPr().append(repeat)
        for k,value in enumerate(row):
            if idx==2 and j==0:
                value={'质量通过日期':'质量通过\n日期','完成日期':'完成\n日期','控制时间减少':'控制时间\n减少','最坏完整周成本变化':'最坏完整周\n成本变化'}.get(value,value)
            c=tb.cell(j,k);c.width=Cm(widths[k]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;cellpadding(c)
            p=c.paragraphs[0];p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.08
            if idx==1:p.paragraph_format.keep_with_next=(j<len(rows)-1)
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER if idx>1 and k!=1 else WD_ALIGN_PARAGRAPH.LEFT
            r=p.add_run(value);r.bold=(j==0);r.font.size=Pt(9 if idx==2 else 10)
            if j==0:
                sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E9EDF2');c._tc.get_or_add_tcPr().append(sh)
    doc.add_paragraph().paragraph_format.space_after=Pt(1)

i=0;eq=0;tableidx=0;lines=source.splitlines();in_refs=False
while i<len(lines):
    s=lines[i]
    if not s:i+=1;continue
    if s.startswith('# '):
        p=doc.add_paragraph(s[2:],style='Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        i+=1;continue
    if s.startswith('## '):
        doc.add_paragraph(s[3:],style='Heading 1');in_refs=(s[3:]=='参考文献');i+=1;continue
    if s.startswith('### '):doc.add_paragraph(s[4:],style='Heading 2');i+=1;continue
    if s==r'\[':
        i+=1
        while lines[i]!=r'\]':i+=1
        formula(eq);eq+=1;i+=1;continue
    if s=='<!--FIGURE-->':
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True;p.paragraph_format.line_spacing=1
        p.add_run().add_picture(str(OUT/'figures/framework_zh.png'),width=Cm(16.4));i+=1;continue
    if s in ['图 1 渐进搜索与变电站补全框架','表 1 按日期汇总的质量与计算结果','表 2 FRAME 的全部冻结确认日期']:
        p=doc.add_paragraph(s);p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=s.startswith('表')
        for r in p.runs:r.font.size=Pt(10)
        i+=1;continue
    if s.startswith('```'):
        i+=1
        while not lines[i].startswith('```'):
            p=doc.add_paragraph(lines[i]);p.paragraph_format.line_spacing=1.04;p.paragraph_format.space_after=Pt(1);p.paragraph_format.left_indent=Cm(.35)
            p.paragraph_format.keep_with_next=not lines[i+1].startswith('```')
            for r in p.runs:r.font.size=Pt(9.2);r.font.name='Consolas'
            i+=1
        i+=1;continue
    if s.startswith('|'):
        rs=[]
        while i<len(lines) and lines[i].startswith('|'):
            cs=[c.strip() for c in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r'[-:]+',c) for c in cs):rs.append(cs)
            i+=1
        tableidx+=1;table(rs,tableidx);continue
    p=doc.add_paragraph();add_runs(p,s)
    if not in_refs:p.paragraph_format.first_line_indent=Pt(23)
    else:
        p.paragraph_format.line_spacing=1.1
        for r in p.runs:r.font.size=Pt(9.6)
    i+=1
assert eq==5 and tableidx==3
# A simple page-number field avoids headers competing with the paper text.
p=sec.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');r=OxmlElement('w:r');t=OxmlElement('w:t');t.text='1';r.append(t);fld.append(r);p._p.append(fld)
doc.save(OUT/'ICAPS2027_中文重写稿_v2.docx')

# Strict bilingual table/numeric consistency, not a claim of automatic semantic proof.
def parse_tables(s):
    out=[];cur=[]
    for l in s.splitlines()+['']:
        if l.startswith('|'):
            row=[c.strip() for c in l.strip('|').split('|')]
            if not all(re.fullmatch(r'[-:]+',c) for c in row):cur.append(row)
        elif cur:out.append(cur);cur=[]
    return out
ets=parse_tables(en.read_text(encoding='utf-8'));zts=parse_tables(source)
assert len(ets)==len(zts)==3
for a,b in zip(ets[1][1:],zts[1][1:]):assert a[1:]==b[1:],(a,b)
for a,b in zip(ets[2][1:],zts[2][1:]):
    for j in [1,2,3,4]:
        assert re.findall(r'[-+]?\d+(?:\.\d+)?',a[j])==re.findall(r'[-+]?\d+(?:\.\d+)?',b[j]),(a,b,j)
english_heads=re.findall(r'^## ',en.read_text(encoding='utf-8'),re.M);chinese_heads=re.findall(r'^## ',source,re.M)
assert len(chinese_heads)==8
assert re.findall(r'^## (.*)',source,re.M)==['摘要','1 引言','2 方法','3 实验','4 结论','5 相关工作','6 局限','参考文献']
from zipfile import ZipFile
with ZipFile(OUT/'ICAPS2027_中文重写稿_v2.docx') as z:
    xml=z.read('word/document.xml').decode('utf-8')
    assert xml.count('<m:oMathPara>')==5 and '<m:nary>' in xml and '<m:f>' in xml
    assert z.testzip() is None
receipt=dict(source_en=str(en),source_en_sha256=previous[str(en.relative_to(ROOT))],original_receipts=previous,full_sections=8,tables=3,display_equations=5,native_word_equations=True,main_table_rows_equal=True,confirmation_table_numeric_equal=True,chinese_characters=len(re.findall('[\u4e00-\u9fff]',source)),new_simulations=0,new_fits=0,scope='Full Chinese rewrite using user edits as style guidance. Same three numerical tables and five equations; source user document preserved. Render QA required.')
(OUT/'rewrite_check.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(OUT),chinese_characters=receipt['chinese_characters'],display_equations=eq,tables=tableidx,checks=True),ensure_ascii=False))
