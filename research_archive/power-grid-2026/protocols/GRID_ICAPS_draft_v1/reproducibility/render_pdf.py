"""Reading-layout PDF, deliberately not a claim of AAAI submission typesetting."""
from pathlib import Path
import re, json, html
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid_icaps_draft_v1'
src=(Path(__file__).resolve().parent/'manuscript.md').read_text(encoding='utf-8')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle('PaperTitle',fontName='Times-Bold',fontSize=19,leading=23,spaceAfter=10,alignment=TA_CENTER))
styles.add(ParagraphStyle('PaperText',fontName='Times-Roman',fontSize=10.8,leading=14,spaceAfter=6,alignment=TA_JUSTIFY))
styles.add(ParagraphStyle('PaperH1',fontName='Times-Bold',fontSize=13.2,leading=17,spaceBefore=10,spaceAfter=6,keepWithNext=True))
styles.add(ParagraphStyle('PaperH2',fontName='Times-Bold',fontSize=11.3,leading=15,spaceBefore=8,spaceAfter=4,keepWithNext=True))
styles.add(ParagraphStyle('PaperNote',fontName='Times-Italic',fontSize=9,leading=12,spaceAfter=8))
styles.add(ParagraphStyle('PaperCell',fontName='Times-Roman',fontSize=9.2,leading=11.5))
styles.add(ParagraphStyle('PaperCellSmall',fontName='Times-Roman',fontSize=8.4,leading=10.4))
styles.add(ParagraphStyle('PaperCode',fontName='Courier',fontSize=8.0,leading=10.2,spaceBefore=5,spaceAfter=8))
styles.add(ParagraphStyle('PaperEquation',fontName='Courier',fontSize=9.4,leading=14,alignment=TA_CENTER,spaceBefore=5,spaceAfter=8))
def markup(s):
    s=html.escape(s)
    return re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
def footer(c,doc):
    c.saveState();c.setFont('Times-Italic',8.3);c.setFillColor(colors.HexColor('#52616b'))
    c.drawString(54,letter[1]-30,'ICAPS 2027 author-editing draft | Reading layout, not AAAI submission formatting')
    c.setStrokeColor(colors.HexColor('#d8dee4'));c.line(54,letter[1]-36,letter[0]-54,letter[1]-36)
    c.setFont('Times-Roman',8.5);c.drawString(54,25,'13 October 2026 | AI-assisted working text; author revision required')
    c.drawRightString(letter[0]-54,25,str(doc.page));c.restoreState()
story=[];lines=src.splitlines();i=0;nt=0;ne=0
equations=[
 '0 < rho(a) < min{rho(x), theta}',
 'f(a) = (fp32(r(a)), -i(a))',
 'L = {rep(i) in A minus C : s(i) = s_C}',
 'Tbar[m,d] = (T[m,d,0] + T[m,d,1]) / 2',
 'S_T(m) = mean_d [1 - Tbar[m,d] / Tbar[FULL,d]]',
]
while i<len(lines):
    s=lines[i]
    if not s:i+=1;continue
    if s.startswith('# '):
        story += [Paragraph(markup(s[2:]),styles['PaperTitle']),Paragraph('Complete English first draft. Evidence, adverse results, and prior manuscripts are retained.',styles['PaperNote'])];i+=1;continue
    if s.startswith('## '):story.append(Paragraph(markup(s[3:]),styles['PaperH1']));i+=1;continue
    if s.startswith('### '):story.append(Paragraph(markup(s[4:]),styles['PaperH2']));i+=1;continue
    if s==r'\[':
        i+=1
        while lines[i]!=r'\]':i+=1
        story.append(Paragraph(html.escape(equations[ne]),styles['PaperEquation']));ne+=1;i+=1;continue
    if s.startswith('```'):
        block=[];i+=1
        while not lines[i].startswith('```'):block.append(lines[i]);i+=1
        story.append(Preformatted('\n'.join(block),styles['PaperCode']));i+=1;continue
    if s.startswith('|'):
        rs=[]
        while i<len(lines) and lines[i].startswith('|'):
            cs=[c.strip() for c in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r'[-:]+',c) for c in cs):rs.append(cs)
            i+=1
        nt+=1
        widths={1:[113,391],2:[69,94,54,56,75,70,86],3:[100,130,94,94,86]}[nt]
        cell_style=styles['PaperCellSmall'] if nt==2 else styles['PaperCell']
        data=[[Paragraph(markup(c if j else '**'+c+'**'),cell_style) for c in row] for j,row in enumerate(rs)]
        tb=Table(data,colWidths=widths,repeatRows=1,hAlign='CENTER')
        tb.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f7')),('LINEABOVE',(0,0),(-1,0),.6,colors.HexColor('#34495e')),('LINEBELOW',(0,0),(-1,0),.4,colors.HexColor('#34495e')),('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#34495e')),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
        labels={1:'Controlled policies with a shared controller and action order.',2:'Table 1. Main date-level results. Positive time/query values mean reduction; positive cost is worse.',3:'Table 2. All frozen confirmation dates for FRAME; zero-search and failed dates are retained.'}
        story += [Spacer(1,4),tb,Paragraph(labels[nt],styles['PaperNote'])];continue
    story.append(Paragraph(markup(s),styles['PaperText']))
    if s.endswith('Figure 1 summarizes the distinction.'):
        im=Image(str(OUT/'figures/framework_preview.png'))
        scale=min(504/im.imageWidth,240/im.imageHeight)
        im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale
        story += [im,Paragraph('Figure 1. Current-state preparation, progressive expansion, and conditional substation completion. Ranking and base control remain replaceable/shared.',styles['PaperNote'])]
    i+=1
assert ne==5 and nt==3
p=OUT/'ICAPS2027_初稿阅读版.pdf'
doc=SimpleDocTemplate(str(p),pagesize=letter,leftMargin=54,rightMargin=54,topMargin=48,bottomMargin=44,title='Progressive Topology Search with Substation Completion for Power Grid Control - working draft',author='Anonymous Working Draft')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
r=PdfReader(p)
texts=[page.extract_text() or '' for page in r.pages]
assert all(t.strip() for t in texts)
for val in ['69.37%','13.11%','83.54%','6.09%','FIRST_SAFE_BEST','198','19842']:
    assert val in '\n'.join(texts),val
(OUT/'pdf_reading_layout_check.json').write_text(json.dumps(dict(pages=len(r.pages),all_pages_have_text=True,three_tables=True,five_equations_reading_notation=True,not_aaai_submission_format=True,visual_review_pending=True),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(pdf=str(p),pages=len(r.pages),text_checks=True)))
