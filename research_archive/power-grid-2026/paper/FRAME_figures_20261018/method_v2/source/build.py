"""FRAME overview V2: asymmetric regions, semantic colors, concrete candidate flow."""
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, re
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path('D:/Codex/GRL研究')
OUT=ROOT/'outputs/frame_method_v2_20261018'; A=OUT/'assets'
W,H=1470,800
CB=json.loads((A/'colorbrewer.json').read_text())
def hx(rgb): return '#'+''.join(f'{int(n):02x}' for n in re.findall(r'\d+',rgb))
B=[hx(x) for x in CB['Blues']['7']]
P=[hx(x) for x in CB['Purples']['7']]
G=[hx(x) for x in CB['BuGn']['7']]
INK='#30343B'; SUB='#727780'; BORDER='#D6DADF'; FONT='Noto Sans SC'
plt.rcParams.update({'font.family':FONT,'svg.fonttype':'none'})

def strip(name, labels, selected, hue, best=None, added=()):
    fig=plt.figure(figsize=(265/96,65/96),dpi=96)
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,265); ax.set_ylim(0,65); ax.axis('off')
    bw=32; gap=10; total=len(labels)*bw+(len(labels)-1)*gap; left=(265-total)/2
    for i,t in enumerate(labels):
        x=left+i*(bw+gap); on=i in selected
        ax.add_patch(FancyBboxPatch((x,15),bw,32,boxstyle='round,pad=0,rounding_size=3',
            facecolor=hue[2] if on else '#FFFFFF',edgecolor=hue[5] if on else '#C4C9CF',
            lw=1.2,linestyle='solid' if on else (0,(2,2))))
        if t: ax.text(x+bw/2,31,t,ha='center',va='center',fontsize=18,color=hue[6] if on else SUB)
        if i==best: ax.plot(x+bw/2,58,marker='*',markersize=10,color=hue[5],clip_on=False)
        if i in added: ax.plot([x+3,x+bw-3],[8,8],color=hue[5],lw=2)
    fig.savefig(A/f'{name}.svg',transparent=True); plt.close(fig)
strip('prefix',['']*6,{0,1,2,3},B,best=2)
strip('batch',['']*6,set(range(6)),P,best=4)
strip('station',list('BACABA'),{1,3,5},G,added={1,3,5})
icon=ET.parse(A/'noun_powergrid_4199252.svg').getroot()
for e in list(icon):
    if e.tag.rsplit('}',1)[-1]=='text':icon.remove(e)
icon.set('viewBox','8 18 104 84')
ET.ElementTree(icon).write(A/'powergrid_artwork.svg',encoding='utf-8',xml_declaration=True)
mx=ET.Element('mxfile',host='app.diagrams.net',version='32.4.1')
dg=ET.SubElement(mx,'diagram',id='frame-method-v2',name='FRAME方法图V2')
md=ET.SubElement(dg,'mxGraphModel',dx=str(W),dy=str(H),grid='1',gridSize='10',page='1',
    pageScale='1',pageWidth=str(W),pageHeight=str(H),math='0',shadow='0',background='#FFFFFF')
rt=ET.SubElement(md,'root'); ET.SubElement(rt,'mxCell',id='0'); ET.SubElement(rt,'mxCell',id='1',parent='0')
n=1
def cell(value,x,y,w,h,style):
    global n
    n+=1; c=ET.SubElement(rt,'mxCell',id=str(n),value=value,style=style,vertex='1',parent='1')
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
    return c
def text(t,x,y,w,h,size=23,bold=False,color=INK,align='left'):
    return cell(t,x,y,w,h,f'text;html=0;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;whiteSpace=wrap;fontFamily={FONT};fontSize={size};fontStyle={int(bold)};fontColor={color};spacing=0;')
def box(x,y,w,h,fill='white',stroke=BORDER,rounded=True,width=1,dashed=False):
    return cell('',x,y,w,h,f'rounded={int(rounded)};arcSize=7;fillColor={fill};strokeColor={stroke};strokeWidth={width};dashed={int(dashed)};dashPattern=5 4;html=0;')
def circle(t,x,y,d,fill,color,size=22):
    return cell(t,x,y,d,d,f'ellipse;html=0;fillColor={fill};strokeColor=none;fontColor={color};fontFamily={FONT};fontSize={size};align=center;verticalAlign=middle;fontStyle=1;')
def line(pts,color=INK,width=1.5,dashed=False,arrow=True,both=False):
    global n
    n+=1; c=ET.SubElement(rt,'mxCell',id=str(n),parent='1',edge='1',
        style=f'edgeStyle=none;rounded=1;arcSize=10;html=0;endArrow={"block" if arrow else "none"};startArrow={"open" if both else "none"};endFill=1;endSize=7;startSize=7;strokeColor={color};strokeWidth={width};dashed={int(dashed)};dashPattern=5 4;')
    geo=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
    ET.SubElement(geo,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),**{'as':'sourcePoint'})
    ET.SubElement(geo,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),**{'as':'targetPoint'})
    if len(pts)>2:
        ar=ET.SubElement(geo,'Array',**{'as':'points'})
        for x,y in pts[1:-1]:ET.SubElement(ar,'mxPoint',x=str(x),y=str(y))
def image(file,x,y,w,h):
    enc=base64.b64encode((A/file).read_bytes()).decode()
    return cell('',x,y,w,h,'shape=image;imageAspect=0;aspect=fixed;image=data:image/svg+xml,'+enc+';')
box(0,0,W,H,fill='#FFFFFF',stroke='none',rounded=False)
text('FRAME',30,20,170,46,34,True)
text('反馈驱动的拓扑搜索',210,25,495,40,27)
for x,hue,label in [(820,B,'前缀'),(965,P,'核查'),(1110,G,'补查')]:
    circle('',x,36,14,hue[3],hue[5]);text(label,x+24,24,95,35,21,color=SUB)
line([(1250,43),(1296,43)],SUB,1.3,True,True,both=True)
text('反馈',1308,24,100,35,21,color=SUB)
# Compact input vignette, asymmetrically placed.
box(30,190,190,345,fill='#FAFBFC',stroke=BORDER,dashed=True)
image('powergrid_artwork.svg',69,212,112,91)
text('当前电网状态',45,320,160,34,23,True,align='center')
text('基础动作 b',45,362,160,30,21,color=SUB,align='center')
for i in range(3):box(69+i*7,420-i*9,100,48,fill='#FFFFFF',stroke=BORDER)
text('O',89,400,90,42,27,True,align='center')
text('有序候选',45,482,160,32,22,align='center')
line([(220,353),(265,353)],SUB,1.5)
# Wide global-search region groups prefix and checking.
box(265,120,640,415,fill='#F7FAFD',stroke='none')
circle('1',288,139,39,B[2],B[6])
text('全局搜索',340,138,250,38,27,True)
text('反馈决定查询深度',340,180,490,30,23,color=SUB)
line([(292,230),(875,230)],'#DCE4ED',1,False,False)
text('前缀筛查',299,250,230,35,25,True,color=B[5])
box(300,294,226,144,fill='#FFFFFF',stroke=B[2])
text('评估前 p 个候选',309,303,208,32,22,align='center')
image('prefix.svg',307,345,212,52)
text('初始最佳 c₀',307,402,212,28,21,color=B[6],align='center')
line([(527,364),(581,364)],B[5],1.7)
text('反馈',529,325,54,27,19,color=SUB,align='center')
text('分批核查',585,250,250,35,25,True,color=P[5])
box(583,294,292,144,fill='#FFFFFF',stroke=P[2])
text('每批追加 h 个候选',594,303,270,32,22,align='center')
image('batch.svg',596,345,265,65)
text('更新最佳动作 c*',597,404,265,28,21,color=P[6],align='center')
line([(858,420),(882,420),(882,468),(626,468),(626,438)],P[4],1.5)
text('未满足停止条件，继续核查',314,473,525,32,22,color=P[6],align='center')
# Gate links depth selection to structural completion.
line([(905,354),(934,354)],INK,1.6)
cell('',934,313,76,82,f'rhombus;html=0;fillColor=#FFFFFF;strokeColor={SUB};strokeWidth=1.2;')
text('需\n补查',944,323,56,60,21,align='center')
line([(1010,354),(1035,354)],G[5],1.7)
text('是',1006,315,35,30,19,color=G[6],align='center')
box(1035,120,405,415,fill='#F3FAF7',stroke='none')
circle('2',1058,139,39,G[2],G[6])
text('同站补查',1110,138,300,38,27,True)
text('结构决定补查位置',1110,180,310,30,23,color=SUB)
line([(1063,230),(1410,230)],'#D7E9DF',1,False,False)
box(1064,256,145,64,fill='#FFFFFF',stroke=P[2])
text('当前最佳 c*',1076,260,120,27,20,color=SUB,align='center')
text('★  A',1088,286,96,28,23,True,color=P[5],align='center')
line([(1209,288),(1250,288)],G[5],1.4)
circle('A站',1250,251,74,G[1],G[6],23)
text('变电站归属',1328,264,100,46,19,color=SUB,align='center')
line([(1287,325),(1287,346),(1164.3,346),(1164.3,385)],G[4],1.2)
line([(1287,346),(1262.6,346),(1262.6,385)],G[4],1.2)
line([(1287,346),(1360.8,346),(1360.8,385)],G[4],1.2)
image('station.svg',1083,368,310,76)
text('沿原序列补查该站',1060,458,355,32,23,color=G[6],align='center')
text('跳过已查询候选',1060,494,355,28,21,color=SUB,align='center')
# Shared feedback shelf has small colored record tags.
box(265,625,640,112,fill='#F5F6F8',stroke='none')
text('共享仿真反馈',288,641,210,32,24,True)
text('已查询集合 E · 当前最佳 c*',288,686,576,30,22,color=SUB)
for i,(label,hue) in enumerate([('奖励',B),('负载率',P),('执行标记',G)]):
    x=520+i*117
    box(x,643,106,31,fill=hue[0],stroke='none')
    text(label,x,643,106,31,19,color=hue[6],align='center')
for x in [413,729]:line([(x,535),(x,625)],SUB,1.1,True,True,both=True)
line([(972,395),(972,687),(1035,687)],SUB,1.4)
text('否',981,565,38,30,20,color=SUB)
line([(1238,535),(1238,625)],G[5],1.7)
box(1035,625,405,112,fill='#FFFFFF',stroke=G[3],width=1.5)
circle('✓',1061,657,45,G[1],G[6],28)
text('最佳已评估拓扑动作',1120,644,295,36,24,True)
text('交给基础控制器执行',1120,690,295,28,21,color=SUB)
ET.indent(mx)
ET.ElementTree(mx).write(OUT/'FRAME_方法图_V2.drawio',encoding='utf-8',xml_declaration=True)
caption='图2：FRAME以仿真反馈控制全局搜索深度，并以当前最佳动作的变电站归属选择补查位置。全局搜索包括前缀筛查与分批核查；满足补查条件时，按原候选顺序评估同站未查询代表。各阶段复用已查询集合与最佳动作，最终将最佳已评估拓扑动作交给基础控制器。候选格与A站为机制示意，数量不对应实际预算。'
credit='power grid by Ary Prasetyo from Noun Project (CC BY 3.0), cropped and scaled. https://thenounproject.com/icon/power-grid-4199252/ ; https://creativecommons.org/licenses/by/3.0/ .'
plan={'utc':datetime.now(timezone.utc).isoformat(),'caption_first':caption,
    'external_caption_credit':credit,'canvas_px':[W,H],'figure_width_mm':170,
    'guide':'https://dy8q0bnq8y.feishu.cn/wiki/OGpcw6zaRiQ6OwkNxfYcl7IKnBh',
    'guide_reading':'Re-read workflow and all seven method-figure examples; actually viewed their images. Adopted asymmetric grouping, concrete objects, functional colors, differentiated feedback arrows and a compact legend.',
    'layout':'compact left input; wide global-search region with prefix and checking stages; smaller right structural-completion region; bottom feedback shelf and action output',
    'palette':{'prefix':{'scheme':'Blues 7','colors':B},'checking':{'scheme':'Purples 7','colors':P},'completion':{'scheme':'BuGn 7','colors':G}},
    'tools':['Draw.io native vector layout','Matplotlib SVG components','ColorBrewer official palettes','Noun Project free CC BY icon'],
    'attribution_location':'external figure caption only','prior_versions_preserved':True,
    'new_experiments':0,'model_or_parameter_changes':0}
(OUT/'制作说明与图注.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'图注与素材来源.md').write_text('# 方法图V2\n\n'+caption+'\n\n'+credit+'''

## 图例与语义

蓝色表示前缀筛查，紫色表示反馈核查，绿色表示同站补查。实线是搜索主流程，虚线是仿真反馈存取；空心候选未查询，实心候选已查询，星号标示当前最佳，下划线表示同站追加查询。颜色同时由名称、字母和线型辅助编码。

## 绘图前的五步

1. 核心句：反馈决定查多深，结构决定查哪里。
2. 必要元素：状态、基础动作、有序候选；前缀最佳；追加批次和反馈回路；条件补查；站归属及分散候选；共享反馈；最终动作。
3. 不对称布局：全局搜索占主区域，前缀与核查在内部分工；结构补查另成一组。共享反馈置于下方，最终输出在右下方。
4. 名称与正文一致：前缀筛查、分批核查、同站补查。图面不堆停止公式，正文算法负责精确条件。
5. 低饱和蓝、紫、绿分别编码功能；统一字体、轻边框与候选形状。图内无署名，外部图注保留许可。

## 与实现的对应

图为方法概览。精确补查条件仍为：最佳可用候选存在、仍有未查询代表，且前缀未达阈值或核查改变了初始最佳。站归属在补查入口固定；映射到代表后去除已查询项，按原相对顺序执行一轮。无可用拓扑动作时返回空，基础控制器执行基础动作。

本图不改变规则、模型或参数。原方法图V1和动机图全部保留。
''',encoding='utf-8')
print(OUT/'FRAME_方法图_V2.drawio')
