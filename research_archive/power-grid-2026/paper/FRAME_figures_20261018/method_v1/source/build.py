"""FRAME main figure: caption, module I/O, layout, names, visual language."""
from pathlib import Path
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
import base64,hashlib,json,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path('D:/Codex/GRL研究')
OUT=ROOT/'outputs/frame_method_v1_20261018';ASSET=OUT/'assets'
W,H=1200,870
CB=json.loads((ASSET/'colorbrewer.json').read_text())
def hx(rgb):return '#'+''.join(f'{int(n):02x}' for n in re.findall(r'\d+',rgb))
BL=[hx(x) for x in CB['Blues']['7']];GY=[hx(x) for x in CB['Greys']['9']]
BLUE=BL[5];PALE=BL[0];INK=GY[7];MID=GY[5];FONT='Noto Sans SC'
plt.rcParams.update({'font.family':FONT,'svg.fonttype':'none','font.size':18})

def strip(name,labels,queried,best=None,station=False):
    fig=plt.figure(figsize=(230/96,58/96),dpi=96)
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,230);ax.set_ylim(0,58);ax.axis('off')
    n=len(labels);gap=7;bw=(230-(n-1)*gap)/n
    for i,label in enumerate(labels):
        x=i*(bw+gap)
        active=station and label=='A'
        r=FancyBboxPatch((x+.8,7),bw-1.6,32,boxstyle='round,pad=0,rounding_size=1.5',
            facecolor=BL[2] if active else GY[2] if i in queried else 'white',
            edgecolor=BLUE if active else INK if i in queried else GY[3],lw=1,
            linestyle='solid' if i in queried else (0,(2.3,1.7)))
        ax.add_patch(r)
        if label:ax.text(x+bw/2,23,label,ha='center',va='center',fontsize=17,color=BLUE if active else MID)
        if i==best:ax.plot(x+bw/2,49,marker='*',markersize=11,color=INK,clip_on=False)
    fig.savefig(ASSET/f'{name}.svg',transparent=True);plt.close(fig)

strip('prefix',['']*7,range(4),2)
strip('batch',['']*6,range(6),4)
strip('same_station',list('ABACBA'),[0,2,5],station=True)

icon=ET.parse(ASSET/'noun_powergrid_4199252.svg').getroot()
assert not icon.findall('.//{http://www.w3.org/2000/svg}script')
for e in list(icon):
    if e.tag.rsplit('}',1)[-1]=='text':icon.remove(e)
icon.set('viewBox','8 18 104 84')
ET.ElementTree(icon).write(ASSET/'powergrid_artwork.svg',encoding='utf-8',xml_declaration=True)

mx=ET.Element('mxfile',host='app.diagrams.net',version='32.4.1')
d=ET.SubElement(mx,'diagram',id='frame-method-cn',name='FRAME方法图')
model=ET.SubElement(d,'mxGraphModel',dx=str(W),dy=str(H),grid='1',gridSize='10',
    guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',
    pageWidth=str(W),pageHeight=str(H),math='0',shadow='0',background='#FFFFFF')
root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
idx=1
def cell(value,x,y,w,h,style):
    global idx
    idx+=1;c=ET.SubElement(root,'mxCell',id=str(idx),value=value,style=style,vertex='1',parent='1')
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
    return c
def text(value,x,y,w,h,size=23,bold=False,color=INK,align='left'):
    return cell(value,x,y,w,h,f'text;html=0;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;whiteSpace=wrap;fontFamily={FONT};fontSize={size};fontStyle={int(bold)};fontColor={color};spacing=0;')
def box(x,y,w,h,fill='white',stroke=GY[3],rounded=True,width=1.2):
    return cell('',x,y,w,h,f'rounded={int(rounded)};arcSize=5;fillColor={fill};strokeColor={stroke};strokeWidth={width};html=0;')
def edge(points,color=BLUE,width=1.5,dashed=False,arrow=True,both=False):
    global idx
    idx+=1;c=ET.SubElement(root,'mxCell',id=str(idx),parent='1',edge='1',
        style=f'edgeStyle=none;rounded=0;html=0;endArrow={"block" if arrow else "none"};startArrow={"open" if both else "none"};endFill=1;endSize=7;startSize=7;strokeColor={color};strokeWidth={width};dashed={int(dashed)};dashPattern=5 5;')
    g=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
    ET.SubElement(g,'mxPoint',x=str(points[0][0]),y=str(points[0][1]),**{'as':'sourcePoint'})
    ET.SubElement(g,'mxPoint',x=str(points[-1][0]),y=str(points[-1][1]),**{'as':'targetPoint'})
    if len(points)>2:
        arr=ET.SubElement(g,'Array',**{'as':'points'})
        for x,y in points[1:-1]:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
def image(path,x,y,w,h):
    content=base64.b64encode(path.read_bytes()).decode()
    return cell('',x,y,w,h,'shape=image;imageAspect=0;aspect=fixed;image=data:image/svg+xml,'+content+';')

box(0,0,W,H,fill='#FFFFFF',stroke='none',rounded=False)
text('FRAME：反馈驱动的拓扑搜索',35,20,620,40,28,True)
edge([(760,41),(815,41)]);text('主流程',828,23,122,35,21,color=MID)
edge([(973,41),(1028,41)],MID,1.3,True,both=True);text('反馈存取',1040,23,130,35,21,color=MID)

# Input data is available throughout this search. The main execution enters prefix.
box(35,90,1130,90,fill=GY[1],stroke='none')
image(ASSET/'powergrid_artwork.svg',54,102,76,62)
text('当前电网状态 s  ·  基础动作 b  ·  全局候选序列 O',154,100,975,34,25)
text('代表映射 π 与原始动作的变电站归属 g',154,140,975,30,22,color=MID)
edge([(180,180),(180,240)])

# Three modules use consistent titles and shared visual grammar.
box(35,240,285,260,fill='white',stroke=BL[2])
box(385,240,300,260,fill='white',stroke=BL[2])
box(895,240,270,260,fill='white',stroke=BL[2])
for number,name,x,w in [(1,'前缀筛查',35,285),(2,'分批核查',385,300),(3,'同站补查',895,270)]:
    box(x+1,241,w-2,58,fill=PALE,stroke='none')
    text(f'{number}  {name}',x+20,253,w-40,35,25,True,color=BLUE)

text('评估前 p 个代表候选',57,312,243,32,22,align='center')
image(ASSET/'prefix.svg',62,350,230,58)
text('初始最佳 c₀',56,421,243,30,23,align='center')
text('阈值标记 H',56,460,243,30,22,color=MID,align='center')

text('每批追加 h 个候选',405,312,260,32,22,align='center')
image(ASSET/'batch.svg',420,350,230,58)
box(425,415,220,43,fill=GY[1],stroke='none')
text('仿真并更新 c*、t',436,420,198,32,22,align='center')
edge([(645,436),(665,436),(665,378),(653,378)],BLUE,1.2)
text('未满足停止条件时继续',403,469,267,26,20,color=MID,align='center')

text('定位变电站 v = g(c*)',913,312,234,32,21,align='center')
image(ASSET/'same_station.svg',915,350,230,58)
text('按原序列补查该站',913,422,234,30,22,align='center')
text('尚未查询的代表',913,461,234,30,22,color=MID,align='center')

edge([(320,375),(385,375)])
text('c₀, H',321,337,63,27,20,color=MID,align='center')
edge([(685,375),(730,375)])
text('核查结束',691,290,193,31,21,color=MID,align='center')
cell('',730,325,120,100,f'rhombus;html=0;fillColor={PALE};strokeColor={BLUE};strokeWidth=1.3;')
text('触发\n补查？',750,344,80,62,22,align='center')
edge([(850,375),(895,375)]);text('是',856,341,35,28,20,color=MID,align='center')
edge([(790,425),(790,580)])
text('否',804,444,45,30,21,color=MID)
text('沿用已有最佳',808,531,250,30,21,color=MID)

# Shared oracle feedback. Dashed lines are state exchanges, not extra phases.
box(35,580,1130,118,fill=GY[1],stroke='none')
for x in [180,535,1030]:edge([(x,500),(x,580)],MID,1.2,True,both=True)
text('共享仿真反馈与当前最佳',58,594,1084,37,25,True,align='center')
text('已查询集合 E   ·   奖励、负载率与执行标记   ·   最佳动作 c*',58,644,1084,35,23,color=MID,align='center')

edge([(600,698),(600,758)])
box(275,758,650,88,fill=PALE,stroke=BL[2])
text('返回最佳已评估拓扑动作 c*',295,769,610,34,25,True,color=BLUE,align='center')
text('交给基础控制器组合执行',295,809,610,29,22,color=MID,align='center')

ET.indent(mx)
ET.ElementTree(mx).write(OUT/'FRAME_方法图_V1.drawio',encoding='utf-8',xml_declaration=True)

caption='图2：FRAME以当前最佳动作连接前缀筛查、分批核查与同站补查，根据仿真反馈控制全局搜索深度，并利用动作的变电站归属选择补查范围。各阶段共用已查询集合E与最佳动作c*。H表示前缀最佳动作是否达到低负载率阈值，t为连续未改善批数；存在可用动作和未查询候选时，H为假或核查改变了初始最佳动作c₀，才触发一轮同站补查。没有可用拓扑动作时，基础控制器执行其基础控制操作。'
credit='power grid by Ary Prasetyo from Noun Project (CC BY 3.0), cropped and scaled. https://thenounproject.com/icon/power-grid-4199252/ ; https://creativecommons.org/licenses/by/3.0/ .'
plan={'utc':datetime.now(timezone.utc).isoformat(),'caption_first':caption,'external_caption_credit':credit,
 'canvas_px':[W,H],'figure_width_mm':170,'tools':['Matplotlib SVG','Draw.io native editable layout and export','ColorBrewer Blues/Greys','The Noun Project free CC BY 3.0 icon'],
 'palette':{'accent':BLUE,'pale':PALE,'text':INK,'secondary':MID},'guide':'https://dy8q0bnq8y.feishu.cn/wiki/OGpcw6zaRiQ6OwkNxfYcl7IKnBh',
 'guide_reading':'Read method workflow steps and core principles again, then visually viewed Agentic Reasoning Method and WAPITI examples, including the legend and differing arrow conventions.',
 'source_files':['outputs/frame_cn_v4_20261018/FRAME_ICAPS2027_中文重写稿_V4.md','work/grid70/policies.py','work/grid89_review/policies.py'],
 'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['outputs/frame_cn_v4_20261018/FRAME_ICAPS2027_中文重写稿_V4.md','work/grid70/policies.py','work/grid89_review/policies.py']},
 'scope':'Method overview; exact stopping branches remain in Algorithm 1 and this caption. The diamond does not imply unconditional same-station completion.',
 'attribution_location':'external figure caption only','new_experiments':0,'parameter_changes':0}
(OUT/'制作说明与图注.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'图注与素材来源.md').write_text('# 方法图V1的外部图注\n\n'+caption+'\n\n'+credit+'\n\n## 指南五步落实\n\n1. 先写核心句：反馈决定深度，当前最佳动作的站归属决定位置。\n2. 元素与流转：输入s/b/O/π/g；前缀输出c₀和H；分批核查更新c*与t；条件分支决定是否补查；共享结果E供各模块读写；输出交给基础控制器。\n3. 横向三个核心模块，分批核查包含继续核查回路；补查前有显式条件菱形。\n4. 模块命名逐字对应正文。\n5. 蓝灰配色、统一字体与候选符号；实线主流程、虚线反馈。\n\n## 实现核对\n\n- 核查停止：候选耗尽；或存在可用动作且查询数不少于q_min、连续L批无改善；前缀未达到阈值的分支还可在后续最佳动作低于阈值时结束。\n- 补查触发：可用最佳存在、有未查询代表，且前缀标记H为假或c*改变。\n- 补查集合：先按原始动作的变电站归属选择，再经π映射到代表，去除E；按原相对顺序执行一轮。图中短序列是示意。\n- 图中不增加训练阶段、全局质量保证或新超参数。\n- 图内没有署名行；图标授权署名置于本外部图注，原SVG保持不变。\n',encoding='utf-8')
print(OUT/'FRAME_方法图_V1.drawio')
