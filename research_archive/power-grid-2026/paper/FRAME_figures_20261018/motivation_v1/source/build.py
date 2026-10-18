"""Compose an editable diagrams.net figure from Matplotlib SVG components.

This is a conceptual illustration. No plotted positions or block counts are
experimental measurements, policy parameters, or measured performance gains.
"""
from pathlib import Path
import base64, hashlib, json, re, xml.etree.ElementTree as ET
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path('D:/Codex/GRL研究')
OUT = ROOT/'outputs/frame_motivation_v1_20261018'
ASSET = OUT/'assets'
OUT.mkdir(parents=True, exist_ok=True)
CB = json.loads((ASSET/'colorbrewer.json').read_text())
def hexcolor(rgb):
    return '#'+''.join(f'{int(n):02x}' for n in re.findall(r'\d+',rgb))
set2 = [hexcolor(x) for x in CB['Set2']['3']]
COLORS = {'A':set2[2], 'B':set2[0], 'C':set2[1]}
INK='#30343B'; MID='#636972'; GRAY='#D5D9DD'; LIGHT='#F4F5F6'
FONT='Noto Sans SC'
plt.rcParams.update({'font.family':FONT,'svg.fonttype':'none','font.size':20})

def strip(name, n, queried, additional=(), stations=None, best=None, potential=None):
    """Exact physical size in diagrams.net CSS px, preserving editable SVG text."""
    w=366 if stations is None else 450
    h=52
    fig=plt.figure(figsize=(w/96,h/96),dpi=96)
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,w);ax.set_ylim(0,h);ax.axis('off')
    gap=6; bw=(w-gap*(n-1))/n
    for i in range(n):
        x=i*(bw+gap)
        c=COLORS[stations[i]] if stations else GRAY
        r=Rectangle((x+.7,5.5),bw-1.4,32,facecolor=c if i in queried else 'white',
                    edgecolor=INK if i in queried else '#AAB0B6',
                    linewidth=2.5 if i in additional else 1.0,
                    linestyle='solid' if i in queried else (0,(2.3,1.7)))
        r.set_gid(f'candidate-{i+1}-'+('added' if i in additional else 'queried' if i in queried else 'unqueried'))
        ax.add_patch(r)
        if stations:
            # A/B/C also preserves group identity in grayscale and for CVD.
            ax.text(x+bw/2,21.5,stations[i],ha='center',va='center',fontsize=18,color=INK)
        if i==best:
            ax.plot(x+bw/2,46,marker='*',markersize=10,color=INK,clip_on=False)
        if i==potential:
            ax.plot(x+bw/2,46,marker='*',markersize=10,markerfacecolor='white',
                    markeredgecolor=INK,markeredgewidth=1,clip_on=False)
    fig.savefig(ASSET/f'{name}.svg',transparent=True)
    plt.close(fig)
    return w,h

for name,q,b,p in [('fixed_a',range(6),2,None),('fixed_b',range(6),None,8),
                  ('frame_a',range(4),2,None),('frame_b',range(10),8,None)]:
    strip(name,12,q,best=b,potential=p)
STATIONS=list('BCA CBCB A C A B A'.replace(' ',''))
assert len(STATIONS)==12 and STATIONS[2]=='A'
assert [i for i,s in enumerate(STATIONS) if s=='A']==[2,7,9,11]
strip('global',12,range(7),range(4,7),STATIONS,best=2)
strip('local',12,[0,1,2,3,7,9,11],[7,9,11],STATIONS,best=2)

mx=ET.Element('mxfile',host='app.diagrams.net',agent='FRAME figure source',version='32.4.1')
diagram=ET.SubElement(mx,'diagram',id='frame-motivation-cn',name='动机图')
model=ET.SubElement(diagram,'mxGraphModel',dx='1100',dy='730',grid='1',gridSize='10',
                    guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',
                    pageScale='1',pageWidth='1100',pageHeight='730',math='0',shadow='0',background='#FFFFFF')
root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
counter=1
def cell(value,x,y,w,h,style):
    global counter
    counter+=1
    c=ET.SubElement(root,'mxCell',id=str(counter),value=value,style=style,vertex='1',parent='1')
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
    return c

def text(value,x,y,w,h,size=22,bold=False,color=INK,align='left'):
    return cell(value,x,y,w,h,f'text;html=0;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;whiteSpace=wrap;rounded=0;fontFamily={FONT};fontSize={size};fontStyle={1 if bold else 0};fontColor={color};spacing=0;')

def rect(x,y,w,h,fill='none',stroke='none',width=1,rounded=False,dashed=False):
    return cell('',x,y,w,h,f'rounded={1 if rounded else 0};arcSize=12;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};strokeWidth={width};dashed={1 if dashed else 0};')

def edge(points,color=MID,width=1.3,dashed=False,arrow=False):
    global counter
    counter+=1
    c=ET.SubElement(root,'mxCell',id=str(counter),edge='1',parent='1',
        style=f'edgeStyle=none;rounded=0;html=0;endArrow={"block" if arrow else "none"};endFill=1;endSize=6;strokeColor={color};strokeWidth={width};dashed={int(dashed)};dashPattern=4 4;')
    g=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
    ET.SubElement(g,'mxPoint',x=str(points[0][0]),y=str(points[0][1]),**{'as':'sourcePoint'})
    ET.SubElement(g,'mxPoint',x=str(points[-1][0]),y=str(points[-1][1]),**{'as':'targetPoint'})
    if len(points)>2:
        a=ET.SubElement(g,'Array',**{'as':'points'})
        for x,y in points[1:-1]:ET.SubElement(a,'mxPoint',x=str(x),y=str(y))

def image(path,x,y,w,h):
    content=path.read_bytes()
    # Preserve all SVG content in the drawing, no external file dependency.
    encoded=base64.b64encode(content).decode()
    return cell('',x,y,w,h,'shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed;image=data:image/svg+xml,'+encoded+';')

def bracket(x1,x2,y,label=None,color=MID):
    edge([(x1,y-5),(x1,y),(x2,y),(x2,y-5)],color,1.2)
    if label:text(label,x1-12,y+7,x2-x1+24,25,19,color=color,align='center')

def cutoff(x,y1,y2):edge([(x,y1),(x,y2)],MID,1.4,True)

# White page rectangle also preserves deliberate margins in all cropped exports.
rect(0,0,1100,730,fill='#FFFFFF')
# Context, two independent decisions, and intentionally restrained visual hierarchy.
icon=(ASSET/'noun_electricity_pole_1364802.svg').read_text().replace('#000000',INK)
(ASSET/'electricity_pole_recolored.svg').write_text(icon,encoding='utf-8')
image(ASSET/'electricity_pole_recolored.svg',25,16,36,46)
text('沿既定候选顺序，决定查多深、查哪里',78,17,920,42,26,True)
edge([(24,82),(1076,82)],'#C9CDD2',1)
text('(a) 不同状态需要不同搜索深度',24,99,510,36,24,True)
text('(b) 同站方案分散在候选序列中',564,99,512,36,24,True)
edge([(536,102),(536,598)],'#DADDE1',1)

# Left: same candidate-batch positions in both strategies, different cutoff.
text('固定预算',24,147,450,30,22,True)
text('状态 A',24,197,82,30,21);image(ASSET/'fixed_a.svg',116,181,366,52)
text('后续查询可能冗余',116,240,386,27,20,color=MID)
text('状态 B',24,291,82,30,21);image(ASSET/'fixed_b.svg',116,275,366,52)
text('遗漏后部改善机会',116,334,386,27,20,color=MID)
cutoff(299,178,327)
text('统一截断',306,149,194,27,20,color=MID)

text('FRAME：反馈核查',24,389,460,30,22,True)
text('状态 A',24,439,82,30,21);image(ASSET/'frame_a.svg',116,423,366,52)
text('及时停止',116,480,380,27,20,color=MID)
text('状态 B',24,533,82,30,21);image(ASSET/'frame_b.svg',116,517,366,52)
text('继续核查',116,574,386,27,20,color=MID)
cutoff(237,420,475);cutoff(423,514,569)

# Right: exact equal extra budget and unchanged candidate ordering.
text('当前最佳：A 站动作',564,147,500,30,21,color=MID)
text('全局扩展',564,200,360,30,22,True)
text('追加 3 次',955,200,121,30,21,True,align='right')
image(ASSET/'global.svg',596,244,450,52)
cutoff(745,239,301)
bracket(596,741,309,'共同前缀')
bracket(748,855,309,'连续扩展')

text('FRAME：同站补查',564,408,360,30,22,True)
text('追加 3 次',955,408,121,30,21,True,align='right')
image(ASSET/'local.svg',596,488,450,52)
cutoff(745,483,545)
bracket(596,741,553,'共同前缀')
# A sparse branched connector groups only the three previously unqueried A actions.
edge([(687,486),(687,462),(1030,462)],COLORS['A'],2)
for x in [879,955,1030]:edge([(x,462),(x,484)],COLORS['A'],2,arrow=True)

# Shared legend avoids conflating a station category with queried status.
edge([(24,623),(1076,623)],'#C9CDD2',1)
rect(25,643,21,20,GRAY,INK)
text('已查询',55,639,125,29,20,color=MID)
rect(210,643,21,20,'#FFFFFF','#AAB0B6',dashed=True)
text('未查询',240,639,125,29,20,color=MID)
rect(398,643,21,20,GRAY,INK,2.5)
text('追加查询',428,639,160,29,20,color=MID)
text('★ 当前最佳',639,639,200,29,20,color=MID)
text('☆ 潜在改善',882,639,194,29,20,color=MID)
text('示意：左格为候选批次，右格为单个候选；A/B/C 为变电站归属',24,680,1052,26,19,color=MID)

ET.indent(mx)
ET.ElementTree(mx).write(OUT/'FRAME_动机图_V1.drawio',encoding='utf-8',xml_declaration=True)
plan={
 'caption':'图1：FRAME的搜索动机。不同运行状态所需的评估数量存在差异，同一变电站的连接方案又可能分散在候选序列中。FRAME根据已有仿真反馈调整全局搜索深度，并依据当前最佳动作的变电站归属确定补查位置。右侧两种方式追加相同数量的查询，候选顺序保持不变。此图为机制示意，方格数量不对应实际参数或实验结果。',
 'icon_attribution':'Electricity Pole by Georgiana Ionescu, The Noun Project, CC BY 3.0 US. Recolored and resized. Original work: https://thenounproject.com/icon/electricity-pole-1364802/ ; licensed SVG redistribution: https://commons.wikimedia.org/wiki/File:Electric_Pole.svg ; license: https://creativecommons.org/licenses/by/3.0/us/ .',
 'icon_note':'Flaticon SVG entry required a premium download; iconfont detail required login. No gated assets were extracted. Selected a Noun Project icon with a previously CC-licensed SVG distribution whose license review is preserved by Wikimedia Commons. Noun current page labels it Royalty-Free; the reused file is the older CC BY 3.0 US distribution.',
 'tools':{'sequence_components':'Matplotlib SVG export','layout_and_export':'diagrams.net / draw.io 32.4.1 native editable text, connectors, SVG image components','palette':'ColorBrewer 2.0 Set2, 3 classes; direct official JSON export'},
 'palette':COLORS,'figure_width_mm':170,'canvas_px':[1100,730],
 'font':'Noto Sans SC','minimum_label_css_px':19,
 'schematic':True,'new_experiments':0,
 'invariants':{'same_candidate_order':True,'global_additional_queries':3,'local_additional_queries':3,'local_additional_station':'A','left_blocks_are_batches':True,'right_blocks_are_candidates':True},
 'guideline':'https://dy8q0bnq8y.feishu.cn/wiki/OGpcw6zaRiQ6OwkNxfYcl7IKnBh',
 'utc':datetime.now(timezone.utc).isoformat()
}
(OUT/'制作说明与图注.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'图注与素材署名.md').write_text('# 动机图V1\n\n'+plan['caption']+'\n\n## 图标署名\n\n'+plan['icon_attribution']+'\n\n## 配色\n\nColorBrewer 2.0 Set2（三类）：A '+COLORS['A']+'，B '+COLORS['B']+'，C '+COLORS['C']+'。字母辅助区分变电站；实线边框表示已查询，虚线边框表示未查询，较粗边框表示本轮追加。左侧空心星号表示截断后尚未评估的潜在改善机会。\n\n图注放在正文中，素材署名随正式插图保留。图中不包含实验曲线或虚构的测量数值。所有原稿和实验记录不改。\n',encoding='utf-8')
print(OUT/'FRAME_动机图_V1.drawio')
