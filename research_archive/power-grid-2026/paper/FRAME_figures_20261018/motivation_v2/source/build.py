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
from matplotlib.patches import Rectangle, FancyBboxPatch

ROOT = Path('D:/Codex/GRL研究')
OUT = ROOT/'outputs/frame_motivation_v2_20261018'
ASSET = OUT/'assets'
OUT.mkdir(parents=True, exist_ok=True)
CB = json.loads((ASSET/'colorbrewer.json').read_text())
def hexcolor(rgb):
    return '#'+''.join(f'{int(n):02x}' for n in re.findall(r'\d+',rgb))
BLUES = [hexcolor(x) for x in CB['Blues']['7']]
GREYS = [hexcolor(x) for x in CB['Greys']['9']]
BLUE=BLUES[5]
COLORS = {'A':BLUES[2], 'B':GREYS[1], 'C':GREYS[1]}
INK=GREYS[7]; MID=GREYS[5]; GRAY=GREYS[2]; LIGHT=GREYS[1]
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
        ec=BLUE if stations and stations[i]=='A' else INK if i in queried else GREYS[3]
        r=FancyBboxPatch((x+.8,6.5),bw-1.6,30,
                    boxstyle='round,pad=0,rounding_size=1.5',
                    facecolor=c if i in queried else 'white',edgecolor=ec,
                    linewidth=1.0,linestyle='solid' if i in queried else (0,(2.3,1.7)))
        r.set_gid(f'candidate-{i+1}-'+('added' if i in additional else 'queried' if i in queried else 'unqueried'))
        ax.add_patch(r)
        if i in additional:
            ax.plot([x+2,x+bw-2],[2,2],color=BLUE if stations[i]=='A' else INK,lw=2.7,
                    solid_capstyle='butt')
        if stations:
            # A/B/C also preserves group identity in grayscale and for CVD.
            ax.text(x+bw/2,21.5,stations[i],ha='center',va='center',fontsize=17,
                    color=BLUE if stations[i]=='A' else MID)
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
model=ET.SubElement(diagram,'mxGraphModel',dx='1100',dy='762',grid='1',gridSize='10',
                    guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',
                    pageScale='1',pageWidth='1100',pageHeight='762',math='0',shadow='0',background='#FFFFFF')
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
rect(0,0,1100,762,fill='#FFFFFF')
# Context, two independent decisions, and intentionally restrained visual hierarchy.
# Preserve the downloaded original. Relocate its attribution text into a visible
# credit line on the full figure, and crop only the artwork's whitespace.
icon=ET.parse(ASSET/'noun_powergrid_4199252.svg').getroot()
ns={'s':'http://www.w3.org/2000/svg'}
assert not icon.findall('.//s:script',ns)
for el in list(icon):
    if el.tag.rsplit('}',1)[-1]=='text':icon.remove(el)
icon.set('viewBox','8 18 104 84')
ET.ElementTree(icon).write(ASSET/'powergrid_artwork.svg',encoding='utf-8',xml_declaration=True)
image(ASSET/'powergrid_artwork.svg',28,12,77,62)
text('仿真查询：查多深，查哪里',130,18,925,42,26,True)
edge([(24,82),(1076,82)],'#C9CDD2',1)
text('(a) 不同状态需要不同搜索深度',24,99,510,36,23,True)
text('(b) 同站方案分散在候选序列中',564,99,512,36,23,True)
edge([(536,102),(536,598)],'#DADDE1',1)

# Left: same candidate-batch positions in both strategies, different cutoff.
text('固定预算',24,147,450,30,22)
text('状态 A',24,197,82,30,21);image(ASSET/'fixed_a.svg',116,181,366,52)
text('后续查询可能冗余',116,240,386,27,20,color=MID)
text('状态 B',24,291,82,30,21);image(ASSET/'fixed_b.svg',116,275,366,52)
text('遗漏后部改善机会',116,334,386,27,20,color=MID)
cutoff(299,178,327)
text('统一截断',306,149,194,27,20,color=MID)

text('FRAME：反馈核查',24,389,460,30,22,True,color=BLUE)
text('状态 A',24,439,82,30,21);image(ASSET/'frame_a.svg',116,423,366,52)
text('及时停止',116,480,380,27,20,color=MID)
text('状态 B',24,533,82,30,21);image(ASSET/'frame_b.svg',116,517,366,52)
text('继续核查',116,574,386,27,20,color=MID)
cutoff(237,420,475);cutoff(423,514,569)

# Right: exact equal extra budget and unchanged candidate ordering.
text('当前最佳：A 站动作',564,147,500,30,21,color=MID)
text('全局扩展',564,200,360,30,22)
text('追加 3 次',955,200,121,30,20,align='right')
image(ASSET/'global.svg',596,244,450,52)
cutoff(745,239,301)
bracket(596,741,309,'共同前缀')
bracket(748,855,309,'连续扩展')

text('FRAME：同站补查',564,408,360,30,22,True,color=BLUE)
text('追加 3 次',955,408,121,30,20,align='right')
image(ASSET/'local.svg',596,488,450,52)
cutoff(745,483,545)
bracket(596,741,553,'共同前缀')
# A sparse branched connector groups only the three previously unqueried A actions.
edge([(687,486),(687,462),(1030,462)],BLUE,1.5)
for x in [879,955,1030]:edge([(x,462),(x,484)],BLUE,1.5,arrow=True)

# Shared legend avoids conflating a station category with queried status.
edge([(24,623),(1076,623)],'#C9CDD2',1)
rect(25,643,21,20,GRAY,INK)
text('已查询',55,639,125,29,20,color=MID)
rect(210,643,21,20,'#FFFFFF',GREYS[3],dashed=True)
text('未查询',240,639,125,29,20,color=MID)
rect(398,643,21,20,GRAY,INK)
edge([(398,669),(419,669)],INK,2.7)
text('追加查询',428,639,160,29,20,color=MID)
text('★ 当前最佳',639,639,200,29,20,color=MID)
text('☆ 潜在改善',882,639,194,29,20,color=MID)
text('示意：左格为候选批次，右格为单个候选；A/B/C 为变电站归属',24,680,1052,26,19,color=MID)
text('power grid by Ary Prasetyo / Noun Project (CC BY 3.0). Cropped and scaled.',24,720,1052,26,17,color=MID)

ET.indent(mx)
ET.ElementTree(mx).write(OUT/'FRAME_动机图_V2.drawio',encoding='utf-8',xml_declaration=True)
plan={
 'caption':'图1：FRAME的搜索动机。不同运行状态所需的评估数量存在差异，同一变电站的连接方案又可能分散在候选序列中。FRAME根据已有仿真反馈调整全局搜索深度，并依据当前最佳动作的变电站归属确定补查位置。右侧两种方式追加相同数量的查询，候选顺序保持不变。此图为机制示意，方格数量不对应实际参数或实验结果。',
 'icon_attribution':'power grid by Ary Prasetyo from Noun Project (CC BY 3.0). Cropped and scaled; embedded attribution relocated to the visible figure credit. https://thenounproject.com/icon/power-grid-4199252/ ; https://creativecommons.org/licenses/by/3.0/ .',
 'icon_note':'After account registration the free With Attribution option explicitly offered CC BY-3.0 SVG. The user downloaded noun_powergrid_4199252.svg and supplied it from Downloads; the original is preserved unchanged on D:. No paid assets or subscriptions used.',
 'tools':{'sequence_components':'Matplotlib SVG export','layout_and_export':'diagrams.net / draw.io 32.4.1 native editable text, connectors, SVG image components','palette':'ColorBrewer 2.0 Blues (7) and Greys (9), official JSON export'},
 'palette':{'station_fill':COLORS,'accent':BLUE,'body_text':INK,'secondary_text':MID},'figure_width_mm':170,'canvas_px':[1100,762],
 'font':'Noto Sans SC','minimum_body_label_css_px':19,'credit_css_px':17,
 'schematic':True,'new_experiments':0,
 'invariants':{'same_candidate_order':True,'global_additional_queries':3,'local_additional_queries':3,'local_additional_station':'A','left_blocks_are_batches':True,'right_blocks_are_candidates':True},
 'guideline':'https://dy8q0bnq8y.feishu.cn/wiki/OGpcw6zaRiQ6OwkNxfYcl7IKnBh',
 'utc':datetime.now(timezone.utc).isoformat()
}
(OUT/'制作说明与图注.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'图注与素材署名.md').write_text('# 动机图V2\n\n'+plan['caption']+'\n\n## 图标署名\n\n'+plan['icon_attribution']+'\n\n## 配色\n\n采用ColorBrewer官方Blues与Greys。目标A站用浅蓝 '+COLORS['A']+' 填充、深蓝 '+BLUE+' 描边；其它变电站用灰色，字母辅助区分归属。实线边框表示已查询，虚线边框表示未查询，下方短线标出本轮追加。左侧空心星号表示截断后尚未评估的潜在改善机会。\n\n本轮移除橙绿配色，降低边框与标题重量，原V1保留。原始下载SVG保持逐字节不变；派生图只裁切空白并缩放，原图底部作者文字改放到整张动机图的可见署名行中。图中不包含实验曲线或虚构测量数值，论文正文和实验记录不改。\n',encoding='utf-8')
print(OUT/'FRAME_动机图_V2.drawio')
