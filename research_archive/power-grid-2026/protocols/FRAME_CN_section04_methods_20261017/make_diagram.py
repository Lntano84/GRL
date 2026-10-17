from pathlib import Path
from html import escape
import xml.etree.ElementTree as ET

ROOT=Path('D:/Codex/GRL研究')
OUT=ROOT/'outputs/frame_paper_sections_20261017/04_方法/figures'
OUT.mkdir(parents=True,exist_ok=True)
font='Microsoft YaHei, Noto Sans CJK SC, sans-serif'
ink='#263846';muted='#5D6D79';blue='#6D89A8';teal='#487C73'
nodes=[
('input',40,320,180,140,'输入',['当前状态 s','基础动作 b','候选序列 O'],'#F3F5F7','#A5AFB8',16),
('cache',280,125,1050,85,'候选仿真与结果缓存',['共享本次搜索的反馈 · 每个代表最多仿真一次'],'#F0F6F4',teal,18),
('prefix',280,280,280,225,'01  前缀筛查',['评估前 min(p, M) 个候选','保存仿真结果','记录前缀最佳动作','按负载率选择后续分支'],'#EFF4FA',blue,17),
('verify',620,280,360,225,'02  分批核查',['每批追加至多 h 个候选','更新最佳动作与无改进批数','核查分支：最低查询量与停滞条件','扩展分支：负载率或停滞条件','候选耗尽时结束全局搜索'],'#E6F1ED',teal,16),
('local',1040,280,290,225,'03  同站补查',['检查分支与动作变化条件','固定最佳动作所属变电站','评估该站尚未查询的候选','更新最佳动作，只补查一次'],'#EFF4FA',blue,16),
('output',1390,320,175,140,'输出',['最佳已评估动作','无合格候选则返回空'],'#F3F5F7','#A5AFB8',14),
]
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="740" viewBox="0 0 1600 740"><rect width="1600" height="740" fill="white"/>',
 '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 Z" fill="#526571"/></marker><marker id="feedback" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto-start-reverse"><path d="M0,0 L7,3.5 L0,7 Z" fill="#77958D"/></marker></defs>',
 f'<g font-family="{font}" fill="{ink}">',
 '<text x="45" y="52" font-size="28" font-weight="bold">FRAME：反馈驱动的拓扑搜索</text>',
 f'<text x="45" y="88" fill="{muted}" font-size="18">根据仿真反馈控制搜索深度，按变电站信息补查相关候选</text>']
for key,x,y,w,h,title,lines,fill,stroke,fs in nodes:
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    svg.append(f'<text x="{x+w/2}" y="{y+36}" text-anchor="middle" font-size="22" font-weight="bold">{escape(title)}</text>')
    if key=='cache':
        svg.append(f'<text x="{x+w/2}" y="{y+65}" text-anchor="middle" font-size="{fs}" fill="{muted}">{escape(lines[0])}</text>')
    else:
        gap=28 if key not in ('input','output') else 26
        yy=y+75 if key not in ('input','output') else y+68
        for line in lines:
            svg.append(f'<text x="{x+w/2}" y="{yy}" text-anchor="middle" font-size="{fs}">{escape(line)}</text>');yy+=gap
paths=[('M220 390 H280',False),('M560 390 H620',False),('M980 390 H1040',False),('M1330 390 H1390',False),
 ('M800 505 V560 H1477.5 V460',False),('M420 280 V210',True),('M800 280 V210',True),('M1185 280 V210',True)]
for path,dashed in paths:
    svg.append(f'<path d="{path}" fill="none" stroke="{"#77958D" if dashed else "#526571"}" stroke-width="2.1" stroke-linejoin="round" marker-end="url(#{"feedback" if dashed else "arrow"})"'+(' stroke-dasharray="7 5" marker-start="url(#feedback)"' if dashed else '')+'/>')
svg.append(f'<text x="1100" y="549" text-anchor="middle" font-size="16" fill="{muted}">未触发同站补查时直接返回</text>')
svg.append('<path d="M310 645 H380" stroke="#526571" stroke-width="2.1" marker-end="url(#arrow)"/><text x="398" y="651" font-size="17">搜索主流程</text>')
svg.append('<path d="M720 645 H790" stroke="#77958D" stroke-width="2.1" stroke-dasharray="7 5" marker-end="url(#feedback)" marker-start="url(#feedback)"/><text x="808" y="651" font-size="17">候选仿真与反馈复用</text>')
svg.append(f'<text x="800" y="708" text-anchor="middle" font-size="17" fill="{muted}">p = 128　 h = 64　 q_min = 256　 L = 2　 ρ_safe = 0.9</text></g></svg>')
(OUT/'FRAME_方法流程.svg').write_text('\n'.join(svg),encoding='utf-8')

# Native editable diagrams.net document: individual vertices and routed edges.
mx=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17')
diagram=ET.SubElement(mx,'diagram',name='FRAME方法流程',id='FRAME-main')
model=ET.SubElement(diagram,'mxGraphModel',dx='1600',dy='740',grid='1',gridSize='10',page='1',pageWidth='1600',pageHeight='740')
root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
for key,x,y,w,h,title,lines,fill,stroke,fs in nodes:
    label='<div style="line-height:1.6"><b style="font-size:22px">'+escape(title)+'</b><br/>'+('<br/>'.join(escape(l) for l in lines))+'</div>'
    cell=ET.SubElement(root,'mxCell',id=key,value=label,style=f'rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontColor={ink};fontFamily=Microsoft YaHei;fontSize={fs};spacing=12;arcSize=8;',vertex='1',parent='1')
    ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
def edge(id,source,target,style='',points=None,label=''):
    c=ET.SubElement(root,'mxCell',id=id,value=label,edge='1',parent='1',source=source,target=target,style='edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeWidth=2;strokeColor=#526571;fontFamily=Microsoft YaHei;fontSize=16;'+style)
    g=ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    if points:
        a=ET.SubElement(g,'Array',attrib={'as':'points'})
        for x,y in points:ET.SubElement(a,'mxPoint',x=str(x),y=str(y))
for a,b in [('input','prefix'),('prefix','verify'),('verify','local'),('local','output')]:edge('e-'+a,a,b,'exitX=1;exitY=0.5;entryX=0;entryY=0.5;')
edge('bypass','verify','output','exitX=0.5;exitY=1;entryX=0.5;entryY=1;',[(800,560),(1477.5,560)],'未触发同站补查')
for key,x in [('prefix',420),('verify',800),('local',1185)]:
    edge('feedback-'+key,key,'cache',f'exitX=0.5;exitY=0;entryX={(x-280)/1050};entryY=1;dashed=1;startArrow=block;strokeColor=#77958D;')
for id,text,x,y,w,fs in [('title','FRAME：反馈驱动的拓扑搜索',45,15,1100,28),('subtitle','根据仿真反馈控制搜索深度，按变电站信息补查相关候选',45,62,1200,18),('params','p = 128　 h = 64　 q_min = 256　 L = 2　 ρ_safe = 0.9',280,680,1050,17)]:
    c=ET.SubElement(root,'mxCell',id=id,value=text,vertex='1',parent='1',style=f'text;html=0;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontFamily=Microsoft YaHei;fontSize={fs};fontColor={ink};')
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height='40',attrib={'as':'geometry'})
ET.ElementTree(mx).write(OUT/'FRAME_方法流程.drawio',encoding='utf-8',xml_declaration=True)
print('Wrote editable drawio and SVG')
