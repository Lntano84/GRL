"""Package only the reviewed figure; retain the manuscript and experiments."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,xml.etree.ElementTree as ET
from pypdf import PdfReader

ROOT=Path('D:/Codex/GRL研究')
OUT=ROOT/'outputs/frame_motivation_v1_20261018'
COLLAB=ROOT/'论文协作资料/FRAME_论文配图_20261018/图1_动机图_V1'
REPO=ROOT/'work/GRL-results-integration'
ARCH=REPO/'research_archive/power-grid-2026/paper/FRAME_figures_20261018/motivation_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    doc=ET.parse(OUT/'FRAME_动机图_V1.drawio')
    assert len(doc.findall('.//mxCell[@vertex="1"]'))>30
    p=PdfReader(OUT/'FRAME_动机图_V1.pdf')
    assert len(p.pages)==1 and len(p.pages[0].images)==0
    assert abs(float(p.pages[0].mediabox.width)/72*25.4-170)<.01
    review={'utc':datetime.now(timezone.utc).isoformat(),'all_pages_reviewed':True,'pages':1,
            'reviewed':['PDF-rendered full-size PNG','670px paper-width preview','grayscale preview'],
            'visual_findings':'No clipped labels, glyph loss, overlapping text, or obstructed connectors. A/B/C letters preserve station groups without color; border styles distinguish queried/unqueried/extra. User requested lighter text; repeated prose shortened or removed before final rendering.',
            'pdf_vector_only':True,'paper_width_mm':170,'minimum_nominal_text_pt':19*170/25.4*72/1100,
            'native_drawio_text_and_connectors':True,'sequence_components':'editable Matplotlib SVG assets',
            'same_right_candidate_sequence':True,'equal_extra_queries':[3,3],
            'source_icon_platform':'The Noun Project, SVG CC BY 3.0 US distribution via Wikimedia Commons',
            'palette':'ColorBrewer 2.0 Set2 (3 classes)','synthetic_results':False,'schematic_only':True,
            'new_experiments':0,'manuscript_modified':False,
            'files':{p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.drawio','.svg','.pdf','.png']}}
    (OUT/'visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    instructions='''# 图1：动机图 V1

这张图说明两个问题：不同状态需要不同搜索深度；相关变电站的连接方案可能分散在候选序列中。图为概念示意，不是实验数据图。

## 文件

- FRAME_动机图_V1.png：查看预览。
- FRAME_动机图_V1.pdf：170毫米宽的矢量排版文件。
- FRAME_动机图_V1.svg：矢量版本，含diagrams.net嵌入源。
- FRAME_动机图_V1.drawio：用Draw.io打开；文字、分隔线、括号和箭头是原生可编辑对象。候选条带和图标为嵌入SVG，组件原文件在assets内，可以替换或进一步编辑。
- source/build.py：用Matplotlib生成候选条带，构造可编辑版式。
- source/export.py：调用实际Draw.io程序导出，再用pypdf固定物理尺寸、Poppler渲染检查。
- 图注与素材署名.md：正式插入论文时使用的图注和素材署名。
- visual_review.json：最终尺寸、矢量检查与实际视觉检查记录。

## 视觉设计

ColorBrewer 2.0 Set2三类颜色，原始官方色表在assets/colorbrewer.json。变电站用A/B/C字母辅助区分；已查询、未查询、追加查询用边框区分。全局扩展与同站补查的顺序完全相同，均追加3次查询。

电力图标采用The Noun Project的Georgiana Ionescu作品Electricity Pole（1364802），使用Wikimedia Commons保存并核验授权的CC BY 3.0 US版本，已缩放和改色。具体来源、作者及许可证链接见图注与素材署名。

## 本轮修改

依据作者对文字密度的反馈，删去右侧重复说明，精简左侧四行注释，改用图形图例。实际检查全尺寸、论文宽度和灰度版本。图中没有加入实验结论或量化节省数据；没有重跑实验，没有改论文正文。原密集版保留于canonical输出的intermediate文件夹。
'''
    (OUT/'README.md').write_text(instructions,encoding='utf-8')
    for target in [COLLAB,ARCH]:
        target.mkdir(parents=True,exist_ok=True)
        for f in OUT.iterdir():
            if f.is_file() and f.name!='FRAME_动机图_V1.raw.pdf':
                shutil.copy2(f,target/f.name)
                assert sha(f)==sha(target/f.name)
        shutil.copytree(OUT/'assets',target/'assets',dirs_exist_ok=True)
        shutil.copytree(Path(__file__).parent,target/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    index=REPO/'docs/PAPER_MATERIALS_INDEX.md'
    content=index.read_text(encoding='utf-8')
    if 'FRAME_figures_20261018/motivation_v1' not in content:
        content+='\n\n## 2026年10月18日 动机图V1\n\n按绘图指南，以Matplotlib组件、Draw.io实际布局导出、ColorBrewer Set2和The Noun Project矢量素材制作。作者反馈后精简文字。保存原生Draw.io、SVG、170mm矢量PDF与预览，检查全尺寸、缩小和灰度版本。概念示意，不新增实验或修改中文V4。\n\n- [配图及编辑说明](../research_archive/power-grid-2026/paper/FRAME_figures_20261018/motivation_v1/README.md)\n- D盘协作：`D:/Codex/GRL研究/论文协作资料/FRAME_论文配图_20261018/图1_动机图_V1`。\n'
        index.write_text(content,encoding='utf-8')
    print(json.dumps({'collaboration':str(COLLAB),'archive':str(ARCH),'drawio_sha256':sha(OUT/'FRAME_动机图_V1.drawio')},ensure_ascii=False))

if __name__=='__main__':main()
