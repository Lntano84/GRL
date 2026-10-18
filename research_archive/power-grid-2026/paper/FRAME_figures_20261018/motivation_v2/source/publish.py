"""Package the reviewed blue/gray motivation figure without changing V1."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,xml.etree.ElementTree as ET
from pypdf import PdfReader

ROOT=Path('D:/Codex/GRL研究')
OUT=ROOT/'outputs/frame_motivation_v2_20261018'
COLLAB=ROOT/'论文协作资料/FRAME_论文配图_20261018/图1_动机图_V2_蓝灰版'
REPO=ROOT/'work/GRL-results-integration'
ARCH=REPO/'research_archive/power-grid-2026/paper/FRAME_figures_20261018/motivation_v2'
BASE='FRAME_动机图_V2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    doc=ET.parse(OUT/(BASE+'.drawio'))
    assert len(doc.findall('.//mxCell[@vertex="1"]'))>30
    reader=PdfReader(OUT/(BASE+'.pdf'))
    assert len(reader.pages)==1 and len(reader.pages[0].images)==0
    assert abs(float(reader.pages[0].mediabox.width)/72*25.4-170)<.01
    credit='power grid by Ary Prasetyo / Noun Project (CC BY 3.0). Cropped and scaled.'
    assert credit in [c.get('value') for c in doc.findall('.//mxCell')]
    source=OUT/'assets/noun_powergrid_4199252.svg'
    assert sha(source)==sha(Path('C:/Users/windows/Downloads/noun_powergrid_4199252.svg'))
    review={'utc':datetime.now(timezone.utc).isoformat(),'all_pages_reviewed':True,'pages':1,
            'reviewed':['PDF-rendered full-size PNG','670px paper-width preview','grayscale preview'],
            'visual_findings':'No clipped labels, glyph loss, overlapping text or obstructed connectors. All station memberships remain identifiable by letters in grayscale. The supplied new SVG is visibly used in the header; its CC attribution is retained as a readable full-figure credit. Blue highlights station A; other stations are neutral. Added queries are marked by underlines, with lighter strokes and fewer bold labels than V1.',
            'pdf_vector_only':True,'paper_width_mm':170,'height_mm':170*762/1100,
            'native_drawio_text_and_connectors':True,'sequence_components':'editable Matplotlib SVG assets',
            'same_right_candidate_sequence':True,'equal_extra_queries':[3,3],
            'original_svg_sha256':sha(source),'original_svg_unchanged':True,
            'source_icon_platform':'The Noun Project; user-supplied free CC BY 3.0 SVG 4199252',
            'palette':'ColorBrewer 2.0 Blues (7) and Greys (9)',
            'orange_green_removed':True,'schematic_only':True,'new_experiments':0,
            'manuscript_modified':False,'v1_preserved':True,
            'files':{p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.drawio','.svg','.pdf','.png']}}
    (OUT/'visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    readme='''# 图1：动机图 V2 蓝灰版

本版按作者反馈移除原Set2橙绿配色，采用ColorBrewer官方Blues和Greys，只用蓝色突出当前最佳动作的A站归属，B/C用灰色和字母区分。下方短线标出追加查询。减轻候选边框与策略标题的粗细，保留原V1的两组对比。未修改论文正文或实验记录。

## 查看与编辑

- FRAME_动机图_V2.png：完整预览。
- FRAME_动机图_V2.pdf：170毫米宽的矢量排版文件，保存为单页，未嵌入栅格图像。
- FRAME_动机图_V2.svg：矢量版本，含diagrams.net嵌入源。
- FRAME_动机图_V2.drawio：用Draw.io打开；文字、箭头、括号、图例为原生可编辑对象。候选序列为Matplotlib生成的SVG组件，原文件在assets内。
- source/build.py：生成SVG组件和可编辑图形布局。
- source/export.py：调用实际Draw.io程序导出，pypdf设置纸面尺寸，Poppler渲染检查。
- 论文宽度检查.png、灰度检查.png：实际检查的预览版本。
- 图注与素材署名.md：完整图注、作者和许可信息。
- visual_review.json：源文件校验、矢量属性及视觉检查记录。

## 免费图标与署名

power grid by Ary Prasetyo from Noun Project (CC BY 3.0).

- 素材：https://thenounproject.com/icon/power-grid-4199252/
- 许可：https://creativecommons.org/licenses/by/3.0/
- 用户在注册后从免费With Attribution选项下载SVG并提供；未使用VIP素材或订阅。
- 原始下载SVG按逐字节保存于assets/noun_powergrid_4199252.svg。
- 派生文件只裁切图形周围空白并缩放；原图底部署名改放到整张动机图下方的可见署名行。正式论文使用时须保留作者、来源、许可证及修改说明。

## 图意范围

这是机制示意图，方格数量不对应实际预算参数。左侧一格为候选批次，右侧一格为单个候选；右侧候选顺序一致，全局扩展与同站补查均追加3次。示意图没有展示测量曲线，也没有宣称同站补查始终得到最佳结果。全部旧稿、原图和实验结果继续保留。
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    for target in [COLLAB,ARCH]:
        target.mkdir(parents=True,exist_ok=True)
        for f in OUT.iterdir():
            if f.is_file():
                shutil.copy2(f,target/f.name)
                assert sha(f)==sha(target/f.name)
        shutil.copytree(OUT/'assets',target/'assets',dirs_exist_ok=True)
        shutil.copytree(Path(__file__).parent,target/'source',dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    index=REPO/'docs/PAPER_MATERIALS_INDEX.md'
    content=index.read_text(encoding='utf-8')
    if 'FRAME_figures_20261018/motivation_v2' not in content:
        content+='\n\n## 2026年10月18日 动机图V2蓝灰版\n\n按作者反馈，采用ColorBrewer Blues/Greys单一蓝色重点、减轻边框与粗体，替换为用户注册后免费取得的The Noun Project线性SVG（Ary Prasetyo，CC BY 3.0）。署名直接保留在图上。实际检查全尺寸、论文宽度和灰度版本；原V1、中文V4和全部实验记录不改。\n\n- [V2配图与编辑说明](../research_archive/power-grid-2026/paper/FRAME_figures_20261018/motivation_v2/README.md)\n- D盘协作：`D:/Codex/GRL研究/论文协作资料/FRAME_论文配图_20261018/图1_动机图_V2_蓝灰版`。\n'
        index.write_text(content,encoding='utf-8')
    print(json.dumps({'collaboration':str(COLLAB),'archive':str(ARCH),
                      'drawio_sha256':sha(OUT/(BASE+'.drawio')),
                      'original_svg_sha256':sha(source)},ensure_ascii=False))

if __name__=='__main__':main()
