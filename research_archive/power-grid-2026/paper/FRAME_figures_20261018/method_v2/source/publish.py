"""Archive the actual reviewed outputs and their editable sources."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,xml.etree.ElementTree as ET
from pypdf import PdfReader
ROOT=Path('D:/Codex/GRL研究')
OUT=ROOT/'outputs/frame_method_v2_20261018'
WORK=ROOT/'work/frame_method_v2_20261018'
REPO=ROOT/'work/GRL-results-integration'
COLLAB=ROOT/'论文协作资料/FRAME_论文配图_20261018/图2_方法图_V2_分区与多色版'
ARCH=REPO/'research_archive/power-grid-2026/paper/FRAME_figures_20261018/method_v2'
BASE='FRAME_方法图_V2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    reader=PdfReader(OUT/(BASE+'.pdf')); assert len(reader.pages)==1
    assert len(reader.pages[0].images)==0
    assert abs(float(reader.pages[0].mediabox.width)/72*25.4-170)<.01
    vals=[c.get('value','') for c in ET.parse(OUT/(BASE+'.drawio')).findall('.//mxCell')]
    assert not any('Ary Prasetyo' in v for v in vals)
    assert sha(OUT/'assets/noun_powergrid_4199252.svg')=='c86df2e8e36c32c9713c8445ab63dba3b01b450b20c3a07456e900236879d353'
    assert 'Ary Prasetyo' in (OUT/'图注与素材来源.md').read_text(encoding='utf-8')
    sources=['outputs/frame_cn_v4_20261018/FRAME_ICAPS2027_中文重写稿_V4.md','work/grid89_review/policies.py']
    review={'utc':datetime.now(timezone.utc).isoformat(),'all_pages_reviewed':True,'pages':1,
        'reviewed':['PDF-rendered full-size PNG','670px paper-width preview','grayscale preview'],
        'visual_findings':'No clipping or overlapping labels in the final render. Prefix tiles now sit within their panel. All three station-selection arrows terminate above A representatives. Branch labels and the no-completion output path remain visible. Named stages, A/B/C letters, solid/dashed borders and underlines supplement color in grayscale.',
        'resolved_during_review':['prefix SVG had extended to its panel border','station arrows were not aligned to all A positions'],
        'pdf_vector_only':True,'paper_width_mm':170,'height_mm':170*800/1470,
        'native_drawio_text_and_connectors':True,
        'palette':'ColorBrewer Blues 7 / Purples 7 / BuGn 7',
        'attribution_location':'external figure caption only',
        'source_sha256':{s:sha(ROOT/s) for s in sources},
        'prior_versions_preserved':True,'manuscript_modified':False,'new_experiments':0,
        'files':{p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.drawio','.svg','.pdf','.png']}}
    (OUT/'visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    readme='''# FRAME方法图V2：分区与多色版

按作者反馈重新学习指南案例，改变原V1三框平铺的布局。全局搜索占左侧主要区域，前缀筛查与分批核查在组内展开；同站补查位于右侧，具体展示站归属与非连续候选。前缀为蓝色、核查为紫色、补查为绿色，均来自ColorBrewer官方配色；正文和旧图保留。

## 查看、编辑与排版

- FRAME_方法图_V2.png：完整预览。
- FRAME_方法图_V2.pdf：170毫米宽、单页纯矢量排版文件。
- FRAME_方法图_V2.svg：矢量图，含嵌入的diagrams.net源。
- FRAME_方法图_V2.drawio：原生可编辑布局、文字、箭头、形状。
- assets/：Matplotlib生成的候选SVG组件、免费Noun图标原件、裁切派生图和ColorBrewer源。
- source/：构建、实际Draw.io导出与归档脚本。
- 论文宽度检查.png、灰度检查.png、visual_review.json：实际查看并核对的预览和记录。
- 图注与素材来源.md：外部图注、素材署名、精确规则对应说明。

## 参考的布局做法

指南：https://dy8q0bnq8y.feishu.cn/wiki/OGpcw6zaRiQ6OwkNxfYcl7IKnBh

本次重新阅读方法图工作流和七个案例，并实际查看案例图片。

1. Agentic Reasoning综述案例：按内容归类，区域大小服务信息层次。
2. Agentic Reasoning方法案例：使用低饱和度功能分区和不同箭头。
3. WAPITI：紧凑图例、清楚的模块命名与主要信息流。
4. Language in the Flow of Time：输入分区、具体序列对象和分组。
5. V-IRL：反馈回路和功能不同的形状。
6. Reasoning Boundary：将抽象机制转成可识别的小对象。
7. Mem-Gallery：主体区域与辅助区域具有不同规模和阅读层级。

本图采用两项贡献为主的非等宽布局，未复制案例内容或加入与FRAME无关的装饰。

## 图标署名在外部图注

power grid by Ary Prasetyo from Noun Project (CC BY 3.0), cropped and scaled.

素材：https://thenounproject.com/icon/power-grid-4199252/

许可：https://creativecommons.org/licenses/by/3.0/

原始SVG保持不变。派生图裁切、缩放，作者署名置于配套外部图注末尾，正式论文使用时随图保留。

## 制作

Draw.io 32.4.1生成布局并实际导出SVG/PDF，Matplotlib生成矢量组件，Poppler渲染核对。未使用图像生成模型，未新增实验或调整模型、规则、预算。
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    for target in [COLLAB,ARCH]:
        target.mkdir(parents=True,exist_ok=True)
        for p in OUT.iterdir():
            if p.is_file():
                shutil.copy2(p,target/p.name);assert sha(p)==sha(target/p.name)
        shutil.copytree(OUT/'assets',target/'assets',dirs_exist_ok=True)
        (target/'source').mkdir(exist_ok=True)
        for p in WORK.glob('*.py'):shutil.copy2(p,target/'source'/p.name)
    index=REPO/'docs/PAPER_MATERIALS_INDEX.md'
    text=index.read_text(encoding='utf-8')
    if 'FRAME_figures_20261018/method_v2' not in text:
        text+='''

## 2026年10月18日 方法图V2：分区与多色版

按作者反馈，重新阅读绘图指南的方法图流程与七个案例并查看图片，采用非等宽主次分区、蓝紫绿功能配色、具体候选和站归属示意。实线主流程、虚线反馈，署名放到外部图注。检查全尺寸、论文宽度与灰度版，修正候选边界和站归属箭头。纯矢量170毫米宽；原V1、动机图、正文和实验保留。

- [方法图V2与编辑说明](../research_archive/power-grid-2026/paper/FRAME_figures_20261018/method_v2/README.md)
- D盘协作：D:/Codex/GRL研究/论文协作资料/FRAME_论文配图_20261018/图2_方法图_V2_分区与多色版。
'''
        index.write_text(text,encoding='utf-8')
    print(json.dumps({'collaboration':str(COLLAB),'archive':str(ARCH),'pdf_vector_only':True},ensure_ascii=True))
if __name__=='__main__':main()
