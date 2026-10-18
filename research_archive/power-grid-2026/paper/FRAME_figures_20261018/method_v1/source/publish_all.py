"""Package the visually reviewed motivation V3 and method V1 figures."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pypdf import PdfReader

ROOT = Path('D:/Codex/GRL研究')
REPO = ROOT / 'work/GRL-results-integration'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def publish(kind, version, folder, collab_name, canvas):
    out = ROOT / 'outputs' / folder
    work = ROOT / 'work' / folder
    label = '动机图' if kind == 'motivation' else '方法图'
    base = f'FRAME_{label}_{version}'
    pdf = PdfReader(out / (base + '.pdf'))
    assert len(pdf.pages) == 1 and len(pdf.pages[0].images) == 0
    assert abs(float(pdf.pages[0].mediabox.width) / 72 * 25.4 - 170) < .01
    doc = ET.parse(out / (base + '.drawio'))
    values = [c.get('value', '') for c in doc.findall('.//mxCell')]
    assert all('Ary Prasetyo' not in value for value in values)
    caption = out / ('图注与素材署名.md' if kind == 'motivation' else '图注与素材来源.md')
    assert 'Ary Prasetyo' in caption.read_text(encoding='utf-8')
    assert sha(out / 'assets/noun_powergrid_4199252.svg') == 'c86df2e8e36c32c9713c8445ab63dba3b01b450b20c3a07456e900236879d353'
    review = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'all_pages_reviewed': True, 'pages': 1,
        'reviewed': ['PDF-rendered full-size PNG', '670px paper-width preview', 'grayscale preview'],
        'visual_findings': 'Labels and symbols are legible; no clipped text, overlapping labels or obstructed connectors. Distinct line styles and station letters remain interpretable in grayscale.',
        'pdf_vector_only': True, 'paper_width_mm': 170,
        'height_mm': 170 * canvas[1] / canvas[0],
        'attribution_location': 'external figure caption only',
        'icon_license': 'The Noun Project, Ary Prasetyo, CC BY 3.0; cropped and scaled',
        'original_svg_unchanged': True,
        'native_drawio_text_and_connectors': True,
        'palette': 'ColorBrewer Blues 7 and Greys 9',
        'prior_versions_preserved': True, 'manuscript_modified': False, 'new_experiments': 0,
        'files': {p.name: sha(p) for p in out.iterdir() if p.suffix in ['.drawio', '.svg', '.pdf', '.png']}
    }
    (out / 'visual_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf-8')
    purpose = ('保留蓝灰版结构，移除图内左下角英文署名，署名放到外部图注。旧V1、V2完整保留。'
               if kind == 'motivation' else
               '展示前缀筛查、分批核查和条件触发的同站补查。实线表示主流程，虚线表示反馈存取。完整停止规则见正文算法与配套图注。')
    readme = f'''# {label} {version}

{purpose}

## 文件使用

- {base}.png：完整预览。
- {base}.pdf：170毫米宽、单页纯矢量排版文件。
- {base}.svg：含diagrams.net源的矢量文件。
- {base}.drawio：可编辑文件；文字、箭头和布局为Draw.io原生对象，候选示意为Matplotlib SVG组件。
- assets/：原始图标、派生SVG组件及ColorBrewer官方配色源。
- source/：生成与导出脚本。
- 论文宽度检查.png、灰度检查.png、visual_review.json：已实际检查的渲染与记录。
- {caption.name}：外部图注和素材许可。

## 外部图注署名

图内无署名文字；正式使用时将以下署名放在图片下方的论文图注末尾，随图保留。

power grid by Ary Prasetyo from Noun Project (CC BY 3.0), cropped and scaled.

素材：https://thenounproject.com/icon/power-grid-4199252/

许可：https://creativecommons.org/licenses/by/3.0/

原始SVG保持不变，派生图裁切空白并缩放。使用免费授权，无VIP素材。

## 制作工具

Draw.io 32.4.1完成原生布局和实际矢量导出；Matplotlib生成候选SVG；ColorBrewer Blues/Greys配色；Poppler渲染检查。未使用图像生成模型。论文正文、实验与历史版本保留。
'''
    (out / 'README.md').write_text(readme, encoding='utf-8')
    collab = ROOT / '论文协作资料/FRAME_论文配图_20261018' / collab_name
    archive = REPO / 'research_archive/power-grid-2026/paper/FRAME_figures_20261018' / f'{kind}_{version.lower()}'
    for target in [collab, archive]:
        target.mkdir(parents=True, exist_ok=True)
        for p in out.iterdir():
            if p.is_file():
                shutil.copy2(p, target / p.name)
                assert sha(p) == sha(target / p.name)
        shutil.copytree(out / 'assets', target / 'assets', dirs_exist_ok=True)
        (target / 'source').mkdir(exist_ok=True)
        for p in work.glob('*.py'):
            shutil.copy2(p, target / 'source' / p.name)
    return {'kind': kind, 'output': str(out), 'collaboration': str(collab), 'archive': str(archive)}

def main():
    results = [publish('motivation', 'V3', 'frame_motivation_v3_20261018', '图1_动机图_V3_图注外置', (1100,730)),
               publish('method', 'V1', 'frame_method_v1_20261018', '图2_方法图_V1', (1200,870))]
    index = REPO / 'docs/PAPER_MATERIALS_INDEX.md'
    content = index.read_text(encoding='utf-8')
    if 'FRAME_figures_20261018/motivation_v3' not in content:
        content += '''

## 2026年10月18日 动机图V3与方法图V1

按作者确认，将免费Noun Project素材署名移到外部论文图注，移除动机图图内英文行。新增方法图展示前缀筛查、分批核查、条件触发同站补查及共享仿真反馈，模块名称与中文V4及冻结实现核对。两图用Draw.io原生布局与Matplotlib SVG制作，采用ColorBrewer Blues/Greys，已检查全尺寸、论文宽度和灰度渲染，PDF为170毫米宽纯矢量。未改论文正文或实验。

- [动机图V3与外部图注](../research_archive/power-grid-2026/paper/FRAME_figures_20261018/motivation_v3/README.md)
- [方法图V1与编辑说明](../research_archive/power-grid-2026/paper/FRAME_figures_20261018/method_v1/README.md)
- D盘协作入口：`D:/Codex/GRL研究/论文协作资料/FRAME_论文配图_20261018`。
'''
        index.write_text(content, encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False))

if __name__ == '__main__':
    main()
