"""Copy the reviewed V4 to the D-drive collaboration folder and Git archive."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pypdf import PdfReader

ROOT = Path('D:/Codex/GRL研究')
OUT = ROOT / 'outputs/frame_cn_v4_20261018'
NAME = 'FRAME_ICAPS2027_中文重写稿_V4'
COLLAB = ROOT / '论文协作资料/FRAME_指南逐节重写_V4_20261018'
REPO = ROOT / 'work/GRL-results-integration'
ARCHIVE = REPO / 'research_archive/power-grid-2026/paper/FRAME_CN_rewrite_V4_20261018'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    receipt = json.loads((OUT / 'source_receipt.json').read_text(encoding='utf-8'))
    visual = json.loads((OUT / 'visual_review.json').read_text(encoding='utf-8'))
    docx = OUT / (NAME + '.docx')
    assert visual['all_pages_reviewed'] and visual['docx_sha256'] == sha(docx)
    pdf_pages = len(PdfReader(OUT / 'qa_final' / (NAME + '.pdf')).pages)
    assert pdf_pages == visual['pages'] == len(visual['reviewed_pages'])
    assert len(list((OUT / 'qa_final').glob('page-*.png'))) == pdf_pages
    for path, value in receipt['input_sha256'].items():
        assert sha(ROOT / path) == value, path
    receipt['visual_review_pending'] = False
    receipt['visual_review'] = visual
    receipt['files'] = {p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['source_receipt.json','交付完成.json']}
    (OUT / 'source_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    final = dict(utc=datetime.now(timezone.utc).isoformat(), docx_sha256=sha(docx), pages=pdf_pages,
                 all_pages_reviewed=True, figures=0, tables=0, new_experiments=0,
                 former_drafts_preserved=True, experimental_records_preserved=True)
    (OUT / '交付完成.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    for folder in [COLLAB, ARCHIVE]:
        folder.mkdir(parents=True, exist_ok=True)
        for p in OUT.iterdir():
            if p.is_file() and p.suffix in ['.md','.docx','.json','.bib']:
                shutil.copy2(p, folder / p.name)
                assert sha(p) == sha(folder / p.name)
    scripts = ARCHIVE / 'source'
    scripts.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copy2(p, scripts / p.name)
    index = REPO / 'docs/PAPER_MATERIALS_INDEX.md'
    content = index.read_text(encoding='utf-8')
    label = 'FRAME_CN_rewrite_V4_20261018'
    if label not in content:
        content += '\n\n## 2026年10月18日 中文全文重写V4\n\n按用户最新要求逐节阅读指南及对应论文后重写全文，无图表，原生公式和文字伪代码，已逐页检查。主评价沿用G2，未新增实验或训练。\n\n- [V4中文Word](../research_archive/power-grid-2026/paper/' + label + '/' + NAME + '.docx)\n- [写作依据与逐节核对](../research_archive/power-grid-2026/paper/' + label + '/写作依据与逐节核对.md)\n- D盘协作目录：`D:/Codex/GRL研究/论文协作资料/FRAME_指南逐节重写_V4_20261018`。旧版本全部保留。\n'
        index.write_text(content, encoding='utf-8')
    state = REPO / 'docs/RESEARCH_STATE.md'
    content = state.read_text(encoding='utf-8')
    if '中文全文重写V4' not in content:
        first, rest = content.split('\n', 1)
        content = first + '\n\n- **2026-10-18写作：** 中文全文重写V4按指南及对应论文章节重新组织，突出反馈搜索深度与同站查询位置的联系，12页、46个可编辑公式、10项参考文献，无图表；逐页检查通过。主结果与所有原始数据不改，未新增实验。V3及用户编辑稿保留。\n' + rest
        state.write_text(content, encoding='utf-8')
    nxt = REPO / 'docs/NEXT_STEPS.md'
    nxt.write_text('# Next steps\n\n1. 用户通读中文全文重写V4，修改贡献表述与方法解释。\n2. 根据作者反馈精修中文稿，保持G2主结果、等预算机制比较与META适配配置的清楚对应；暂不新增实验。\n3. 中文确认后再开展配图、表格与英文模板排版。历史文稿和全部实验记录继续保留。\n', encoding='utf-8')
    print(json.dumps(dict(collaboration=str(COLLAB), archive=str(ARCHIVE), **final), ensure_ascii=False))

if __name__ == '__main__':
    main()
