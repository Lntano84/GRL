"""Render a new editorial version without changing experimental evidence."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path('D:/Codex/GRL研究')
OUT = ROOT / 'outputs/frame_cn_v4_20261018'
NAME = 'FRAME_ICAPS2027_中文重写稿_V4'
PREVIOUS = ROOT / 'outputs/frame_cn_v3_20261018'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (OUT / '交付完成.json').exists()
    source = OUT / (NAME + '.md')
    text = source.read_text(encoding='utf-8')
    if not text.split('# 参考文献', 1)[1].strip():
        references = (PREVIOUS / 'FRAME_ICAPS2027_中文合稿_V3.md').read_text(encoding='utf-8').split('# 参考文献', 1)[1]
        text = text.rstrip() + '\n\n' + references.strip() + '\n'
        source.write_text(text, encoding='utf-8')
    assert not re.search(r'^\s*\|', text, re.M)
    assert '![' not in text
    assert not any(s in text for s in ['不是', '而不是', '下一控制时刻重新建立', '周毅'])
    shutil.copy2(PREVIOUS / 'references.bib', OUT / 'references.bib')
    supplemental = (PREVIOUS / '补充材料_V3.md').read_text(encoding='utf-8')
    supplemental = supplemental.replace('FRAME中文稿V3', 'FRAME中文稿V4').replace('本文V3与V2', '本文V4与V3、V2')
    (OUT / '补充材料_V4.md').write_text(supplemental, encoding='utf-8')
    # Reuse the tested native Word equation and paragraph renderer.
    code = (ROOT / 'work/grid89_review/source_v1/build_reference.py').read_text(encoding='utf-8')
    renderer = code[code.index('def mn(tag)'):code.index("receipt={'utc'")]
    namespace = dict(globals(), merged=text)
    exec(compile(renderer, str(Path(__file__).resolve()), 'exec'), namespace)
    doc = namespace['doc']
    assert len(doc.tables) == len(doc.inline_shapes) == 0
    decision = json.loads((ROOT / 'outputs/grid89_review/decision.json').read_text(encoding='utf-8'))
    compare = next(c for c in decision['comparisons'] if c['phase'] == 'sealed' and c['method'] == 'FRAME_GUARD' and c['baseline'] == 'FULL')
    assert f"{compare['macro_controller_reduction'] * 100:.2f}" == '14.30'
    assert f"{compare['macro_query_reduction'] * 100:.2f}" == '60.73'
    assert compare['same_horizon_all_dates'] and compare['both_quality_all_dates']
    inputs = [
        PREVIOUS / 'FRAME_ICAPS2027_中文合稿_V3.docx',
        PREVIOUS / 'FRAME_ICAPS2027_中文合稿_V3.md',
        ROOT / 'outputs/grid89_review/decision.json',
        ROOT / 'outputs/grid89_review/review.json',
        ROOT / 'outputs/grid84/decision.json',
        ROOT / 'outputs/grid88/GRID88_report.md',
        ROOT / 'work/grid89_review/policies.py',
    ]
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), version='V4 full Chinese rewrite',
                   input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in inputs},
                   new_experiments=0, new_training=0, results_changed=False,
                   former_drafts_preserved=True, figures=0, tables=0,
                   editable_math_objects=len(doc.element.findall('.//' + qn('m:oMath'))),
                   reference_entries=len(re.findall(r'^@', (OUT / 'references.bib').read_text(encoding='utf-8'), re.M)),
                   visual_review_pending=True,
                   files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    (OUT / 'source_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(docx=str(OUT / (NAME + '.docx')), paragraphs=len(doc.paragraphs),
                         math=receipt['editable_math_objects'], references=receipt['reference_entries']), ensure_ascii=False))

if __name__ == '__main__':
    main()
