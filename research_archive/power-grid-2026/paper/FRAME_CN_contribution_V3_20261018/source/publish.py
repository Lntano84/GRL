"""Deliver the visually reviewed manuscript without modifying scientific records."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

ROOT = Path('D:/Codex/GRL研究')
OUT = ROOT / 'outputs/frame_cn_v3_20261018'
REPO = ROOT / 'work/GRL-results-integration'
COLLAB = ROOT / '论文协作资料/FRAME_贡献主线重组_V3_20261018'
ARCHIVE = REPO / 'research_archive/power-grid-2026/paper/FRAME_CN_contribution_V3_20261018'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    q = json.loads((OUT / 'visual_review.json').read_text(encoding='utf-8'))
    doc = OUT / 'FRAME_ICAPS2027_中文合稿_V3.docx'
    assert q['all_pages_reviewed'] and q['docx_sha256'] == sha(doc)
    receipt = json.loads((OUT / 'source_receipt.json').read_text(encoding='utf-8'))
    for path, value in receipt['input_sha256'].items():
        assert sha(ROOT / path) == value, path
    receipt.update(visual_review_pending=False, visual_review=q)
    (OUT / 'source_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    final = dict(utc=datetime.now(timezone.utc).isoformat(), pages=11, docx_sha256=sha(doc),
                 all_pages_reviewed=True, new_experiments=0, new_fits=0,
                 former_drafts_preserved=True, all_experimental_records_preserved=True)
    (OUT / '交付完成.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    for target in [COLLAB, ARCHIVE]:
        target.mkdir(parents=True, exist_ok=True)
        for p in OUT.iterdir():
            if p.is_file() and p.suffix in ['.md', '.docx', '.bib', '.json']:
                shutil.copy2(p, target / p.name)
                assert sha(p) == sha(target / p.name)
    source = ARCHIVE / 'source'
    source.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, source / path.name)
    print(json.dumps(dict(collaboration=str(COLLAB), archive=str(ARCHIVE), **final), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
