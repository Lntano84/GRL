"""Add only a conditional STATIC_ID ranker to the audited common-cache driver."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=(ROOT/'work/grid34/run.py').read_text(encoding='utf-8')
changes={
 '"""GRID34 common-cache FULL and all existing rankers; staged quality pilot, resumable."""':'"""GRID37 common-cache static versus both frozen GNNs and exhaustive FULL."""',
 "OUT=ROOT/'outputs/grid34'":"OUT=ROOT/'outputs/grid37'",
 "        summary=worker.worker(item,D)":"        if item['rule']=='STATIC_ID':\n            sys.path[:0]=[str(ROOT/'work/grid28'),str(ROOT/'work/grid36')]\n            from engine import Search\n            from cheap_rank import rank\n            original_rank=Search.rank;Search.rank=rank\n            try:summary=worker.worker(item,D)\n            finally:Search.rank=original_rank\n        else:summary=worker.worker(item,D)",
 'all6arms shareDC+safe cache':'all4arms shareDC+safe cache',
}
for before,after in changes.items():
    assert source.count(before)==1,(before,source.count(before));source=source.replace(before,after)
target=ROOT/'work/grid37/run.py';assert not target.exists();target.write_text(source,encoding='utf-8')
print('GRID37 derived; historical control code unchanged.')
