"""Minimal public-ranking substitution in original audited common-cache driver."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=(ROOT/'work/grid34/run.py').read_text(encoding='utf-8')
replacements={
 '"""GRID34 common-cache FULL and all existing rankers; staged quality pilot, resumable."""':'"""GRID36 three cheap public orders,March quality pilot only,resumable."""',
 "OUT=ROOT/'outputs/grid34'":"OUT=ROOT/'outputs/grid36'",
 "        summary=worker.worker(item,D)":"        sys.path[:0]=[str(ROOT/'work/grid28'),str(ROOT/'work/grid36')]\n        from engine import Search\n        from cheap_rank import rank\n        original_rank=Search.rank;Search.rank=rank\n        try:summary=worker.worker(item,D)\n        finally:Search.rank=original_rank",
 "all6arms shareDC+safe cache":"3cheaparms shareDC+safe cache"
}
for before,after in replacements.items():
    assert source.count(before)==1,(before[:100],source.count(before));source=source.replace(before,after)
target=ROOT/'work/grid36/run.py';assert not target.exists();target.write_text(source,encoding='utf-8')
print('Derived GRID36 without changing any previous controller.')
