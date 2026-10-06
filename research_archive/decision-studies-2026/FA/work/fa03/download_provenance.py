"""Archive public upstream provenance for FA03 cost semantics, not outcomes."""
from pathlib import Path
from urllib.request import urlopen, Request
import json, hashlib

root = Path(__file__).resolve().parents[2]
out = root / 'work/fa03/provenance'
out.mkdir(parents=True, exist_ok=True)
items = {
    'QBF-2011_readme.txt': 'https://raw.githubusercontent.com/coseal/aslib_data/master/QBF-2011/readme.txt',
    'ASLib_format.md': 'https://raw.githubusercontent.com/coseal/aslib-spec/master/format.md',
    'kotthoff_evaluation_2012.pdf': 'https://www.cs.uwyo.edu/~larsko/papers/kotthoff_evaluation_2012.pdf',
    'kotthoff_ranking_2013.pdf': 'https://www.eecs.uwyo.edu/~larsko/papers/kotthoff_ranking_2013.pdf',
}
manifest = []
for name, url in items.items():
    with urlopen(Request(url, headers={'User-Agent': 'FA03 qualification'}), timeout=40) as response:
        blob = response.read()
    (out / name).write_bytes(blob)
    manifest.append(dict(file=name, url=url, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
print(json.dumps([{k:v for k,v in x.items() if k!='sha256'} for x in manifest], indent=2))
