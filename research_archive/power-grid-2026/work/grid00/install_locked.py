"""Download only artifacts chosen by pip's saved plan; verify hashes, install offline."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid00'
WHEELS = ROOT / 'work/grid00/wheels'
WHEELS.mkdir(exist_ok=True)
plan = json.loads((OUT / 'install_plan.json').read_text(encoding='utf-8'))
records = []
for item in plan['install']:
    url = item['download_info']['url']
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=30) as response:
        size = int(response.headers['Content-Length'])
    records.append({'name': item['metadata']['name'], 'version': item['metadata']['version'],
                    'url': url, 'bytes': size,
                    'sha256': item['download_info']['archive_info']['hashes']['sha256']})
total = sum(r['bytes'] for r in records)
# The dry run downloaded these wheels once already. Reserve 3 MiB for source,
# index/metadata payloads; this is a conservative envelope, not packet metering.
assert 2 * total + 3 * 1024**2 <= 200 * 1024**2, total
print('Wheel payload per pass:', total, 'two passes + reserve:', 2 * total + 3 * 1024**2, flush=True)
for rec in records:
    dst = WHEELS / rec['url'].rsplit('/', 1)[1]
    if not dst.exists():
        with urllib.request.urlopen(rec['url'], timeout=60) as src, dst.open('wb') as dest:
            while data := src.read(1024 * 1024):
                dest.write(data)
    assert dst.stat().st_size == rec['bytes']
    assert hashlib.sha256(dst.read_bytes()).hexdigest() == rec['sha256']
    print(rec['name'], rec['version'], rec['bytes'], flush=True)
(OUT / 'wheel_manifest.json').write_text(json.dumps({'wheels': records,
    'payload_per_pass': total, 'two_passes_plus_3MiB_reserve': 2 * total + 3 * 1024**2}, indent=2))
lock = OUT / 'requirements.lock'
lock.write_text('\n'.join(f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}" for r in records) + '\n')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', str(WHEELS),
                '--require-hashes', '-r', str(lock)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
