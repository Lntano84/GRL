import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid02'
SRC = ROOT / 'work/grid02/original_nn'
SHA = 'ca0637eab9f098be7f206ed0e46a3900cd4deec0'
FILES = ['modules/topology_nn_policy.py', 'gym_assets/action_space.py',
         'gym_assets/__init__.py', 'requirements.txt',
         'models/RL_training_PPO.zip',
         'assets/nn_act_space/action_12_unsafe_nn.npz']
OUT.mkdir(parents=True, exist_ok=True)
manifest = []
for name in FILES:
    url = f'https://raw.githubusercontent.com/lajavaness/l2rpn-2023-ljn-agent/{SHA}/{name}'
    path = SRC / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f'Refuse to replace source file: {path}')
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read(60 * 1024**2)
    assert len(data) < 60 * 1024**2
    path.write_bytes(data)
    manifest.append({'path': name, 'source': url, 'bytes': len(data),
                     'sha256': hashlib.sha256(data).hexdigest()})
    (OUT / 'source_manifest.json').write_text(json.dumps({'commit': SHA, 'files': manifest}, indent=2), encoding='utf-8')
    print(name, len(data), flush=True)
assert sum(x['bytes'] for x in manifest) < 60 * 1024**2
with zipfile.ZipFile(SRC / 'models/RL_training_PPO.zip') as z:
    info = {'members': [{'name': x.filename, 'size': x.file_size} for x in z.infolist()],
            'system_info': z.read('system_info.txt').decode() if 'system_info.txt' in z.namelist() else None,
            'data': json.loads(z.read('data'))}
(OUT / 'model_archive_metadata.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
for package in ['stable-baselines3', 'gymnasium']:
    with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/json', timeout=30) as r:
        d = json.load(r)
    (OUT / f'{package}_metadata.json').write_text(json.dumps({'version': d['info']['version'], 'requires_dist': d['info']['requires_dist']}, indent=2), encoding='utf-8')
    print(package, d['info']['version'], flush=True)
print('Archive system info:', info['system_info'], flush=True)
