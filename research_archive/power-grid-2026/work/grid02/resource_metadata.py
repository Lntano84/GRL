"""Read distribution Content-Length only; never download another wheel or dataset."""
import importlib.metadata
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid02'
groups=[]
unique={}
for filename in ['install_report.json','matplotlib_dryrun.json','matplotlib_install_report.json']:
    data=json.loads((OUT/filename).read_text(encoding='utf-8'))
    group=[]
    for item in data['install']:
        url=item['download_info']['url']
        if url not in unique:
            request=urllib.request.Request(url,method='HEAD')
            with urllib.request.urlopen(request,timeout=30) as response:
                size=int(response.headers['Content-Length'])
            unique[url]={'name':item['metadata']['name'],'version':item['metadata']['version'],'bytes':size,
                         'url':url,'sha256':item['download_info']['archive_info']['hashes']['sha256']}
        group.append(unique[url])
    groups.append({'report':filename,'files':group,'payload_bytes':sum(x['bytes'] for x in group)})
unique_payload=sum(x['bytes'] for x in unique.values())
cumulative_payload=sum(x['payload_bytes'] for x in groups)
source=json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8'))
names={d.metadata['Name'] for d in importlib.metadata.distributions() if d.metadata.get('Name')}
versions={name:importlib.metadata.version(name) for name in names}
prior=json.loads((ROOT/'outputs/grid01/scenario0_full/metadata.json').read_text(encoding='utf-8'))['versions']
assert all(importlib.metadata.version(k)==v for k,v in prior.items())
result={'groups':groups,'unique_dependency_payload_bytes':unique_payload,
        'cumulative_wheel_payload_including_dry_run_bytes':cumulative_payload,
        'frozen_dependency_payload_cap_bytes':30*1024**2,
        'cumulative_dependency_payload_cap_met':cumulative_payload<=30*1024**2,
        'source_and_asset_bytes':sum(x['bytes'] for x in source['files']),
        'torch_downloaded':False,'GRID01_selected_versions_unchanged':True,'installed_versions':versions,
        'note':'Payload estimate from distribution Content-Length and documented downloads, not packet-level network metering. Dry-run and installation logs both show wheel downloads; conservatively count both. Do not retroactively erase the original 30 MiB resource ceiling.'}
(OUT/'resource_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Unique dependency payload',unique_payload,'cumulative',cumulative_payload,'cap_met',result['cumulative_dependency_payload_cap_met'],flush=True)

metadata_path=OUT/'formal_dataset_metadata.json'
metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
entry=metadata['entries']['l2rpn_idf_2023']
url=entry['base_url']+entry['filename']
try:
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=20) as response:
        metadata['HEAD_only']={'url':url,'status':response.status,'headers':dict(response.headers)}
except Exception as error:
    metadata['HEAD_only']={'url':url,'error':type(error).__name__+': '+str(error)}
metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('Formal dataset HEAD only; archive not downloaded.',flush=True)
