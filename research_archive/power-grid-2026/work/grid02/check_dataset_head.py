"""HEAD only with certificate verification enabled; no archive download."""
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'outputs/grid02/formal_dataset_metadata.json'
data=json.loads(path.read_text(encoding='utf-8'))
entry=data['entries']['l2rpn_idf_2023']
url=entry['base_url']+entry['filename']
result={'verification':'requests default TLS certificate verification, HEAD only'}
try:
    response=requests.head(url,allow_redirects=True,timeout=20)
    result.update(status=response.status_code,url=response.url,headers=dict(response.headers))
except Exception as error:
    result['error']=type(error).__name__+': '+str(error)
data['requests_HEAD_only']=result
path.write_text(json.dumps(data,indent=2),encoding='utf-8')
print({k:v for k,v in result.items() if k!='headers'})
