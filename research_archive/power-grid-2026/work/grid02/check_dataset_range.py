"""A one-byte GET range probe; signed GET URLs may legitimately reject HEAD."""
import json
from pathlib import Path
from urllib.parse import urlsplit
import requests

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'outputs/grid02/formal_dataset_metadata.json'
data=json.loads(path.read_text(encoding='utf-8'))
entry=data['entries']['l2rpn_idf_2023']
url=entry['base_url']+entry['filename']
result={'method':'GET','requested_range':'bytes=0-0','TLS_verification':True,'archive_downloaded':False}
try:
    with requests.get(url,headers={'Range':'bytes=0-0'},allow_redirects=True,stream=True,timeout=(15,20)) as response:
        parts=urlsplit(response.url)
        result.update(status=response.status_code,final_url_without_query=f'{parts.scheme}://{parts.netloc}{parts.path}',
                      content_range=response.headers.get('Content-Range'),content_length=response.headers.get('Content-Length'),
                      content_type=response.headers.get('Content-Type'))
        if response.status_code==206 and response.headers.get('Content-Range','').startswith('bytes 0-0/'):
            first=response.raw.read(1)
            result.update(application_body_bytes_read=len(first),first_byte_hex=first.hex())
        else:
            result['application_body_bytes_read']=0
except Exception as error:
    result['error']=type(error).__name__+': '+str(error)
data['one_byte_GET_range_probe']=result
data['HEAD_interpretation']='HEAD failure does not by itself establish GET/download failure; redirects can be presigned for GET.'
path.write_text(json.dumps(data,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
