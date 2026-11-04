"""Use original numerical policy/reveal/NN; inject only public identity state."""
from pathlib import Path
import gzip,hashlib,importlib.util,io,json,sys
from state import State,KEYS
from guard import R,O,read,sha
def install(Search,item,models,expected=None):
 prefix=None
 if item.get('target') is not None:
  rel=item['baseline_folder']+'/steps.jsonl.gz';ref=read(O/'baseline_raw_manifest.json')[rel];data=(R/rel).read_bytes();assert hashlib.sha256(data).hexdigest()==ref['sha256']
  prefix=[]
  with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
   for _ in range(item['target']['common']['step']-1):
    r=json.loads(stream.readline());prefix.append({k:r[k] for k in KEYS})
  del data
 state=State(item,prefix);sys.path.insert(0,str(R/'work/grid297_paired_collection_preparation'))
 spec=importlib.util.spec_from_file_location('g300_unchanged_adapter',R/'work/grid297_paired_collection_preparation/adapter.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.EpisodeIntervention=lambda arm,unused:state
 got=m.install(Search,item,models,None);assert got is state;return state
