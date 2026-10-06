"""BC04 fixed paths and read-only history semantics."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json

BASE=Path(__file__).resolve().parents[2]
WORK=BASE/'work/bc04'
OUT=BASE/'outputs'
REPO=BASE/'work/bc01/Baleen-FAST24'
PYTHON=BASE/'work/bc01/python311-embed/python.exe'
CONFIG=REPO/'runs/example/baleen/prefetch_ml-on-partial-hit/config.json'
REFERENCE=BASE/'work/bc03/STATIC-BASE/raw/ml-ap-0.798545_6_lru_366.475GB/full_0_0.1_cache_perf.txt.stats.lzma'
START=1572074461.57806
EVAL_START=86401.23277902603
CATEGORIES=['FIRST','ADMITTED-BEFORE','REJECTED-BEFORE','OTHER','MIXED']
CHUNK_CATEGORIES=CATEGORIES[:4]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def typed(value):
    if isinstance(value,tuple):
        return {'tuple':[typed(v) for v in value]}
    if isinstance(value,list):
        return {'list':[typed(v) for v in value]}
    if isinstance(value,(str,int,float,bool)) or value is None:
        return value
    raise TypeError(type(value))

@dataclass(slots=True)
class Entry:
    candidate_count:int=0
    written_count:int=0
    rejection_count:int=0
    eviction_count:int=0
    qualification_accept_count:int=0
    qualification_reject_count:int=0
    last_candidate_request:int|None=None
    last_write_request:int|None=None
    last_rejection_request:int|None=None
    last_eviction_request:int|None=None
    last_candidate_s:float|None=None
    last_write_s:float|None=None
    last_rejection_s:float|None=None
    last_eviction_s:float|None=None

class History:
    def __init__(self):
        self.seen_get_blocks=set()
        self.entries={}

    def entry(self,key):
        if key not in self.entries:
            self.entries[key]=Entry()
        return self.entries[key]

    def snapshot(self,block,chunks,pending):
        first=block not in self.seen_get_blocks
        result={}
        for chunk in chunks:
            key=(block,chunk)
            entry=self.entries.get(key)
            # Detached value snapshot, never aliases the live history.
            entry=entry if entry is not None else Entry()
            facts={field:getattr(entry,field) for field in Entry.__dataclass_fields__}
            facts['pending_before_request']=key in pending
            facts['chunk']=chunk
            facts['category']=('FIRST' if first else 'ADMITTED-BEFORE' if facts['written_count'] else
                               'REJECTED-BEFORE' if facts['rejection_count'] else 'OTHER')
            result[chunk]=facts
        return first,result

    def finish_get(self,block):
        self.seen_get_blocks.add(block)

    def event(self,key,kind,request,s):
        e=self.entry(key)
        count={'candidate':'candidate_count','write':'written_count','rejection':'rejection_count','eviction':'eviction_count'}[kind]
        stem={'candidate':'candidate','write':'write','rejection':'rejection','eviction':'eviction'}[kind]
        setattr(e,count,getattr(e,count)+1)
        setattr(e,'last_'+stem+'_request',request)
        setattr(e,'last_'+stem+'_s',s)

    def qualification(self,key,accepted):
        e=self.entry(key)
        field='qualification_accept_count' if accepted else 'qualification_reject_count'
        setattr(e,field,getattr(e,field)+1)

def request_category(facts):
    if not facts:
        return 'HIT',[]
    combo=[c for c in CHUNK_CATEGORIES if any(f['category']==c for f in facts)]
    assert all(f['category'] in CHUNK_CATEGORIES for f in facts)
    return (combo[0] if len(combo)==1 else 'MIXED'),combo
