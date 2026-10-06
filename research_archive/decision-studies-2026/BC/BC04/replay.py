"""One author STATIC-BASE replay, with observational hooks only.

Classify using detached history at run_get ENTRY; log actual need_fetch at
_log_st. Real writes are counted only after QueueCache.admit succeeds. Real
rejections are committed only after process_admit_buffer completes and its
rejection counter delta verifies the recorded decisions. check_only is separate.
"""
import gzip
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
sys.path.insert(0,str(REPO))
import compress_json
import numpy as np
from BCacheSim.cachesim import sim_cache, admission_policies as aps, eviction_policies as evp
from BCacheSim.cachesim.utils import ods
from BCacheSim.cachesim.simulate_ap import get_parsed_args
from BCacheSim.episodic_analysis.episodes import service_time

class Recorder:
    def __init__(self):
        self.history=History()
        self.request_sequence=0
        self.get_sequence=0
        self.current=None
        self.frames=[]
        self.totals=Counter()
        self.max_per_request_cost_error_s=0.
        self.start=time.perf_counter()
        self.requests=gzip.open(WORK/'all_get_requests.jsonl.gz','wt',encoding='utf-8',compresslevel=1)
        self.qualifiers=gzip.open(WORK/'prefetch_qualification_checks.jsonl.gz','wt',encoding='utf-8',compresslevel=1)
        ref=compress_json.load(str(REFERENCE))['batches']
        self.ends=np.asarray(ref['time_elapsed_phy'],dtype=float)
        assert len(self.ends)==1008 and self.ends[143]==EVAL_START
        self.durations=np.diff(self.ends,prepend=0.)

    def event(self,key,kind,ts):
        assert self.current is not None,'unexpected event outside a GET'
        s=ts.physical-START
        self.history.event(key,kind,self.request_sequence,s)
        self.current['actions'][kind].append([typed(key),s])
        self.totals[kind]+=1

    def qualify(self,batch,decisions,ts):
        assert self.current is not None
        for key,accepted in decisions.items():
            self.history.qualification(key,bool(accepted))
        record={'request_sequence':self.request_sequence,'physical_time_s':ts.physical,
                'trace_elapsed_s':ts.physical-START,'window_index':self.current['window_index'],
                'check_only':True,'decisions':[[typed(k),bool(v)] for k,v in decisions.items()]}
        self.qualifiers.write(json.dumps(record,separators=(',',':'))+'\n')
        self.totals['qualification_calls']+=1
        self.totals['qualification_items']+=len(batch)

    def begin(self,sim,acc):
        assert self.current is None and sim.ram_cache is None and sim.cache.cache_size==3002
        self.get_sequence+=1
        first,facts=self.history.snapshot(acc.block_id,acc.chunks,sim.cache.admit_buffer)
        relative=acc.ts.physical-START
        # Author _stats already executed. Validate its actual pre-request
        # checkpoint assignment against the pinned reference, including last IO.
        window=len(ods.batches.get('time_elapsed_phy',[]))
        assert 0<=window<1008
        assert int(np.searchsorted(self.ends,relative,side='right'))==window or (window==1007 and relative==self.ends[-1])
        self.current={'request_sequence':self.request_sequence,'get_sequence':self.get_sequence,
                      'author_logical_time':acc.ts.logical,'physical_time_s':acc.ts.physical,
                      'trace_elapsed_s':relative,'window_index':window,
                      'full_block_key':typed(acc.block_id),'requested_chunks':list(acc.chunks),
                      'block_had_prior_get':not first,'snapshot_phase':'before_run_get',
                      '_facts':facts,'actions':{kind:[] for kind in ['candidate','write','rejection','eviction']}}

    def cost(self,need_fetch,need_prefetch,acc,before,after):
        record=self.current
        assert record is not None and record['author_logical_time']==acc.ts.logical
        assert len(set(need_fetch))==len(need_fetch)
        facts=[record['_facts'][chunk] for chunk in need_fetch]
        category,combo=request_category(facts)
        if need_fetch:
            demand_span=max(need_fetch)-min(need_fetch)+1
            full=list(need_fetch)+list(need_prefetch)
            extra=max(full)-min(full)+1-demand_span
            demand=float(service_time(1,demand_span))
            prefetch=float(service_time(0,extra))
        else:
            assert not need_prefetch
            demand_span=extra=0
            demand=prefetch=0.
        errors=[abs(after[0]-before[0]-demand),abs(after[1]-before[1]-prefetch),
                abs(after[2]-before[2]-demand-prefetch)]
        error=max(errors)
        assert error<=1e-9,('request metering mismatch',error)
        self.max_per_request_cost_error_s=max(self.max_per_request_cost_error_s,error)
        coefficient=100/(36*.001*self.durations[record['window_index']])
        record.update(need_fetch=list(need_fetch),need_prefetch=list(need_prefetch),
                      actual_missing_chunk_history=facts,request_category=category,category_combination=combo,
                      demand_read_span_chunks=demand_span,prefetch_extra_read_span_chunks=extra,
                      demand_service_time_s=demand,prefetch_extra_service_time_s=prefetch,
                      demand_dt_pct_of_window=demand*coefficient,prefetch_extra_dt_pct_of_window=prefetch*coefficient)
        self.totals['missing_chunks']+=len(facts)
        self.totals['get_requests']+=1
        self.totals['category/'+category]+=1
        for fact in facts:self.totals['chunk_category/'+fact['category']]+=1

    def finish(self,acc):
        record=self.current
        assert 'request_category' in record
        del record['_facts']
        self.requests.write(json.dumps(record,separators=(',',':'))+'\n')
        self.history.finish_get(acc.block_id)
        self.current=None

    def close(self):
        self.requests.close()
        self.qualifiers.close()

def main():
    os.chdir(REPO)
    recorder=Recorder()
    original_stats=sim_cache.CacheSimulator._stats
    original_get=sim_cache.CacheSimulator.run_get
    original_log_st=sim_cache.CacheSimulator._log_st
    original_insert=evp.QueueCache.insert
    original_admit=evp.QueueCache.admit
    original_process=evp.QueueCache.process_admit_buffer
    original_accept=aps.NewMLAP.batchAccept
    original_eviction=evp.EvictionPolicy.log_eviction
    original_checkpoint=sim_cache.CacheSimulator._checkpoint

    def stats(self,ts):
        original_stats(self,ts)
        recorder.request_sequence+=1
        assert self.start_ts.physical==START
    def run_get(self,acc):
        recorder.begin(self,acc)
        result=original_get(self,acc)
        recorder.finish(acc)
        return result
    def log_st(self,need_fetch,need_prefetch,all_chunks_hit,acc):
        before=[ods.get(k) for k in ['service_time_used_demand','service_time_used_prefetch','service_time_used']]
        result=original_log_st(self,need_fetch,need_prefetch,all_chunks_hit,acc)
        after=[ods.get(k) for k in ['service_time_used_demand','service_time_used_prefetch','service_time_used']]
        recorder.cost(need_fetch,need_prefetch,acc,before,after)
        return result
    def insert(self,key,ts,keyfeaturelist,*,metadata=None):
        # Original insert asserts a real candidate absent from the cache.
        # Recording before original execution captures batch candidates even when
        # this call immediately triggers processing; request history is detached.
        assert key not in self.cache
        recorder.event(key,'candidate',ts)
        return original_insert(self,key,ts,keyfeaturelist,metadata=metadata)
    def admit(self,key,ts,**kwargs):
        before=self.keys_written
        result=original_admit(self,key,ts,**kwargs)
        assert self.keys_written-before==1
        recorder.event(key,'write',ts)
        return result
    def accept(self,batch,ts,*,metadata=None,check_only=False):
        assert self.threshold==.798545,'BC04 must retain fixed original threshold'
        decisions=original_accept(self,batch,ts,metadata=metadata,check_only=check_only)
        if check_only:
            recorder.qualify(batch,decisions,ts)
        else:
            assert recorder.frames,'actual admission call outside process_admit_buffer'
            recorder.frames[-1].extend(k for k,v in decisions.items() if not v)
        return decisions
    def process(self,ts):
        frame=[]
        before=self.rejections
        recorder.frames.append(frame)
        try:
            result=original_process(self,ts)
            assert self.rejections-before==len(frame)
            for key in frame:recorder.event(key,'rejection',ts)
            return result
        finally:
            assert recorder.frames.pop() is frame
    def eviction(self,ts,evicted):
        before=self.evictions
        result=original_eviction(self,ts,evicted)
        assert self.evictions-before==1
        recorder.event(evicted[0],'eviction',ts)
        return result
    def checkpoint(self,ts,*args,**kwargs):
        result=original_checkpoint(self,ts,*args,**kwargs)
        (WORK/'progress.json').write_text(json.dumps({'windows':len(ods.batches.get('time_elapsed_phy',[])),
                                                   'trace_elapsed_s':ts.physical-START,
                                                   'wall_seconds':time.perf_counter()-recorder.start}))
        return result
    sim_cache.CacheSimulator._stats=stats
    sim_cache.CacheSimulator.run_get=run_get
    sim_cache.CacheSimulator._log_st=log_st
    evp.QueueCache.insert=insert
    evp.QueueCache.admit=admit
    evp.QueueCache.process_admit_buffer=process
    aps.NewMLAP.batchAccept=accept
    evp.EvictionPolicy.log_eviction=eviction
    sim_cache.CacheSimulator._checkpoint=checkpoint
    sys.argv=['bc04/replay.py','--config',str(WORK/'config.json')]
    options=get_parsed_args()
    assert options.peak_strategy is None and options.write_mbps==0 and options.ap_threshold==.798545
    audit={'completed':False,'mode':'read_only_observation'}
    try:
        sim_cache.simulate_cache_driver(options)
        assert recorder.current is None and not recorder.frames
        assert recorder.totals['get_requests']==127305 and recorder.request_sequence==147794
        assert recorder.totals['write']==ods.get('flashcache/keys_written')
        assert recorder.totals['rejection']==ods.get('flashcache/rejections')
        assert recorder.totals['eviction']==ods.get('flashcache/evictions')
        assert sum(recorder.totals['chunk_category/'+c] for c in CHUNK_CATEGORIES)==recorder.totals['missing_chunks']
        audit['completed']=True
    finally:
        recorder.close()
        audit.update(totals=dict(recorder.totals),requests=recorder.request_sequence,
                     unique_seen_get_blocks=len(recorder.history.seen_get_blocks),
                     max_per_request_cost_error_s=recorder.max_per_request_cost_error_s,
                     wall_seconds=time.perf_counter()-recorder.start)
        (WORK/'replay_audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':main()
