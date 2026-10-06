import compress_json,sys
for stats_path,summary_path in zip(sys.argv[1::2],sys.argv[2::2]):
 s=compress_json.load(stats_path); j=compress_json.load(summary_path)
 print('\n',stats_path)
 print('summary keys',list(j.keys()))
 print('options', {k:j.get('options',{}).get(k) for k in ['size_gb','size_opt','log_interval','stats_start','trace','sample_ratio','ap','ap_threshold','prefetch_when','prefetch_range','write_mbps']})
 print('results selected',{k:v for k,v in j.get('results',{}).items() if any(x in k.lower() for x in ['cache','write','fetch','prefetch','hit','miss','sample'])})
 b=s['batches']; keys=[k for k in b if k in ['time_phy','time_elapsed_phy','time_log','duration','realtime_elapsed','iops_requests_stats','chunk_queries_stats','fetches_ios_stats','fetches_chunks_stats','fetches_chunks_demandmiss_stats','fetches_chunks_prefetch_stats','puts_ios_stats','puts_chunks_stats','service_time_writes_stats','flashcache/keys_written_stats','flashcache/evictions_stats','flashcache/admits_stats'] or k.startswith('flashcache/')]
 print('selected series')
 for k in sorted(keys):
  v=b[k]
  print(k, len(v),v[:3],v[-3:])
 print('freq keys',[k for k in s.get('freq',{}) if any(x in k.lower() for x in ['hit','prefetch','write','miss','fetch'])])
