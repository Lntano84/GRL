import compress_json, pathlib, sys
for name in sys.argv[1:]:
    j=compress_json.load(name)
    print('\nFILE',name)
    print('TOP',list(j.keys()))
    print('OPTIONS',{k:j.get('options',{}).get(k) for k in ('log_interval','stats_start','size_gb','sample_ratio','write_mbps','trace','ap','ap_threshold','prefetch_when','prefetch_range')})
    print('RESULT_KEYS',list(j.get('results',{}).keys()))
    b=j.get('batches',{})
    print('BATCH_KEYS',list(b.keys()))
    for k,v in b.items():
        if hasattr(v,'__len__'):
            print('  ',k,len(v),v[:3] if isinstance(v,list) else type(v).__name__)
