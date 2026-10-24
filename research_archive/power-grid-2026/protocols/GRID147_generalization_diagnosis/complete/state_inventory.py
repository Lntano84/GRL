"""Inventory existing pre-search physical vectors; do not initialize an environment."""
import gzip,hashlib,json,collections
import run as D

def main():
    D.verify();assert not (D.O/'saved_state_inventory.json').exists()
    before_sources={};steps={};records=[]
    for ref in D.B.refs():
        p=D.R/ref['source'];c=D.B.call_at(ref['source']);v=p.with_name(p.stem+'_state.npz');folder=p.parent
        if folder not in steps:
            st=folder/'steps.jsonl.gz';before_sources[st.relative_to(D.R).as_posix()]=D.sha(st)
            with gzip.open(st,'rt',encoding='utf8') as f:steps[folder]={r['step']:r['before_hash'] for r in map(json.loads,f)}
        h=D.sha(v);before_sources[v.relative_to(D.R).as_posix()]=h
        with D.np.load(v,allow_pickle=False) as z:
            assert set(z.files)=={'observation','base_action'}
            obs=z['observation'];base=z['base_action']
            digest=hashlib.sha256(D.np.ascontiguousarray(obs).tobytes()).hexdigest()
            assert digest==c['state_hash']==steps[folder][c['step']]
            records.append(dict(call_id=ref['call_id'],partition=ref['partition'],date=ref['date'],
                observation_file=v.relative_to(D.R).as_posix(),file_sha256=h,observation_shape=list(obs.shape),base_action_shape=list(base.shape),
                observation_dtype=str(obs.dtype),observation_nonfinite=int((~D.np.isfinite(obs)).sum()),state_hash_matches_call_and_before_step=True))
    for p,h in before_sources.items():assert D.sha(D.R/p)==h
    D.save('saved_state_source_hashes.json',before_sources)
    assert len(records)==305
    D.save('saved_state_inventory.json',dict(passed=True,records=records,calls=305,folders=len(steps),
        dates=len({r['date'] for r in records}),partitions=dict(collections.Counter(r['partition'] for r in records)),
        observation_shapes=sorted({tuple(r['observation_shape']) for r in records}),
        source='GRID28 runner saved pending search state;GRID100 extract independently checked hash against physical before-step vector.',
        no_environment_initialized=True,new_physical_queries=0,new_samples_collected=0,
        field_layout_decoded=False,next='Read frozen observation layout/schema before deriving line/topology features. Never guess vector offsets or use later observations/hidden query results.'))
    D.verify();print('SAVED_STATE_INVENTORY',len(records),'calls',len(steps),'folders')

if __name__=='__main__':main()
