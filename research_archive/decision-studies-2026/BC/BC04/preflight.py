import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def main():
    h=History()
    # FIRST takes priority and PUT is not a historical GET.
    first,freeze=h.snapshot('A',[1],set())
    assert first and freeze[1]['category']=='FIRST'
    h.event(('A',1),'candidate',1,0.)
    h.event(('A',1),'rejection',1,0.)
    assert freeze[1]['category']=='FIRST' and freeze[1]['rejection_count']==0
    h.finish_get('A')
    first,freeze=h.snapshot('A',[2],set())
    assert not first and freeze[2]['category']=='OTHER'
    h.event(('A',2),'candidate',2,1.)
    h.event(('A',2),'rejection',2,1.)
    # This is the trap: current rejection must not rewrite current history.
    assert freeze[2]['category']=='OTHER' and freeze[2]['rejection_count']==0
    assert h.snapshot('A',[2],set())[1][2]['category']=='REJECTED-BEFORE'
    # check_only accepts and rejects both leave real writes/rejections/candidates untouched.
    for accepted,chunk in [(True,3),(False,4)]:
        h.qualification(('A',chunk),accepted)
        fact=h.snapshot('A',[chunk],set())[1][chunk]
        assert fact['category']=='OTHER' and fact['written_count']==fact['rejection_count']==fact['candidate_count']==0
    h.event(('A',2),'write',3,2.)
    h.event(('A',2),'eviction',4,3.)
    fact=h.snapshot('A',[2],set())[1][2]
    assert fact['category']=='ADMITTED-BEFORE' and fact['rejection_count']==1
    # FIRST is block based, not chunk based; new chunk on old block is OTHER.
    assert h.snapshot('A',[5],set())[1][5]['category']=='OTHER'
    h.event(('A',6),'candidate',5,4.)
    fact=h.snapshot('A',[6],{('A',6)})[1][6]
    assert fact['category']=='OTHER' and fact['candidate_count']==1 and fact['pending_before_request']
    mixed=h.snapshot('A',[1,2,5],set())[1]
    assert request_category(list(mixed.values()))==('MIXED',['ADMITTED-BEFORE','REJECTED-BEFORE','OTHER'])
    assert request_category([])==('HIT',[])
    # Complete native block key retained, including a hostname if present.
    h.finish_get(('block','hostA'))
    assert h.snapshot(('block','hostB'),[1],set())[0]
    WORK.mkdir(exist_ok=True,parents=True)
    source=json.loads(CONFIG.read_text())
    assert source['ap_threshold']==.798545 and source['write_mbps']==0 and source['ap']=='mlnew'
    config={**source,'output_dir':str(WORK/'raw').replace('\\','/')}
    assert {k for k in source if source[k]!=config[k]}=={'output_dir'}
    (WORK/'config.json').write_text(json.dumps(config,indent=2))
    files=[CONFIG,REFERENCE,REPO/'data/tectonic/201910/Region1/full_0_0.1.trace',
           *list((REPO/'tmp/example/201910_Region1_0_0.1').glob('*.model'))]
    result={'passed':True,'checks':['current_request_events_do_not_backdate','check_only_accept_and_reject_do_not_make_real_history',
                                  'first_block_get_priority','admission_over_rejection_priority','new_chunk_on_seen_block_other',
                                  'pending_and_prior_candidate_preserved','mixed_request_not_split','hit_zero_cost','hostname_identity_preserved'],
            'inputs_sha256':{str(p):sha(p) for p in files},'config_sha256':sha(WORK/'config.json'),
            'replay_budget_seconds':1800,'source_example_config':source}
    (WORK/'preflight.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'preflight_passed':True,'checks':result['checks']}))

if __name__=='__main__':main()
