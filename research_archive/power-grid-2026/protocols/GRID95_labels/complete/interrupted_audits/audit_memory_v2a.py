"""Correct an over-inclusive witness check; preserve all simulations and original audit."""
from pathlib import Path
import copy,hashlib,json,re
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid95_labels'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
IGNORED=['base_optimizer','restore_optimizer','reconnect','recover_topology','topology_search']
RECORD=[]


def normalize(w):
    data=copy.deepcopy(w['inventory']);excluded={}
    for name in IGNORED:
        if 'space_prng' in data[name]['values']:
            excluded[name]=data[name]['values'].pop('space_prng')
    return data,excluded


def qualified_memory_equal(a,b):
    aa,ra=normalize(a);bb,rb=normalize(b)
    assert aa==bb,'A decision-relevant numeric or explicitly seeded RNG witness differs'
    RECORD.append(dict(original_hash_equal=a['sha256']==b['sha256'],
        normalized_hash=hashlib.sha256(json.dumps(aa,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        unseeded_unused_module_rngs_complete=ra,unseeded_unused_module_rngs_skip=rb))
    return True


def main():
    # These modules construct inherited BaseAgent RNGs but none of their executed
    # get_act/optimization paths reads those RNGs. Root agent/environment/opponent
    # RNGs are deliberately NOT excluded. Actual topology get_act is replaced by
    # the deterministic recorded search callback.
    paths=['work/grid02/ljn_nn/modules/base_module.py','work/grid02/ljn_nn/modules/topology_heuristic.py',
           'work/grid02/ljn_nn/modules/convex_optim.py']
    for rel in paths:
        text=(ROOT/rel).read_text(encoding='utf-8')
        assert not re.search(r'space_prng|np\.random|random\.|\.sample\(|\.shuffle\(|\.permutation\(',text),rel
    frozen=read(OUT/'code_freeze.json')
    for rel in paths:assert sha(ROOT/rel)==frozen[rel]
    assert (OUT/'audit_original_stdout_stderr.log').exists()
    source=W/'audit.py';code=source.read_text(encoding='utf-8')
    old1="assert wa['before_search']['sha256']==wb['before_search']['sha256']"
    old2="assert wa['after_global']['sha256']==wb['after_global']['sha256']"
    assert code.count(old1)==code.count(old2)==1
    code=code.replace(old1,"assert qualified_memory_equal(wa['before_search'],wb['before_search'])")
    code=code.replace(old2,"assert qualified_memory_equal(wa['after_global'],wb['after_global'])")
    env={'__file__':str(source),'__name__':'grid95_readonly_audit_v2','qualified_memory_equal':qualified_memory_equal}
    exec(compile(code,str(source),'exec'),env);env['main']()
    assert len(RECORD)==8
    receipt=dict(passed=True,kind='Read-only witness normalization; no physical or protocol change',
        original_audit_sha256=sha(source),original_failure_log_sha256=sha(OUT/'audit_original_stdout_stderr.log'),
        excluded_only=IGNORED,reason='Inherited but unseeded and unused RNG objects were incorrectly included in a check specified for explicitly seeded objects. Their values remain in raw witnesses.',
        proof_sources={rel:sha(ROOT/rel) for rel in paths},checked_memory_pairs=RECORD,
        controller_numeric_and_used_seeded_rngs_equal=True,new_replays=0,new_fits=0,
        unchanged_checks='All physical prefixes, actions, feedback, costs, same-choice negative control, env/agent/opponent RNG witnesses and numerical optimizer parameters still required.')
    (OUT/'audit_memory_v2_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    review=read(OUT/'review.json');review['memory_witness_normalization']='audit_memory_v2_receipt.json; original failure and all raw witnesses preserved'
    (OUT/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
