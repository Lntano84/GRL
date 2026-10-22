"""Extend the frozen37 prefix summaries to a locked-station query history."""
import numpy as np
from common import ROOT
source=ROOT/'work/grid101_gain_learning/features.py'
code=source.read_text(encoding='utf8')
assert code.count('def project(x):')==1
code=code.replace('def project(x):','def project(x, locked_targets=None):')
marker="    targets = {int(rep) for original, rep in x['public_aliases'].items() if subs[int(original)] == station}"
assert code.count(marker)==1
code=code.replace(marker,"    targets = set(locked_targets) if locked_targets is not None else {int(rep) for original, rep in x['public_aliases'].items() if subs[int(original)] == station}")
namespace={'__name__':'grid102_prefix_projection'}
exec(compile(code,str(source),'exec'),namespace)
NAMES=namespace['NAMES']+['local_batches_done','local_budget_used_fraction','log_next_batch_queries','locked_remaining_fraction']
best=namespace['best'];reward=namespace['reward']


def observe(visible,order,aliases,subids,rho_limit,locked_targets,remaining,original_budget,batches_done):
    """Only purchased feedback is accepted; no oracle or target argument."""
    ids=[r['pool_id'] for r in visible];seen=set(ids)
    public_order=ids+[i for i in order if i not in seen]
    x=dict(visible=visible,public_order=public_order,public_aliases=aliases,public_subids=subids,
        visible_count=len(visible),incumbent_id=best(visible)['pool_id'],
        remaining_same_station_ids=remaining,known_completion_queries=len(remaining),rho_limit=rho_limit)
    base=namespace['project'](x,locked_targets)
    extra=[batches_done,(original_budget-len(remaining))/original_budget,
           np.log1p(min(64,len(remaining))),len(remaining)/original_budget]
    result=np.concatenate([base,extra]).astype(np.float32)
    assert len(result)==41 and np.isfinite(result).all()
    return result
