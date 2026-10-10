"""Targeted checks for V1 gradient separation and bootstrap representation."""
from common import *
from networks_v1 import ActorCritic
checks={}
native_buffer=[{'features':(None,None,np.array([0,0,0,0,0,0,a],dtype=np.float32))} for a in [0.,1.]]
active=torch.tensor([bool(b['features'][2][6]>.5) for b in native_buffer],dtype=torch.bool)
assert active.tolist()==[False,True]
checks['native_numpy_mask_conversion']=True
torch.manual_seed(20261009)
x=torch.randn(4,8,14);g=torch.randn(4,11);g[:,6]=torch.tensor([0.,1.,0.,1.])
edges=[torch.tensor([[0,1,2,3],[1,2,3,0]]) for _ in range(4)]
for kind in ['gnn','mlp']:
    m=ActorCritic(kind,8,max_edges=12)
    logits,v=m(x,edges,g)
    assert torch.isfinite(logits).all() and torch.isfinite(v).all()
    assert (logits[[0,2],1]==-1.e9).all()
    m.zero_grad();v.sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in m.policy.parameters())
    assert any(p.grad is not None and p.grad.any() for p in m.value.parameters())
    checks[kind+'_critic_cannot_update_actor']=True
    m.zero_grad();logits,v=m(x,edges,g)
    torch.nn.functional.cross_entropy(logits[[1,3]],torch.tensor([0,1])).backward()
    assert any(p.grad is not None and p.grad.any() for p in m.policy.actor.parameters())
    assert all(p.grad is None for p in m.value.parameters())
    checks[kind+'_actor_cannot_update_critic']=True
    changed=g.clone();changed[:,6:]=torch.randn_like(changed[:,6:])*100
    with torch.no_grad():_,a=m(x,edges,g);_,b=m(x,edges,changed)
    assert torch.equal(a,b)
    checks[kind+'_critic_bootstrap_proposal_invariance']=True
    with torch.no_grad():
        inactive=g.clone();inactive[:,6]=0.
        disabled,_=m(x,edges,inactive)
        assert (disabled.argmax(-1)==0).all()
    checks[kind+'_inactive_mask']=True
write_json(OUT/'v1/preflight.json',{'passed':True,'checks':checks,
    'scope':'Neural update and representation checks; physical proposal qualification reuses unchanged V0 preflight.'})
print(json.dumps(checks))
