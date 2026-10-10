"""V1 isolates the actor representation and critic on sparse residual decisions."""
from common import torch,ActorCritic as OriginalActorCritic

class ActorCritic(torch.nn.Module):
    def __init__(self,kind,n_nodes,hidden=64,max_edges=372):
        super().__init__();self.kind=kind;self.n_nodes=n_nodes;self.max_edges=max_edges
        self.policy=OriginalActorCritic(kind,n_nodes,hidden,max_edges)
        for p in self.policy.critic.parameters():p.requires_grad_(False)
        # Same independent critic for both actors. It receives observed pooled
        # node features and public global state; proposal availability and
        # forecast-only fields are excluded so rollout bootstrap is consistent.
        self.value=torch.nn.Sequential(torch.nn.Linear(34,128),torch.nn.Tanh(),
            torch.nn.Linear(128,64),torch.nn.Tanh(),torch.nn.Linear(64,1))

    def forward(self,x,edges,glob):
        logits,_=self.policy(x,edges,glob)
        if x.ndim==2:x=x.unsqueeze(0);glob=glob.unsqueeze(0)
        pooled=torch.cat([x.mean(1),x.max(1).values,glob[:,:6]],dim=-1)
        return logits,self.value(pooled).squeeze(-1)
