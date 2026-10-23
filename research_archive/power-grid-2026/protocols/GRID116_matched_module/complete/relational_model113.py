"""Same tensors, paired init; only feedback association differs."""
import math,torch
from torch import nn
def mlp(a,b):return nn.Sequential(nn.Linear(a,16),nn.ReLU(),nn.Linear(16,b),nn.ReLU())
class RelationalValue(nn.Module):
    def __init__(self,arm):
        super().__init__();assert arm in ['JOINT','FACTOR'];self.arm=arm
        self.A=mlp(68,16);self.F=mlp(8,16);self.V=mlp(32,16)
        self.Q=nn.Linear(16,8,bias=False);self.K=nn.Linear(16,8,bias=False)
        self.G=mlp(58,16);self.H=mlp(48,16)
        self.head=nn.Sequential(nn.Linear(48,16),nn.ReLU(),nn.Linear(16,2))
    def forward(self,context_action,context_feedback,next_action,anchor_action,global41,known_query_cost,context_mask,next_mask):
        # Mask before encoding too: padded values, including NaN, cannot enter the model.
        ca=torch.where(context_mask[...,None],context_action,0.)
        cf=torch.where(context_mask[...,None],context_feedback,0.)
        na=torch.where(next_mask[...,None],next_action,0.)
        a=self.A(ca);f=self.F(cf);q=self.A(na);anchor=self.A(anchor_action)
        cm=context_mask[...,None];nm=next_mask[...,None]
        if self.arm=='FACTOR':f=(f*cm).sum(1,keepdim=True)/cm.sum(1,keepdim=True);f=f.expand_as(a)
        v=self.V(torch.cat([a,f],-1))
        logits=self.Q(q)@self.K(a).transpose(1,2)/math.sqrt(8)
        logits=logits.masked_fill(~context_mask[:,None,:],-torch.inf)
        r=logits.softmax(-1)@v
        g=self.G(torch.cat([global41,anchor,known_query_cost],-1))
        h=self.H(torch.cat([q,r,g[:,None,:].expand(-1,q.shape[1],-1)],-1))
        avg=(h*nm).sum(1)/nm.sum(1)
        mx=h.masked_fill(~nm,-torch.inf).max(1).values
        return self.head(torch.cat([avg,mx,g],-1))
