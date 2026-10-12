"""Original residual network, removing only the final hard tanh bound."""
from prior import ResidualScorer

class FreeResidualScorer(ResidualScorer):
    def forward(self,x,edges,edge_attr,glob,base,plans):
        # Same inner parameters, zero initial head, graph decoder and prior KL.
        # This changes the scoring function only; no extra encoder or RL head.
        assert self.kind=='gnn'
        return self.inner(x,edges,edge_attr,glob,base,plans)
