"""One soft-compression followup:original scale,unit local slope,no hard ceiling."""
import torch
from prior import ResidualScorer

class CompressedResidualScorer(ResidualScorer):
    def forward(self,x,edges,edge_attr,glob,base,plans):
        assert self.kind=='gnn'
        raw=self.inner(x,edges,edge_attr,glob,base,plans)
        # Original scale2 is retained,not tuned. Asinh grows logarithmically and
        # retains nonzero slope rather than tanh's saturating hard score range.
        return 2.*torch.asinh(raw/2.)
