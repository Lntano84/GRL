"""New scorer only;original fast policy,legal inputs and timing stay identical."""
import hashlib,time
from pathlib import Path
from free_residual import FreeResidualScorer
from components import public_inputs,tensors,canonical_groups
from prior import public_prior
from common import np,torch

def install_free_adapter(Search):
    old_init,old_rank=Search.__init__,Search.rank
    def init(self,env,ctl,rule,budget,model_path=None):
        if rule!='FREEGNN':return old_init(self,env,ctl,rule,budget,model_path)
        old_init(self,env,ctl,rule,budget,None);assert model_path is not None
        features=public_inputs(env.get_obs(),env.action_space(),env,self.plans)
        self.model=FreeResidualScorer('gnn',2*env.n_sub,2*env.n_line,len(features['base']),env.dim_topo,self.subids)
        self.model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True));self.model.eval()
        self.checkpoint_sha=hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    def rank(self,obs,base,eligible,reps,aliases):
        if self.rule!='FREEGNN':return old_rank(self,obs,base,eligible,reps,aliases)
        t=time.perf_counter();prior,_=public_prior(obs,self.env,self.ctl,eligible,aliases);prior_s=time.perf_counter()-t
        t=time.perf_counter();features=public_inputs(obs,base,self.env,self.plans)
        a,b=canonical_groups(features,eligible);assert a==reps and b==aliases;feature_s=time.perf_counter()-t
        t=time.perf_counter()
        with torch.no_grad():residual=self.model(**tensors(features))[0].numpy()
        assert np.isfinite(residual).all();values=residual+prior;order=sorted(reps,key=lambda i:(-float(values[i]),i))
        assert len(order)==len(reps) and set(order)==set(reps)
        return order,prior,features,{'feature_s':feature_s,'prior_s':prior_s,'forward_rank_s':time.perf_counter()-t}
    Search.__init__,Search.rank=init,rank
    return old_init,old_rank
