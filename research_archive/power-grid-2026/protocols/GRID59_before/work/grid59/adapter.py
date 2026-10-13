"""Frequency prior + unchanged residual graph scorer; shared stop/control remain intact."""
import hashlib,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'work/grid58'),str(ROOT/'work/grid36')]
from compressed_residual import CompressedResidualScorer
from frequency_prior import frequency_logs
from cheap_rank import rank as static_rank
from components import public_inputs,tensors,canonical_groups
from common import np,torch

def install_frequency_adapter(Search):
    old_init,old_rank=Search.__init__,Search.rank
    def init(self,env,ctl,rule,budget,model_path=None):
        if rule!='FREQUENCYGNN':return old_init(self,env,ctl,rule,budget,model_path)
        old_init(self,env,ctl,rule,budget,None)
        assert model_path is not None
        f=public_inputs(env.get_obs(),env.action_space(),env,self.plans)
        self.model=CompressedResidualScorer('gnn',2*env.n_sub,2*env.n_line,len(f['base']),env.dim_topo,self.subids)
        self.model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True));self.model.eval()
        self.checkpoint_sha=hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    def rank(self,obs,base,eligible,reps,aliases):
        if self.rule=='STATIC_ID':return static_rank(self,obs,base,eligible,reps,aliases)
        if self.rule!='FREQUENCYGNN':return old_rank(self,obs,base,eligible,reps,aliases)
        t=time.perf_counter();prior=frequency_logs(reps,len(self.library));prior_s=time.perf_counter()-t
        t=time.perf_counter();f=public_inputs(obs,base,self.env,self.plans)
        a,b=canonical_groups(f,eligible);assert a==reps and b==aliases;feature_s=time.perf_counter()-t
        t=time.perf_counter()
        with torch.no_grad():residual=self.model(**tensors(f))[0].numpy()
        assert np.isfinite(residual).all();values=residual+prior;order=sorted(reps,key=lambda i:(-float(values[i]),i))
        return order,prior,f,{'prior_s':prior_s,'feature_s':feature_s,'forward_rank_s':time.perf_counter()-t}
    Search.__init__,Search.rank=init,rank

