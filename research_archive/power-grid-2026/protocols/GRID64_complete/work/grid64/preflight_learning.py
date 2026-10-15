"""Pre-fit mathematical and causal interface tests; uses train split only."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
import metacontrol as M
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid64'
def main():
    torch.set_num_threads(1);checks=[]
    records=json.loads((OUT/'records.json').read_text());train=[r for r in records if r['split']=='train']
    torch.manual_seed(20261015);net=M.QModel()
    assert [m.out_features for m in net.modules() if isinstance(m,torch.nn.Linear)]==[64,32,2]
    assert all(torch.count_nonzero(m.bias)==0 for m in net.modules() if isinstance(m,torch.nn.Linear));checks.append('source_architecture_and_zero_bias')
    # Explicit numeric Double-Q target and MSE over two action outputs.
    online=torch.tensor([[.1,.2],[.4,.3]]);states=torch.zeros(2,12);states[0,2]=0;states[1,2]=1
    probabilities=torch.softmax(M.mask_q(online,states)/.001,dim=1)
    target=torch.tensor([[99.,.7],[.5,.8]])
    expected=(probabilities*target).sum(1);assert torch.allclose(expected,torch.tensor([.7,.5]))
    y=.2+.99*expected;assert abs(float(y[0])-.893)<1e-6;checks.append('double_q_masked_target_manual')
    pred=torch.tensor([[1.,3.]],requires_grad=True);desired=torch.tensor([[2.,3.]])
    loss=torch.nn.functional.mse_loss(pred,desired);assert float(loss)==.5
    loss.backward();assert torch.equal(pred.grad,torch.tensor([[-1.,0.]]));checks.append('source_two_output_mse_manual')
    q=[.4,.45,.47];t=[.1,.4,1.];beta=math.log1p(.25)
    rs=[M.utility(q[i+1],t[i+1],beta)-M.utility(q[i],t[i],beta) for i in range(2)]
    assert abs(sum(rs)-(M.utility(q[-1],1.,beta)-M.utility(q[0],.1,beta)))<1e-12;checks.append('utility_difference_telescoping_undiscounted')
    # Source gamma=.99 is retained, not claimed equivalent to undiscounted optimal stopping.
    stop=M.QModel()
    with torch.no_grad():
        for p in stop.parameters():p.zero_()
        stop.net[-1].bias[0]=1.
    order=list(range(400))
    def observe(i,phase):return {'pool_id':i,'reward':1. if i!=399 else 1.8,'rho':1. if i!=399 else .2,'strict_admissible':True,'phase':phase}
    a=M.run(order,observe,stop,1.2);assert a['public_queries']==128 and a['chosen_pool_id']==0
    b=M.run(order,lambda i,p:{**observe(i,p),'reward':999. if i>=128 else 1.},stop,1.2)
    assert a==b;checks.append('suffix_cannot_affect_current_stop')
    invalid=M.run(order,lambda i,p:{**observe(i,p),'strict_admissible':False},stop,1.2)
    assert invalid['public_queries']==400 and invalid['chosen_pool_id'] is None;checks.append('no_admissible_forces_complete_search')
    for r in train[:12]:
        truth={v['pool_id']:v for v in r['rows']};order=[v['pool_id'] for v in r['rows']]
        got=M.run(order,lambda i,p:{**truth[i],'phase':p},net,r['rho_limit']);replay=M.replay(net,r['rows'],r['rho_limit'])
        assert got['public_queries']==replay['end'] and got['chosen_pool_id']==replay['chosen']
    checks.append('12_training_trace_online_vs_cached_prefix_equivalence')
    h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'learning_preflight.json').write_text(json.dumps({'passed':True,'checks':checks,'script_sha256':h,'evaluation_records_used':0,'new_fits':0},indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'checks':len(checks)}))
if __name__=='__main__':main()
