"""Label-independent cyclic feedback reassociation of fixed inputs; sensitivity only."""
from analyze import *
def main():
    assert not (O/'binding_sensitivity.json').exists();configure();verify113();saved=np.load(S/'state_predictions.npz');result={}
    for rec in read(S/'fits.json')['records']:
        name=f"{rec['arm']}_seed{rec['seed']}";m=RelationalValue(rec['arm']);cp=torch.load(S/'models'/f'{name}.pt',map_location='cpu',weights_only=False);m.load_state_dict(cp['state_dict']);before=digest(m);m.eval();result[name]={}
        for part in ['train','validation']:
            rows=pickle.loads((S/f'{part}_public.pkl').read_bytes());y=np.load(S/f'{part}_targets.npz')['target'];outs=[]
            with torch.no_grad():
                for j in range(0,len(rows),32):
                    rr=[dict(r,context_feedback=np.roll(r['context_feedback'],1,axis=0)) for r in rows[j:j+32]]
                    outs.append(m(**collate(rr)).numpy())
            q=np.concatenate(outs);base=saved[name+'__'+part];delta=(q[:,1]-q[:,0])-(base[:,1]-base[:,0]);changed=(q[:,1]>q[:,0])!=(base[:,1]>base[:,0]);eligible=np.array([len(r['context_feedback'])>1 for r in rows]);badv=base[:,1]-base[:,0]
            result[name][part]=dict(all_abs_advantage_change=stats(abs(delta)),positive_abs_advantage_change=stats(abs(delta[y>0])),
                eligible_contexts=int(eligible.sum()),decision_sign_changes=int(changed.sum()),positive_decision_sign_changes=int((changed&(y>0)).sum()),
                base_positive_abs_margin=stats(abs(badv[y>0])),max_output_absolute_change=float(np.max(abs(q-base))))
            if rec['arm']=='FACTOR':assert np.allclose(q,base,atol=3e-5,rtol=3e-5)
        assert before==digest(m)==rec['final_hash']
    save(O/'binding_sensitivity.json',dict(passed=True,results=result,
        perturbation='Exactly one label-independent cyclic shift of paid feedback rows within each existing context. Geometry,feedback multiset,global41 and query set fixed.',
        meaning='Sensitivity diagnostic, not a new baseline fit or performance/causal claim. A single reassociation cannot establish all conditional dependence or its usefulness.',
        no_policy_deployment=True,new_fits=0,new_physical_calls=0,source_sha256=sha(Path(__file__))))
    print(json.dumps({n:r['train'] for n,r in result.items()},indent=2))
if __name__=='__main__':main()
