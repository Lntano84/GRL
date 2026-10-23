"""Post-hoc exhaustive action-cell accounting; no selected cases or thresholds."""
import diagnose as d
np=d.np

def main():
    assert not (d.O/'action_cell_accounting.json').exists();d.verify()
    plan=dict(posthoc=True,purpose='After the frozen gradient diagnosis, account for all main Huber and sign-regret changes by the exhaustive baseline/candidate STOP-CONT action cells, without selecting cases.',
      cells=['STOP_STOP','STOP_CONT','CONT_STOP','CONT_CONT'],rule='Saved main q1>0, unchanged from GRID123. No new threshold, predictions, model updates or case selection.',new_fits=0)
    d.save(d.O/'posthoc_partition_plan.json',dict(**plan,script_sha256=d.sha(d.W/'partition.py')))
    Q=np.load(d.S/'predictions.npz');ptrain=np.load(d.S/'train_data.npz')['probability'];rows=[]
    cases=d.src.pickle.loads((d.R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cc=d.collections.Counter(c['date'] for c in cases if c['partition']=='validation');by={c['sample_id']:c for c in cases};meta=d.read(d.S/'validation_metadata.json');pv=np.array([1/len(cc)/cc[m['date']]/len(by[m['sample_id']]['states']) for m in meta])
    for split in ['full','f0','f1','f2','f3']:
        part='validation' if split=='full' else 'train';_,y,e,ids,p=d.src.getdata(split,part)
        if split=='full':ids=np.arange(len(y));p=pv
        else:ids=np.setdiff1d(np.arange(len(y)),ids);p=ptrain[ids]/ptrain[ids].sum()
        y=y[ids];dom='validation' if split=='full' else 'fold_held'
        for arm in ['AUX_NETSIGN','AUX_GAIN']:
            for seed in range(3):
                bn=f'{split}__NET_ZERO_seed{seed}';an=f'{split}__{arm}_seed{seed}';b=Q[bn+'__'+part][ids,1].astype(float);a=Q[an+'__'+part][ids,1].astype(float);bg=b>0;ag=a>0
                def hub(v):
                    er=abs(v-y);return np.where(er<1,.5*er**2,er-.5)
                hd=hub(a)-hub(b);rd=(bg.astype(float)-ag.astype(float))*y;cells={}
                for v,w in [(0,0),(0,1),(1,0),(1,1)]:
                    mask=(bg==bool(v))&(ag==bool(w));key=('CONT' if v else 'STOP')+'_'+('CONT' if w else 'STOP');cells[key]=dict(states=int(mask.sum()),mass=float(p[mask].sum()),huber_difference=float(p[mask]@hd[mask]),sign_regret_difference=float(p[mask]@rd[mask]),profitable_mass=float(p[mask]@(y[mask]>0)))
                assert sum(c['states'] for c in cells.values())==len(y)
                assert abs(sum(c['huber_difference'] for c in cells.values())-p@hd)<1e-12
                assert abs(sum(c['sign_regret_difference'] for c in cells.values())-p@rd)<1e-12
                rows.append(dict(name=an,domain=dom,cells=cells,huber_difference=float(p@hd),sign_regret_difference=float(p@rd)))
    summary={}
    for dom in ['validation','fold_held']:
        summary[dom]={}
        for arm in ['AUX_NETSIGN','AUX_GAIN']:
            rr=[r for r in rows if r['domain']==dom and '__'+arm+'_seed' in r['name']];ss={}
            for cell in plan['cells']:ss[cell]={k:float(np.mean([r['cells'][cell][k] for r in rr])) for k in ['mass','huber_difference','sign_regret_difference','profitable_mass']}
            summary[dom][arm]=ss
    d.save(d.O/'action_cell_accounting.json',dict(plan=plan,rows=rows,summary=summary,conserved_all30_pair_differences=True))
    print(d.json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
