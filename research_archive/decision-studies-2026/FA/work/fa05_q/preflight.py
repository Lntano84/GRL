"""Hand-computed and finite-world checks of the information semantics."""
import json
from pathlib import Path
from information import describe, proxy_possible, signed_difference, update_visible
from run import OUT, ROOT, load_json


def main():
    checks=[]
    def record(name, fn):
        fn(); checks.append(dict(name=name,passed=True))
    def expect(ka,va,kb,vb,pref,lower,upper):
        d=describe(ka,va,kb,vb)
        assert d['preference']==pref,d
        assert d['regret_lower']==lower and d['regret_upper']==upper,d
    record('two_completions_exact',lambda: expect(2,20,2,50,'a_better',30,30))
    record('single_censor_known_preference_unknown_cost',lambda: expect(2,20,1,100,'a_better',80,5980))
    record('reverse_pair',lambda: expect(1,100,2,20,'b_better',80,5980))
    def double_censor():
        d=describe(1,100,1,100)
        assert d['preference']=='unidentified'
        assert (d['signed_lower'],d['signed_upper'])==(-5900,5900)
    record('both_censored_preference_unidentified',double_censor)
    record('native_timeout_exact_score',lambda: expect(2,20,3,600,'a_better',5980,5980))
    record('two_native_timeouts_exact_tie',lambda: expect(3,600,3,600,'tie',0,0))
    record('completion_at_native_cutoff_is_completion',lambda: expect(2,600,3,600,'a_better',5400,5400))
    record('completion_at_acquisition_cutoff_vs_censor',lambda: expect(2,100,1,100,'a_better',0,5900))
    def boundary():
        d=describe(2,20,1,34.5)
        assert (d['regret_lower'],d['regret_upper'])==(14.5,5980)
        assert proxy_possible(d,325)
    record('budget_tail_not_assumed_100_seconds',boundary)
    def insufficient():
        assert describe(2,50,1,20)['preference']=='unidentified'
    record('small_lower_bound_does_not_resolve_preference',insufficient)
    def refine():
        old=describe(2,20,1,100); new=describe(2,20,1,200); complete=describe(2,20,2,201)
        assert new['regret_lower']>old['regret_lower'] and new['regret_upper']==old['regret_upper']
        assert complete['regret_exact'] and complete['regret_lower']==181
    record('later_feedback_can_refine_cost_without_new_preference',refine)
    record('lower_bound_preserves_maximum',lambda: expect_update())
    def gap():
        d=describe(2,20,1,100)
        assert not proxy_possible(d,980)
        assert proxy_possible(d,580) and proxy_possible(d,5980)
        assert not proxy_possible(d,80) # open censor endpoint
    record('native_regret_set_has_gap',gap)
    record('completed_tie_zero_regret',lambda: expect(2,20,2,20,'tie',0,0))
    def finite_worlds():
        # Independent true-score calculation; worlds are mathematical toys,
        # never rows of the hidden benchmark matrix.
        states=[(2,20),(2,100),(1,100),(1,34.5),(3,600),(0,0)]
        def outcomes(k,v):
            if k==2: return [(v,True)]
            if k==3: return [(1000,False)]
            l=v if k==1 else 0
            return [(l+0.25,True),(300,True),(600,True),(1000,False)]
        n=0
        for ka,va in states:
            for kb,vb in states:
                parts=signed_difference(ka,va,kb,vb)
                for ta,success_a in outcomes(ka,va):
                    for tb,success_b in outcomes(kb,vb):
                        # Reject any toy outcome inconsistent with the visible state.
                        if ka==1 and ta<=va or kb==1 and tb<=vb: continue
                        sa=ta if success_a and ta<=600 else 6000
                        sb=tb if success_b and tb<=600 else 6000
                        assert any(s.contains(sb-sa) for s in parts),(ka,va,kb,vb,sa,sb,parts)
                        n+=1
        # 15 compatible outcomes across the six visible-state fixtures,
        # hence 15*15 ordered two-algorithm worlds.
        assert n==225,n
    record('finite_compatible_worlds_contained',finite_worlds)
    def perturb():
        # Actual feedback generation from two complete toy worlds, followed by
        # the same update and information functions used by the audit.
        def observe(t,cap): return (2,t) if t<=cap else (1,cap)
        def audit(world):
            a=update_visible((0,0),*observe(world[0],100))
            b=update_visible((0,0),*observe(world[1],100))
            return describe(*a,*b)
        x=audit([20,200]); y=audit([20,1000])
        assert x==y and not x['regret_exact']
        assert audit([20,50])!=x # revealed changes are allowed to matter
    record('two_world_feedback_perturbation_real_path',perturb)
    def whitelist():
        for p in [ROOT/'outputs/fa00/FA00_data.npz',
                  ROOT/'outputs/fa04/runs/COARSE-PC-100_s7/predictions.npy',
                  ROOT/'outputs/fa02_end/models/FIXED-100_s7_ALL/selector.pkl']:
            try: load_json(p)
            except PermissionError: pass
            else: raise AssertionError(f'Forbidden input was accepted: {p}')
    record('semantic_input_whitelist_rejects_matrix_predictions_model',whitelist)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'FA05_preflight.json').write_text(json.dumps(dict(checks=checks,count=len(checks),
         passed=True,new_acquisition=0,new_fits=0),indent=2),encoding='utf8')
    print(json.dumps(dict(passed=True,checks=len(checks))))


def expect_update():
    state=update_visible((0,0),1,100)
    state=update_visible(state,1,20)
    assert state==(1,100)
    assert update_visible(state,2,101)==(2,101)


if __name__=='__main__': main()
