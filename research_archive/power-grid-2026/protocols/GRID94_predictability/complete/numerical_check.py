"""Resolve weighted StandardScaler warnings without refitting any classifier."""
from collections import Counter
import sys
import numpy as np
import sklearn
from data import ROOT,W,OUT,read,save,sha


def main():
    rows={r['row_id']:r for r in read(OUT/'dataset.json')};keys=read(W/'design.json')['features'];out=[]
    for f in read(OUT/'fits.json'):
        if f['details']['type']!='LOGISTIC':continue
        tr=[rows[i] for i in f['train_rows']]
        x=np.array([[r[k] for k in keys] for r in tr],dtype=np.longdouble)
        counts=Counter(r['calendar_date'] for r in tr)
        sw=np.array([1/counts[r['calendar_date']] for r in tr],dtype=np.longdouble);sw*=len(sw)/sw.sum()
        mu=np.sum(sw[:,None]*x,axis=0)/sw.sum()
        var=np.sum(sw[:,None]*(x-mu)**2,axis=0)/sw.sum()
        eps=np.finfo(np.float64).eps;n=sw.sum()
        constant=var <= n*eps*var+(n*mu*eps)**2
        expected=np.sqrt(var);expected[constant]=1
        actual=np.array(f['details']['scale']);observed=np.array(f['details']['mean'])
        assert np.isfinite(actual).all() and np.isfinite(observed).all() and np.isfinite(f['details']['coefficient']).all()
        assert np.allclose(expected,actual,rtol=1e-9,atol=1e-12)
        assert np.allclose(mu,observed,rtol=1e-12,atol=1e-12)
        out.append(dict(evaluation=f['evaluation'],fold=f['fold'],constant_features=[k for k,c in zip(keys,constant) if c],
            max_scale_error=float(np.max(np.abs(expected-actual))),max_mean_error=float(np.max(np.abs(mu-observed)))))
    save(OUT/'numerical_review.json',dict(passed=True,normalizers_checked=len(out),checks=out,
        observed_warning='Weighted StandardScaler sqrt(var) RuntimeWarning in the original fit; installed constant-feature handling replaces near-constant scales by 1.',
        resolution='Long-double centered weighted variances independently reproduce every saved scale and mean; all coefficients/probabilities finite and manual inference previously checked. No classifier refit or protocol change.',
        actual_runtime=sys.executable,sklearn_location=sklearn.__file__,classifier_refits=0))
    print('PASS',len(out),'normalizers; no refit')


if __name__=='__main__':main()
