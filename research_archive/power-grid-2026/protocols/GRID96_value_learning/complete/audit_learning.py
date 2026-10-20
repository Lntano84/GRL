"""Recalculate exported model scores, splits, weights and fixed-threshold errors."""
from pathlib import Path
from collections import Counter
import importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid96_value_learning'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    sp=importlib.util.spec_from_file_location('g96_export',W/'learn.py');L=importlib.util.module_from_spec(sp);sp.loader.exec_module(L)
    D=read(OUT/'design.json');rows={r['case_id']:r for r in read(OUT/'labels.json')};fits=read(OUT/'fits.json');pred=read(OUT/'predictions.json');names=D['learning']['features'];checks=Counter()
    for fit in fits:
        train=[rows[i] for i in fit['train_ids']];test=[rows[i] for i in fit['test_ids']]
        assert set(fit['train_ids']).isdisjoint(fit['test_ids']);assert all(r['label']!='UNRESOLVED' for r in train+test)
        if fit['scheme']=='calendar_loo':assert {r['calendar_date'] for r in train}.isdisjoint(r['calendar_date'] for r in test)
        else:assert all(r['cohort']=='old13' for r in train) and all(r['cohort'] in ['G1','G2'] for r in test)
        cnt=Counter(r['calendar_date'] for r in train);ww=np.asarray([1/cnt[r['calendar_date']] for r in train]);ww/=ww.mean();assert np.array_equal(ww,fit['weights'])
        x=np.asarray([[r['features'][n] for n in names] for r in train]);xt=np.asarray([[r['features'][n] for n in names] for r in test]);m=fit['export']
        if m['kind']=='LOGISTIC':
            mean=np.average(x,axis=0,weights=ww);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=ww));scale[scale<1e-12]=1
            assert np.array_equal(mean,m['mean']) and np.array_equal(scale,m['scale']);assert np.max(m['iterations'])<2000
        if m['kind']=='FOREST64':
            assert len(m['trees'])==64
            for tree in m['trees']:
                def depth(n):return 0 if tree['left'][n]==-1 else 1+max(depth(tree['left'][n]),depth(tree['right'][n]))
                assert depth(0)<=4
        expected=np.tile(m['prior'],(len(test),1)) if m['kind']=='TRAIN_PRIOR' else L.exported_predict(m,xt)
        pp=[r for r in pred if r['fit_id']==fit['id']];assert [r['case_id'] for r in pp]==fit['test_ids']
        assert np.allclose(expected,[r['probabilities'] for r in pp],atol=1e-12,rtol=0)
        for r,p in zip(pp,expected):assert r['label']==rows[r['case_id']]['label'] and r['pilot_known']==rows[r['case_id']]['pilot_known'] and r['prediction']==L.CLASSES[int(np.argmax(p))]
        checks['fits']+=1;checks['predictions']+=len(pp)
    results=read(OUT/'learning_results.json')
    for r in results:
        pp=[v for v in pred if v['scheme']==r['scheme'] and v['model']==r['model']]
        assert r['headline']==L.metrics([v for v in pp if not v['pilot_known']]);assert r['known_pilots_diagnostic']==L.metrics([v for v in pp if v['pilot_known']])
        checks['metric_groups']+=1
    review=read(OUT/'learning_review.json');review['independent_saved_export_audit']=dict(passed=True,checks=dict(checks));L.save('learning_review.json',review)
    print(json.dumps(review,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
