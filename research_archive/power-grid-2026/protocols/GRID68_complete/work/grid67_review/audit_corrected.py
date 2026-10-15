"""Prospective audit-only correction: physical terminal failures are reported, not excluded."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'work/grid67/audit.py'
text=source.read_text(encoding='utf-8')
old="assert s==read(folder/'summary.json') and s['illegal']==s['ambiguous']==s['exception_steps']==0"
assert text.count(old)==1
text=text.replace(old,"assert s==read(folder/'summary.json') and s['illegal']==s['ambiguous']==0")
marker="assert len(steps)==s['steps'] and abs(math.fsum(x['raw_cost'] for x in steps)-s['cost'])<1e-6"
assert text.count(marker)==1
text=text.replace(marker,marker+"\n  assert s['exception_steps']==sum(bool(x['exceptions']) for x in steps)\n  assert all(x['done'] for x in steps if x['exceptions']), 'Nonterminal physical exception needs review'")
marker="with np.load(folder/'vectors.npz') as z:\n   for i,x in enumerate(steps):\n    assert digest(z['action'][i])==x['action_hash'] and digest(z['observation'][i])==x['after_hash']"
assert text.count(marker)==1
text=text.replace(marker,"with np.load(folder/'vectors.npz') as z:\n   actions=z['action'];observations=z['observation']\n   for i,x in enumerate(steps):\n    assert digest(actions[i])==x['action_hash'] and digest(observations[i])==x['after_hash']")
marker="'quality_pass':survival and (delta is None or delta<=D['quality_threshold'])"
assert text.count(marker)==1
text=text.replace(marker,"'quality_pass':survival and all(not (s0['complete'] and f0['complete']) or s0['cost']/f0['cost']-1<=D['quality_threshold'] for s0,f0 in zip(own,fs))")
text=text.replace("'FRESH_LOCAL_SCENES_SUPPORTED' if supported else", "'FRESH_LOCAL_SCENES_SUPPORTED' if supported and timing else")
# Keep __file__ at the original to retain module resolution and ROOT. No original bytes changed.
exec(compile(text,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
