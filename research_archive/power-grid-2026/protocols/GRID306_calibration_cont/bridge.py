"""Verify the whole physical prefix and flip one frozen original STOP."""
import math
import sys
from bindings import LEGACY, ledger, module, state, validate_event


class LegacyState(state.State):
    def force(self, raw, native, ordinary):
        if self.event is not None:
            return ordinary
        position = self.call['step'], self.call['call_index'], self.offer['batch']
        target = self.target['common']
        wanted = target['step'], target['call_index'], target['batch']
        if position < wanted:
            return ordinary
        q = native['native_q']
        assert len(q) == 2 and all(math.isfinite(x) for x in q) and q[1] <= q[0]
        common = {**self.call, **self.offer, 'prefix_signature':state.digest(self.prefix),
                  'prefix_cost':self.prefix_cost, 'raw41_hash':state.digest(raw), 'native_q':q,
                  'public_NN_input_hash':state.digest(native['NN_input'])}
        marker = {'common':common, 'ordinary261_continue':bool(ordinary['continue_'])}
        validate_event(marker, self.target)
        assert self.item['arm'] == 'CONT_NEXT'
        self.event = {**marker, 'arm':self.item['arm'], 'forced_continue':True, 'delivered':None}
        return {'continue_':True,'log':{**ordinary['log'],'paired_intervention':self.item['arm'],
                                       'fixed_legacy_marker':True}}


def install(Search, item, models, expected=None):
    assert expected is None
    prefix = ledger(item)[:item['target']['common']['step']-1]
    cls = LegacyState if item['year'] == 2037 else state.State
    replay = cls(item,prefix)
    sys.path.insert(0,str(LEGACY))
    adapter = module('grid306_original_adapter',LEGACY/'adapter.py')
    adapter.EpisodeIntervention = lambda arm,unused:replay
    assert adapter.install(Search,item,models,None) is replay
    return replay
