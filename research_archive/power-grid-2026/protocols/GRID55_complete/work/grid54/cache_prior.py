"""Cache input-only action metadata; recheck fresh vectors on every use.

Do not use action.to_vect(): the historical audit found stale action caches.
The dynamic PPO inference, graph-pressure tail, aliases, and arithmetic are
compiled from the unchanged original function with only metadata substitutions.
"""
import inspect, sys, time
import prior as prior_module
from common import np, action_vector, digest

ORIGINAL_PRIOR = prior_module.public_prior
source = inspect.getsource(ORIGINAL_PRIOR)
replacements = {
    'def public_prior(obs, env, ctl, ids, aliases):':
        'def dynamic_prior(obs, env, ctl, ids, aliases, metadata):',
    '    subs = [int(a.as_dict()["set_bus_vect"]["modif_subs_id"][0]) for a in library]':
        '    subs = metadata.subs',
    '    keys = {digest(action_vector(a)):i for i,a in enumerate(library)}':
        '    keys = metadata.keys',
    '    mapped = {i:keys[key] for i,a in enumerate(neural.gym_env.action_space.topo_actions_list) if (key:=digest(action_vector(a))) in keys}':
        '    mapped = metadata.mapped',
}
for a,b in replacements.items():
    assert source.count(a) == 1, a
    source = source.replace(a,b)
namespace = dict(vars(prior_module))
exec(compile(source, '<grid54-original-prior-with-static-metadata>', 'exec'), namespace)
DYNAMIC_PRIOR = namespace['dynamic_prior']

class PriorMetadata:
    def __init__(self, ctl):
        self.builds = 0; self.validations = 0; self.build_s = 0.; self.validation_s = 0.
        self.rebuild(ctl)

    def rebuild(self, ctl, vectors=None):
        started = time.perf_counter()
        library = ctl.base.topo_n1_unsafe.topo_act_list
        neural = ctl.base.topo_12_unsafe
        actions = neural.gym_env.action_space.topo_actions_list
        self.identity = (id(library), id(neural), id(actions),
                         tuple(id(a) for a in library), tuple(id(a) for a in actions))
        if vectors is None:
            vectors = ([action_vector(a) for a in library], [action_vector(a) for a in actions])
        self.library_vectors = vectors[0]; self.neural_vectors = vectors[1]
        self.subs = [int(a.as_dict()['set_bus_vect']['modif_subs_id'][0]) for a in library]
        # Original duplicate handling is last-key-wins; preserve it exactly.
        self.keys = {digest(v):i for i,v in enumerate(self.library_vectors)}
        self.mapped = {i:self.keys[key] for i,v in enumerate(self.neural_vectors)
                       if (key:=digest(v)) in self.keys}
        self.builds += 1; self.build_s += time.perf_counter()-started

    def validate(self, ctl):
        started = time.perf_counter()
        library = ctl.base.topo_n1_unsafe.topo_act_list
        neural = ctl.base.topo_12_unsafe
        actions = neural.gym_env.action_space.topo_actions_list
        identity = (id(library), id(neural), id(actions),
                    tuple(id(a) for a in library), tuple(id(a) for a in actions))
        vectors = ([action_vector(a) for a in library], [action_vector(a) for a in actions])
        same = identity == self.identity and \
            all(np.array_equal(a,b, equal_nan=True) for a,b in zip(vectors[0],self.library_vectors)) and \
            all(np.array_equal(a,b, equal_nan=True) for a,b in zip(vectors[1],self.neural_vectors))
        self.validations += 1; self.validation_s += time.perf_counter()-started
        if not same:
            self.rebuild(ctl, vectors)

    def stats(self):
        return {k:getattr(self,k) for k in ['builds','validations','build_s','validation_s']}

def public_prior_cached(obs, env, ctl, ids, aliases):
    metadata = getattr(ctl, '_grid54_prior_metadata', None)
    if metadata is None:
        metadata = PriorMetadata(ctl); ctl._grid54_prior_metadata = metadata
    else:
        metadata.validate(ctl)
    return DYNAMIC_PRIOR(obs, env, ctl, ids, aliases, metadata)

def install_prior_cache():
    """Apply the same equivalent function to all loaded original-prior users."""
    patched = []
    for name,module in list(sys.modules.items()):
        if module is not None and getattr(module, 'public_prior', None) is ORIGINAL_PRIOR:
            module.public_prior = public_prior_cached; patched.append(name)
    assert 'prior' in patched and 'guard_adapter' in patched, patched
    return sorted(patched)
