"""Explicit, hash-bound META interface. No training or simulator entry point."""
from pathlib import Path
import functools
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'outputs/meta_unified_20261019/configuration.json'


def _module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@functools.lru_cache(maxsize=1)
def contract():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    for relative, expected in cfg['runtime_source_hashes'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    return cfg


@functools.lru_cache(maxsize=10)
def model(variant, seed):
    import torch
    cfg = contract()
    assert variant in cfg['variants'] and seed in cfg['seeds']
    entry = cfg['models'][f'{variant}_S{seed}']
    path = ROOT / entry['checkpoint']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], str(path)
    meta = core()
    result = meta.QModel()
    result.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    result.eval()
    return result


@functools.lru_cache(maxsize=1)
def core():
    contract()
    return _module('unified_meta_core', 'work/grid77/metacontrol.py')


def run(order, observe, subids, aliases, *, limit, context, variant, seed):
    """Run the frozen learned policy; variant selects weights, not a runtime cost knob.

    observe(pool_id, phase) is the caller's common paid simulation interface.
    No simulator, implicit selected.json lookup, or weight fitting occurs here.
    """
    assert len(order) == len(set(order))
    assert context is not None
    return core().run(order, observe, seed, subids, aliases, limit, context,
                      model=model(variant, seed))
