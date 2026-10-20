"""Build new, inspectable wrappers from immutable GRID95 source before freezing."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent


def main():
    source=ROOT/'work/grid95_labels'
    h=(source/'label_hooks.py').read_text(encoding='utf-8').replace("OUT=ROOT/'outputs/grid95_labels'","OUT=ROOT/'outputs/grid96_value_learning'")
    old="    # Unsupported engine/solver objects are inventoried, not claimed serialized."
    assert h.count(old)==1
    new="""    # Inherited but unused RNGs are retained outside the equality witness.
    unused_rngs={}
    for name in ['base_optimizer','restore_optimizer','reconnect','recover_topology','topology_search']:
        if 'space_prng' in data[name]['values']:
            unused_rngs[name]=data[name]['values'].pop('space_prng')
    # Unsupported engine/solver objects are inventoried, not claimed serialized."""
    h=h.replace(old,new).replace("inventory=data,global_rngs_record_only=global_rngs,","inventory=data,global_rngs_record_only=global_rngs,unused_module_rngs_record_only=unused_rngs,")
    assert not (W/'label_hooks.py').exists();(W/'label_hooks.py').write_text(h,encoding='utf-8')
    worker=(source/'worker.py').read_text(encoding='utf-8').replace("OUT=ROOT/'outputs/grid95_labels'","OUT=ROOT/'outputs/grid96_value_learning'")
    worker=worker.replace("\"OUT=ROOT/'outputs/grid95_labels'\"","\"OUT=ROOT/'outputs/grid96_value_learning'\"")
    marker="    exec(compile(worker_source,str(ROOT/'work/grid55/worker.py'),'exec'),worker.__dict__);worker.OUT=OUT"
    assert worker.count(marker)==1
    patch="    worker_source=worker_source.replace('\\\"torch_threads\\\": __import__', '\\\"nominal_episode_steps\\\": int(env.max_episode_duration()), \\\"torch_threads\\\": __import__')\n"
    worker=worker.replace(marker,patch+marker)
    (W/'worker.py').write_text(worker,encoding='utf-8')


if __name__=='__main__':main()
