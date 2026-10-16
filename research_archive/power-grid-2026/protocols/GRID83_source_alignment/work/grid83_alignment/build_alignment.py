"""Read-only source/decision attribution. No fitting or simulator calls."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json, statistics, subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid83_alignment'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def main():
    assert read(ROOT/'outputs/grid82/review.json')['passed_saved_data']
    source_rows, calls, summaries = [], [], []
    for seed in (0, 1):
        method = f'META_CONTEXT{seed}'
        for p in sorted((ROOT/'outputs/grid82/runs').glob(f'r0__*__{method}/call*.json')):
            x = read(p)
            events = x.get('meta_events', [])
            assert x['policy'] == method
            assert all(e['action'] in (0, 1, 2) for e in events)
            calls.append(dict(method=method, date=p.parent.name.split('__')[1],
                              call=p.stem, first_action=events[0]['action'] if events else None,
                              decision_count=len(events), stop_reason=x['stop_reason'],
                              source=p.relative_to(ROOT).as_posix(), sha256=sha(p)))
            source_rows.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)))
        own = [r for r in calls if r['method'] == method]
        assert own
        first = Counter(r['first_action'] for r in own)
        summaries.append(dict(method=method, unique_closed_loop_calls=len(own),
                              first_action_counts={str(k): v for k,v in first.items()},
                              immediate_guard_delegations=first[2],
                              immediate_guard_delegation_share=first[2]/len(own),
                              stop_reason_counts=dict(Counter(r['stop_reason'] for r in own))))
    assert [s['unique_closed_loop_calls'] for s in summaries] == [32, 30]
    assert summaries[1]['immediate_guard_delegations'] == 27
    save('seed_behavior.json', dict(scope='GRID82 repeat0 only; timing repeats not duplicated. Different closed-loop states, not paired independent cases.',
                                    action_names={'0':'RETURN','1':'GLOBAL64','2':'DELEGATE_FRAME_GUARD'},
                                    summaries=summaries, calls=calls, new_runs=0, new_fits=0))
    with (OUT/'seed_behavior_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.writer(f);w.writerow(['method','calls','immediate_guard','share','return_first','global_first'])
        for s in summaries:
            c=s['first_action_counts'];w.writerow([s['method'],s['unique_closed_loop_calls'],s['immediate_guard_delegations'],s['immediate_guard_delegation_share'],c.get('0',0),c.get('1',0)])
    r81=read(ROOT/'outputs/grid81/review.json'); dates=sorted({r['date'] for r in r81['groups']})
    pairs=[]
    for d in dates:
        a=next(r for r in r81['groups'] if r['method']=='FRAME_GUARD' and r['date']==d)
        b=next(r for r in r81['groups'] if r['method']=='FULL' and r['date']==d)
        ar=next(r for r in r81['rows'] if r['method']=='FRAME_GUARD' and r['date']==d)
        br=next(r for r in r81['rows'] if r['method']=='FULL' and r['date']==d)
        equal=ar['steps']==br['steps'] and ar['complete']==br['complete']
        pairs.append(dict(date=d, time_reduction=1-a['controller_s']/b['controller_s'],
                          query_reduction=1-a['queries']/b['queries'] if equal and b['queries'] else None))
    metrics=dict(stage='GRID81',method='FRAME_GUARD',baseline='FULL',
                 macro_controller_reduction=statistics.mean(p['time_reduction'] for p in pairs),
                 macro_query_reduction_equal_nonzero_horizons=statistics.mean(p['query_reduction'] for p in pairs if p['query_reduction'] is not None),
                 pairs=pairs)
    assert abs(metrics['macro_controller_reduction']-.1497914817)<1e-8
    save('existing_metrics.json',dict(guard_vs_full=metrics,grid82_decisions=read(ROOT/'outputs/grid82/decision.json'),
                                      selected_meta_configuration=read(ROOT/'outputs/grid77/selected.json')))
    source=ROOT/'work/grid64_baseline_sources/Metareasoning.jl'
    git='C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
    commit=subprocess.check_output([git,'-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert commit=='facb41ad6544b60895b1543e57c70a11dc280224'
    names=['outputs/grid82/review.json','outputs/grid82/decision.json','outputs/grid81/review.json',
           'outputs/grid77/selected.json','outputs/grid77/design.json','work/grid77/metacontrol.py',
           'work/grid77/train.py','work/grid70/policies.py',
           'outputs/grid83_alignment/sources/Bhatia_ICAPS2022.pdf',
           'outputs/grid83_alignment/reading/Bhatia_ICAPS2022.txt',
           'work/grid64_baseline_sources/Metareasoning.jl/aastar_dqn.jl',
           'work/grid64_baseline_sources/Metareasoning.jl/src/rl/dqn.jl']
    source_rows.extend(dict(path=n,sha256=sha(ROOT/n)) for n in names)
    save('source_receipt.json',dict(original_paper=dict(title='Tuning the Hyperparameters of Anytime Planning: A Metareasoning Approach with Deep Reinforcement Learning',
         doi='10.1609/icaps.v32i1.19842', pages='556-564', url='https://ojs.aaai.org/index.php/ICAPS/article/view/19842',
         pdf_url='https://ojs.aaai.org/index.php/ICAPS/article/download/19842/19601/23855',
         author_code='https://github.com/bhatiaabhinav/Metareasoning.jl', code_commit=commit),
         reading=dict(full_pdf_text=True,visually_inspected_pdf_pages=[4,7],
                      qualification='Source reading and saved-log analysis, not original-task reproduction or new simulator audit.'),
         sources=source_rows,new_training=0,new_simulation=0))
    print(json.dumps(dict(summaries=summaries,metrics=metrics),ensure_ascii=False))

if __name__=='__main__':main()
