"""Read native recorded outcomes; no environment, model or counterfactual calls."""
from pathlib import Path
import collections,gzip,pickle,struct,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R=C.R;O=C.O;P=C.P;read=C.read
def reward(v):return struct.unpack('f',struct.pack('f',v['reward']))[0]
def best(rows):return max((v for v in rows if v['strict_admissible']),key=lambda v:(reward(v),-v['pool_id']))
def main():
    C.verify();assert not (O/'native_feedback_audit.json').exists();a=read(O/'analysis.json');top=[v['group'] for v in a['loss_concentration']['all_groups'] if v['gain']<0][:3];assert len(a['loss_concentration']['all_groups'])==222
    C.save('native_detail_plan.json',dict(posthoc=True,source_code_sha256=C.sha(Path(__file__)),scope='Replay recorded candidate selection only for52fixed focus starts and all action-changing starts in top3lossgroups; independently verify prefix selection/gain/public next IDs; zero new simulation or model forwards. All222group decomposition retained.'))
    examples=[v for v in read(P/'paired_action_audit.json')['all_changed_action_examples'] if v['physical_state_group'] in top];ids={v['sample_id'] for v in read(O/'unique_divergence_states.json')}|{v['sample_id'] for v in examples}
    meta={v['sample_id']:v for v in read(R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples']};cases={v['sample_id']:v for v in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if v['sample_id'] in ids};subs=read(R/'outputs/grid28/offline.json')['public_subids'];public={}
    with gzip.open(R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            v=C.json.loads(line)
            if v['sample_id'] in ids:public[v['sample_id'],v['position']]=v['public_input']
    cache={};checks=collections.Counter();receipts=[]
    for sid,c in cases.items():
        m=meta[sid];path=R/m['source_call']
        if m['source_call'] not in cache:
            assert C.sha(path)==m['source_sha256'];cache[m['source_call']]=read(path)
        call=cache[m['source_call']];truth={v['pool_id']:v for v in call['outcomes']};n=m['visible_count'];visible=[truth[i] for i in call['order'][:n]];inc=best(visible);station=subs[inc['pool_id']];target={int(v) for k,v in call['aliases'].items() if subs[int(k)]==station};pending=[i for i in call['order'][n:] if i in target];chosen=[inc['pool_id']];gains=[]
        for j in range(len(c['states'])):
            chunk=pending[j*64:(j+1)*64];pub=public[sid,j];assert pub['next_ids']==chunk;assert pub['incumbent_id']==best(visible)['pool_id'];before=reward(best(visible));visible += [truth[i] for i in chunk];after=best(visible);gains.append(reward(after)-before);chosen.append(after['pool_id']);checks['native_batches']+=1
        assert chosen==c['chosen'] and np.array_equal(gains,c['gains']);checks['native_starts']+=1
        receipts.append(dict(sample_id=sid,group=c['physical_state_group'],source_call=m['source_call'],source_sha256=m['source_sha256'],batches=len(gains)))
    by={}
    for v in read(P/'oof_trajectories.json'):by.setdefault(v['method'],{})[v['sample_id']]=v
    d=read(P/'design.json');details=[]
    for e in examples:
        sid=e['sample_id'];s=e['seed'];c=cases[sid];old=by[f'NET_seed{s}'][sid];new=by[f'CHANGE_seed{s}'][sid];j=min(old['batches'],new['batches']);pub=public[sid,j]
        details.append(dict(**e,position=j,next_queries=len(pub['next_ids']),final_in_next64=int(c['chosen'][-1]) in pub['next_ids'],next_gain=float(c['gains'][j]),next_NET=float((c['gains'][j]-d['cost_per_query']*c['queries'][j])/d['gain_scale']),suffix_gain=float(c['gains'][j:].sum()),old_batches=old['batches'],new_batches=new['batches']))
    C.save('top3_loss_details.json',dict(top3_groups=top,all_action_changing_pairs=details,description='Ranked by signed per-group date-balanced gain delta across all methods/seeds/starts; top3 share is of total negative group contributions, not net loss.'))
    C.save('native_feedback_audit.json',dict(passed=True,checks=dict(checks),native_source_calls=len(cache),source_receipts=receipts,zero_environment_construction=True,zero_model_forwards=True,zero_new_predictions=True));C.verify();C.archive('native_detail');print('GRID188_NATIVE_DETAILS',C.json.dumps(dict(checks=dict(checks),top3_groups=top,top3_pairs=len(details),source_calls=len(cache))))
if __name__=='__main__':main()
