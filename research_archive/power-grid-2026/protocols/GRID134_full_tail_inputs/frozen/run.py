"""Qualify complete public pending plans on train/calibration only; zero fits."""
from pathlib import Path
import collections,copy,datetime,gzip,hashlib,json,shutil,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究'); O=R/'outputs/grid134_full_tail_inputs'; W=Path(__file__).parent
REPO=R/'work/GRL-results-integration'; A=REPO/'research_archive/power-grid-2026/protocols/GRID134_full_tail_inputs'
sys.path.insert(0,str(R/'work/grid111_action_feedback'))
from public import Decoder,best,signature
sys.path.insert(0,str(R/'work/grid112_relational_plan'))
from schema import pack
sys.path.insert(0,str(R/'work/grid107_direct_value'))
from common107 import sealed
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x): (O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def archive(phase):
    d=A/phase; d.mkdir(parents=True,exist_ok=False)
    for p in [*O.iterdir(),Path(__file__)]:
        if p.is_file() and p.suffix!='.gz': shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8')
def verify():
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items(): assert sha(R/p)==h,p
    sealed()
def prepare():
    assert not O.exists(); O.mkdir(parents=True); sealed(); assert read(R/'outputs/grid133_nonlinear_gate/additional_review.json')['passed']
    # Select train records before opening per-call files. No validation call is decoded.
    meta=read(R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples']
    with np.load(R/'outputs/grid129_baseline_risk/train_public_inputs.npz') as z: ids=set(z['sample_ids'].tolist())
    selected=[m for m in meta if m['sample_id'] in ids]; assert len(selected)==2378
    paths=[R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json',R/'outputs/grid129_baseline_risk/train_public_inputs.npz',R/'outputs/grid130_calibrated_gate/train_calibration_scores.npz',R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz',R/'outputs/grid111_action_feedback/analysis_receipt.json',R/'outputs/grid133_nonlinear_gate/information_audit.json']
    for name in sorted({m['source_call'] for m in selected}):
        p=R/name; paths += [p,p.with_name(p.stem+'_state.npz'),p.parent/'metadata.json']
    paths += [R/'work/grid111_action_feedback/public.py',R/'work/grid111_action_feedback/common111.py',R/'work/grid112_relational_plan/schema.py',
        R/'outputs/grid03/scoring_smoke/agent_episode/dict_action_space.json',R/'outputs/grid03/scoring_smoke/agent_episode/dict_observation_space.json',
        R/'outputs/grid28/offline.json',R/'work/grid02/ljn_nn/assets/action_N1_unsafe.npz',R/'outputs/grid98_feature_enrichment/template_metadata.json']
    save('design.json',dict(id='GRID134_FULL_TAIL_INPUTS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_fits=0,
        purpose='Input qualification only, not a claim that full-tail representation will improve learning.',
        scope='All2378 originaltrain starting prefixes, original12fit/4cal split; no validation call reconstruction and final8sealed.',
        source='Saved current forecast observation/base action plus public action library; no environment/reset/step/simulate.',
        public_contract='All already-paid local feedback at starting gate, same locked incumbent station, ALL remaining aliased representatives in original order. No unbought feedback, gains, chosen future action, dates or ids as numeric predictors.',
        qualifications=['first64 equal prior GRID111 public input','full count equals known pending count','visible ids originalprefix only','all pending absent from visible','candidate alias/order fixed','poison future outcomes has no effect on context','reject future fields','bus-label invariance of all encoded partitions','paired context permutation equivariance and pair-multiset identity'],
        caveat='Source qualification reads old full teacher records to verify commands/hashes; numerical feature construction only receives known feedback/public plans. Public local connection intent is not future solved topology/AC flow.',
        prior='GRID98/99/110 structural summaries and GRID113/116/119 relational fits did not establish stable improvement. Full-tail input is a scope correction for one-shot full-completion target, not novelty or performance proof.',
        prohibited=['fit','new labels','validation reconstruction','threshold tuning','new physical call','data expansion','finaltest','paper/figures']))
    paths+=[O/'design.json']; save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths})
    save('code_freeze.json',{Path(__file__).relative_to(R).as_posix():sha(Path(__file__))}); save('state.json',dict(stage='frozen',fits=0)); archive('frozen'); print('GRID134_FROZEN')
def run():
    verify(); assert not (O/'qualification_started.json').exists(); save('qualification_started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with np.load(R/'outputs/grid129_baseline_risk/train_public_inputs.npz') as z: ids=z['sample_ids'].tolist(); X=z['X']; Q=z['known_queries']
    with np.load(R/'outputs/grid130_calibrated_gate/train_calibration_scores.npz') as z: fm=z['fitmask']; cm=z['calmask']
    lookup={i:j for j,i in enumerate(ids)}; old={}
    with gzip.open(R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['sample_id'] in lookup and r['position']==0: old[r['sample_id']]=r['public_input']
    assert len(old)==2378
    meta=[m for m in read(R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples'] if m['sample_id'] in lookup]
    bycall=collections.defaultdict(list)
    for m in meta: bycall[m['source_call']].append(m)
    dec=Decoder(); checks=collections.Counter(); counts={k:collections.Counter() for k in ['fit','calibration']}; index=[]; started=time.perf_counter()
    with gzip.open(O/'train_calibration_public_inputs.jsonl.gz','wt',encoding='utf8') as out:
        for number,(path,ms) in enumerate(bycall.items()):
            decoded=dec.decode(path); call=decoded[0]; truth={r['pool_id']:r for r in call['outcomes']}; aliases=decoded[-1]
            for m in ms:
                sid=m['sample_id']; j=lookup[sid]; n=m['visible_count']; visible=[truth[i] for i in call['order'][:n]]
                station=dec.subs[best(visible)['pool_id']]; target={v for k,v in aliases.items() if dec.subs[k]==station}
                pending=[i for i in call['order'][n:] if i in target]; assert len(pending)==int(Q[j]) and not set(pending)&{r['pool_id'] for r in visible}
                c=dec.context(decoded,visible,pending,station); prev=old[sid]
                assert c['seen_ids']==prev['seen_ids'] and c['seen_plans']==prev['seen_plans'] and c['seen_feedback']==prev['seen_feedback']
                assert c['next_ids'][:64]==prev['next_ids'] and c['next_plans'][:64]==prev['next_plans']
                assert all(c[k]==prev[k] for k in ['incumbent_id','incumbent_reward','incumbent_plan'])
                poisoned=copy.copy(call); poisoned['outcomes']=[{'forbidden_future':'poisoned'}]
                assert dec.context((poisoned,*decoded[1:]),visible,pending,station)==c; checks['future_outcomes_poison_invariance']+=1
                # Same public encoder can retain the full pending set without new weights.
                a=pack(c,X[j],0.,1.); assert a['next_action'].shape==(len(pending),68)
                try: pack(dict(c,future_feedback=[1]),X[j],0.,1.)
                except AssertionError: checks['future_field_rejections']+=1
                else: raise AssertionError('Future input accepted')
                swapped=copy.deepcopy(c)
                for key in ['seen_plans','next_plans']: swapped[key]=[[3-v if v>0 else v for v in p] for p in c[key]]
                swapped['incumbent_plan']=[3-v if v>0 else v for v in c['incumbent_plan']]
                b=pack(swapped,X[j],0.,1.); assert all(np.array_equal(a[k],b[k]) for k in a); checks['bus_label_invariance']+=1
                rev=copy.deepcopy(c)
                for key in ['seen_ids','seen_plans','seen_feedback']: rev[key]=list(reversed(c[key]))
                b=pack(rev,X[j],0.,1.)
                assert np.array_equal(a['context_action'][::-1],b['context_action']) and np.array_equal(a['context_feedback'][::-1],b['context_feedback'])
                assert all(np.array_equal(a[k],b[k]) for k in ['next_action','anchor_action','global41','known_query_cost']); checks['paired_context_permutation_equivariance']+=1
                # Padding width is inherited from public grid metadata, not labels.
                for plan in c['next_plans']+c['seen_plans']:
                    assert len(plan)<=17; checks['public_plan_rows']+=1
                group='fit' if fm[j] else 'calibration'; assert fm[j] or cm[j]
                counts[group]['prefixes']+=1; counts[group]['pending_candidates']+=len(pending); counts[group]['beyond_first_batch']+=max(0,len(pending)-64)
                counts[group]['max_pending']=max(counts[group]['max_pending'],len(pending)); counts[group]['expensive_prefixes']+=int(len(pending)>64)
                encoded=json.dumps(dict(sample_id=sid,public_input=c),ensure_ascii=False,allow_nan=False)
                out.write(encoded+'\n'); index.append(dict(sample_id=sid,split=group,source_call=path,public_sha256=hashlib.sha256(encoded.encode()).hexdigest(),pending=len(pending),seen=len(c['seen_ids'])))
                checks['cases']+=1
            if number%60==0: print('qualified',number+1,'/',len(bycall),'calls',checks['cases'],'prefixes',flush=True)
    assert checks['cases']==2378 and len(bycall)==222 and counts['fit']['prefixes']==1637 and counts['calibration']['prefixes']==741
    # Roundtrip every record and contract; exported inputs never include teacher targets.
    with gzip.open(O/'train_calibration_public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line,m in zip(f,index):
            d=json.loads(line); assert set(d)=={'sample_id','public_input'}; assert d['sample_id']==m['sample_id']
            assert hashlib.sha256(line.rstrip('\n').encode()).hexdigest()==m['public_sha256']; checks['serialized_roundtrips']+=1
    assert checks['serialized_roundtrips']==2378
    save('input_index.json',index); save('dataset_receipt.json',dict(path='train_calibration_public_inputs.jsonl.gz',sha256=sha(O/'train_calibration_public_inputs.jsonl.gz'),bytes=(O/'train_calibration_public_inputs.jsonl.gz').stat().st_size,records=2378))
    verify(); review=dict(passed=True,checks=dict(checks),cohorts={k:dict(v) for k,v in counts.items()},physical_calls=222,public_command_checks=dict(dec.checks),
        model_fits=0,model_inferences=0,new_labels=0,new_physical_predictions=0,validation_call_decodes=0,final_tests_absent=True,source_hashes_unchanged=True,elapsed_s=time.perf_counter()-started)
    save('review.json',review)
    lines=['# GRID134：完整剩余候选输入资格核查','',
        '已还原全部2378条原训练前缀的完整剩余同站接线计划，其中1637条拟合、741条校准；覆盖222个物理搜索调用。没有训练模型、生成新标签、调用仿真或还原验证调用。','',
        '## 核查结果','']
    for group,v in counts.items(): lines.append(f"- {group}：{v['prefixes']}条前缀，{v['expensive_prefixes']}条需多批，剩余候选累计{v['pending_candidates']}，首批后累计{v['beyond_first_batch']}，单条最多{v['max_pending']}个。")
    lines+=['',
        '每条记录首64个候选、初始已购反馈和当前最佳动作均与旧GRID111逐项一致；尾部保持原候选顺序和别名合并。完整输入沿用原68维公开接线编码，补齐候选范围，不增加网络参数。',
        '逐条通过未来反馈字段拒绝、毒化原未用结果不改变公开输入、母线编号互换不变及已购配对行置换等变检查。最后一项是输入编码的性质，不是尚未训练模型的不变性保证。',
        '导出文件只含关联sample_id与公开输入。样本ID仅用于关联，不作为数值预测特征；日期/划分只在独立索引中。标签不写入该输入文件。','',
        '## 能据此推进什么','',
        '可以在不增加物理数据的条件下研究整段补查的公开候选结构。未来如训练，需同样的整段目标与划分，匹配配对/去配对容量，并分开比较首批与完整尾部，才能判断是哪种变化起作用。',
        '这次只证明输入可以完整、可复核地构建，未证明它能预测补查收益。历史结构统计、关系网络和固定主干校正未建立稳定优势的结果全部保留；不直接追加训练或扩大规模。',
        '还原的是动作的站内连接意图，不是执行后的潮流或未来安全状态。相同物理调用的多个前缀相关，不把2378当作独立场景；最终8测试仍未生成。']
    (O/'GRID134_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save('state.json',dict(stage='complete',fits=0)); save('complete.json',dict(passed=True,inputs_qualified=True,predictive_value_established=False,ready_to_scale=False)); archive('complete'); print(json.dumps(review,ensure_ascii=False))
if __name__=='__main__': {'prepare':prepare,'run':run}[sys.argv[1]]()
