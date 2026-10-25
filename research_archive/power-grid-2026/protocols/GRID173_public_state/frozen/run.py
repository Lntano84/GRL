"""Qualify current public electrical-state features on all original training dates."""
from pathlib import Path
import collections,copy,gzip,hashlib,json,shutil,sys
import numpy as np
from feature_core import compute,independent,NAMES
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid173_public_state';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID173_public_state'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def main():
    assert not O.exists();O.mkdir(parents=True)
    assert read(R/'outputs/grid172_stop_opportunity/complete.json')['passed_audit']
    save('design.json',dict(id='GRID173_PUBLIC_ELECTRICAL_STATE',authorization='User continue building recurring small savings; read-only current input qualification before training.',
        question='Current41 summaries and68actiontokens omit full public terminal power and incident-line stress. Is a fixed eight-scalar block available and reproducible BEFORE another query?',
        population='All3582states/2378starts of16originaltrainingdates only. No original4dev newfeatures,final8,simulation or fitting.',
        fields=NAMES,first4='Reuse GRID98 physical descriptors,not invented as new mechanisms. currentmaxrho often duplicates rho_limit; report that explicitly.',
        power='Perterminal signed incomingMW:gen+,load-,line_or/ex-,storage-. Currentdisconnectedterminal=>0. Staticpositions identify terminals; dates/IDs not features.',
        imbalance='For any known intended two-bus plan, sum_b |sum_{terminal on b} current incomingMW|/(1+sum |current incomingMW|). Fixed-current-flow descriptive proxy,NOT a solved future powerflow or guaranteed reward predictor. Base continuous changes and redistribution not recomputed.',
        feature_block='All8 fixed jointly before association/fitting; no feature selection on15losses. Threephysicalstress descriptors add possible information;rawmax may duplicateoldlimit;remaining4 capture scale andknownaction/currentpower relation.',
        qualification='Serialized Grid2Op spaces only,environment construction/reset/step/simulate forbidden. Check everyrawvector/hash,manualvectoroffsets,independentarithmetic,bus/terminal/permutation invariance,future-key rejection.',
        future_use='If qualified,one paired cheap-helper POWER8 versus ZERO8 comparison,oldFACTOR/routed models andone-sidedrule unchanged. Training notice before fits;no current claim improves predictions.',
        official_docs='https://grid2op.readthedocs.io/en/latest/user/observation.html',prohibited=['newforecasts/replays','newfits/modelpredictions','featuretargetranking','newdata/final8','paper']))
    archive('frozen')
    sys.path.insert(0,str(R/'work/grid111_action_feedback'));from public import Decoder,digest
    decoder=Decoder();meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');source_meta={m['sample_id']:m for m in read(R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples'] if m['partition']=='train'}
    contexts={}
    with gzip.open(R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if row['partition']=='train':contexts[row['sample_id'],row['position']]=row['public_input']
    assert len(contexts)==len(meta)==3582
    static=decoder.ac;positions={name:np.asarray(getattr(static,name),int) for name in ['gen_pos_topo_vect','load_pos_topo_vect','line_or_pos_topo_vect','line_ex_pos_topo_vect','storage_pos_topo_vect']}
    assert np.array_equal(np.sort(np.concatenate(list(positions.values()))),np.arange(540))
    obs_by_call={};paths=set();checks=collections.Counter();schema=None;source=[]
    for m in source_meta.values():
        rel=m['source_call']
        if rel in obs_by_call:continue
        path=R/rel;sp=path.with_name(path.stem+'_state.npz');call=read(path)
        assert sha(path)==m['source_sha256'] and call['state_hash']==m['state_hash']
        z=np.load(sp);ob=decoder.oc.from_vect(z['observation']);assert np.array_equal(ob.to_vect(),z['observation']) and digest(ob.to_vect())==m['state_hash'];checks['state_hashes']+=1
        attrs=[];offset=0
        for name in type(ob).attr_list_vect:
            arr=ob._get_array_from_attr_name(name);n=arr.size;assert np.array_equal(arr.astype(np.float32),z['observation'][offset:offset+n]);attrs.append(dict(name=name,offset=offset,size=n));offset+=n
        assert offset==4460
        if schema is None:schema=attrs
        else:assert schema==attrs
        fields={name:np.asarray(getattr(ob,name)).copy() for name in ['gen_p','load_p','p_or','p_ex','storage_power','rho','line_status','topo_vect']}
        manual={a['name']:z['observation'][a['offset']:a['offset']+a['size']].copy() for a in attrs if a['name'] in fields}
        for name,value in fields.items():assert np.array_equal(value,manual[name]);checks['manual_field_slices']+=1
        power=np.zeros(540,float)
        for posname,value,sign in [('gen_pos_topo_vect','gen_p',1),('load_pos_topo_vect','load_p',-1),('line_or_pos_topo_vect','p_or',-1),('line_ex_pos_topo_vect','p_ex',-1),('storage_pos_topo_vect','storage_power',-1)]:power[positions[posname]]=sign*fields[value].astype(float)
        power[fields['topo_vect']<=0]=0
        obs_by_call[rel]=(fields,power,call['rho_limit'])
        paths.update([path,sp,path.parent/'metadata.json']);source.append(dict(source_call=rel,state_sha256=sha(sp),state_hash=m['state_hash'],step=m['step']))
    arrays=[];provenance=[];duplicate=0
    for i,m in enumerate(meta):
        sm=source_meta[m['sample_id']];ctx=contexts[m['sample_id'],m['position']];first=contexts[m['sample_id'],0];station=int(decoder.subs[first['incumbent_id']]);a,b=decoder.bounds[station:station+2];fields,power,limit=obs_by_call[sm['source_call']]
        assert len(ctx['anchor_plan'])==b-a if 'anchor_plan' in ctx else len(ctx['incumbent_plan'])==b-a
        c=dict(terminal_power=power[a:b],rho=fields['rho'],line_status=fields['line_status'],incident_lines=(static.line_or_to_subid==station)|(static.line_ex_to_subid==station),anchor_plan=ctx['incumbent_plan'],next_plans=ctx['next_plans'])
        v=compute(c);assert np.allclose(v,independent(c),atol=1e-6,rtol=1e-6);checks['independent_features']+=8
        swapped=dict(c,anchor_plan=[3-x if x>0 else x for x in c['anchor_plan']],next_plans=[[3-x if x>0 else x for x in plan] for plan in c['next_plans']]);assert np.array_equal(v,compute(swapped));checks['bus_swap']+=1
        perm=np.arange(len(c['terminal_power']))[::-1];changed=dict(c,terminal_power=c['terminal_power'][perm],anchor_plan=np.asarray(c['anchor_plan'])[perm],next_plans=[np.asarray(p)[perm] for p in c['next_plans']]);assert np.allclose(v,compute(changed),atol=1e-6,rtol=1e-6);checks['terminal_relabel']+=1
        rev=dict(c,next_plans=list(reversed(c['next_plans'])));assert np.allclose(v,compute(rev),atol=1e-6,rtol=1e-6);checks['next_permutation']+=1
        for key in ['future_outcomes','next_reward','date','whole_week_cost']:
            try:compute(dict(c,**{key:999}))
            except AssertionError:checks['forbidden_keys']+=1
            else:raise AssertionError('nonpublic field accepted')
        duplicate+=bool(abs(float(v[0])-limit)<1e-7);arrays.append(v);provenance.append(dict(sample_id=m['sample_id'],position=m['position'],date=m['date'],physical_state_group=m['physical_state_group'],source_call=sm['source_call'],station=station,next_queries=len(ctx['next_plans'])))
    X=np.stack(arrays);np.savez_compressed(O/'features.npz',physical8=X);save('metadata.json',provenance);save('vector_schema.json',schema);save('state_sources.json',source)
    assert len(obs_by_call)==222 and X.shape==(3582,8)
    paths.update([R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz',R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json',R/'outputs/grid28/offline.json',R/'outputs/grid03/scoring_smoke/agent_episode/dict_action_space.json',R/'outputs/grid03/scoring_smoke/agent_episode/dict_observation_space.json',R/'outputs/grid98_feature_enrichment/template_metadata.json',R/'work/grid111_action_feedback/public.py',R/'work/grid111_action_feedback/common111.py',O/'design.json',O/'features.npz',O/'metadata.json'])
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in sorted(paths)});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')})
    save('feature_ranges.json',[dict(name=name,minimum=float(X[:,j].min()),maximum=float(X[:,j].max()),distinct_values=len(np.unique(X[:,j]))) for j,name in enumerate(NAMES)])
    for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h
    sys.path.insert(0,str(R/'work/grid139_global_feasibility'));import importlib.util
    sp=importlib.util.spec_from_file_location('g173_sealed',R/'work/grid139_global_feasibility/run.py');safe=importlib.util.module_from_spec(sp);sp.loader.exec_module(safe);safe.sealed()
    save('review.json',dict(passed=True,states=3582,starts=2378,physical_calls=222,dates=16,checks=dict(checks),maxrho_duplicates_old_limit=duplicate,environment_constructions=0,simulate_calls=0,new_fits=0,new_predictions=0,final8_absent=True,validation_feature_rows=0))
    lines=['# GRID173：把查询前已有的电气状态补进输入','',
        '现有输入保留了已查候选的奖励和拓扑分区，却没有把当前设备有功功率、线路负载与候选分区联系起来。本轮没有训练模型，只把固定8项描述在全部16个原训练日期上打包并核验。','',
        '前4项沿用GRID98的当前负载描述，不能作为新发明：全网最大负载率、全网过载程度、当前补查站的最大接线负载率、该站占全网过载线路的比例。后4项描述当前站功率规模，以及在冻结当前功率的情况下，当前最佳与下一批已知分区的母线不平衡量。','',
        '这些量全部来自本次搜索开始时的已记录观测、公开设备位置及尚未查询候选的已知动作。它们不读取下一批仿真反馈，不重新计算潮流。母线不平衡量只是固定当前潮流下的动作描述，不能解释为该动作实际运行后的不平衡或好坏。','',
        f'3582个状态、2378个起点、222个物理调用均通过原向量/哈希和独立算术核验。全网最大负载率有{duplicate}/3582状态与旧rho_limit相同或近似相同，明确保留其冗余身份；真正新增信息不能按8项全部计算。', '',
        '所有状态通过母线标签互换、设备位置同步重排和下一批候选排列不变性核验，额外未来反馈、日期和整周成本字段均被拒绝。使用序列化空间解码，禁止创建环境、reset、step或simulate；没有额外物理预测、学习评分或拟合。原4个开发日期未提取新特征，最终8未生成。','',
        '下一步只改便宜帮助模型的这一块输入：POWER8与同容量ZERO8配对，原FACTOR、路由模型、停止规则、NET目标、费用及四折日期拆分都不改。8项作为一个整体检验，不按15条错例挑单项。是否改善，要由后续全部日期、种子与正反案例共同回答。']
    (O/'GRID173_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');save('complete.json',dict(passed_audit=True,new_fits=0,new_physics=0,features_qualified=True));archive('complete')
    collab=R/'论文协作资料/FRAME_公共电气输入核对_GRID173';collab.mkdir(exist_ok=False)
    for name in ['GRID173_report.md','design.json','review.json','feature_ranges.json','complete.json']:shutil.copy2(O/name,collab/name);assert sha(collab/name)==sha(O/name)
    msg=f'GRID173当前公开电气状态POWER8资格通过：3582状态/222调用/16日期，旧98前4物理描述复用，新增冻结当前功率与已知分区描述；maxrho有{duplicate}行重复旧limit不称新信息。全部向量/手算/标签互换/未来字段拒绝通过，无环境/预测/拟合/原4dev新特征/最终8。下一步仅便宜helper POWER8对ZERO8，同容量四折三种子，原F/routed/NET/单向规则保持，训练前通知。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID173公开电气输入资格\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        p.write_text(old,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID173便宜模型输入配对\n\n1. 公开电气POWER8打包通过，保留现有小改善，不重复换标签/损失。\n2. 训练前通知：仅cheaphelper增加8列，与ZERO8同容量/初始化/采样/目标/预算配对，四旧日期折三种子24拟合，原F/routed/单向规则不变；全权重推送后统一held评分。\n3. 同网旧数据开发，不自动新物理/数据/最终8，不修改论文；全部改善和收益损失保留。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID173查询前电气状态输入\r\n\r\n[报告](protocols/GRID173_public_state/complete/GRID173_report.md)；固定8项公开描述通过打包资格，尚未训练。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID173_COMPLETE',json.dumps(read(O/'review.json'),ensure_ascii=False))
if __name__=='__main__':main()
