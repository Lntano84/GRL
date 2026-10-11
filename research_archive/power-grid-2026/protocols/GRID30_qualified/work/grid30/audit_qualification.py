"""Saved numerical vectors and independently reconstructed action conversions."""
import gzip,hashlib,json,math,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid30'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
sys.path[:0]=[str(ROOT/'work/grid30'),str(ROOT/'work/grid29'),str(ROOT/'work/grid08')]
from common import np,OptimModule,make_env,action_vector,write_json
P=json.loads((OUT/'qualification.json').read_text());assert P['passed']
for name,h in json.loads((OUT/'code_freeze.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
outputs=np.load(OUT/'safe_vectors.npz')['outputs']
assert outputs.shape[:2]==(153,3)
parameters=np.load(OUT/'pre_solve_parameters.npz')
assert all(parameters[k].shape[0]==153 for k in parameters.files)
folder=OUT/'runs/r0__2035-03-05_0__SAFE_QP_SHADOW'
baseline=ROOT/'outputs/grid29c/runs/r0__2035-03-05_0__ORIGINAL'
def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f]
new,old=rows(folder/'steps.jsonl.gz'),rows(baseline/'steps.jsonl.gz')
assert len(new)==len(old)==2017
for a,b in zip(new,old):
    for key in ['step','before_hash','action_hash','after_hash','raw_cost','simulations','done','complete']:
        assert a[key]==b[key],(a['step'],key)
    assert not a['illegal'] and not a['ambiguous'] and not a['exceptions']
with np.load(folder/'vectors.npz') as x,np.load(baseline/'vectors.npz') as y:
    for k in ['action','observation']:assert np.array_equal(x[k],y[k],equal_nan=True)
    observations=x['observation'].copy()
maximum=0.;actions=0
env=make_env();env.seed(0);env.set_id('2035-03-05_0');template=env.reset()
try:
    mod=OptimModule(env,env.action_space)
    assert outputs.shape[2]==env.n_line+3*mod.nb_max_bus
    for index,pair in enumerate(P['pairs']):
        assert pair['order']==([0,1] if index%2==0 else [1,0])
        reference=outputs[index,0]
        for slot,name in [(1,'ORIGINAL'),(2,'CACHED')]:
            assert np.array_equal(np.isfinite(reference),np.isfinite(outputs[index,slot]))
            valid=np.isfinite(reference)
            error=float(np.max(np.abs(reference[valid]-outputs[index,slot,valid]))) if valid.any() else 0.
            assert error<=1e-6 and error==pair['max_numerical_error_MW'][name]
            maximum=max(maximum,error)
        step=pair['step'];assert step>=1
        obs=template.copy();obs.from_vect(observations[step-1]);assert obs.current_step==step
        for name in parameters.files:
            p=getattr(mod,name);value=parameters[name][index]
            if value.shape==():p.value=value.item()
            else:p.value[...]=value
            assert np.array_equal(p.value,value)
        mod._storage_power_obs.value=0.
        converted=[]
        for slot in range(3):
            c,s,r=np.split(outputs[index,slot,env.n_line:],3)
            converted.append(action_vector(mod.to_grid2op(obs,c.copy(),s.copy(),r.copy(),base_action=env.action_space(),safe=True)))
        for slot in (1,2):
            assert np.array_equal(converted[0],converted[slot])
            actions+=1
finally:env.close()
times={name:math.fsum(p['shadow_s'][name] for p in P['pairs']) for name in ['ORIGINAL','CACHED']}
for name in times:assert abs(times[name]-P['shadow_times_s'][name])<1e-6
result={'passed':True,'manual_checks':P['manual_checks'],'live_numerical_triples':153,
    'reconstructed_action_equal_checks':actions,'max_numerical_error_MW':maximum,
    'same_delivered_physical_rows':2017,'shadow_times_s':times,
    'shadow_reduction':1-times['CACHED']/times['ORIGINAL'],
    'qualification_scope':'Saved numerical vectors, real pre-solve parameters, action reconstruction using author conversion; no independent new QP/AC solve.',
    'new_fits':0,'reserved_evaluations':0,
    'warning':'Instrumented original control included shadow work and cannot be used as whole-controller acceleration. Qualification covers one exposed March and8manual+1copy checks.'}
write_json(OUT/'audit.json',result)
(OUT/'GRID30_qualification_report.md').write_text(f'''# GRID30 安全QP参数复用资格

9项手工/复制检查通过。153次真实预求解参数与记忆下，原控制/隔离原QP/隔离缓存QP的最大输出差{maximum:g}MW；306次保存数据重建Grid2Op动作均相同。实际2017步动作/观测/成本/查询与原受保护控制完全相同。

影子原QP{times['ORIGINAL']:.3f}s、缓存QP{times['CACHED']:.3f}s，减少{100*result['shadow_reduction']:.2f}%。影子从未派发，含额外探针的整套计时不能用于提速。

V0影子setter拒绝作者gen_per_bus中的两处负值；V1只按原数组更新方式原样复制，未裁剪/投影。V0源码、协议、335步片段和失败均保留。

仍需完整实际派发的原/DC/DC+安全QP缓存比较。求解器、目标、约束、系数、资格、warm_start=False和控制节奏均保留。这是共享工程优化，不是GNN贡献。
''',encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('QUALIFICATION COMPLETED AND AUDITED:no safe-cache action deployed; whole-arm pilot still required.\n',encoding='utf-8')
print(json.dumps(result,indent=2))
