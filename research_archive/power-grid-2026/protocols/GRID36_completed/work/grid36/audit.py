"""Saved vector/cost/feedback audit, independent public-order reconstruction."""
import gzip,hashlib,json,math,os,struct,sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid36'
sys.path.insert(0,str(ROOT/'work/grid08'))
from common import np,make_env
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f]
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def score(r):return struct.unpack('f',struct.pack('f',r['reward']))[0],-r['pool_id']
F=read(OUT/'finished.json');assert F['passed_engineering'] and len(F['summaries'])==3
for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
baseline=next(s for s in read(ROOT/'outputs/grid34/finished.json')['summaries'] if s['repeat']==0 and s['week']=='2035-03-05_0' and s['method']=='FULL')
subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
# Environment initialization supplies public grid metadata/Observation vector schema.
# No environment step, action forecast or independent AC feasibility solve is used.
env=make_env();template=env.get_obs();report=[];calls_checked=physical=feedback=0
try:
    for s in F['summaries']:
        folder=OUT/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}";actual=rows(folder/'steps.jsonl.gz')
        assert len(actual)==s['steps'] and s['illegal']==0 and s['ambiguous']==0
        assert abs(math.fsum(r['raw_cost'] for r in actual)-s['cost'])<1e-6
        assert abs(math.fsum(r['decision_s'] for r in actual)-s['controller_s'])<1e-6
        assert sum(r['simulations'] for r in actual)==s['public_forecasts']
        with np.load(folder/'vectors.npz') as z:actions=z['action'];observations=z['observation']
        for i,r in enumerate(actual):
            assert digest(actions[i])==r['action_hash'] and digest(observations[i])==r['after_hash']
            if r['ledger'] is not None:assert abs(r['ledger']['recomputed_raw_cost']-r['raw_cost'])<=r['ledger_tolerance']
        calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]
        assert len(calls)==s['n1_calls'] and sum(c['public_queries'] for c in calls)==s['n1_forecasts']
        for i,c in enumerate(calls):
            with np.load(folder/f'call{i:03d}_state.npz') as z:vec=z['observation'].copy()
            obs=template.copy();obs.from_vect(vec);assert digest(obs.to_vect())==c['state_hash']
            ids=c['representatives'];alias={int(i):int(v) for i,v in c['aliases'].items()};live=np.flatnonzero(obs.line_status)
            pressure=[0.]*env.n_sub;adj=[set() for _ in range(env.n_sub)]
            for l in live:
                u,v=int(env.line_or_to_subid[l]),int(env.line_ex_to_subid[l]);p=float(obs.rho[l]);pressure[u]=max(pressure[u],p);pressure[v]=max(pressure[v],p);adj[u].add(v);adj[v].add(u)
            if s['method']=='STATIC_ID':expected=sorted(ids)
            elif s['method']=='LOCAL_STRESS':expected=sorted(ids,key=lambda i:(-pressure[subs[i]],i))
            else:
                assert s['method']=='PHYSICS_TAIL'
                top=sorted(live,key=lambda l:(-float(obs.rho[l]),int(l)))[:3]
                front={int(x) for l in top for x in [env.line_or_to_subid[l],env.line_ex_to_subid[l]]};known=set();dist=[env.n_sub+1]*env.n_sub;level=0
                while front:
                    for v in front:dist[v]=level
                    known.update(front);front={t for v in front for t in adj[v]}-known;level+=1
                raw=sorted(c['eligible'],key=lambda i:(dist[subs[i]],-pressure[subs[i]],i));expected=list(dict.fromkeys(alias[i] for i in raw))
            assert expected==c['order'] and c['prior_s']==0.
            assert len({r['pool_id'] for r in c['outcomes']})==c['public_queries']
            good=[]
            for r in c['outcomes']:
                valid=bool(0<r['rho']<c['rho_limit'] and not r['done'] and not r['exceptions'] and not r['illegal'] and not r['ambiguous'])
                assert valid==r['strict_admissible']
                if valid:good.append(r)
            assert (max(good,key=score)['pool_id'] if good else None)==c['chosen_pool_id'];calls_checked+=1;feedback+=c['public_queries']
        comparable=s['steps']==baseline['steps'];regret=(s['cost']-baseline['cost'])/max(1,abs(baseline['cost'])) if comparable else None
        qualified=s['complete'] and comparable and regret<=.005
        report.append({**s,'comparable_cost_regret':regret,'quality_qualified':bool(qualified)})
        physical+=s['steps']
finally:env.close()
R={'passed_saved_data':True,'runs':3,'physical_rows':physical,'order_checks':calls_checked,'feedback_checked':feedback,'rows':report,
   'new_fits':0,'new_action_forecasts':0,'reserved_evaluations':0,
   'warning':'Single exposed March quality pilot percheaprule. Setup environment only supplies vector schema/public metadata; no independent new AC feasibility verification. Cross-batch timers descriptive,not speed certification. Cheap ranking does not remove original author base-controller PPO branches.'}
(OUT/'review.json').write_text(json.dumps(R,indent=2),encoding='utf-8')
lines=['# GRID36 非学习排序闭环质量试点','','相同停止规则、相同共有编译复用、相同安全与目标计量，完整保存数据核验。', '',
    '| 顺序 | 完成/步数 | 成本差(FULL) | 质量资格 | N1查询 | 控制秒(描述) |','|---|---|---:|---|---:|---:|']
for r in report:
    value='不可比' if r['comparable_cost_regret'] is None else f"{100*r['comparable_cost_regret']:.5f}%"
    lines.append(f"| {r['method']} | {r['complete']}/{r['steps']} | {value} | {r['quality_qualified']} | {r['n1_forecasts']} | {r['controller_s']:.3f} |")
lines+=['',R['warning'],'','所有廉价规则结果保留；0.5%是投资筛查尺度，不是质量等价。通过只允许后续公平比较，不证明跨日期优势；不通过只阻断当前顺序/停止配置。']
(OUT/'GRID36_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('COMPLETED AND AUDITED:3cheapqualitypilots; readperrulegate before widening.\n',encoding='utf-8')
print(json.dumps({'passed_saved_data':True,'runs':3,'rows':[{k:r[k] for k in ['method','steps','complete','cost','comparable_cost_regret','quality_qualified','n1_forecasts','controller_s']} for r in report]},indent=2))
