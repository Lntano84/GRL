"""Freeze one development rule, no test-outcome driven selection or tuning."""
import hashlib,json,difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid05'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
old=load(ROOT/'outputs/grid04/design_freeze.json')
assert not (OUT/'design_freeze.json').exists()
reused=load(ROOT/'outputs/grid04/reused_assets_frozen.json')
for p,h in reused.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
for p in ['work/grid04/io_utils.py','outputs/grid04/per_run.json','outputs/grid04/paired_trajectories.json','outputs/grid04/pool_metadata.json','outputs/grid04/GRID04_audit.json']:
    reused[p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
plan=[]
for j,s in enumerate(old['selected_development_weeks']):
    for p in (['NN20','FALLBACK90'] if j%2==0 else ['FALLBACK90','NN20']):plan.append({'scenario':s,'policy':p})
design={'stage':'GRID05','environment_seed':0,'data_root':old['data_root'],
    'selected_development_weeks':old['selected_development_weeks'],'plan':plan,'policies':['NN20','FALLBACK90'],
    'rule':'Evaluate original NN top20. If its selected topology has recorded predicted max rho >= original rho_safe=0.9, or no candidate is returned, evaluate remaining332; merge cached original float32 rewards in NN352 full order. Keep original eligibility, null reward, tie order and downstream control.',
    'no_threshold_sweep':True,'initial_top_k':20,'expanded_top_k':352,
    'metrics_order':['native termination and full week completion','raw cost only on jointly completed paired weeks','actual candidate simulations and same-prior-prefix time; no production deadline claim'],
    'calibration':'New NN20 must reproduce all four GRID04 NN20 action/observation traces; otherwise document failure of exact calibration and do not claim a clean incremental effect.',
    'quality_screen':{'january_end_at_least':1719,'april_recovery_of_historical_NN352_cost_gap_at_least':0.8,
        'no_earlier_end_than_new_NN20_any_week':True,'maximum_relative_cost_regression_any_jointly_complete_week':0.02},
    'verdict':'Rule supported on this development set only if all quality screen items hold. Earlier termination or >2% complete-week cost regression makes the rule unsafe to promote here. Other results mixed/inconclusive; no threshold/model/seed tuning or automatic training.',
    'historical_NN352':'GRID04 quality reference only, not a concurrent wall-time or significance comparison.',
    'preflight':{'physical_steps':0,'maximum_native_simulates':800,'maximum_wall_s':180,'description':'Synthetic semantic checks plus forced expansion vs source full greedy on the first already exposed development initial observation; force switch absent from production calls.'},
    'caps':{'main_runs':8,'physical_steps':16200,'per_run_after_import_s':900,'total_runner_including_preflight_s':1800,'output_bytes':2147483648},
    'scope':'Post-GRID04 development rule, not independent confirmation; synthetic public grid; pretrained training IDs unknown; validation/test not run; no training or normalized competition score.'}
OUT.mkdir(parents=True,exist_ok=True)
payload=json.dumps(design,indent=2)
(OUT/'design_freeze.json').write_text(payload,encoding='utf-8')
(OUT/'design_freeze.sha256').write_text(hashlib.sha256((OUT/'design_freeze.json').read_bytes()).hexdigest())
(OUT/'reused_assets_frozen.json').write_text(json.dumps(reused,indent=2),encoding='utf-8')
oldsrc=(ROOT/'work/grid04/run_one.py').read_text();newsrc=(ROOT/'work/grid05/run_one.py').read_text()
(OUT/'worker_derivation.diff').write_text(''.join(difflib.unified_diff(oldsrc.splitlines(True),newsrc.splitlines(True),fromfile='GRID04 frozen worker',tofile='GRID05 worker')),encoding='utf-8')
(OUT/'GRID05_protocol.md').write_text('''# GRID05 固定回退检查

这是一项已看开发数据后的规则验证，不是独立确认。阈值直接取作者rho_safe=0.9，不扫参。没有合格Top-20动作或其选中拓扑的已执行预测rho>=0.9时才扩到同库352项；复用前20项的原始奖励，完整排序破并列。安全阈值只是作者控制器阈值，不是安全认证。

四个原开发周、环境种子0；8条交错轨迹。新NN20对照先校验与GRID04逐步相同；旧NN352只作质量参照，不能比较跨批生产延迟。恢复、重连、N-1搜索、连续优化、原生规则、成本适配器及评分范围不变。不增加候选池，不训练，不进入验证或测试。

先做零实际推进预检：模拟反馈/阈值/缓存/并列/基动作语义，以及在首个旧开发初始状态比较强制扩展与原生全扫描。预检最多800次一步模拟，180秒；强制开关仅用于该检查。实际运行不启用强制扩展。

主要检验：各周不比新NN20更早终止；1月结束步至少达到旧NN352的1719；4月完整成本至少恢复旧NN352相对NN20差距的80%；其他共同完整周成本回退不得超过2%。全部满足仅支持这条开发规则；任一更早终止或完整周>2%成本退化则不推广；其他记混合/不确定。不存在任何自动启动GRL的分支。

成本只比较共同完整周，不把提前终止算节省。一步拓扑预测与最终连续控制后的实际rho分别记录，不作同一动作残差解释。主运行包络含预检共1800秒、16200次实际推进，逐轨迹900秒，输出2GiB。错误/超限保留，不自动重跑或换周。审计另计时间，保存全部动作、反馈、扩展决策与原始模拟记录。
''',encoding='utf-8')
print(json.dumps({'runs':len(plan),'design_sha256':hashlib.sha256((OUT/'design_freeze.json').read_bytes()).hexdigest()}))
