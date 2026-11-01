from pathlib import Path
import json,hashlib,ast
R=Path("D:/Codex/GRL研究");O=R/"outputs/grid286_cost_objective_contract"
def rd(p):return json.loads(p.read_text(encoding="utf8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
assert not (O/"execution_started.json").exists()
for p,h in rd(O/"source_freeze.json").items():assert sha(R/p)==h,p
save("execution_started.json",{"static_source_read_and_synthetic_algebra_only":True,"fits":0})
rules=[("work/grid112_relational_plan/schema.py",["cost_per_query*len(q)/gain_scale"]),("work/grid113_relational_learning/prepare.py",["['next_queries']=len(inp['next_action'])"]),("work/grid183_online_contract/runtime.py",["ids=pending[:64]","provider.context(rows,ids,station,targets)","action,details=self.models.decide", "for i in ids:","r=reveal(i,'local_completion')"]),("work/grid208_public_amount/run.py",["bool(pp*mm>q)","fee*c['queries'][j]/scale"]),("work/grid257_reference_labels/run.py",["net=gain/scale-fee*queries/scale","y=fn>0","risk=fm[mask]*abs(fn[mask])","weight[mask]=.5*risk/total"]),("work/grid219_prior_objective/run.py",["np.logaddexp(0,z)-y*z",".5*(theta@theta)"]),("work/grid261_bounded_basis/run.py",["STEP.choose_alpha","z['y'],z['weight'],z['b']"])]
evidence=[]
for rel,tokens in rules:
 text=(R/rel).read_text(encoding="utf-8-sig");ast.parse(text)
 for token in tokens:
  assert token in text,(rel,token)
  evidence.append(dict(path=rel,token=token,line=next(i for i,s in enumerate(text.splitlines(),1) if token in s),sha256=sha(R/rel)))
live=(R/"work/grid183_online_contract/runtime.py").read_text(encoding="utf8")
start=live.index("class CompletionPolicy:");live=live[start:]
assert live.index("ids=pending[:64]")<live.index("action,details=self.models.decide")<live.index("for i in ids:")<live.index("r=reveal(i,'local_completion')")
# Synthetic signs/regrets only, no real parameter fit/threshold or policy score.
for net in [-4.,-.25,0.,.25,4.]:
 for cont in [False,True]:
  utility=net*int(cont);oracle=max(net,0.)
  regret=abs(net)*int(cont!=(net>0))
  assert abs((oracle-utility)-regret)<1e-12
for nets,masses in [([-2.,1.],[1.,1.]),([-2.,4.],[.2,.8]),([0.,3.],[.5,.5])]:
 positive=sum(m*abs(n) for n,m in zip(nets,masses) if n>0);negative=sum(m*abs(n) for n,m in zip(nets,masses) if n<=0)
 assert abs(positive-negative-sum(m*n for n,m in zip(nets,masses)))<1e-12
q=rd(R/"outputs/grid257_reference_labels/qualification.json");assert q['passed'] and len(q['pools'])==12
risk=[dict(pool=p['pool'],KEEP=p['fallback_KEEP'],STOP_risk=p['risk_STOP'],CONT_risk=p['risk_CONT']) for p in q['pools']]
save("source_evidence.json",evidence);save("published_domain_risks.json",risk)
save("qualification.json",dict(passed=True,known_charge_before_forecast=True,known_charge_unit="original fee * planned next candidate count / original gain scale; not elapsed time",future_gain_unavailable=True,parent208_already_subtracts_charge=True,weighted_classification_already_encodes_signed_NET=True,domain_renormalization_changes_pooled_target=True,regularized_anchored_finite_model_not_Bayes_optimum=True,new_features_or_target_information_established=False,admit_new_training=False,newfits=0,new_model_scores=0,new_outerlabels=0,new_routes=0))
print("GRID286 static contract and synthetic regret identities passed; duplicate-target training rejected")
