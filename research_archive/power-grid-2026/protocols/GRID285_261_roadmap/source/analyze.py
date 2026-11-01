from pathlib import Path
import json,hashlib,collections,statistics
R=Path("D:/Codex/GRL研究"); O=R/"outputs/grid285_261_roadmap"
def rd(p): return json.loads(p.read_text(encoding="utf8"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v): (O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
assert not (O/"analysis_started.json").exists()
for p,h in rd(O/"source_freeze.json").items(): assert sha(R/p)==h,p
save("analysis_started.json",{"scope":"saved ownfit rows only; no recomputed score or policy", "fits":0})
rows=rd(R/"outputs/grid263_training_residual/training_row_diagnostics.json")
old=rd(R/"outputs/grid263_training_residual/summary.json")
active=[r for r in rows if not r["KEEP"]]
assert len(active)==2725 and len(rows)==2859
out={}; callsout={}
for domain in ["STOP","CONT"]:
 rr=[r for r in active if r["domain"]==domain]
 bad=[r for r in rr if r["label"]==0 and r["continue_"]]
 pos=[r for r in rr if r["label"]==1]
 assert len(bad)==old["domain"][domain]["falseCONT_negative"]
 hist=collections.defaultdict(lambda:dict(rows=0,raw_queries=0,risk_fraction_pool_sum=0.,calls=set()))
 calls=collections.defaultdict(list)
 for r in bad:
  h=hist[r["next_queries"]];h["rows"]+=1;h["raw_queries"]+=r["next_queries"];h["risk_fraction_pool_sum"]+=r["risk_fraction_total"];h["calls"].add(r["call"]);calls[r["call"]].append(r)
 for h in hist.values():h["calls"]=len(h["calls"])
 callrows=[]
 for call,rs in sorted(calls.items()):
  callrows.append(dict(call=call,rows=len(rs),distinct_pool_ids=len({(r["pool"],r["metadata_id"]) for r in rs}),unique_metadata_ids=len({r["metadata_id"] for r in rs}),pools=sorted({r["pool"] for r in rs}),max_single_record_queries=max(r["next_queries"] for r in rs),repeated_record_query_sum=sum(r["next_queries"] for r in rs),dates=sorted({r["date"] for r in rs})))
 callsout[domain]=callrows
 masses=collections.defaultdict(float)
 for r in pos:masses[r["call"]]+=r["risk_fraction_total"]
 total=sum(masses.values());ordered=sorted(masses.values(),reverse=True)
 out[domain]=dict(rows=len(rr),positive_rows=len(pos),positive_calls=len(masses),positive_risk_top_call_fraction=ordered[0]/total,positive_risk_top5_call_fraction=sum(ordered[:5])/total,falseCONT_records=len(bad),falseCONT_calls=len(calls),full_query_count_histogram=dict(sorted(hist.items())),record_query_sum=sum(r["next_queries"] for r in bad),call_max_query_sum=sum(v["max_single_record_queries"] for v in callrows),risk_share_domain=sum(r["risk_fraction_total"] for r in bad)/sum(r["risk_fraction_total"] for r in rr),nonpositive_with_strictly_positive_NET=sum(r["actual_NET"]>0 for r in bad))
 assert out[domain]["record_query_sum"]==old["domain"][domain]["falseCONT_raw_queries"]
save("cost_support_summary.json",dict(domains=out,all_rows=len(rows),active_rows=len(active),KEEP_rows=134,new_fits=0,new_scores=0,new_routes=0,limits=["stored-context diagnostic, not deployment-context E; reuse278 for that distinction","call_max_query_sum is descriptive normalization, not executable savings or an oracle bound","cross-pool call reuse remains; unique calls are not independent unseen scenarios","no original arrays, outer inputs, new labels, predictions, model selection or thresholds read"]))
save("cost_support_calls.json",callsout)
save("complete.json",dict(passed=True,saved_counts_and_query_totals_match263=True,new_fits=0,new_scores=0,new_routes=0,all_original_rows_preserved=True))
print(json.dumps({d:{k:v for k,v in x.items() if k!="full_query_count_histogram"} for d,x in out.items()},ensure_ascii=False))
