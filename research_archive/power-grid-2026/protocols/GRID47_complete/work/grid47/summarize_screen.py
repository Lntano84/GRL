"""Report the predeclared macro development screen without changing raw outcomes."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid47'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
R=read(OUT/'review.json');D=read(OUT/'design.json');assert R['passed_saved_data']
own=[]
for method in ['LOCAL_GNN0','LOCAL_GNN1']:
    c=next(c for c in R['comparisons'] if c['method']==method and c['opponent']=='FULL')
    q=[q for q in R['quality'] if q['method']==method]
    v=[p['time_reduction'] for p in c['pairs'] if p['time_reduction'] is not None]
    macro=math.fsum(v)/len(v);passed=all(qi['quality_qualified'] for qi in q) and macro>=.05 and sum(x>0 for x in v)>=3
    own.append({'method':method,'quality_pairs':sum(qi['quality_qualified'] for qi in q),'macro_control_reduction':macro,
       'pooled_control_reduction':c['pooled_control_reduction'],'positive_dates':sum(x>0 for x in v),'screen_passed':passed})
result={'own':own,'both_seeds_screen_passed':all(r['screen_passed'] for r in own),
        'interpretation':'Quality repaired on old development trajectories,whole speed below frozen5%screen. Keep as strict endpoint,not absolute research failure or independent confirmation. PPO has same query count and is slightly faster.',
        'new_model_fits':0,'new_GRID44_evaluations':0}
(OUT/'development_screen.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
p=OUT/'GRID47_report.md';p.write_text(p.read_text(encoding='utf-8')+'\n\n预定开发联合筛查：'+str(result['both_seeds_screen_passed'])+'。原两模型实际轨迹逐步等同FULL，控制汇总减少3.46%/3.53%，拓扑搜索时间减少22.32%/22.77%；宏平均未达5%。PPO查询相同且略快，不宣称严格保护下图模型增量已经建立。\n',encoding='utf-8')
print(json.dumps(result,indent=2))
