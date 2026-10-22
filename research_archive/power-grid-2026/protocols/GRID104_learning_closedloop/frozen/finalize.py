"""Report every fixed arm and repeat; no model selection or automatic next experiment."""
import csv,shutil
import numpy as np
from registry import ROOT,OUT,read,save

def main():
    review=read(OUT/'review.json');assert review['passed_saved_data']
    D=read(OUT/'design.json');decision=read(OUT/'decision.json');rows=review['descriptive_runs']
    lines=['# GRID104：学习补查的开发场景闭环比较','',
      '固定使用四个已暴露验证日期的第一份生成时序：2037-03-02、05-04、08-03、10-19，各scenario_id100。它们来自同一公开电网，不是新独立测试集。六臂各两次交错反向计时，共48条串行回放；没有新训练、场景生成、模型选择或参数调整。',
      '', '六臂共享原FRAME全局前缀128、批次64、停滞2、阈值0.9、候选顺序/别名、初始补查站及公共控制器。FRAME查完原同站候选；LOCAL_BATCH1仅补查第一批；LOCAL_RF按下一批收益预测和原固定查询费用决定；LOCAL_DQN0_0/1/2为三份固定当前批DQN最终权重。特征投影和单状态推理完整计入控制时间。载入和一次空输入预热只计入初始化，所有费用记录保留。',
      '', '旧FULL教师仅用于存活、完整周成本和时域参照，未混入本轮计时。以下整周成本只在双方完整结束时比较；早停成本不能当作节省。计时重复、模型种子与同日期时序不是新的独立场景。',
      '', '## 本轮实际结果','',
      '| 方法 | 完整周/4 | 四场景平均累计成本* | 控制秒数均值 | 候选查询均值 | 本站补查均值 | 全步骤P95秒 | 搜索步骤P95秒 | 特征/推理秒数均值 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    methods=[m['method'] for m in D['methods']]
    for method in methods:
        rs=[r for r in rows if r['method']==method];r0=[r for r in rs if r['repeat']==0]
        mean=lambda key:float(np.mean([r[key] for r in rs if r[key] is not None])) if any(r[key] is not None for r in rs) else None
        searchp=mean('search_p95')
        lines.append(f"| {method} | {sum(r['complete'] for r in r0)}/4 | {mean('cost'):.3f} | {mean('controller_s'):.3f} | {mean('candidate_queries'):.1f} | {mean('local_queries'):.1f} | {mean('all_p95'):.6f} | {searchp:.6f} | {mean('local_feature_s'):.4f}/{mean('local_inference_s'):.4f} |")
    lines+=['','*累计成本列包含早停场景，仅作记录，不可直接排序为经济性优劣。质量与效率以逐场景、相同时域比较判读；方法间日期宏平均减少率见下表。P95是先逐次计算再平均，不是将所有步骤混合后计算。',
      '', '## 学习策略相对两个对照','',
      '| 学习策略 | 对照 | 质量达标 | 时域全部相同 | 控制时间减少 | 活跃等时域候选查询减少 | 更快日期 | 扩大条件 |',
      '|---|---|---|---|---:|---:|---:|---|']
    for method,gates in decision['gates'].items():
        for ref,g in gates.items():
            q=g['macro_active_query_reduction']
            lines.append(f"| {method} | {ref} | {g['quality_all4']} | {g['same_horizon_all4']} | {g['macro_controller_reduction']:.2%} | {q:.2%} | {g['faster_dates']}/4 | {g['pass_gate']} |")
    lines+=['','控制时间为整个控制器一次决策的墙钟时间，不是仅模型推理或候选模拟。若时域不相同，上述原始减少率只作描述，不具同质量提速资格；匹配时域控制时间另存decision.json。',
      '', '## 逐场景质量与成本','',
      '| 场景 | 方法 | 步数 | 完整结束 | 累计成本 | 相对FRAME整周成本 | 相对BATCH1整周成本 |',
      '|---|---|---:|---|---:|---:|---:|']
    for selected in D['selected']:
        week=selected['scenario']
        for method in methods:
            r=next(r for r in rows if r['repeat']==0 and r['week']==week and r['method']==method)
            b=next(r for r in rows if r['repeat']==0 and r['week']==week and r['method']=='FRAME_GUARD')
            c=next(r for r in rows if r['repeat']==0 and r['week']==week and r['method']=='LOCAL_BATCH1')
            cost=lambda base:f"{r['cost']/base['cost']-1:+.3%}" if r['complete'] and base['complete'] else '不可比（至少一方早停）'
            lines.append(f"| {week} | {method} | {r['steps']} | {r['complete']} | {r['cost']:.3f} | {cost(b)} | {cost(c)} |")
    lines+=['','## 冻结判读','',
       '达到预设开发扩大条件，可向用户提出下一阶段规模方案；尚未授权扩大，本流水线不会继续生成、训练或回放。' if decision['ready_to_scale'] else
       '未达到预设扩大条件。保留质量/时间/查询权衡及所有种子，不能用本轮结果宣布学习模块全面优于便宜固定策略；不自动扩规模或追加实验救结果。',
       '', f"RF扩大资格：{decision['rf_ready']}；三个DQN作为整体的资格：{decision['dqn_family_ready']}。不挑选单个最好种子。",
       '', '预设条件：相对FRAME和固定一批，全部四场景质量达标，计算比较时域一致；实际控制时间与匹配活跃查询宏平均均减少至少5%，至少3/4日期更快，两次计时均同向。参考完整结束时，质量是完整周成本不超过参考100.5%；参考早停时仅要求存活不短于参考，不作完整周经济性结论。DQN整体另要求全部种子质量/时域通过、至少2/3种子计算通过及整体均值通过。',
       '', '## 机制与开销','',
       'review.json保留每次回放的连续优化、全部预测、搜索开销和剩余控制时间互斥分解；学习特征与推理属于搜索开销的子项，不重复加总。还保存P99、只在触发搜索步骤计算的P95/P99、特征标准化裁剪次数、实际付费反馈及每批决策。可用它们区分“少查了但推理吃掉节省”和“即时增益没有转化为闭环质量”。',
       '', '重复轨迹要求逐步动作/观测/成本一致。保存向量哈希、合法反馈、候选别名/组合动作、公共上下文、模型输入/动作、最优已观察选择和逐步成本算术完成核对；没有独立AC重求解。审计只初始化一次模板环境，不发预测查询。',
       '', f"审计计数：{review['checks']}；确定性重复对数：{review['deterministic_repeat_pairs']}。",
       '', '最终8测试场景保持未生成。本轮是已有分区上的开发比较；不声称跨电网泛化、正式测试成功或已部署。扩大前由用户审阅统一报告。']
    (OUT/'GRID104_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    with (OUT/'run_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    collab=ROOT/'论文协作资料/FRAME_学习补查闭环_GRID104_20261022';collab.mkdir(exist_ok=False)
    for name in ['GRID104_report.md','review.json','decision.json','design.json','run_metrics.csv']:shutil.copy2(OUT/name,collab/name)
    print('Report and collaboration copy written; ready_to_scale='+str(decision['ready_to_scale']),flush=True)
if __name__=='__main__':main()
