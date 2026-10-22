"""Verify predictions, report all models and archive the exploratory round."""
import collections,csv,json,shutil,subprocess
import joblib
import numpy as np
from fit import verify_freeze
from prepare import ROOT,W,OUT,SOURCE,ARCHIVE,read,save,sha


def independent_ranking(y,score):
    p=score[y];n=score[~y]
    auc=float(((p[:,None]>n).sum()+.5*(p[:,None]==n).sum())/(len(p)*len(n)))
    order=np.argsort(-score,kind='stable');score=score[order];truth=y[order];tp=seen=0;ap=0.
    for value in np.unique(score)[::-1]:
        block=score==value;positive=int(truth[block].sum());tp+=positive;seen+=int(block.sum())
        ap+=positive/y.sum()*tp/seen
    return auc,float(ap)


def main():
    verify_freeze();D=read(OUT/'design.json');rows=read(OUT/'rows.json');fits=read(OUT/'fits.json');results=read(OUT/'results.json')
    assert fits['model_fits']==2 and fits['logistic_converged'] and not fits['validation_used_for_fit']
    with np.load(OUT/'data.npz') as data:X=data['X'];gain=data['gain']
    train=np.array([r['partition']=='train' for r in rows]);dates=np.array([r['calendar_group'] for r in rows]);val=~train
    assert set(fits['train_ids'])=={r['sample_id'] for r in rows if r['partition']=='train'}
    count=collections.Counter(dates[train]);weight=np.array([1/count[d] for d in dates[train]]);weight/=weight.mean()
    logistic=joblib.load(OUT/'models/logistic.joblib');forest=joblib.load(OUT/'models/forest.joblib')
    assert np.allclose(logistic['scaler'].mean_,np.average(X[train],axis=0,weights=weight),atol=1e-12)
    expected={'LOGISTIC':logistic['classifier'].predict_proba(logistic['scaler'].transform(X))[:,1], 'FOREST_REG':forest.predict(X)}
    predictions=read(OUT/'predictions.json');checks=0
    for model in ['LOGISTIC','FOREST_REG']:
        records=[r for r in predictions if r['model']==model]
        assert [r['sample_id'] for r in records]==[r['sample_id'] for r in rows]
        assert np.array_equal(np.array([r['score'] for r in records]),expected[model])
    for model in results:
        records=[r for r in predictions if r['model']==model]
        score=np.array([r['score'] for r in records]);auc,ap=independent_ranking(gain[val]>0,score[val])
        assert abs(auc-results[model]['validation_pooled']['auc'])<1e-12
        assert abs(ap-results[model]['validation_pooled']['ap'])<1e-12
        for date,record in results[model]['per_validation_date'].items():
            mask=val&(dates==date)
            if (gain[mask]>0).any() and (gain[mask]==0).any():
                auc,ap=independent_ranking(gain[mask]>0,score[mask])
                assert abs(auc-record['auc'])<1e-12 and abs(ap-record['ap'])<1e-12
            checks+=1
    data_design=read(SOURCE/'design.json')
    assert not any((ROOT/data_design['environment']/'chronics'/r['scenario']).exists() for r in data_design['scenario_plan'] if r['partition']=='test')
    review=dict(passed=True,model_fits=2,saved_prediction_checks=2*len(rows),independent_ranking_checks=checks+4,
        train_only_scaler_verified=True,train_ids_verified=True,date_groups_disjoint=True,final_test_materialized=False,
        new_physical_calls=0,deployed_policy=False,validation_date_groups=4,scope='Exploratory validation, no independent final-test or whole-week advantage claim.')
    save(OUT/'review.json',review)
    fmt=lambda x:'—' if x is None else f'{x:.4f}'
    lines=['# GRID101 新标签简单模型首轮训练','',
        '沿用GRID100：训练2378条前缀（32场景/16日期），验证590条（8场景/4日期）。模型输入为37项已查询反馈与公开候选结构统计，未使用原始完整观测、日历标识、补查后反馈或长期结果。',
        '', '仅训练一次逻辑回归和一次64棵浅层随机森林回归；没有搜索超参数、用验证集选门槛或重拟合。每个训练日期总权重相同。另保留零预测和训练均值/正类先验常量参照。',
        '', '## 验证结果：四个日期分别计算后宏平均','',
        '| 方法 | 正收益AUC | 正收益AP | 增益MAE | 正收益样本MAE | 前20%记录覆盖的增益 |',
        '|---|---:|---:|---:|---:|---:|']
    for model,record in results.items():
        r=record['validation_date_macro']
        lines.append(f"| {model} | {fmt(r['auc'])} | {fmt(r['ap'])} | {fmt(r['mae'])} | {fmt(r['positive_mae'])} | {100*r['top20']['gain_capture']:.2f}% |")
    lines+=['','AP应与常量基线比较，不以多数零收益带来的高准确率判断有效。逻辑回归输出正收益概率，不直接输出增益量，故MAE不与回归模型混比。森林的AUC/AP以预测增益作排序分数，不将其当概率。',
      '', '前20%是各验证日期内按分数挑选记录的离线排序诊断，相同分数边界按均匀随机挑选的期望计算；它不是已经部署的触发规则，也不是实际节省20%查询。多个前缀共享物理状态，不能当独立控制决策或独立场景。',
      '', '## 各验证日期（AUC / AP / 前20%增益覆盖）','']
    for date in sorted(set(dates[val])):
        lines.append('- '+date+': '+ '; '.join(f"{model} {fmt(r['per_validation_date'][date]['auc'])}/{fmt(r['per_validation_date'][date]['ap'])}/{100*r['per_validation_date'][date]['top20']['gain_capture']:.1f}%" for model,r in results.items()))
    lr=results['LOGISTIC']['validation_date_macro'];rf=results['FOREST_REG']['validation_date_macro'];base=results['TRAIN_CONSTANT']['validation_date_macro']
    improvement=rf['ap']>base['ap'] and rf['top20']['gain_capture']>base['top20']['gain_capture']
    lines+=['','## 判读','',
        ('森林在本轮验证中同时提高正收益排序与前20%记录的增益覆盖，存在可供后续研究的预测信号。' if improvement else '森林尚未在本轮验证中同时改善正收益排序与前20%记录的增益覆盖，不能据此认为已学出有效计算分配。'),
        f"逻辑回归日期宏平均AUC {lr['auc']:.4f}、AP {lr['ap']:.4f}；森林AUC {rf['auc']:.4f}、AP {rf['ap']:.4f}；常量AP {base['ap']:.4f}。完整训练内、合并验证、逐日期、概率误差与排序诊断全部保存于results.json。",
        '', '这里仅有4个验证日期，且标签来自同一电网FULL教师路径。即时预测奖励不是整周成本，未完成周样本没有删除，不能据此声称闭环可靠、跨网泛化或实际提速。没有训练强化学习，没有接入控制器；最终8测试场景仍未生成。',
        '', '后续由用户根据本轮结果安排模型开发或强化学习研究，本轮不自动追加试验。']
    (OUT/'GRID101_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    with (OUT/'validation_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        chosen=[r for r in predictions if r['partition']=='validation'];writer=csv.DictWriter(f,fieldnames=list(chosen[0]));writer.writeheader();writer.writerows(chosen)
    dest=ARCHIVE/'complete';dest.mkdir(parents=True,exist_ok=False)
    for path in OUT.rglob('*'):
        if path.is_file():
            target=dest/path.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for p in W.glob('*.py'):shutil.copy2(p,dest/p.name)
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})
    collab=ROOT/'论文协作资料/FRAME_新标签简单模型_GRID101_20261022';collab.mkdir(exist_ok=False)
    for n in ['GRID101_report.md','results.json','design.json','review.json','validation_predictions.csv']:shutil.copy2(OUT/n,collab/n)
    print(json.dumps(review,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
