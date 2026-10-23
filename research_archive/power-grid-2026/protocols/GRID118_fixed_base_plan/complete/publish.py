"""Archive a checked proposal, with no training or trained-result claim."""
from pathlib import Path
import json
import shutil
import sys
R = Path('D:/Codex/GRL研究')
W = Path(__file__).parent
O = R / 'outputs/grid118_fixed_base_plan'
REPO = R / 'work/GRL-results-integration'
sys.path.insert(0, str(R / 'work/grid113_relational_learning'))
from common import read, save, sha, verify, sealed
import numpy as np


def main():
    assert not (O / 'complete.json').exists()
    verify(); sealed()
    c = read(O / 'matching_check.json')
    p = read(O / 'plan.json')
    assert c['passed'] and c['new_fits'] == c['optimizer_instances'] == c['backward_calls'] == 0
    assert all(sha(R/n) == h for n,h in read(O / 'input_manifest.json').items())
    assert p['future_fits']['count'] == 9
    assert p['future_fits']['new_scalar_targets_total'] == 9*7250*32*2
    assert c['corrected_total_parameters'] == 6978+5938
    y1 = np.load(R / 'outputs/grid113_relational_learning/train_targets.npz')['target']
    y = np.column_stack([np.zeros_like(y1), y1])
    cache = np.load(O / 'training_target_and_exposure_check.npz')
    original_fits = read(R / 'outputs/grid107_direct_value/fits.json')['records']
    for seed in range(3):
        s = c['seeds'][str(seed)]
        assert s['base_final_hash'] == original_fits[seed]['final_q_hash']
        assert s['base_frozen_and_eval'] and s['initial_total_equals_base_exact']
        assert s['paired_initial_tensors_identical'] and s['correction_parameter_ids_only']
        assert np.array_equal(cache[f'correction_target_seed{seed}'], y-cache[f'base_q_seed{seed}'])
        assert int(cache[f'extra_stage_exposure_seed{seed}'].sum()) == 7250*32
        assert c['sampling_streams'][str(seed)]['proposed_extra_stage']['scalar_targets'] == 464000
    save(O / 'review.json', dict(passed=True, scope='Revised proposal engineering, not model effectiveness or execution qualification.',
        source_hashes_unchanged=True, original_three_trained_backbones_verified=True,
        correction_target_cache_rechecked=True, original_huber_equivalence_checked_to_float32_tolerance=True,
        freeze_flags_and_parameter_identity_checked=True, init_output_exact=True,
        new_continuation_streams_matched=True, extra_training_control_included=True,
        parameter_and_compute_boundaries_explicit=True,
        backward_calls=0, optimizer_instances=0, optimizer_steps=0, new_fits=0,
        validation_model_predictions=0, physical_calls=0, final_test_materialized=False,
        future_checks=['Actual trainer freeze and public-input isolation',
            'Synthetic backward plus gradient-flow check inside approved update budget']))
    report = '''# GRID118：固定基础模型，只学习校正量

本轮完成了修改方案和工程核验，没有开始训练。下一轮拟比较四组：原模型、原模型额外训练、固定原模型加配对校正、固定原模型加汇总校正。

## 为什么这样改

GRID116中，基础网络和校正模块从头一起训练。两部分可以互相补偿：基础输出加上某个偏移、模块减去相同偏移，总输出完全不变。因此，不能仅凭模块输出为正，就解释为它找到了值得继续补查的证据。

这次固定GRID107已训练好的基础模型，让它作为不变的参照，新增模块只校正这份预测。这样可以明确“校正的是谁的误差”。这是针对上一轮归因问题的改动，并不预先保证收益变好。上一轮多查的定位和反例继续保留。

## 模块具体学什么

基础模型输入原41维状态摘要，输出“停止”和“继续查询”的两项预测，记为 b。原监督目标仍为 y=[0,下一批即时净收益]，净收益已扣除原定查询费用并采用原收益尺度。固定 b 后，模块需要学习的校正目标就是 y−b。

例如，若基础模型预测继续查询的净收益为0.20，原标签为−0.05，模块对应的校正目标就是−0.25；若原标签为0.30，校正目标就是+0.10。这里的数值只是说明计算方式，不是实验结果。

最终预测仍为 b+δ，继续项大于停止项才继续查询，平局停止。模块接收既有公开连接信息、已查询反馈、下一批候选和状态摘要，不接收未来反馈。最后输出层从0开始，因此加上模块后，初始决策与原模型一致。

拟执行的训练仍直接计算原Huber损失 Huber(b+δ,y)，不更换损失、标签、费用或决策门槛。固定基础输出后，它在数学上等价于 Huber(δ,y−b)。此次已核对3582个训练状态、三个基础模型，浮点计算的逐项损失最大差为0.00000191。实际训练保留原总输出形式，避免引入不必要的数值差别。

这仍是即时补查净收益的预测。校正量不是整周收益，也不是某个输入特征的因果贡献。模块仍能使用41维摘要，因此也可能只学到普通摘要修正；是否利用配对信息，要看下面的同容量对照。

## 四组各自回答什么

1. **BASE_FIXED：原模型保持不动。** 直接复用GRID107三个种子的固定终点，不重复训练，作为当前水平的参照。
2. **BASE_EXTRA：原模型额外训练一轮。** 从同一份旧权重出发，再更新7250次。它检验新增模块的收益是否只来自额外训练。
3. **FIXED_JOINT：固定原模型，加配对校正。** 每个已查动作和它的反馈先结合，再供下一批候选使用。
4. **FIXED_FACTOR：固定原模型，加汇总校正。** 先汇总已查反馈，再与动作结合。它接收相同来源的信息，改变的是反馈聚合的位置。

JOINT和FACTOR均有12916个总参数，其中6978个原模型参数固定、5938个模块参数可训练；两组对应种子的模块初始张量逐项相同。BASE_EXTRA只有原来的6978个可训练参数。

JOINT与FACTOR是主要机制对照，容量、初始化、基础模型、数据、抽样和优化条件一致。与BASE_EXTRA比较时，匹配的是额外更新次数和监督目标数量，并非参数量、计算量或训练耗时。不能把这两种“匹配”混为一谈。

## 训练预算与数据

若执行，新增训练为BASE_EXTRA、FIXED_JOINT、FIXED_FACTOR各三个种子，共9次；原模型三个终点直接复用。每次新增7250次更新，每批32状态、64个标量目标，固定最终一步，不选择最好检查点。新增总计65250次更新、2088000次状态抽样和4176000个标量目标。

三组都从已经完成7250次训练的旧基础模型出发。原预训练是共同前提；新增模块的7250次训练属于额外预算，不能把它写成与原模型只训练一轮具有相同总训练投入。

所有新增组都使用新建Adam，不载入旧优化器动量。BASE_EXTRA应称为“从旧权重额外训练，并重置优化器”，不能当作原训练不中断地延长。学习率0.0003、betas=0.9/0.999、eps=1e−8、权重衰减0、梯度裁剪10、Huber beta=1和CPU单线程均不变。

各种子先复现原7250批抽样序列，再把同一个随机流的后续7250批用于新增三组。抽样概率不变，不增加正样本权重。这次已实际重算前后两段索引流，原段与旧哈希、计数一致，后段已记录独立哈希与计数。

仍使用32训练场景、8验证场景，对应16/4日期、3582/856个可达状态、2378/590条起始前缀。最终8个测试场景不生成、不读取标签。新增九次训练总墙钟上限1800秒，不自动延长，不重复完整训练项。全部训练结束后才统一计算验证结果，保留所有种子和不利结果。

## 怎样解释结果

主指标仍是完整补查序列的四日期宏平均净效用，同时报告实际收益、查询比例、正收益漏停、无利继续、物理调用均衡结果，以及逐日期、逐种子的结果。

- JOINT超过原模型，却没有超过BASE_EXTRA：不能排除额外训练的解释。
- JOINT和FACTOR都超过BASE_EXTRA，但两者接近：支持新增校正模块有用，不足以支持配对结构有额外价值。
- JOINT稳定超过FACTOR，并超过原模型及BASE_EXTRA：才支持这条配对校正路线继续推进。
- 只在个别种子改善：保留差异，不能挑最好种子作为结论。

配对机制信号沿用此前判据：JOINT家族均值高于FACTOR、三个配对种子差值均为正、种子平均后至少3/4日期同向。继续推进还需JOINT家族均值高于BASE_FIXED、BASE_EXTRA、固定一批BATCH1和原gamma0 DQN家族。它们是开发阶段的判读条件，不是统计显著性证明。

本次与GRID116还存在“先训练基础模型、再训练模块”的阶段差异，因此不能把未来相对GRID116的变化全部归因于冻结操作。当前先回答“相对固定模型的配对校正是否有用”，不再同时增加更多变体。

## 本次核验与剩余工作

已直接从原模型文件中读取在线Q网络权重，三个种子的权重SHA256均与原记录一致，没有实例化旧训练器或载入优化器。原输入、标签、抽样概率逐值一致；3582个训练输入上的基础输出与历史预测最大差小于0.000001，并用独立NumPy计算核对了前向结果。

已检查基础参数全部冻结、基础网络保持评估模式、待训练参数列表只包含校正模块；JOINT/FACTOR初始张量相同，实际输入上的初始总输出与基础模型逐值一致。原权重、数据和封存源码未改变。

本轮没有优化器实例、反向传播、参数更新、验证模型预测、新仿真或测试集生成。这里的“核验通过”仅指修改方案的工程核验。实际训练器尚未编写、封存或启动；训练前还需检查执行器的输入隔离、掩码和梯度流。零输出初始化使模块上游首步梯度为0，首步及后续梯度检查必须计入正式7250次预算，不额外热身训练。

本研究仍限于同一电网的已暴露验证日期和固定FULL教师路径。前缀相关、日期重复，以及即时收益与整周质量的区别继续保留。达到上述条件也不会自动扩大规模、上电网闭环或读取最终测试。
'''
    (O / '固定主干校正方案_中文.md').write_text(report, encoding='utf8')
    archive = REPO / 'research_archive/power-grid-2026/protocols/GRID118_fixed_base_plan/complete'
    collab = R / '论文协作资料/FRAME_固定主干校正方案_GRID118'
    archive.mkdir(parents=True, exist_ok=False)
    collab.mkdir(exist_ok=False)
    save(O / 'complete.json', dict(passed=True, scope='Proposal and engineering qualification only.',
        training_started=False, new_fits=0, planned_fits=9, ready_to_scale=False,
        report=str(O / '固定主干校正方案_中文.md'), archive=str(archive), collaboration=str(collab)))
    for path in O.iterdir():
        if path.is_file(): shutil.copy2(path, archive/path.name)
    for path in W.glob('*.py'): shutil.copy2(path, archive/path.name)
    for name in ['固定主干校正方案_中文.md','plan.json','matching_check.json','review.json','complete.json']:
        shutil.copy2(O/name, collab/name)
    save(archive / 'archive_manifest.json', {x.name:sha(x) for x in archive.iterdir() if x.is_file()})
    for x in O.iterdir():
        if x.is_file(): assert sha(x) == sha(archive/x.name)
    for x in collab.iterdir(): assert sha(x) == sha(O/x.name)
    note = '''## 2026-10-23 — GRID118 fixed-base correction proposal qualified, no training

User requests revision after GRID117. Freeze exact original trained GRID107 DIRECT seeds0/1/2, not jointly coadapted GRID116 bases. Define detached correction reference y-b while retaining original total-output Huber(b+delta,y). Proposed BASE_FIXED reuse plus BASE_EXTRA/FIXED_JOINT/FIXED_FACTOR three seeds each,9futurefits; each extra7250x32, original fee/loss/Adam/splits/final endpoint, fresh optimizer all extra arms. JOINT/FACTOR12916total,6978frozen,5938trainable; BASE_EXTRA6978trainable, so equal extra updates/targets not equal capacity/FLOPs/time. Original pretraining7250 counted separately. All3basefinalhashes exact; independent NumPy train forwards and historical outputs checked; residual targets/float32 Huber equivalence/freeze flags/parameter identity/zero initial outputs checked. Original index streams reproduced then common next7250 streams frozen. No optimizer/backward/fits/newvalidationprediction/physical/test. Proposed1800s total future cap, all9finish beforevalidation. Pair gate unchanged and progression also requires exceeding BASE_EXTRA. Fixes moving output reference, not guaranteed pairing use or quality; retains global41 and staged-training confound vsGRID116. Actual trainer/gradient/input qualifications still required before any fit. No automatic training/scaling.
'''
    for name in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/name).open('a', encoding='utf8') as f: f.write('\n\n'+note)
    (REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps — GRID118固定基础模型校正方案已核对，未训练

1. 将四组方案交用户检查：原BASE复用；BASE_EXTRA与固定BASE+JOINT/FACTOR各三个种子，拟9次新增训练。
2. 获准训练后编写、核验并封存实际执行器；保持原目标/Huber/费用/数据/种子，使用已核对的后续抽样序列，首步与梯度检查计入7250次。
3. JOINT/FACTOR配对判据不变；同时比较原BASE和同额外训练BASE_EXTRA。不自动改目标、扩网络、上电网闭环或读取最终测试。

方案：research_archive/power-grid-2026/protocols/GRID118_fixed_base_plan/complete/固定主干校正方案_中文.md
''', encoding='utf8')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a', encoding='utf8') as f:
        f.write('\n- [GRID118固定基础模型校正方案](protocols/GRID118_fixed_base_plan/complete/固定主干校正方案_中文.md)：固定旧预测参照，补充相同额外训练对照；工程核验完成，未训练。\n')
    print('GRID118_PROPOSAL_ARCHIVED_NO_TRAINING', flush=True)


if __name__ == '__main__':
    main()
