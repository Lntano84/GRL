"""Finite serial generation/teacher/extraction pipeline; no training or test use."""
import argparse
import collections
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from bulk_common import ROOT, W, OUT, RUNTIME, items, read, save, verify, verify_dataset, test_sealed, utc


def state(stage, **extra):
    save(OUT / 'bulk_state.json', dict(stage=stage, pid=os.getpid(), utc=utc(), model_fits=0,
                                     final_test_materialized=False, **extra))


def run_child(command, logfile, timeout, stage, generation=False, **extra):
    with logfile.open('x', encoding='utf8') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        state(stage, child_pid=child.pid, **extra)
        started = time.monotonic()
        watched, item_start = None, started
        try:
            while child.poll() is None:
                assert time.monotonic() - started < timeout, f'{stage} wall budget exceeded'
                if generation:
                    progress = read(OUT / 'generation_progress.json')
                    if progress.get('mode') == 'trainval' and progress.get('phase') == 'generating':
                        scenario = progress['item']['scenario']
                        if scenario != watched:
                            watched, item_start = scenario, time.monotonic()
                        assert time.monotonic() - item_start < 1920, f'Generation watchdog: {scenario}'
                time.sleep(2)
            assert child.returncode == 0, f'{stage} child exit {child.returncode}; preserved log: {logfile}'
        except BaseException:
            if child.poll() is None:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], stdout=stream, stderr=subprocess.STDOUT)
                child.wait(timeout=30)
            raise


def receipts():
    completed = []
    for item in items():
        path = OUT / 'dataset_receipts' / (item['scenario'] + '.json')
        if path.exists():
            record = read(path); verify_dataset(record); completed.append(record)
    return completed


def refresh_manifest(completed):
    summaries = [{k: v for k, v in r.items() if k != 'files'} for r in completed]
    save(OUT / 'bulk_manifest.json', dict(completed=len(summaries), planned=40, scenarios=summaries))


def summarize():
    completed = receipts(); assert len(completed) == 40
    all_samples, ids, roles = [], set(), {}
    by_partition = {}
    for role in ['train', 'validation']:
        chosen = [r for r in completed if r['partition'] == role]
        counts = collections.Counter()
        for r in chosen:
            counts.update(r['counts'])
        by_partition[role] = dict(scenarios=len(chosen), date_groups=len({r['calendar_group'] for r in chosen}),
             samples=sum(r['samples'] for r in chosen), excluded_prefixes=sum(r['excluded_prefixes'] for r in chosen),
             completed_weeks=sum(r['teacher_complete'] for r in chosen),
             physical_steps=sum(r['teacher_steps'] for r in chosen), counts=dict(counts))
    for r in completed:
        manifest = verify_dataset(r)
        roles.setdefault(r['calendar_group'], set()).add(r['partition'])
        for sample in manifest['samples']:
            assert sample['sample_id'] not in ids
            ids.add(sample['sample_id'])
            all_samples.append(dict(sample, dataset_path=r['dataset_path']))
    assert len(roles) == 20 and all(len(x) == 1 for x in roles.values())
    assert by_partition['train']['scenarios'] == 32 and by_partition['validation']['scenarios'] == 8
    test_sealed()
    save(OUT / 'bulk_dataset_manifest.json', dict(samples=all_samples, partitions=by_partition,
         samples_are_correlated_prefixes=True, calendar_group_roles={k: next(iter(v)) for k, v in roles.items()},
         final_test_materialized=False, model_fits=0))
    save(OUT / 'bulk_review.json', dict(passed=True, scenarios=40, partitions=by_partition,
         checks=['all saved source and dataset hashes', 'action/observation chain', 'paid feedback admissibility',
                 'float32 best reward selection and station aliases', 'input/target separation',
                 'local reward gain recomputation', 'scenario and date split coverage', 'test remains unmaterialized'],
         reused_pilot=1, new_teacher_episodes=39, model_fits=0, final_test_materialized=False, utc=utc(),
         scope='Saved-data engineering audit, not independent AC re-solving or a model evaluation.'))
    train, val = by_partition['train'], by_partition['validation']
    report = f'''# GRID100 学习数据第一阶段\n\n本批40个场景完成采集与数据核验，包含32训练场景、8验证场景，共20个日期组；首个试点已复用。最终测试8场景尚未生成，未拟合模型。\n\n训练部分：{train['samples']}条相关搜索前缀，{train['counts'].get('positive_gain',0)}条补查即时正收益，{train['counts'].get('zero_gain',0)}条零收益；完整周{train['completed_weeks']}/32。\n\n验证部分：{val['samples']}条相关搜索前缀，{val['counts'].get('positive_gain',0)}条即时正收益，{val['counts'].get('zero_gain',0)}条零收益；完整周{val['completed_weeks']}/8。未完成周、无搜索与无可用补查前缀均保留，不替换日期或种子。\n\n标签回答“在当前状态，查完已选动作所属变电站的剩余候选，最佳可用奖励能增加多少”。查询数在决策前已知，作为成本输入；这不是整周成本收益标签。原始反馈、动作与观测向量、输入和目标分别保存，逐场景清单见bulk_manifest.json，样本索引见bulk_dataset_manifest.json。\n\n同一物理状态的多个前缀、同一天的两个生成重复都保持同一分区。样本数量不是独立场景数量。数据来自同一公开电网的FULL教师路径；后续仍需检验FRAME状态分布和学习策略闭环效果。\n\n训练输入必须遵守training_input_contract.json，原始观测包含日期字段，不能将整个向量直接喂给模型并声称已排除日期。当前无新模型训练、准确率结论或计算优势结论。\n'''
    (OUT / 'GRID100_report.md').write_text(report, encoding='utf8')


def main(check_only=False):
    verify(read(OUT / 'bulk_code_freeze.json')); test_sealed()
    assert read(OUT / 'bulk_preflight.json')['passed']
    completed = receipts(); refresh_manifest(completed)
    if check_only:
        print(json.dumps(dict(passed=True, existing=len(completed), planned=40,
                              pilot_reused=True, final_test_materialized=False, model_fits=0)))
        return
    assert not (OUT / 'bulk_error.json').exists(), 'Preserved error requires recovery review'
    assert not (OUT / 'generation_error.json').exists(), 'Preserved generation error requires review'
    assert not (OUT / 'bulk_complete.json').exists(), 'Already complete'
    mutex = ctypes.windll.kernel32.CreateMutexW(None, True, 'GRL_GRID100_BULK_COLLECTION')
    assert ctypes.windll.kernel32.GetLastError() != 183, 'Existing bulk collector'
    plan = read(OUT / 'bulk_plan.json'); started = time.monotonic()
    logs = OUT / 'bulk_logs'; logs.mkdir(exist_ok=True)
    command = [sys.executable, '-u', '-X', 'utf8']
    try:
        if not (OUT / 'generation_trainval_finished.json').exists():
            run_child(command + [str(W / 'generate.py'), '--mode', 'trainval'], logs / 'generation_trainval.log',
                      plan['generation_wall_seconds'], 'generating_trainval', generation=True)
        gen = read(OUT / 'generation_trainval_finished.json')
        assert gen['passed'] and gen['completed'] == 40
        for index, item in enumerate(items()):
            test_sealed()
            if (OUT / 'dataset_receipts' / (item['scenario'] + '.json')).exists():
                continue
            remaining = plan['total_wall_seconds'] - (time.monotonic() - started)
            assert remaining > 0, 'Overall data collection budget exhausted'
            folder = RUNTIME / 'runs' / f"r0__{item['scenario']}__FULL_TEACHER"
            result = RUNTIME / f'worker_result_{index:02d}.json'
            assert not folder.exists() and not result.exists(), 'Partial or complete unreceipted teacher preserved; inspect before recovery'
            run_child(command + [str(W / 'bulk_teacher_worker.py'), '--index', str(index)],
                      logs / f"teacher_{item['scenario']}.log", min(1320, remaining),
                      'collecting_teacher', index=index, scenario=item['scenario'], completed=len(completed))
            run_child(command + [str(W / 'bulk_extract.py'), '--index', str(index)],
                      logs / f"extract_{item['scenario']}.log", 300, 'extracting_saved_data',
                      index=index, scenario=item['scenario'], completed=len(completed))
            record = read(OUT / 'dataset_receipts' / (item['scenario'] + '.json'))
            verify_dataset(record); completed.append(record); refresh_manifest(completed)
            print(json.dumps(dict(scenario=item['scenario'], completed=len(completed), samples=record['samples'])), flush=True)
        state('checking_completed_dataset', completed=40)
        summarize()
        state('archiving_completed_dataset', completed=40)
        run_child(command + [str(W / 'bulk_archive.py')], logs / 'archive.log', 14400,
                  'archiving_completed_dataset', completed=40)
        state('complete', completed=40)
    finally:
        ctypes.windll.kernel32.ReleaseMutex(mutex); ctypes.windll.kernel32.CloseHandle(mutex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        main(args.check)
    except BaseException:
        if not args.check:
            save(OUT / 'bulk_error.json', dict(utc=utc(), traceback=traceback.format_exc()))
            state('stopped_preserving_error')
        raise
