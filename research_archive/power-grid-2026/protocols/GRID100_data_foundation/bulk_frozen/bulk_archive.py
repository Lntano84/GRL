"""Archive all completed train/validation records and publish without fitting."""
import shutil
import subprocess
import zipfile
from bulk_common import ROOT, W, OUT, RUNTIME, ARCHIVE, COLLAB, items, read, sha, save, verify, verify_dataset, test_sealed, utc

REPO = ROOT / 'work/GRL-results-integration'


def package(destination, files):
    batches, batch, size = [], [], 0
    for path in sorted(set(files)):
        length = path.stat().st_size
        assert length < 80_000_000, f'Single file requires separate archive review: {path}'
        if batch and size + length > 80_000_000:
            batches.append(batch); batch, size = [], 0
        batch.append(path); size += length
    if batch:
        batches.append(batch)
    for number, group in enumerate(batches):
        path = destination / f'records_{number:02d}.zip'
        assert not path.exists(), 'Preserve any partial archive; inspect before resuming'
        temp = path.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temp, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for source in group:
                archive.write(source, source.relative_to(ROOT).as_posix())
        assert temp.stat().st_size < 95_000_000
        with zipfile.ZipFile(temp) as archive:
            assert archive.testzip() is None
        temp.replace(path)


def main():
    verify(read(OUT / 'bulk_code_freeze.json')); test_sealed()
    assert read(OUT / 'bulk_review.json')['passed']
    complete = ARCHIVE / 'bulk_complete'
    complete.mkdir(parents=True, exist_ok=True)
    planned = items()
    for item in planned:
        receipt = read(OUT / 'dataset_receipts' / (item['scenario'] + '.json'))
        verify_dataset(receipt)
        if receipt['reused_pilot']:
            continue  # Original lossless pilot archive already published; avoid duplicating it.
        dest = complete / item['scenario']
        index = dest / 'archive_receipt.json'
        if index.exists():
            for name, expected in read(index)['files'].items():
                assert sha(dest / name) == expected
            continue
        assert not dest.exists(), 'Partial archive requires review, never overwritten'
        dest.mkdir()
        generation = OUT / 'generation_receipts' / (item['scenario'] + '.json')
        gen_receipt = read(generation); verify(gen_receipt['files'])
        paths = [ROOT / name for name in receipt['files']]
        paths += [ROOT / name for name in gen_receipt['files']]
        paths += [generation, OUT / 'dataset_receipts' / (item['scenario'] + '.json')]
        log = OUT / 'bulk_logs' / f"teacher_{item['scenario']}.log"
        if log.exists():
            paths.append(log)
        package(dest, paths)
        save(index, dict(scenario=item['scenario'], files={p.name: sha(p) for p in dest.iterdir() if p.is_file()},
                         original_file_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in paths}))
    for name in ['bulk_plan.json', 'bulk_preflight.json', 'bulk_code_freeze.json', 'bulk_review.json',
                 'bulk_dataset_manifest.json', 'generation_trainval_finished.json', 'GRID100_report.md',
                 'training_input_contract.json', '数据协议_中文.md', 'HANDOFF.md']:
        shutil.copy2(OUT / name, complete / name)
        shutil.copy2(OUT / name, COLLAB / name)
    logs = complete / 'pipeline_logs'; logs.mkdir(exist_ok=True)
    for path in (OUT / 'bulk_logs').glob('*.log'):
        if path.name.startswith('generation'):
            shutil.copy2(path, logs / path.name)
    save(complete / 'archive_manifest.json', dict(
         files={p.relative_to(complete).as_posix(): sha(p) for p in complete.rglob('*') if p.is_file() and p.name != 'archive_manifest.json'},
         pilot_reference='../pilot_complete', final_test_materialized=False, model_fits=0, utc=utc()))
    entry = ('\n\n## 2026-10-21 — GRID100 train/validation data collection completed\n\n'
             '40个预定场景（32训练/8验证，20日期组）数据采集与保存数据核验完成；首个试点复用，'
             '所有未完周和零搜索场景按实际结果保留。局部补查连续增益标签不等同长期收益；'
             '8最终测试场景未生成，未训练新模型。逐场景数量与完成情况见GRID100_report.md。\n')
    for name in ['RESEARCH_STATE.md', 'EXPERIMENT_LOG.md']:
        path = REPO / 'docs' / name
        text = path.read_text(encoding='utf8')
        if 'GRID100 train/validation data collection completed' not in text:
            path.write_text(text + entry, encoding='utf8')
    next_steps = REPO / 'docs/NEXT_STEPS.md'
    text = next_steps.read_text(encoding='utf8')
    if 'GRID100批量采集运行中' in text:
        next_steps.write_text(text.replace('GRID100批量采集运行中', 'GRID100训练/验证数据已完成，等待用户确认下一阶段'), encoding='utf8')
    status = subprocess.check_output(['git', 'status', '--porcelain', '-z'], cwd=REPO).decode('utf8')
    allowed = {'docs/RESEARCH_STATE.md', 'docs/EXPERIMENT_LOG.md', 'docs/NEXT_STEPS.md'}
    prefix = 'research_archive/power-grid-2026/protocols/GRID100_data_foundation/'
    for record in status.split('\0'):
        if record:
            name = record[3:]
            assert name in allowed or name.startswith(prefix), f'Unrelated Git changes preserved, publication stopped: {name}'
    env = __import__('os').environ.copy()
    env['TEMP'] = env['TMP'] = str(ROOT / 'tmp/grid100')
    log = OUT / 'bulk_logs/archive_push.log'
    with log.open('x', encoding='utf8') as stream:
        subprocess.run(['cmd.exe', '/d', '/c', str(REPO / 'scripts/git_push.cmd'),
                        'GRID100 archive qualified training and validation data'], cwd=REPO,
                       stdout=stream, stderr=subprocess.STDOUT, env=env, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    local = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'lntano-src', 'refs/heads/main'], cwd=REPO, text=True).split()[0]
    assert local == remote
    save(OUT / 'bulk_complete.json', dict(passed=True, local_sha=local, remote_sha=remote,
         scenarios=40, train=32, validation=8, date_groups=20,
         final_test_materialized=False, model_fits=0, utc=utc()))
    shutil.copy2(OUT / 'bulk_complete.json', COLLAB / 'bulk_complete.json')


if __name__ == '__main__':
    main()
