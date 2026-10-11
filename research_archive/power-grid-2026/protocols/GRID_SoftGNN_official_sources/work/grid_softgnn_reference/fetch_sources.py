"""Small official-source snapshot only; no install, fitting, or grid replay."""
import hashlib, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid_softgnn_reference'
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / 'manifest.json').exists()
headers = {'User-Agent': 'GRL-research-source-review'}

def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return r.read()

repo = 'ainetus/soft_label_gnn'
commit = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/main'))['sha']
tree = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'))['tree']
assert not any(r['path'].endswith(('.pt', '.pth', '.ckpt')) for r in tree)
names = ['README.md', 'LICENSE', 'gnn/gnn_models.py', 'gnn/obs_converter.py',
         'gnn/gnn_prediction.py', 'gnn/torch_geometric_datasets.py',
         'data/best_model/paper_train_config.pkl']
records = []
for name in names:
    url = f'https://raw.githubusercontent.com/{repo}/{commit}/{name}'
    b = fetch(url)
    p = OUT / 'official' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b)
    records.append({'path': p.relative_to(ROOT).as_posix(), 'source': url,
                    'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)})
manifest = {'repository': f'https://github.com/{repo}', 'commit': commit,
    'files': records, 'public_pretrained_checkpoint_in_tree': False,
    'source_snapshot_only': True, 'new_installs': 0, 'new_fits': 0,
    'new_forecasts': 0, 'warning': 'Official paper configuration is present, but no trained checkpoint in the checked source tree. WCCI2022 actions/weights cannot be assumed compatible with IDF2023. No binary configuration deserialization/execution performed.'}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
(OUT / 'SOURCE_SCOPE.md').write_text('''# SoftGNN对照：公开材料范围

候选是ECML-PKDD2025的SoftGNN，来源为作者正式仓库。只保存必要原始源码、许可证和论文配置，未安装依赖、拟合或回放。

本次检查的源码树没有.pt/.pth/.ckpt训练权重；有paper_train_config.pkl（只保存字节，不执行反序列化）。因此不能声称已经运行作者训练好的SoftGNN。原论文为WCCI2022拓扑控制；当前为IDF2023含连续恢复的控制器，动作输出与评价必须适配。

下一步若使用，限定为资源匹配的公开方法改编：保留GAT与soft监督的核心定义，在相同合法训练数据上拟合，使用相同物理后端和连续控制接口；说明与原论文的差异。不重建分布式超参数搜索，不把它降为自造弱规则，也不把当前GRL编码器冒充SoftGNN。

这是一项来源与兼容范围记录，不是新增基线性能实验或工程退出门槛。主对照数量仍限定2–3项；当前GRID39继续。
''', encoding='utf-8')
print(json.dumps({'source_snapshot': True, 'commit': commit,
                  'files': len(records), 'weights_in_tree': False, 'executed_model': False}))
