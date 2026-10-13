"""Publication-exportable plots retain repeat variability and inactive dates."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid_paper_v2'


def main():
    assert (OUT / 'paper_status.json').exists()
    assert not (OUT / 'fresh_confirmation.pdf').exists()
    r = json.loads((ROOT / 'outputs/grid62/review.json').read_text())
    f = json.loads((ROOT / 'outputs/grid62/finished.json').read_text())
    dates = sorted({x['week'] for x in r['family_rows']})
    methods = ['FULL', 'FIRST_SAFE_BEST', 'NO_LOCAL', 'FRAME']
    labels = ['Full search', 'First-safe + fallback', 'No local completion', 'Framework']
    colors = ['#707780', '#dc9f45', '#679a94', '#4269b0']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.3))
    width = .18
    ix = {(x['week'], x['method']): x for x in r['family_rows']}
    for j, (method, label, color) in enumerate(zip(methods, labels, colors)):
        positions = np.arange(3) + (j-1.5)*width
        times = [ix[date, method]['controller_s'] for date in dates]
        axes[0].bar(positions, times, width, color=color, label=label)
        for k, date in enumerate(dates):
            observed = [s['controller_s'] for s in f['summaries'] if s['week'] == date and s['method'] == method]
            axes[0].scatter([positions[k]]*2, observed, c='black', s=11, zorder=4, alpha=.65)
        q = [ix[date, method]['n1_queries'] for date in dates]
        axes[2].bar(positions, q, width, color=color)
    axes[0].set_title('(a) Actual controller time\nblack dots: both timing repeats')
    axes[0].set_ylabel('Seconds per episode')
    complete_dates = [d for d in dates if ix[d, 'FULL']['complete']]
    for j, (method, color) in enumerate(zip(methods, colors)):
        positions = np.arange(len(complete_dates)) + (j-1.5)*width
        values = [100*ix[date, method]['cost_delta_full'] for date in complete_dates]
        axes[1].bar(positions, values, width, color=color)
    axes[1].axhline(.5, color='#aa4343', linestyle='--', linewidth=1.2, label='Frozen 0.5% screen')
    axes[1].axhline(0, color='#888888', linewidth=.6)
    axes[1].set_title('(b) Complete-week benchmark cost\nfailed May excluded from cost ratios')
    axes[1].set_ylabel('Change relative to FULL (%)')
    axes[1].set_xticks(np.arange(len(complete_dates)), [d[:10] for d in complete_dates])
    axes[2].set_title('(c) Topology-search queries\nMay: zero queries, all methods')
    axes[2].set_ylabel('Queries per episode')
    for ax in (axes[0], axes[2]):
        ax.set_xticks(np.arange(3), [d[:10] for d in dates])
    for ax in axes:
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelrotation=20)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4,
               bbox_to_anchor=(.5, .02), frameon=False)
    fig.text(.5, .99, 'Frozen three-date confirmation: shared frequency ranking, base controller and caches',
             ha='center', va='top', fontsize=13)
    fig.tight_layout(rect=(0, .13, 1, .91))
    for suffix in ('pdf', 'png'):
        fig.savefig(OUT / ('fresh_confirmation.'+suffix), dpi=190, bbox_inches='tight')
    plt.close(fig)
    (OUT / 'figure_receipts.json').write_text(json.dumps([
        {'path': p.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in [ROOT / 'outputs/grid62/review.json', ROOT / 'outputs/grid62/finished.json',
                  OUT / 'fresh_confirmation.pdf', OUT / 'fresh_confirmation.png']], indent=2), encoding='utf-8')
    print(json.dumps({'figures': 2, 'raw_timing_repeats_retained': True}))


if __name__ == '__main__':
    main()
