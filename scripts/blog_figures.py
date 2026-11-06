"""Simple figures for the blog post, for readers outside interpretability.

Usage: .venv/bin/python scripts/blog_figures.py
Reads the result tables in outputs/figure1, figure3 and figure4 and writes outputs/blog/blog_*.png.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import scripts.diagrams as D
from lewm_probe import common as C

OUT = C.ROOT / 'outputs' / 'blog'
HIGHLIGHT = '#2a78d6'
MUTED = '#9AA5B1'
DARK = '#52606D'
INK = '#1F2933'
INK_2 = '#52514e'


def style(ax):
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#CBD2D9')
    ax.tick_params(colors=INK_2, length=0)
    ax.xaxis.grid(True, color='#E4E7EB', lw=0.8)
    ax.set_axisbelow(True)


def title(fig, main, sub):
    fig.text(0.01, 0.98, main, fontsize=15, fontweight='bold', color=INK, va='top')
    fig.text(0.01, 0.90, sub, fontsize=10.5, color=INK_2, va='top')


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def overview():
    g = D.Diagram('Reading the plan before the agent moves',
                  'The world model imagines where each candidate plan leads; a small monitor reads that imagined future',
                  xlim=135, ylim=36)
    cam = g.box(1, 15, 17, 12, 'What the agent sees', 'one camera frame\nand a picture\nof the goal')
    wm = g.box(24, 13, 26, 16, 'World model imagines', 'for each candidate plan:\n"where will I be\nafter 25 steps?"\n(192 numbers per step,\nno pictures, no words)', 'model')
    plan = g.box(56, 15, 21, 12, 'Planner picks', 'the plan whose imagined\nend is closest to the goal', 'planner')
    mon = g.box(83, 15, 23, 12, 'Monitor reads', 'the imagined future:\n"will this plan enter\nthe hazard zone?"', 'probe')
    act = g.box(112, 21, 22, 6, 'Looks safe: act', '', 'env')
    flag = g.box(112, 12, 22, 6, 'Risky: flag or re-plan', '', 'result')
    g.path([(cam['r'], cam['cy']), (wm['x'], cam['cy'])])
    g.path([(wm['r'], plan['cy']), (plan['x'], plan['cy'])])
    g.path([(plan['r'], plan['cy']), (mon['x'], mon['cy'])])
    g.path([(mon['r'], 24), (act['x'], act['cy'])])
    g.path([(mon['r'], 18), (flag['x'], flag['cy'])])
    g.note(83, 11.5, 'The hazard zone is never shown to the\nworld model or the planner. The monitor\nlearned it from examples of plans that\nentered it and plans that did not.', size=9.5)
    g.note(1, 8.5, 'The monitor is a linear readout: one weighted sum of the world model\'s\ninternal numbers, so it is cheap to run and easy to inspect.', size=9.5)
    g.ax.set_ylim(4.5, 36)
    D.OUT = OUT
    g.save('blog_overview.png')


def reading():
    r = pd.read_csv(C.ROOT / 'outputs' / 'figure1' / 'results.csv')
    r = r[r.target == 'hazard']
    def get(model, features, probe):
        return float(r[(r.model == model) & (r.features == features) & (r.probe == probe)]['mean'].iloc[0])
    rows = [
        ('Monitor on the trained world model', get('trained', 'selected layer', 'linear'), HIGHLIGHT),
        ('Best guess from start and goal\n(boosted trees on the geometry)', get('baseline', 'position+goal', 'trees'), DARK),
        ('Same network, weights shuffled', get('shuffled', 'selected layer', 'linear'), MUTED),
        ('Same network, untrained', get('random_init', 'selected layer', 'linear'), MUTED),
        ('Coin flip', 0.5, MUTED),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.subplots_adjust(top=0.78, left=0.31)
    title(fig, 'The monitor separates risky plans from safe ones better than geometry does',
          'Score for telling risky plans from safe ones (AUROC): 1.0 is perfect, 0.5 is a coin flip. 2,308 plans.')
    ys = range(len(rows))[::-1]
    for y, (label, v, c) in zip(ys, rows):
        ax.plot([0.5, v], [y, y], color='#E4E7EB', lw=2, zorder=1)
        ax.scatter([v], [y], s=110, color=c, zorder=3, edgecolor='white', linewidth=2)
        ax.text(v + 0.012, y, f'{v:.3f}' if v != 0.5 else '0.50', va='center', fontsize=10.5,
                color=INK if c == HIGHLIGHT else INK_2, fontweight='bold' if c == HIGHLIGHT else 'normal')
    ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=10.5, color=INK)
    ax.set_xlim(0.48, 1.04)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    style(ax)
    save(fig, 'blog_reading.png')


def compression():
    q = pd.read_csv(C.ROOT / 'outputs' / 'figure3' / 'quantization.csv').set_index('precision')
    order = [('fp32', 'Full\nprecision'), ('int8', '8-bit'), ('w6', '6-bit'), ('w4', '4-bit'), ('w3', '3-bit')]
    x = range(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(top=0.70, wspace=0.3)
    title(fig, 'Compressed to 4 bits, the monitor still catches risky plans; at 3 bits it slips',
          'The monitor stays frozen at full precision; the world model is compressed. Same 2,000 episodes each time.\n'
          'Lines: 95% intervals. Grey band: the full-precision interval.')
    panels = [('miss_rate', 'Risky plans the monitor misses', 0.25),
              ('success', 'Episodes where the agent reaches its goal', 1.0)]
    for ax, (col, label, top) in zip(axes, panels):
        lo, hi = q.loc['fp32', f'{col}_lo'], q.loc['fp32', f'{col}_hi']
        ax.axhspan(lo, hi, color='#E4E7EB', zorder=0)
        for i, (p, _) in enumerate(order):
            v, vlo, vhi = q.loc[p, col], q.loc[p, f'{col}_lo'], q.loc[p, f'{col}_hi']
            c = '#e34948' if p == 'w3' else HIGHLIGHT
            ax.plot([i, i], [vlo, vhi], color=c, lw=2, zorder=2)
            ax.scatter([i], [v], s=90, color=c, zorder=3, edgecolor='white', linewidth=2)
            ax.text(i + 0.12, v, f'{v:.0%}' if col == 'success' else f'{v:.1%}', va='center', fontsize=10, color=INK)
        ax.set_xticks(list(x), [o[1] for o in order], fontsize=10, color=INK)
        ax.set_ylim(0, top)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(label, fontsize=11.5, color=INK, loc='left')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.spines['left'].set_color('#CBD2D9')
        ax.spines['bottom'].set_color('#CBD2D9')
        ax.tick_params(colors=INK_2, length=0)
        ax.yaxis.grid(True, color='#E4E7EB', lw=0.8)
        ax.set_axisbelow(True)
    save(fig, 'blog_compression.png')


def steering():
    s = pd.read_csv(C.ROOT / 'outputs' / 'figure4' / 'steering.csv')
    s = s[s.set == 'unsafe']
    def get(mode, strength):
        return s[(s['mode'] == mode) & (s.strength == strength)].iloc[0]
    rows = [
        ('Re-plan with a new random seed', get('unguided, new seed', 0.0), MUTED),
        ('Penalty in a random direction', get('random', 100.0), MUTED),
        ('Monitor\'s warning added to the planner\'s cost', get('probe', 100.0), HIGHLIGHT),
        ('Hand-built penalty that knows the zone', get('geometry', 100.0), DARK),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    fig.subplots_adjust(top=0.76, left=0.36, right=0.86)
    title(fig, 'The monitor can also steer the planner away from the hazard',
          '120 episodes whose original plan entered the zone. Bars: share that still enter. Right: share that reach the goal.')
    ys = list(range(len(rows)))[::-1]
    for y, (label, row, c) in zip(ys, rows):
        ax.barh(y, row.hazard, height=0.55, color=c)
        ax.text(row.hazard + 0.015, y, f'{row.hazard:.0%}', va='center', fontsize=10.5,
                color=INK, fontweight='bold' if c == HIGHLIGHT else 'normal')
        ax.text(1.08, y, f'{row.success:.0%}', va='center', ha='center', fontsize=10.5, color=INK_2,
                transform=ax.get_yaxis_transform())
    ax.text(1.08, len(rows) - 0.55, 'reach goal', ha='center', va='bottom', fontsize=9.5, color=INK_2,
            transform=ax.get_yaxis_transform())
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=10.5, color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style(ax)
    save(fig, 'blog_steering.png')


if __name__ == '__main__':
    overview()
    reading()
    compression()
    steering()
    print(f'wrote {OUT.relative_to(C.ROOT)}/')
