"""Figure 4 (RQ4): can the monitor steer the planner away from the hazard?

Unsafe set: 120 episodes whose full-precision first plan entered the zone. Random set: 60 episodes.
For each guidance method and strength: how often the planner still enters the zone and how often
it reaches the goal. References come from data already collected for the same episode ids:
  - unguided full precision (unsafe set: 100% enter by selection)
  - the same episodes with only the planner's seed changed (data/fp32_reseed): any change to
    the search re-rolls borderline routes, so this is the floor a real effect must beat
Intervals: Wilson 95% for rates; paired episode bootstrap for probe-minus-random differences.

Usage: .venv/bin/python scripts/figure4.py
Writes outputs/figure4/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lewm_probe import common as C
from lewm_probe import dataset as D
from scripts.figure3 import wilson
from scripts.steering import OUT as RUNS

OUT = C.ROOT / 'outputs' / 'figure4'
METHODS = {'probe': ('monitor penalty (frozen hazard probe)', 'tab:blue', '-o'),
           'random': ('random-direction penalty', '0.5', '--s'),
           'geometry': ('geometry penalty (knows the zone)', 'tab:green', ':^'),
           'steer': ('literal steering (block 4)', 'tab:purple', '-D'),
           'steer_random': ('random-direction steering', '0.7', '--D')}


def reference(name, ids):
    d = D.load(sorted((C.ROOT / 'data' / name).glob('episodes_*.npz')))
    idx = {e: i for i, e in enumerate(d['ep/id'])}
    rows = [idx[e] for e in ids]
    return d['ep/hazard'][rows], d['ep/success'][rows]


def load_runs():
    out = []
    for f in sorted((RUNS / 'main').glob('*.npz')):
        mode, strength, which = f.stem.rsplit('_', 2)
        d = dict(np.load(f))
        out.append(dict(mode=mode, strength=float(strength), set=which, ids=d['id'], hazard=d['hazard'],
                        success=d['success'], min_dist=d['min_dist']))
    return out


def rate(x):
    p, n = float(np.mean(x)), len(x)
    lo, hi = wilson(np.array([p]), np.array([n]))
    return p, float(lo[0]), float(hi[0]), n


def paired_diff(a, b, n_boot=2000, seed=0):
    """Mean of a - b over episodes (same ids, same order), with a bootstrap 95% interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    boot = (a[idx] - b[idx]).mean(1)
    return float((a - b).mean()), *np.percentile(boot, [2.5, 97.5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    rows = []
    for r in runs:
        h, s = rate(r['hazard']), rate(r['success'])
        rows.append(dict(set=r['set'], mode=r['mode'], strength=r['strength'], n=h[3], hazard=h[0], hazard_lo=h[1],
                         hazard_hi=h[2], success=s[0], success_lo=s[1], success_hi=s[2],
                         min_dist_median=float(np.median(r['min_dist']))))
    refs = {}
    for which in ('unsafe', 'random'):
        ids = next((r['ids'] for r in runs if r['set'] == which), None)
        if ids is None:
            continue
        for name, label in [('full', 'unguided'), ('fp32_reseed', 'unguided, new seed')]:
            h, s = reference(name, ids)
            hh, ss = rate(h), rate(s)
            refs[(which, label)] = (hh, ss)
            rows.append(dict(set=which, mode=label, strength=0.0, n=hh[3], hazard=hh[0], hazard_lo=hh[1],
                             hazard_hi=hh[2], success=ss[0], success_lo=ss[1], success_hi=ss[2]))
    table = pd.DataFrame(rows).sort_values(['set', 'mode', 'strength'])
    table.to_csv(OUT / 'steering.csv', index=False)

    # paired differences on the unsafe set: monitor penalty vs random penalty at the same strength
    by = {(r['set'], r['mode'], r['strength']): r for r in runs}
    diffs = []
    for (which, mode, lam), r in by.items():
        if which == 'unsafe' and mode == 'probe' and ('unsafe', 'random', lam) in by:
            q = by[('unsafe', 'random', lam)]
            assert np.array_equal(r['ids'], q['ids'])
            for key in ('hazard', 'success'):
                m, lo, hi = paired_diff(r[key], q[key])
                diffs.append(dict(strength=lam, metric=key, probe_minus_random=m, ci_lo=lo, ci_hi=hi))
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(OUT / 'probe_vs_random.csv', index=False)

    plot(table, refs, OUT / 'figure4.png')
    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.precision', 3):
        print(table[['set', 'mode', 'strength', 'n', 'hazard', 'hazard_lo', 'hazard_hi', 'success', 'success_lo',
                     'success_hi']].to_string(index=False))
        if len(diffs):
            print('\nprobe minus random penalty, unsafe set (paired):')
            print(diffs.to_string(index=False))
    print(f'wrote {OUT.relative_to(C.ROOT)}/')


def plot(table, refs, path):
    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
    unsafe = table[table.set == 'unsafe']

    def curves(ax, key, modes, xlabel):
        for m in modes:
            t = unsafe[unsafe['mode'] == m].sort_values('strength')
            if not len(t):
                continue
            label, color, style = METHODS[m]
            v = t[key].to_numpy()
            ax.errorbar(t['strength'], v, yerr=np.vstack([v - t[f'{key}_lo'], t[f'{key}_hi'] - v]), fmt=style,
                        color=color, capsize=3, label=label, ms=5)
        for lbl, ls in [('unguided', '-'), ('unguided, new seed', ':')]:
            if ('unsafe', lbl) in refs:
                r = refs[('unsafe', lbl)][0 if key == 'hazard' else 1]
                ax.axhline(r[0], color='k', ls=ls, lw=1, label=f'{lbl} ({r[0]:.0%})')
        ax.set_xscale('log')
        ax.set_xticks([30, 100, 300], ['30', '100', '300'])
        ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.05)

    ax = axes[0]
    curves(ax, 'hazard', ['probe', 'random', 'geometry'], 'penalty weight λ')
    ax.set_ylabel('episodes that still enter the zone')
    ax.set_title('A. Penalty in the planner\'s cost:\nunsafe episodes still entering the zone', fontsize=10)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    curves(ax, 'success', ['probe', 'random', 'geometry'], 'penalty weight λ')
    ax.set_ylabel('episodes that reach the goal')
    ax.set_title('B. ...and still reaching the goal', fontsize=10)
    ax.legend(fontsize=7.5, loc='lower left')
    ax = axes[2]
    for key, marker in [('hazard', 'o'), ('success', 's')]:
        for m in ['steer', 'steer_random']:
            t = unsafe[unsafe['mode'] == m].sort_values('strength')
            if not len(t):
                continue
            label, color, style = METHODS[m]
            v = t[key].to_numpy()
            ax.errorbar(t['strength'], v, yerr=np.vstack([v - t[f'{key}_lo'], t[f'{key}_hi'] - v]),
                        fmt=style[:-1] + marker, color=color, capsize=3, ms=5,
                        label=f'{label}: {"enter zone" if key == "hazard" else "reach goal"}')
    ax.set_xscale('log')
    ax.set_xticks([3, 10], ['3', '10'])
    ax.minorticks_off()
    ax.set_xlabel('steering scale α (std of the projection)')
    ax.set_ylim(0, 1.05)
    ax.set_title('C. Literal steering of block 4 (unsafe episodes)', fontsize=10)
    ax.legend(fontsize=7.5)

    ax = axes[3]
    rnd = table[table.set == 'random']
    labels, hz, sc, herr, serr = [], [], [], [], []
    order = [('unguided', 0.0), ('unguided, new seed', 0.0)] + [(m, s) for m, s in
             rnd[~rnd['mode'].str.startswith('unguided')][['mode', 'strength']].itertuples(index=False)]
    for m, s in order:
        r = rnd[(rnd['mode'] == m) & (rnd['strength'] == s)]
        if not len(r):
            continue
        r = r.iloc[0]
        labels.append(m if m.startswith('unguided') else f'{m} {s:g}')
        hz.append(r['hazard']); sc.append(r['success'])
        herr.append([r['hazard'] - r['hazard_lo'], r['hazard_hi'] - r['hazard']])
        serr.append([r['success'] - r['success_lo'], r['success_hi'] - r['success']])
    y = np.arange(len(labels))
    ax.barh(y - 0.2, hz, 0.38, xerr=np.array(herr).T if herr else None, color='tab:red', alpha=0.75, label='enters zone',
            capsize=2)
    ax.barh(y + 0.2, sc, 0.38, xerr=np.array(serr).T if serr else None, color='tab:green', alpha=0.75,
            label='reaches goal', capsize=2)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_title('D. Side effects on 60 random episodes', fontsize=10)
    ax.legend(fontsize=7.5, loc='lower right')

    fig.suptitle('LeWM TwoRooms: turning the hazard monitor into a planning constraint (RQ4)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
