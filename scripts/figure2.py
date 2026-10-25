"""Figure 2 (RQ2): where does the predictor turn the plan into its predicted path around the hazard?

Each pair shares start and goal frames and differs only in the plan: one plan's executed path
enters the zone (unsafe), the other's avoids it (safe). Patches copy activations from one run into
the other at the same (layer, imagined step):
  - residual stream at block 0's input or a block's output, at one predictor call or all of them
  - the action conditioning one block receives (alone, or blocks 0..i cumulatively), or that all
    blocks receive at one predictor call
Directions: denoising = run the unsafe plan, patch in the safe run (is this enough to move the
imagined path off the zone?); noising = run the safe plan, patch in the unsafe run.

Effect = fraction of the safe-vs-unsafe gap recovered, for
  - decoded path (headline): closest approach of the imagined path to the zone centre, decoded by
    a position decoder trained only on real frames
  - latent: projection onto the safe-minus-unsafe difference of the imagined latents
  - hazard probe (secondary): Figure 1's frozen linear probe on the predicted latents, applied
    out-of-fold, in logit units
Aggregates are ratios of summed effects over pairs, with 95% CIs from resampling pairs.

Usage: .venv/bin/python scripts/figure2.py   (after make_pairs.py)
Writes outputs/figure2/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Circle

from lewm_probe import common as C
from lewm_probe import dataset as D
from lewm_probe import patching as PT
from lewm_probe import position_decoder as PD
from lewm_probe.frozen import load_probes
from lewm_probe.hazard import segment_distance
from scripts.feasibility_check import draw_room
from scripts.make_pairs import Planner

OUT = C.ROOT / 'outputs' / 'figure2'
MIN_GAP_PX = 2.0  # pairs whose decoded paths barely differ cannot show a fraction recovered
N_BOOT = 1000
LAYER_NAMES = {PT.INPUT: 'input', **{i: f'b{i}' for i in range(6)}}


def closest_approach(start, points, center):
    """Closest distance from `center` to the polyline start -> imagined block ends."""
    poly = np.vstack([start, points])
    return float(segment_distance(poly[:-1], poly[1:], center).min())


class Readouts:
    def __init__(self, d):
        self.decoder = PD.load()['model']
        self.center = D.zone(d).center
        fz = load_probes(C.ROOT / 'outputs' / 'figure1' / 'frozen_probes.npz')
        self.probe_models = fz['probes'][('trained', 'pred_proj', 'all', 'hazard', 'linear')]['models']
        self.test_fold = {}
        for f, (_, te) in enumerate(fz['folds']):
            for e in np.unique(fz['episodes'][te]):
                self.test_fold[e] = f

    def __call__(self, z, start, episode):
        points = self.decoder.predict(z[1:])
        probe = self.probe_models[self.test_fold[episode]]
        logit = float(probe.decision_function(z[1:].reshape(1, -1))[0])
        return dict(z=z, points=points, dist=closest_approach(start, points, self.center), logit=logit)


def experiments(n_steps, n_blocks):
    """(name, family, builder(source_record) -> patch dict)."""
    exps = []
    step_sets = [(f'call {t + 1}', [t]) for t in range(n_steps)] + [('all calls', list(range(n_steps)))]
    for layer in [PT.INPUT] + list(range(n_blocks)):
        for label, ts in step_sets:
            exps.append((f'res|{LAYER_NAMES[layer]}|{label}', 'residual',
                         lambda rec, l=layer, ts=ts: PT.residual_patch(rec, l, ts)))
    for b in range(n_blocks):
        exps.append((f'cond|b{b}', 'conditioning, one block',
                     lambda rec, b=b: PT.conditioning_patch(rec, [b], range(n_steps))))
        exps.append((f'cond|b0-b{b}', 'conditioning, blocks 0..i',
                     lambda rec, b=b: PT.conditioning_patch(rec, range(b + 1), range(n_steps))))
    for t in range(n_steps):
        exps.append((f'cond|call {t + 1}', 'conditioning, one call',
                     lambda rec, t=t: PT.conditioning_patch(rec, range(n_blocks), [t])))
    return exps


def fraction(patched, target, source, key):
    """Per-pair numerator and denominator of the fraction of the target->source gap recovered."""
    if key == 'latent':
        diff = source['z'][1:] - target['z'][1:]
        return float(((patched['z'][1:] - target['z'][1:]) * diff).sum()), float((diff * diff).sum())
    return patched[key] - target[key], source[key] - target[key]


def ratio_ci(num, den, rng):
    est = num.sum() / den.sum()
    idx = rng.integers(0, len(num), size=(N_BOOT, len(num)))
    boot = num[idx].sum(1) / den[idx].sum(1)
    return est, *np.percentile(boot, [2.5, 97.5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = D.load()
    pairs = [dict(np.load(p)) for p in sorted((C.ROOT / 'data' / 'pairs').glob('pairs_*.npz'))]
    pairs = {k: np.concatenate([p[k] for p in pairs]) for k in pairs[0]}
    n_pairs = len(pairs['row'])
    print(f'{n_pairs} pairs; executed closest approach unsafe {np.median(pairs["min_dist_unsafe"]):.1f}px, '
          f'safe {np.median(pairs["min_dist_safe"]):.1f}px (median)')

    planner = Planner()
    patcher = PT.Patcher(planner.model)
    read = Readouts(d)
    exps = experiments(C.PLAN['horizon'], len(patcher.blocks))

    records, examples = [], []
    for i in range(n_pairs):
        start, goal, ep = pairs['start'][i], pairs['goal'][i], pairs['episode'][i]
        info = planner.info(start, goal)
        plans = {k: torch.from_numpy(pairs[f'plan_{k}'][i])[None] for k in ('unsafe', 'safe')}
        base, rec = {}, {}
        for k, plan in plans.items():
            z, rec[k] = patcher.run(info, plan, record=True)
            base[k] = read(z, start, ep)
        row = dict(pair=i, base_dist_unsafe=base['unsafe']['dist'], base_dist_safe=base['safe']['dist'],
                   base_logit_unsafe=base['unsafe']['logit'], base_logit_safe=base['safe']['logit'])
        for direction, target, source in [('denoise', 'unsafe', 'safe'), ('noise', 'safe', 'unsafe')]:
            for name, family, build in exps:
                z, _ = patcher.run(info, plans[target], patches=build(rec[source]))
                patched = read(z, start, ep)
                for key in ('dist', 'latent', 'logit'):
                    num, den = fraction(patched, base[target], base[source], key)
                    records.append(dict(pair=i, direction=direction, experiment=name, family=family,
                                        metric=key, num=num, den=den))
                if direction == 'denoise' and name in ('cond|b0', 'cond|b2', 'cond|b5', 'res|b2|all calls'):
                    row[f'points|{name}'] = patched['points']
        row['points|unsafe'], row['points|safe'] = base['unsafe']['points'], base['safe']['points']
        row['path_unsafe'], row['path_safe'], row['start'] = pairs['path_unsafe'][i], pairs['path_safe'][i], start
        examples.append(row)
        if (i + 1) % 25 == 0:
            print(f'    {i + 1}/{n_pairs} pairs patched', flush=True)
    patcher.remove()

    rec_df = pd.DataFrame(records)
    base_df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in examples])
    gap = (base_df['base_dist_safe'] - base_df['base_dist_unsafe']).to_numpy()
    keep = np.nonzero(gap >= MIN_GAP_PX)[0]
    print(f'decoded imagined paths: safe plan farther from the zone in {(gap > 0).mean():.0%} of pairs; '
          f'{len(keep)} pairs with a gap >= {MIN_GAP_PX}px used for the decoded-path metric '
          f'(median gap {np.median(gap):.1f}px)')
    print(f'hazard probe logit: unsafe {base_df["base_logit_unsafe"].median():+.2f}, '
          f'safe {base_df["base_logit_safe"].median():+.2f} (median)')

    rng = np.random.default_rng(0)
    summary = []
    for (direction, name, family, metric), g in rec_df.groupby(['direction', 'experiment', 'family', 'metric']):
        g = g.set_index('pair')
        if metric == 'dist':
            g = g.loc[g.index.intersection(keep)]
        est, lo, hi = ratio_ci(g['num'].to_numpy(), g['den'].to_numpy(), rng)
        summary.append(dict(direction=direction, experiment=name, family=family, metric=metric,
                            fraction=est, ci_lo=lo, ci_hi=hi, n=len(g)))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'patching.csv', index=False)
    base_df.to_csv(OUT / 'pairs.csv', index=False)
    rec_df.to_csv(OUT / 'per_pair_effects.csv.gz', index=False)
    plot(summary, examples, keep, read.center, OUT / 'figure2.png')
    report(summary)
    print(f'wrote {OUT.relative_to(C.ROOT)}/')


def _frac(summary, direction, experiment, metric='dist'):
    r = summary[(summary.direction == direction) & (summary.experiment == experiment) & (summary.metric == metric)]
    return r.iloc[0]


def plot(summary, examples, keep, center, path):
    layers = ['input'] + [f'b{i}' for i in range(6)]
    cols = [f'call {t}' for t in range(1, 6)] + ['all calls']
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    for k, (direction, title) in enumerate([('denoise', 'A. Residual patch, safe into unsafe'),
                                            ('noise', 'B. Residual patch, unsafe into safe')]):
        ax = fig.add_subplot(gs[0, k])
        M = np.array([[_frac(summary, direction, f'res|{l}|{c}')['fraction'] for c in cols] for l in layers])
        im = ax.imshow(M, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        for (r, c), v in np.ndenumerate(M):
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=7, color='w' if v < 0.6 else 'k')
        ax.set_xticks(range(len(cols)), cols, fontsize=8)
        ax.set_yticks(range(len(layers)), layers)
        ax.set_xlabel('predictor call patched (call t produces imagined step t)')
        ax.set_title(title + '\nfraction of decoded closest-approach gap moved', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(6)
    for off, (direction, family, label, color) in zip(
            [-0.3, -0.1, 0.1, 0.3],
            [('denoise', 'b{}', 'one block, safe into unsafe', 'tab:blue'),
             ('denoise', 'b0-b{}', 'blocks 0..i, safe into unsafe', 'tab:cyan'),
             ('noise', 'b{}', 'one block, unsafe into safe', 'tab:red'),
             ('noise', 'b0-b{}', 'blocks 0..i, unsafe into safe', 'tab:orange')]):
        rows = [_frac(summary, direction, 'cond|' + family.format(b)) for b in range(6)]
        v = np.array([r['fraction'] for r in rows])
        err = np.array([[r['fraction'] - r['ci_lo'], r['ci_hi'] - r['fraction']] for r in rows]).T
        ax.bar(x + off, v, width=0.2, color=color, yerr=err, capsize=2, label=label)
    ax.axhline(1, color='k', lw=0.6, ls=':')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x, [f'b{b}' for b in range(6)])
    ax.set_ylabel('fraction of decoded closest-approach gap moved')
    ax.set_title('C. Which blocks use the plan? (swap the actions a block sees)', fontsize=10)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 0])
    t = np.arange(1, 6)
    for direction, color in [('denoise', 'tab:blue'), ('noise', 'tab:red')]:
        rows = [_frac(summary, direction, f'cond|call {s}') for s in t]
        v = np.array([r['fraction'] for r in rows])
        err = np.array([[r['fraction'] - r['ci_lo'], r['ci_hi'] - r['fraction']] for r in rows]).T
        ax.errorbar(t, v, yerr=err, fmt='-o', color=color, capsize=2,
                    label='safe into unsafe' if direction == 'denoise' else 'unsafe into safe')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(t, [f'call {s}' for s in t])
    ax.set_ylabel('fraction of decoded closest-approach gap moved')
    ax.set_title('D. Which imagined step\'s actions matter? (all blocks, one call)', fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    for direction, color in [('denoise', 'tab:blue'), ('noise', 'tab:red')]:
        for metric, style in [('dist', '-o'), ('latent', '--s'), ('logit', ':^')]:
            v = [_frac(summary, direction, f'cond|b0-b{b}', metric)['fraction'] for b in range(6)]
            ax.plot(range(6), v, style, color=color, ms=4,
                    label=f'{"safe into unsafe" if direction == "denoise" else "unsafe into safe"}, '
                          f'{dict(dist="decoded path", latent="latent", logit="hazard probe")[metric]}')
    ax.axhline(1, color='k', lw=0.6, ls=':')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(range(6), [f'b0-b{b}' for b in range(6)])
    ax.set_ylabel('fraction of gap moved')
    ax.set_title('E. Cumulative action swap, three readouts agree?', fontsize=10)
    ax.legend(fontsize=6.5)

    ax = fig.add_subplot(gs[1, 2])
    gaps = np.array([examples[i]['base_dist_safe'] - examples[i]['base_dist_unsafe'] for i in keep])
    ex = examples[keep[np.argmin(np.abs(gaps - np.median(gaps)))]]  # a typical pair, not the extreme
    draw_room(ax, dict(axis=1, pos=112.0, thickness=10, door_centers=[49.0], door_half_sizes=[14.0]))
    ax.add_patch(Circle(center, 10.0, color='tab:red', alpha=0.2, lw=0))
    start = ex['start']
    for key, color, label in [('path_unsafe', 'tab:red', 'executed, unsafe plan'),
                              ('path_safe', 'tab:blue', 'executed, safe plan')]:
        ax.plot(*ex[key].T, color=color, lw=1, alpha=0.5, label=label)
    for key, color, style, label in [('unsafe', 'tab:red', '-o', 'imagined (decoded), unsafe'),
                                     ('safe', 'tab:blue', '-o', 'imagined (decoded), safe'),
                                     ('cond|b0', 'tab:green', '--^', 'unsafe plan, b0 sees safe actions'),
                                     ('cond|b5', 'tab:purple', '--v', 'unsafe plan, b5 sees safe actions')]:
        pts = np.vstack([start, ex[f'points|{key}']])
        ax.plot(*pts.T, style, color=color, ms=4, lw=1.2, label=label)
    allpts = np.vstack([ex['path_unsafe'], ex['path_safe']] + [ex[k] for k in ex if k.startswith('points|')])
    lo, hi = allpts.min(0) - 12, allpts.max(0) + 12
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(hi[1], lo[1])
    ax.set_title('F. A typical pair (median decoded gap)', fontsize=10)
    ax.legend(fontsize=6.5, loc='best')

    n = int(summary[(summary.metric == 'dist')]['n'].iloc[0])
    fig.suptitle(f'LeWM TwoRooms: where the predictor turns the plan into its path around the hazard '
                 f'({n} same-start pairs, safe vs unsafe plan)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(summary):
    def line(direction, exp, metric='dist'):
        r = _frac(summary, direction, exp, metric)
        return f'{r["fraction"]:+.2f} [{r["ci_lo"]:+.2f}, {r["ci_hi"]:+.2f}]'
    print('\nfraction of the decoded closest-approach gap moved (95% CI)')
    print('residual, all calls:   ' + '  '.join(
        f'{l}: {_frac(summary, "denoise", f"res|{l}|all calls")["fraction"]:.2f}' for l in ['input'] + [f'b{i}' for i in range(6)]))
    for b in range(6):
        print(f'    b{b}: action swap alone  denoise {line("denoise", f"cond|b{b}")}  noise {line("noise", f"cond|b{b}")}'
              f'   | blocks 0..{b}  denoise {line("denoise", f"cond|b0-b{b}")}')
    for t in range(1, 6):
        print(f'    call {t}: all-block action swap  denoise {line("denoise", f"cond|call {t}")}')
    print('cumulative action swap, three readouts (denoise): ' + '  '.join(
        f'b0-b{b}: dist {_frac(summary, "denoise", f"cond|b0-b{b}", "dist")["fraction"]:.2f} '
        f'latent {_frac(summary, "denoise", f"cond|b0-b{b}", "latent")["fraction"]:.2f} '
        f'probe {_frac(summary, "denoise", f"cond|b0-b{b}", "logit")["fraction"]:.2f}' for b in range(6)))


if __name__ == '__main__':
    main()
