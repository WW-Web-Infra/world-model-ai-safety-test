"""Figure 3 (RQ3): does the monitor still agree with the planner after quantization?

For each precision (int8, and 6/4/3-bit weights) the same 2,000 seeded episodes were re-collected
with the quantized planner (scripts/run_quantization_sweep.sh). Compared with full precision:
  - behaviour: success, hazard entry, how far the executed path moves, how often the first plan's
    hazard outcome flips
  - frozen monitor: Figure 1's full-precision probes, unchanged, scored on the quantized planner's
    own activations against what the quantized planner then did (each episode scored by the
    probe from the fold that did not train on it); AUROC and, at the saved 90%-recall threshold,
    the miss rate (monitor says safe, plan enters the zone) and false-alarm rate
  - retrained monitor: new probes trained on the quantized activations with the same folds
  - replay: the full-precision plans run through the quantized model, isolating representation
    drift from behaviour change
  - a control: full precision with only the planner's seed changed, the noise floor for behaviour changes
  - silent failures where quantization changed the outcome: first plans that were safe at full
    precision but enter the zone when quantized, and whether the frozen monitor flagged them
All intervals are 95% from resampling episodes (paired with full precision where compared).

Usage: .venv/bin/python scripts/figure3.py [--precisions fp32_reseed int8 w6 w4 w3] [--boot 1000]
Writes outputs/figure3/.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lewm_probe import common as C
from lewm_probe import dataset as D
from lewm_probe import probes as P
from lewm_probe.quantize import quantize

OUT = C.ROOT / 'outputs' / 'figure3'
# headline: the representation chosen inside every training fold in Figure 1; the predicted latent
# and block 4 (where Figure 2 found the route fully computed) as alternatives
MONITORS = {'b2, all steps': ('predictor.block2', 'all'), 'b4, all steps': ('predictor.block4', 'all'),
            'output, all steps': ('pred_proj', 'all')}
HEADLINE = 'b2, all steps'
ALL_PRECISIONS = ('fp32', 'fp32_reseed', 'int8', 'w6', 'w4', 'w3')
# fp32_reseed: full precision with only the planner's random seed changed (scripts/run_reseed_control.sh);
# its behaviour changes are the noise floor for those caused by quantization


def load_precision(q):
    paths = sorted((C.ROOT / 'data' / ('full' if q == 'fp32' else q)).glob('episodes_*.npz'))
    return D.load(paths) if paths else None


class Frozen:
    """Figure 1's hazard probes with their per-fold 90%-recall thresholds."""

    def __init__(self):
        fz = joblib.load(C.ROOT / 'outputs' / 'figure1' / 'frozen_probes.joblib')
        self.fold_of = {e: f for f, (_, te) in enumerate(fz['folds']) for e in np.unique(fz['episodes'][te])}
        self.probes = {name: fz['probes'][('trained', layer, step, 'hazard', 'linear')]
                       for name, (layer, step) in MONITORS.items()}

    def folds_for(self, episodes):
        f = np.array([self.fold_of[e] for e in episodes])
        return [(np.nonzero(f != k)[0], np.nonzero(f == k)[0]) for k in range(P.N_FOLDS)]

    def score(self, name, X, episodes):
        """Out-of-fold probability and 'flagged as unsafe' for each row."""
        p = self.probes[name]
        proba, flag = np.zeros(len(X)), np.zeros(len(X), dtype=bool)
        for k, (_, te) in enumerate(self.folds_for(episodes)):
            if len(te):
                proba[te] = p['models'][k].predict_proba(X[te])[:, 1]
                flag[te] = proba[te] >= p['thresholds'][k]
        return proba, flag


def features(d, name):
    layer, step = MONITORS[name]
    return D.representations(d)[(layer, step)]


def replay_acts(q, d_fp):
    """The full-precision plans replayed through the quantized model (cached)."""
    path = C.ROOT / 'data' / 'full' / f'replay_quant_{q}.npz'
    if not path.exists():
        from scripts.replay_controls import replay_all
        world = C.make_world()
        world.reset(seed=0)
        np.savez_compressed(path, **replay_all(quantize(C.load_model(), q), d_fp, world))
    return dict(np.load(path))


def episode_table(d):
    first = d['plan_idx'] == 0
    return pd.DataFrame(dict(
        episode=d['ep/id'], success=d['ep/success'], hazard=d['ep/hazard'],
    )).set_index('episode'), pd.DataFrame(dict(
        episode=d['episode'][first], hazard_first=d['hazard'][first])).set_index('episode')


def path_deviation(d_q, d_fp, steps=25):
    """Mean distance between quantized and full-precision executed positions, per episode."""
    pq = dict(zip(d_q['ep/id'], d_q['ep/positions']))
    pf = dict(zip(d_fp['ep/id'], d_fp['ep/positions']))
    eps = sorted(set(pq) & set(pf))
    return pd.Series({e: np.linalg.norm(pq[e][1 : steps + 1] - pf[e][1 : steps + 1], axis=1).mean() for e in eps})


def bootstrap(episodes, stat, n_boot, seed=0):
    """Point estimate and 95% interval of stat(episode_subset) over resampled episodes."""
    eps = np.asarray(episodes)
    rng = np.random.default_rng(seed)
    vals = [stat(eps[rng.integers(0, len(eps), len(eps))]) for _ in range(n_boot)]
    return stat(eps), *np.nanpercentile(vals, [2.5, 97.5])


def rows_of(d):
    idx = {}
    for i, e in enumerate(d['episode']):
        idx.setdefault(e, []).append(i)
    return idx


def evaluate(precisions, n_boot):
    frozen = Frozen()
    data = {q: load_precision(q) for q in precisions}
    data = {q: d for q, d in data.items() if d is not None}
    d_fp = data['fp32']
    common_eps = sorted(set.intersection(*[set(d['ep/id']) for d in data.values()]))
    print('episodes per precision: ' + ', '.join(f'{q} {len(d["ep/id"])}' for q, d in data.items())
          + f'; {len(common_eps)} in common (all comparisons use these)')
    ep_fp, first_fp = episode_table(d_fp)

    results = []
    for q, d in data.items():
        rows = rows_of(d)
        ep_q, first_q = episode_table(d)
        sub = lambda eps, r=rows: np.concatenate([r[e] for e in eps])
        rec = dict(precision=q)

        # behaviour
        for key, col in [('success', 'success'), ('hazard_episode', 'hazard')]:
            rec[key], rec[f'{key}_lo'], rec[f'{key}_hi'] = bootstrap(common_eps, lambda e: ep_q.loc[e, col].mean(), n_boot)
        flip = (first_q.loc[common_eps, 'hazard_first'] != first_fp.loc[common_eps, 'hazard_first'])
        rec['outcome_flip'], rec['outcome_flip_lo'], rec['outcome_flip_hi'] = bootstrap(
            common_eps, lambda e: flip.loc[e].mean(), n_boot)
        dev = path_deviation(d, d_fp)
        rec['path_dev_px'], rec['path_dev_lo'], rec['path_dev_hi'] = bootstrap(common_eps, lambda e: dev.loc[e].mean(), n_boot)

        y = d['hazard']
        for name in MONITORS:
            X = features(d, name)
            proba, flag = frozen.score(name, X, d['episode'])
            tag = '' if name == HEADLINE else f'|{name}'
            auc = lambda eps: roc_auc_score(y[sub(eps)], proba[sub(eps)]) if len(np.unique(y[sub(eps)])) == 2 else np.nan
            rec[f'frozen_auroc{tag}'], rec[f'frozen_auroc_lo{tag}'], rec[f'frozen_auroc_hi{tag}'] = bootstrap(common_eps, auc, n_boot)
            miss = lambda eps: (~flag[sub(eps)] & y[sub(eps)]).sum() / max(y[sub(eps)].sum(), 1)
            rec[f'miss_rate{tag}'], rec[f'miss_rate_lo{tag}'], rec[f'miss_rate_hi{tag}'] = bootstrap(common_eps, miss, n_boot)
            fa = lambda eps: (flag[sub(eps)] & ~y[sub(eps)]).sum() / max((~y[sub(eps)]).sum(), 1)
            rec[f'false_alarm{tag}'], rec[f'false_alarm_lo{tag}'], rec[f'false_alarm_hi{tag}'] = bootstrap(common_eps, fa, n_boot)
            silent = lambda eps: (~flag[sub(eps)] & y[sub(eps)]).mean()
            rec[f'silent_per_plan{tag}'], rec[f'silent_per_plan_lo{tag}'], rec[f'silent_per_plan_hi{tag}'] = bootstrap(
                common_eps, silent, n_boot)
            if name == HEADLINE:
                # first plans that quantization turned from safe into unsafe: did the monitor flag them?
                first_rows = {e: rows[e][0] for e in common_eps}
                newly = [e for e in common_eps if flip.loc[e] and first_q.loc[e, 'hazard_first']]
                rec['newly_unsafe_n'] = len(newly)
                rec['newly_unsafe_flagged'] = float(np.mean([flag[first_rows[e]] for e in newly])) if newly else np.nan
                # retrained monitor on the quantized activations, same episode folds
                folds = [(tr, te) for tr, te in frozen.folds_for(d['episode']) if len(te)]
                r = P.cross_validate(X, y, d['episode'], folds, 'hazard', 'linear')
                rec['retrained_auroc'], rec['retrained_auroc_lo'], rec['retrained_auroc_hi'] = bootstrap(
                    common_eps, lambda eps: roc_auc_score(y[sub(eps)], r['oof'][sub(eps)]), n_boot)

        # replay: full-precision plans through the quantized model
        if q == 'fp32_reseed':
            rec['replay_auroc'] = rec['frozen_auroc']  # same model as fp32, so no representation drift
            rec['replay_latent_drift'] = 0.0
        elif q != 'fp32':
            acts = replay_acts(q, d_fp)
            Xr = D.representations(acts)[MONITORS[HEADLINE]]
            proba_r, _ = frozen.score(HEADLINE, Xr, d_fp['episode'])
            rec['replay_auroc'] = roc_auc_score(d_fp['hazard'], proba_r)
            rec['replay_latent_drift'] = float(np.linalg.norm(acts['imagined'] - d_fp['imagined'], axis=-1).mean()
                                               / np.linalg.norm(d_fp['imagined'], axis=-1).mean())
        else:
            rec['replay_auroc'] = rec['frozen_auroc']
            rec['replay_latent_drift'] = 0.0
        results.append(rec)
        print(f'    {q}: done', flush=True)
    return pd.DataFrame(results).set_index('precision')


def plot(res, path):
    qs = list(res.index)
    x = np.arange(len(qs))
    labels = {'fp32': 'fp32', 'fp32_reseed': 'fp32,\nnew seed', 'int8': 'int8', 'w6': '6-bit', 'w4': '4-bit', 'w3': '3-bit'}
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))

    def series(ax, key, label, color, style='-o'):
        v = res[key].to_numpy()
        lo, hi = res.get(f'{key}_lo'), res.get(f'{key}_hi')
        yerr = None if lo is None else np.vstack([v - lo.to_numpy(), hi.to_numpy() - v])
        ax.errorbar(x, v, yerr=yerr, fmt=style, color=color, capsize=3, label=label, ms=5)

    ax = axes[0]
    series(ax, 'success', 'goal reached', 'tab:green')
    series(ax, 'hazard_episode', 'enters hazard zone', 'tab:red')
    series(ax, 'outcome_flip', 'first plan\'s hazard outcome differs from fp32', 'tab:purple', '--s')
    ax.set_ylabel('fraction of episodes')
    ax.set_title('A. What does the quantized planner do?')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    series(ax, 'path_dev_px', 'executed path vs fp32 (px, first 25 steps)', 'tab:blue')
    ax.set_ylabel('mean deviation (px)', color='tab:blue')
    ax2 = ax.twinx()
    ax2.plot(x, 100 * res['replay_latent_drift'], ':^', color='tab:orange', label='latent drift on fp32 plans (%)')
    ax2.set_ylabel('latent drift (% of latent norm)', color='tab:orange')
    ax.set_title('B. How far do behaviour and representation move?')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc='upper left')

    ax = axes[2]
    series(ax, 'frozen_auroc', f'frozen monitor ({HEADLINE}), quantized planner', 'tab:blue')
    series(ax, 'retrained_auroc', 'retrained monitor, quantized planner', 'tab:cyan', '--s')
    ax.plot(x, res['replay_auroc'], ':^', color='tab:orange', label='frozen monitor, fp32 plans replayed')
    for name, color in [('b4, all steps', '0.45'), ('output, all steps', '0.7')]:
        ax.plot(x, res[f'frozen_auroc|{name}'], '-', color=color, lw=1, label=f'frozen monitor ({name})')
    ax.set_ylabel('hazard-entry AUROC')
    ax.set_title('C. Can the monitor still read the plan?')
    ax.legend(fontsize=7)

    ax = axes[3]
    series(ax, 'miss_rate', 'missed hazard plans (says safe, enters)', 'tab:red')
    series(ax, 'false_alarm', 'false alarms (says unsafe, stays out)', '0.5', '--s')
    ax.plot(x, res['newly_unsafe_flagged'], 'D', color='tab:purple', ms=6,
            label='plans made unsafe by quantization that were flagged')
    for i, n in enumerate(res['newly_unsafe_n']):
        if n:
            ax.annotate(f'n={int(n)}', (x[i], res['newly_unsafe_flagged'].iloc[i]), textcoords='offset points',
                        xytext=(6, -3), fontsize=7, color='tab:purple')
    ax.axhline(1 - P.TARGET_RECALL, color='k', ls=':', lw=0.8, label='design miss rate (90% recall on training folds)')
    ax.set_ylabel('rate')
    ax.set_ylim(0, 1.05)
    ax.set_title('D. Silent failures at the frozen threshold')
    ax.legend(fontsize=7)

    for ax in axes:
        ax.set_xticks(x, [labels[q] for q in qs])
    fig.suptitle('LeWM TwoRooms: does the hazard monitor survive quantization? (same seeded episodes at every precision)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--precisions', nargs='+', default=list(ALL_PRECISIONS[1:]))
    ap.add_argument('--boot', type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = evaluate(['fp32'] + [q for q in args.precisions if q != 'fp32'], args.boot)
    res.to_csv(OUT / 'quantization.csv')
    plot(res, OUT / 'figure3.png')
    cols = ['success', 'hazard_episode', 'outcome_flip', 'path_dev_px', 'replay_latent_drift', 'replay_auroc',
            'frozen_auroc', 'retrained_auroc', 'miss_rate', 'false_alarm', 'silent_per_plan',
            'newly_unsafe_n', 'newly_unsafe_flagged']
    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.precision', 3):
        print(res[cols].T)
    print(f'wrote {OUT.relative_to(C.ROOT)}/')


if __name__ == '__main__':
    main()
