"""Figure 1 (RQ1): can the planner's intended path and hazard entry be read from its activations?

Probes every latent representation (encoder, predictor blocks, final norm, predicted latent) at
each imagined step for hazard entry (AUROC) and displacement at +5..+25 steps (R2; predictor
features are paired with the current-frame latent for this target), against:
  - geometry baselines that know position (+ goal) but not the plan, linear and boosted trees;
    trees on position + goal are the headline bar
  - the same probes on a randomly initialised and a weight-shuffled LeWM
  - shuffled labels (noise floor)
  - reading the plan's actions directly (reference line, not a bar to beat)
plus the decisive test: does adding the latent to flexible geometry improve prediction?

All splits are GroupKFold(5) by episode; probe regularisation is tuned in inner folds; the
headline layer is selected inside each training fold. Run after collect.py, validate_data.py
and replay_controls.py.
Usage: .venv/bin/python scripts/figure1.py [--jobs 8]
Writes outputs/figure1/.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lewm_probe import common as C
from lewm_probe import dataset as D
from lewm_probe import probes as P
from lewm_probe.hazard import segment_distance

OUT = C.ROOT / 'outputs' / 'figure1'
CONTROLS = ('random_init', 'shuffled')
LAYER_LABELS = {'encoder': 'enc', 'encoder+goal': 'enc+goal', 'predictor.norm': 'norm', 'pred_proj': 'out',
                **{f'predictor.block{i}': f'b{i}' for i in range(6)}}
BANDS = [(0, 7), (7, 14), (14, 22), (22, np.inf)]  # distance from the plan's start-goal line to the zone centre
N_PERMUTATIONS = 20


def latent_features(reps, layer, step, tname):
    """Imagined-step activations encode where the agent will be, not where it is, so for
    displacement targets they are paired with the current-frame encoder latent (which a monitor
    also has); hazard entry depends only on the absolute path."""
    X = reps[(layer, step)]
    if kind(tname) == 'disp' and not layer.startswith('encoder'):
        X = np.c_[reps[('encoder', '-')], X]
    return X


def target(d, name):
    return d['hazard'] if name == 'hazard' else D.displacement(d, int(name[1:]))


def kind(name):
    return 'hazard' if name == 'hazard' else 'disp'


def job_list(reps_by_model, baselines):
    jobs = []
    horizons = [f'+{h}' for h in D.HORIZONS]
    for model, reps in reps_by_model.items():
        probes = ('linear', 'mlp') if model == 'trained' else ('linear',)
        for (layer, step) in reps:
            for probe in probes:
                jobs.append((model, layer, step, 'hazard', probe))
                for t in horizons:
                    aligned = step in ('-', 'all') or int(step) * D.BLOCK == int(t[1:])
                    if aligned:
                        jobs.append((model, layer, step, t, probe))
    for name, (probes, _) in baselines.items():
        for probe in probes:
            for t in ['hazard'] + horizons:
                jobs.append(('baseline', name, '-', t, probe))
    return jobs


def run_job(job, X, y, groups, folds):
    model, layer, step, tname, probe = job
    keep = model == 'trained' and probe == 'linear'
    res = P.cross_validate(X, y, groups, folds, kind(tname), probe, keep_models=keep)
    if keep and tname == 'hazard':
        res['thresholds'] = [P.recall_threshold(m, X[tr], y[tr], groups[tr])
                             for m, (tr, _) in zip(res['models'], folds)]
    return job, res


def increment_job(layer, Xlat, geo, y, groups, folds, tname):
    """Boosted trees on geometry vs on geometry + the latent probe's (nested, out-of-fold) prediction."""
    k = kind(tname)
    nested = P.nested_inner_predictions(Xlat, y, groups, folds, k)
    with_latent = np.zeros(y.shape)
    geo_only = np.zeros(y.shape)
    for (tr, te), (ptr, pte) in zip(folds, nested):
        ptr, pte = ptr.reshape(len(tr), -1), pte.reshape(len(te), -1)
        m, _ = P.ESTIMATORS[(k, 'trees')](np.c_[geo[tr], ptr], y[tr], groups[tr])
        with_latent[te] = P.predict(m, np.c_[geo[te], pte], k)
        m, _ = P.ESTIMATORS[(k, 'trees')](geo[tr], y[tr], groups[tr])
        geo_only[te] = P.predict(m, geo[te], k)
    return layer, tname, geo_only, with_latent


def permutation_floor(X, y, groups, folds, k, seed):
    rng = np.random.default_rng(seed)
    yp = y[rng.permutation(len(y))]
    preds = np.zeros(y.shape)
    for tr, te in folds:
        est = LogisticRegression(C=1.0, max_iter=5000) if k == 'hazard' else Ridge(alpha=100.0)
        m = make_pipeline(StandardScaler(), est).fit(X[tr], yp[tr])
        preds[te] = m.predict_proba(X[te])[:, 1] if k == 'hazard' else m.predict(X[te])
    return np.mean([P.score(yp[te], preds[te], k) for _, te in folds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    d = D.load()
    groups = d['episode']
    folds = P.outer_folds(groups)
    n = len(groups)
    reps_by_model = {'trained': D.representations(d)}
    for m in CONTROLS:
        reps_by_model[m] = D.representations(dict(np.load(C.ROOT / 'data' / 'full' / f'replay_{m}.npz')))
    baselines = {
        'position': (('linear', 'trees'), D.geometry(d, with_goal=False)),
        'position+goal': (('linear', 'trees'), D.geometry(d)),
        'plan actions': (('trees',), D.plan_reference(d)),
    }
    targets = {t: target(d, t) for t in ['hazard'] + [f'+{h}' for h in D.HORIZONS]}
    print(f'{n} plans, {len(np.unique(groups))} episodes, hazard rate {d["hazard"].mean():.2f}')

    def features(job):
        model, layer, step, tname = job[:4]
        if model == 'baseline':
            return baselines[layer][1]
        return latent_features(reps_by_model[model], layer, step, tname)

    jobs = job_list(reps_by_model, baselines)
    print(f'running {len(jobs)} probe fits x {P.N_FOLDS} folds on {args.jobs} workers', flush=True)
    done = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(run_job)(j, features(j), targets[j[3]], groups, folds) for j in jobs)
    results = dict(done)
    print(f'    probes done ({time.perf_counter() - t0:.0f}s)', flush=True)

    # --- results table -----------------------------------------------------------------------
    rows = []
    for (model, layer, step, tname, probe), r in results.items():
        y = targets[tname]
        rows.append(dict(model=model, features=layer, step=step, target=tname, probe=probe,
                         mean=r['scores'].mean(), sd=r['scores'].std(),
                         pooled=P.score(y, r['oof'], kind(tname))))

    # --- nested layer selection (headline probe) ---------------------------------------------
    selected = {}
    for model in reps_by_model:
        for tname in targets:
            cands = [(j, r) for j, r in results.items() if j[0] == model and j[3] == tname
                     and j[4] == 'linear' and j[1] not in ('encoder', 'encoder+goal')]
            oof = np.zeros(targets[tname].shape)
            chosen, fold_scores = [], []
            for f, (_, te) in enumerate(folds):
                j, r = max(cands, key=lambda c: c[1]['inner'][f])
                oof[te] = r['oof'][te]
                chosen.append(f'{LAYER_LABELS[j[1]]}@{j[2]}')
                fold_scores.append(P.score(targets[tname][te], oof[te], kind(tname)))
            selected[(model, tname)] = oof
            rows.append(dict(model=model, features='selected layer', step=' '.join(chosen), target=tname,
                             probe='linear', mean=np.mean(fold_scores), sd=np.std(fold_scores),
                             pooled=P.score(targets[tname], oof, kind(tname))))

    # --- noise floor from shuffled labels ------------------------------------------------------
    floor_X = reps_by_model['trained'][('pred_proj', 'all')]
    for tname in ('hazard', '+10'):
        vals = Parallel(n_jobs=args.jobs)(
            delayed(permutation_floor)(floor_X, targets[tname], groups, folds, kind(tname), s)
            for s in range(N_PERMUTATIONS))
        rows.append(dict(model='baseline', features='shuffled labels', step='-', target=tname, probe='linear',
                         mean=np.mean(vals), sd=np.std(vals), pooled=np.percentile(vals, 95)))
    print(f'    noise floor done ({time.perf_counter() - t0:.0f}s)', flush=True)

    # --- does the latent add to flexible geometry? ---------------------------------------------
    geo = D.geometry(d)
    inc_reps = [('encoder+goal', '-')] + [(l, 'all') for l in D.LAYERS]
    inc = Parallel(n_jobs=args.jobs)(
        delayed(increment_job)(layer, latent_features(reps_by_model['trained'], layer, step, t), geo, targets[t],
                               groups, folds, t)
        for layer, step in inc_reps for t in ['hazard', '+5', '+10', '+15'])
    line = segment_distance(d['pos'], d['goal'], D.zone(d).center)
    inc_rows = []
    for layer, tname, geo_only, with_lat in inc:
        y, k = targets[tname], kind(tname)
        diff = lambda rows_, a=with_lat, b=geo_only: P.score(y[rows_], a[rows_], k) - P.score(y[rows_], b[rows_], k)
        lo, hi = P.episode_bootstrap(groups, diff, n=500)
        row = dict(features=layer, target=tname, geometry=P.score(y, geo_only, k),
                   geometry_plus_latent=P.score(y, with_lat, k), gain=diff(np.arange(n)), ci_lo=lo, ci_hi=hi)
        if tname == 'hazard':
            for a, b in BANDS:
                m = (line >= a) & (line < b)
                if len(np.unique(y[m])) == 2:
                    row[f'band_{a}-{b}_n'] = int(m.sum())
                    row[f'band_{a}-{b}_geometry'] = roc_auc_score(y[m], geo_only[m])
                    row[f'band_{a}-{b}_plus_latent'] = roc_auc_score(y[m], with_lat[m])
        inc_rows.append(row)
    print(f'    increments done ({time.perf_counter() - t0:.0f}s)', flush=True)

    table = pd.DataFrame(rows)
    increments = pd.DataFrame(inc_rows)
    table.to_csv(OUT / 'results.csv', index=False)
    increments.to_csv(OUT / 'increments.csv', index=False)
    np.savez_compressed(OUT / 'oof.npz', folds=np.array([np.isin(np.arange(n), te) for _, te in folds]),
                        **{'|'.join(j): r['oof'] for j, r in results.items()},
                        **{f'selected|{m}|{t}': v for (m, t), v in selected.items()})
    frozen = {j: dict(models=r['models'], thresholds=r.get('thresholds')) for j, r in results.items() if r['models']}
    joblib.dump(dict(probes=frozen, folds=folds, episodes=groups), OUT / 'frozen_probes.joblib', compress=3)
    plot(table, increments, OUT / 'figure1.png')
    summarize(table, increments)
    print(f'wrote {OUT.relative_to(C.ROOT)}/ in {time.perf_counter() - t0:.0f}s')


def _get(table, **kw):
    m = np.ones(len(table), dtype=bool)
    for k, v in kw.items():
        m &= table[k] == v
    return table[m]


def plot(table, inc, path):
    layers = ['encoder', 'encoder+goal'] + D.LAYERS
    xs = np.arange(len(layers))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # A: hazard AUROC by representation
    ax = axes[0]
    def band(features, probe, color, label):
        r = _get(table, model='baseline', features=features, target='hazard', probe=probe).iloc[0]
        ax.axhspan(r['mean'] - r['sd'], r['mean'] + r['sd'], color=color, alpha=0.18, lw=0)
        ax.axhline(r['mean'], color=color, lw=1.2, label=label)
    band('position+goal', 'trees', 'tab:red', 'position+goal, trees (bar)')
    band('position+goal', 'linear', 'tab:orange', 'position+goal, linear')
    band('position', 'trees', 'tab:brown', 'position, trees')
    r = _get(table, model='baseline', features='plan actions', target='hazard').iloc[0]
    ax.axhline(r['mean'], color='k', ls=':', lw=1.2, label='plan actions (reference)')
    r = _get(table, model='baseline', features='shuffled labels', target='hazard').iloc[0]
    ax.axhline(r['pooled'], color='0.5', ls='-.', lw=1, label='shuffled labels (95th pct)')
    for step, color in zip(['1', '2', '3', '4', '5'], plt.cm.Blues(np.linspace(0.35, 0.85, 5))):
        vals = [_get(table, model='trained', features=l, step=step, target='hazard', probe='linear')['mean']
                for l in D.LAYERS]
        ax.plot(xs[2:], [v.iloc[0] for v in vals], '-', color=color, lw=1, label=f'imagined step {step}')
    for model, style, label in [('trained', '-o', 'all steps, trained'), ('random_init', '--s', 'all steps, random init'),
                                ('shuffled', '--^', 'all steps, shuffled weights')]:
        vals = []
        for l in layers:
            step = '-' if l.startswith('encoder') else 'all'
            vals.append(_get(table, model=model, features=l, step=step, target='hazard', probe='linear')['mean'].iloc[0])
        ax.plot(xs, vals, style, color='tab:blue' if model == 'trained' else '0.4', lw=2 if model == 'trained' else 1,
                ms=4, label=label)
    vals = [_get(table, model='trained', features=l, step='-' if l.startswith('encoder') else 'all',
                 target='hazard', probe='mlp')['mean'].iloc[0] for l in layers]
    ax.plot(xs, vals, ':o', color='tab:purple', ms=3, label='all steps, trained, MLP')
    ax.set_xticks(xs, [LAYER_LABELS[l] for l in layers])
    ax.set_ylabel('hazard-entry AUROC (5-fold mean)')
    ax.set_title('A. Will this plan enter the hazard zone?')
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=6.5, loc='lower right', ncol=2)

    # B: displacement R2 by horizon
    ax = axes[1]
    hs = [f'+{h}' for h in D.HORIZONS]
    hx = np.array(D.HORIZONS)
    def line(sel, style, color, label, lw=1.5):
        vals = [sel(t) for t in hs]
        ax.plot(hx, [v['mean'].iloc[0] for v in vals], style, color=color, lw=lw, label=label, ms=4)
    for model, color in [('trained', 'tab:blue'), ('random_init', '0.55'), ('shuffled', '0.3')]:
        line(lambda t, m=model: _get(table, model=m, features='selected layer', target=t),
             '-o' if model == 'trained' else '--s', color, f'latent probe, {model.replace("_", " ")} (layer chosen in-fold)',
             lw=2 if model == 'trained' else 1)
    line(lambda t: _get(table, model='trained', features='pred_proj', step=str(int(t[1:]) // D.BLOCK), target=t, probe='linear'),
         '-', 'tab:cyan', 'predicted latent at matching imagined step')
    ax.text(0.02, 0.03, 'latent probes also see the current-frame latent', transform=ax.transAxes, fontsize=7,
            va='bottom', color='0.3')
    line(lambda t: _get(table, model='baseline', features='position+goal', target=t, probe='trees'), '-', 'tab:red',
         'position+goal, trees (bar)')
    line(lambda t: _get(table, model='baseline', features='position', target=t, probe='trees'), '-', 'tab:brown',
         'position, trees')
    line(lambda t: _get(table, model='baseline', features='plan actions', target=t, probe='trees'), ':', 'k',
         'plan actions (reference)')
    r = _get(table, model='baseline', features='shuffled labels', target='+10').iloc[0]
    ax.axhline(r['pooled'], color='0.5', ls='-.', lw=1, label='shuffled labels (95th pct, +10)')
    ax.set_xticks(hx)
    ax.set_xlabel('steps ahead (one imagined step = 5 env steps)')
    ax.set_ylabel('displacement R² (5-fold mean)')
    ax.set_title('B. Where will the agent be?')
    ax.set_ylim(-0.3, 1.0)
    ax.legend(fontsize=6.5, loc='lower right')

    # C: gain from adding the latent to flexible geometry
    ax = axes[2]
    inc_layers = ['encoder+goal'] + D.LAYERS
    for i, (t, color) in enumerate(zip(['hazard', '+5', '+10', '+15'], ['tab:red', 'tab:blue', 'tab:cyan', 'tab:green'])):
        rows = [inc[(inc['features'] == l) & (inc['target'] == t)].iloc[0] for l in inc_layers]
        g = np.array([r['gain'] for r in rows])
        err = np.array([[r['gain'] - r['ci_lo'], r['ci_hi'] - r['gain']] for r in rows]).T
        ax.errorbar(np.arange(len(inc_layers)) + (i - 1.5) * 0.18, g, yerr=err, fmt='o', ms=3.5, color=color,
                    capsize=2, label='hazard AUROC' if t == 'hazard' else f'displacement R² {t}')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(np.arange(len(inc_layers)), [LAYER_LABELS[l] + (' (all)' if l in D.LAYERS else '') for l in inc_layers],
                  rotation=30, fontsize=8)
    ax.set_ylabel('gain over position+goal trees (95% CI, episodes resampled)')
    ax.set_title('C. Does the latent add anything beyond geometry?')
    ax.legend(fontsize=7)

    fig.suptitle('LeWM TwoRooms: reading the planner\'s intent from its activations (2,308 plans, 2,000 episodes)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize(table, inc):
    def show(**kw):
        r = _get(table, **kw)
        return f'{r["mean"].iloc[0]:.3f} ± {r["sd"].iloc[0]:.3f}' if len(r) else 'n/a'
    print('\nhazard AUROC (5-fold mean ± sd)')
    for label, kw in [('selected latent layer (trained)', dict(model='trained', features='selected layer')),
                      ('selected latent layer (random init)', dict(model='random_init', features='selected layer')),
                      ('selected latent layer (shuffled weights)', dict(model='shuffled', features='selected layer')),
                      ('encoder+goal latent', dict(model='trained', features='encoder+goal', probe='linear')),
                      ('position+goal, trees (bar)', dict(model='baseline', features='position+goal', probe='trees')),
                      ('position+goal, linear', dict(model='baseline', features='position+goal', probe='linear')),
                      ('plan actions (reference)', dict(model='baseline', features='plan actions'))]:
        print(f'    {label:42s} {show(target="hazard", **kw)}')
    sel = _get(table, model='trained', features='selected layer', target='hazard')
    print(f'    layers chosen per fold: {sel["step"].iloc[0]}')
    print('\ndisplacement R2 by horizon:      ' + '  '.join(f'+{h:<6d}' for h in D.HORIZONS))
    for label, kw in [('selected latent (trained)', dict(model='trained', features='selected layer')),
                      ('selected latent (random init)', dict(model='random_init', features='selected layer')),
                      ('selected latent (shuffled)', dict(model='shuffled', features='selected layer')),
                      ('position+goal, trees', dict(model='baseline', features='position+goal', probe='trees')),
                      ('plan actions', dict(model='baseline', features='plan actions'))]:
        vals = [_get(table, target=f'+{h}', **kw)['mean'] for h in D.HORIZONS]
        print(f'    {label:30s} ' + '  '.join(f'{v.iloc[0]:6.3f}' for v in vals))
    print('\ngain from adding the latent to position+goal trees (95% CI)')
    for _, r in inc.iterrows():
        print(f'    {r["target"]:7s} {r["features"]:18s} {r["geometry"]:.3f} -> {r["geometry_plus_latent"]:.3f}  '
              f'gain {r["gain"]:+.3f} [{r["ci_lo"]:+.3f}, {r["ci_hi"]:+.3f}]')


if __name__ == '__main__':
    main()
