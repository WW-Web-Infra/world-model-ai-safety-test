"""Paired comparison of each quantized model with full precision, for stating what "unchanged" means.

Each bootstrap sample draws the same episodes for the quantized model and for full precision, so
the interval is for the difference itself. A metric is called unchanged at a precision when the
95% interval of the difference lies inside +/- the bound reported here.

Usage: .venv/bin/python scripts/quant_equivalence.py
Writes outputs/figure3/quant_paired.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lewm_probe import common as C
from scripts.figure3 import HEADLINE, Frozen, episode_table, features, load_precision, rows_of

OUT = C.ROOT / 'outputs' / 'figure3'
CONDITIONS = ['fp32_reseed', 'int8', 'w6', 'w4', 'w3']
N_BOOT = 2000


def per_condition(d, frozen):
    rows = rows_of(d)
    proba, flag = frozen.score(HEADLINE, features(d, HEADLINE), d['episode'])
    ep, _ = episode_table(d)
    return dict(rows=rows, proba=proba, flag=flag, y=d['hazard'], success=ep['success'])


def metrics(c, eps):
    idx = np.concatenate([c['rows'][e] for e in eps])
    y, flag = c['y'][idx], c['flag'][idx]
    return dict(miss_rate=(~flag & y).sum() / max(y.sum(), 1),
                false_alarm=(flag & ~y).sum() / max((~y).sum(), 1),
                frozen_auroc=roc_auc_score(y, c['proba'][idx]),
                success=c['success'].loc[eps].mean())


def main():
    frozen = Frozen()
    data = {q: load_precision(q) for q in ['fp32'] + CONDITIONS}
    common = sorted(set.intersection(*[set(d['ep/id']) for d in data.values()]))
    cond = {q: per_condition(d, frozen) for q, d in data.items()}
    rng = np.random.default_rng(0)
    boots = [np.asarray(common)[rng.integers(0, len(common), len(common))] for _ in range(N_BOOT)]
    base_point = metrics(cond['fp32'], common)
    base_boot = [metrics(cond['fp32'], b) for b in boots]
    out = []
    for q in CONDITIONS:
        point = metrics(cond[q], common)
        boot = [metrics(cond[q], b) for b in boots]
        for m in point:
            diffs = np.array([b[m] - a[m] for a, b in zip(base_boot, boot)])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            out.append(dict(condition=q, metric=m, value=point[m], fp32=base_point[m], diff=point[m] - base_point[m],
                            diff_lo=lo, diff_hi=hi, bound=max(abs(lo), abs(hi))))
    table = pd.DataFrame(out)
    table.to_csv(OUT / 'quant_paired.csv', index=False)
    with pd.option_context('display.width', 200, 'display.precision', 4):
        print(f'{len(common)} common episodes, {N_BOOT} paired bootstrap samples\n')
        print(table.to_string(index=False))
    print(f'wrote {(OUT / "quant_paired.csv").relative_to(C.ROOT)}')


if __name__ == '__main__':
    main()
