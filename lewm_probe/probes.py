"""Cross-validated probes. Every split is grouped by episode; all tuning happens inside training folds."""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_FOLDS = 5
N_INNER = 3
CS = np.logspace(-4, 2, 7)
ALPHAS = np.logspace(-2, 5, 8)
TARGET_RECALL = 0.9  # monitor threshold: catch 90% of hazard entries on training data


def outer_folds(groups):
    return list(GroupKFold(N_FOLDS).split(np.zeros(len(groups)), groups=groups))


def inner_folds(groups, n=N_INNER):
    return list(GroupKFold(n).split(np.zeros(len(groups)), groups=groups))


# --- estimators: fit(X, y, groups) -> (fitted model, inner-CV score or None) ---------------

def linear_classifier(X, y, groups):
    m = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=CS, cv=inner_folds(groups), scoring='roc_auc', max_iter=5000))
    m.fit(X, y)
    lr = m[-1]
    return m, float(lr.scores_[True].mean(axis=0).max())


def linear_regressor(X, Y, groups):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=inner_folds(groups)))
    m.fit(X, Y)
    return m, float(m[-1].best_score_)


def mlp_classifier(X, y, groups):
    m = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500, random_state=0))
    return m.fit(X, y), None


def mlp_regressor(X, Y, groups):
    m = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500, random_state=0)),
        transformer=StandardScaler())
    return m.fit(X, Y), None


def tree_classifier(X, y, groups):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(X, y), None


class _MultiTree:
    """One boosted-tree regressor per output column."""

    def __init__(self, X, Y):
        self.models = [HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)
                       .fit(X, Y[:, j]) for j in range(Y.shape[1])]

    def predict(self, X):
        return np.stack([m.predict(X) for m in self.models], axis=1)


def tree_regressor(X, Y, groups):
    return _MultiTree(X, Y), None


ESTIMATORS = {
    ('hazard', 'linear'): linear_classifier, ('hazard', 'mlp'): mlp_classifier, ('hazard', 'trees'): tree_classifier,
    ('disp', 'linear'): linear_regressor, ('disp', 'mlp'): mlp_regressor, ('disp', 'trees'): tree_regressor,
}


def predict(model, X, kind):
    return model.predict_proba(X)[:, 1] if kind == 'hazard' else model.predict(X)


def score(y, pred, kind):
    return roc_auc_score(y, pred) if kind == 'hazard' else r2_score(y, pred)


def cross_validate(X, y, groups, folds, kind, probe, keep_models=False):
    """Out-of-fold predictions, per-fold test scores, inner-CV scores (for nested layer selection)."""
    fit = ESTIMATORS[(kind, probe)]
    oof = np.zeros(y.shape, dtype=np.float64)
    test_scores, inner_scores, models = [], [], []
    for tr, te in folds:
        m, inner = fit(X[tr], y[tr], groups[tr])
        oof[te] = predict(m, X[te], kind)
        test_scores.append(score(y[te], oof[te], kind))
        inner_scores.append(inner)
        if keep_models:
            models.append(m)
    return dict(oof=oof, scores=np.array(test_scores), inner=inner_scores, models=models)


def recall_threshold(model, X, y, groups, recall=TARGET_RECALL):
    """Score threshold that flags `recall` of hazard plans, from inner out-of-fold predictions
    on the training fold (in-sample scores would be optimistic)."""
    C = model[-1].C_[0]
    scores = np.zeros(len(y))
    for tr, te in inner_folds(groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[tr], y[tr])
        scores[te] = m.predict_proba(X[te])[:, 1]
    return float(np.quantile(scores[y], 1 - recall))


def nested_inner_predictions(X, y, groups, folds, kind, probe='linear'):
    """For stacking: in each outer fold, out-of-fold predictions on the training rows (from inner
    folds) and predictions on the test rows (from a model fit on all training rows)."""
    fit = ESTIMATORS[(kind, probe)]
    out = []
    for tr, te in folds:
        train_pred = np.zeros(y[tr].shape, dtype=np.float64)
        for itr, ite in inner_folds(groups[tr], n=N_FOLDS):
            m, _ = fit(X[tr][itr], y[tr][itr], groups[tr][itr])
            train_pred[ite] = predict(m, X[tr][ite], kind)
        m, _ = fit(X[tr], y[tr], groups[tr])
        out.append((train_pred, predict(m, X[te], kind)))
    return out


def episode_bootstrap(groups, stat, n=1000, seed=0):
    """95% CI of stat(rows) over resampled episodes."""
    rng = np.random.default_rng(seed)
    eps = np.unique(groups)
    rows_of = {e: np.nonzero(groups == e)[0] for e in eps}
    vals = []
    for _ in range(n):
        rows = np.concatenate([rows_of[e] for e in rng.choice(eps, size=len(eps))])
        vals.append(stat(rows))
    return np.percentile(vals, [2.5, 97.5])
