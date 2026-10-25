"""Saved probes and the position decoder as plain arrays (.npz), never pickles.

Pickled scikit-learn objects (joblib) run code when loaded, so a swapped file could execute
anything. Every saved model here is a standard-scaler + linear map, or a small ReLU network, so
each is stored as its arrays and rebuilt with numpy; files are read with `allow_pickle=False`.
"""

import json

import numpy as np


class LinearProbe:
    """StandardScaler followed by a linear map: z = ((x - mean) / scale) @ coef.T + intercept."""

    def __init__(self, mean, scale, coef, intercept):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.coef, self.intercept = np.atleast_2d(coef), np.atleast_1d(intercept)

    def raw(self, X):
        return ((np.asarray(X) - self.mean) / self.scale) @ self.coef.T + self.intercept

    def decision_function(self, X):
        return self.raw(X)[:, 0]

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.stack([1 - p, p], axis=1)

    def predict(self, X):
        return self.raw(X)

    @classmethod
    def from_pipeline(cls, pipe):
        sc, est = pipe[0], pipe[-1]
        return cls(sc.mean_, sc.scale_, est.coef_, est.intercept_)


class MLPDecoder:
    """StandardScaler followed by a ReLU multilayer perceptron with an identity output."""

    def __init__(self, mean, scale, weights, biases):
        self.mean, self.scale, self.weights, self.biases = mean, scale, weights, biases

    def predict(self, X):
        h = (np.asarray(X) - self.mean) / self.scale
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
        return h


KEY_SEP = '|'


def save_probes(path, probes, folds, episodes):
    """probes: {(model, layer, step, target, probe): dict(models=[sklearn pipeline per fold], thresholds)}.
    Only linear probes are saved (the ones later experiments reuse)."""
    arrays = {'episodes': np.asarray(episodes),
              'test_masks': np.array([np.isin(np.arange(len(episodes)), te) for _, te in folds])}
    for key, entry in probes.items():
        if key[4] != 'linear' or not entry['models']:
            continue
        name = KEY_SEP.join(key)
        for k, pipe in enumerate(entry['models']):
            lp = LinearProbe.from_pipeline(pipe)
            for field in ('mean', 'scale', 'coef', 'intercept'):
                arrays[f'{name}{KEY_SEP}fold{k}{KEY_SEP}{field}'] = getattr(lp, field)
        if entry.get('thresholds') is not None:
            arrays[f'{name}{KEY_SEP}thresholds'] = np.asarray(entry['thresholds'], dtype=np.float64)
    np.savez_compressed(path, **arrays)


def load_probes(path):
    """Returns dict(probes={key: dict(models=[LinearProbe per fold], thresholds)}, folds, episodes)."""
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    masks = data.pop('test_masks')
    folds = [(np.nonzero(~m)[0], np.nonzero(m)[0]) for m in masks]
    episodes = data.pop('episodes')
    grouped = {}
    for name, value in data.items():
        parts = name.split(KEY_SEP)
        key = tuple(parts[:5])
        entry = grouped.setdefault(key, {'folds': {}, 'thresholds': None})
        if parts[5] == 'thresholds':
            entry['thresholds'] = list(value)
        else:
            entry['folds'].setdefault(int(parts[5][4:]), {})[parts[6]] = value
    probes = {key: dict(models=[LinearProbe(**e['folds'][k]) for k in sorted(e['folds'])], thresholds=e['thresholds'])
              for key, e in grouped.items()}
    return dict(probes=probes, folds=folds, episodes=episodes)


def save_decoder(path, pipe, name, test_error_px):
    sc, est = pipe[0], pipe[-1]
    arrays = {'mean': sc.mean_, 'scale': sc.scale_, 'meta': np.array(json.dumps({'name': name, 'test_error_px': test_error_px}))}
    if name == 'mlp':
        assert est.activation == 'relu' and est.out_activation_ == 'identity'
        for i, (W, b) in enumerate(zip(est.coefs_, est.intercepts_)):
            arrays[f'W{i}'], arrays[f'b{i}'] = W, b
    else:
        arrays['W0'], arrays['b0'] = est.coef_.T, est.intercept_
    np.savez_compressed(path, **arrays)


def load_decoder(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['meta']))
        n = sum(1 for k in z.files if k.startswith('W'))
        model = MLPDecoder(z['mean'], z['scale'], [z[f'W{i}'] for i in range(n)], [z[f'b{i}'] for i in range(n)])
    return dict(model=model, **meta)
