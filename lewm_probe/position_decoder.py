"""Decode agent position from a LeWM latent, trained only on real rendered frames.

Independent of every hazard label and probe: it is fitted on encoder latents of frames rendered
at random positions, then applied to the predictor's imagined latents to draw the imagined path.
"""

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import common as C
from .frozen import load_decoder, save_decoder

PATH = C.ROOT / 'data' / 'pairs' / 'position_decoder.npz'
RADIUS = 7.0
ARENA = (14.0 + RADIUS, 209.0 - RADIUS)
WALL_X = (107.0 - RADIUS, 117.0 + RADIUS)
DOOR_Y = (35.0, 63.0)


def random_positions(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        x, y = rng.uniform(*ARENA, size=2)
        if WALL_X[0] < x < WALL_X[1] and not DOOR_Y[0] < y < DOOR_Y[1]:
            continue
        out.append((x, y))
    return np.array(out, dtype=np.float32)


def encode_positions(model, env, positions, batch=64):
    tf = C.img_transform()
    latents = []
    for i in range(0, len(positions), batch):
        frames = []
        for p in positions[i : i + batch]:
            env._set_state(p)
            frames.append(tf(torch.from_numpy(env.render().copy()).permute(2, 0, 1)))
        with torch.inference_mode():
            latents.append(model.encode({'pixels': torch.stack(frames)[:, None]})['emb'][:, 0].numpy())
    return np.concatenate(latents)


def train(model, env, n_train=4000, n_test=1000):
    pos = random_positions(n_train + n_test, seed=0)
    z = encode_positions(model, env, pos)
    tr, te = slice(0, n_train), slice(n_train, None)
    candidates = {
        'ridge': make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 7))),
        'mlp': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 256), alpha=1e-3,
                                                            max_iter=1000, random_state=0)),
    }
    errs = {}
    for name, m in candidates.items():
        m.fit(z[tr], pos[tr])
        errs[name] = float(np.linalg.norm(m.predict(z[te]) - pos[te], axis=1).mean())
    best = min(errs, key=errs.get)
    save_decoder(PATH, candidates[best], best, errs)
    return candidates[best], errs


def load():
    return load_decoder(PATH)
