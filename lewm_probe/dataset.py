"""Load collected plans and derive probe targets and baseline features."""

import json
from pathlib import Path

import numpy as np

from . import common as C
from .hazard import Zone, segment_distance

PLAN_STEPS = C.PLAN['horizon'] * C.PLAN['action_block']
BLOCK = C.PLAN['action_block']
HORIZONS = tuple(BLOCK * k for k in range(1, C.PLAN['horizon'] + 1))  # 5, 10, ..., 25: one per imagined step
LAYERS = [f'predictor.block{i}' for i in range(6)] + ['predictor.norm', 'pred_proj']


def load(paths=None):
    """Concatenate collection shards into one dict of per-plan arrays (plus 'ep/' and 'meta/')."""
    paths = sorted(paths or (C.ROOT / 'data' / 'full').glob('episodes_*.npz'))
    parts = [dict(np.load(p)) for p in paths]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if not k.startswith('meta/')}
    d['meta/zone'] = parts[0]['meta/zone']
    ep_row = {e: i for i, e in enumerate(d['ep/id'])}
    d['path'] = d['ep/positions'][[ep_row[e] for e in d['episode']]]  # full episode path for each plan
    return d


def zone(d):
    cx, cy, r = d['meta/zone']
    return Zone((float(cx), float(cy)), float(r))


def displacement(d, h):
    """Executed position h steps after the plan starts, minus the position at plan start."""
    n = len(d['episode'])
    return d['path'][np.arange(n), d['step'] + h] - d['pos']


def raw_actions(d):
    """Chosen plan in env units, (n, 25, 2), clipped as the env clips them."""
    st = json.loads((C.ROOT / 'lewm_probe' / 'action_stats.json').read_text())
    n = len(d['episode'])
    a = d['plan'].reshape(n, PLAN_STEPS, 2) * np.array(st['std']) + np.array(st['mean'])
    return np.clip(a, -1.0, 1.0)


def intended_path(d, speed=5.0):
    """Where the plan's actions would take the agent with no walls, (n, 25, 2)."""
    return d['pos'][:, None] + np.cumsum(raw_actions(d) * speed, axis=1)


def geometry(d, with_goal=True):
    """Baseline features that know where the agent is (and where it is going), not the plan."""
    if not with_goal:
        return d['pos'].astype(np.float64)
    line = segment_distance(d['pos'], d['goal'], zone(d).center)
    return np.c_[d['pos'], d['goal'], line].astype(np.float64)


def plan_reference(d):
    """'Reading the plan directly': position, actions, and the action-integrated path at each block."""
    path = intended_path(d)[:, BLOCK - 1 :: BLOCK].reshape(len(d['episode']), -1)
    return np.c_[d['pos'], raw_actions(d).reshape(len(d['episode']), -1), path].astype(np.float64)


def representations(acts):
    """Latent feature sets to probe, keyed (name, step); step 'all' concatenates the 5 imagined steps.

    `acts` holds 'emb', 'goal_emb' and 'act/<layer>' arrays, either from the collected data or
    from a control model's replay of the same plans.
    """
    reps = {('encoder', '-'): acts['emb'], ('encoder+goal', '-'): np.c_[acts['emb'], acts['goal_emb']]}
    for layer in LAYERS:
        a = acts[f'act/{layer}']
        for k in range(a.shape[1]):
            reps[(layer, str(k + 1))] = a[:, k]
        reps[(layer, 'all')] = a.reshape(len(a), -1)
    return reps
