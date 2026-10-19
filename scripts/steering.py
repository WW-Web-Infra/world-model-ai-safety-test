"""Week 4 (RQ4): run seeded episodes with a guided planner (see lewm_probe/steering.py).

Episodes are the same ids, starts, goals and seeds as the full-precision collection, so mode
'none' reproduces it exactly. Two evaluation sets: 'unsafe' (episodes whose full-precision first
plan entered the zone) and 'random' (any episode).

Usage: .venv/bin/python scripts/steering.py --mode probe --lam 30 [--episodes 120 --set unsafe]
Writes data/steering/<tag>/<mode>_<strength>_<set>.npz (resumable).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from lewm_probe import common as C
from lewm_probe import dataset as D
from lewm_probe.episode import run_episode
from lewm_probe.hazard import pairs_near_zone
from lewm_probe.steering import STEER_LAYER, GuidedCost
from scripts.collect import ACTION_STATS, PAIRS, POOL_SIZE, ZONE
from scripts.figure3 import Frozen

OUT = C.ROOT / 'data' / 'steering'
PROBE_KEY = ('trained', 'predictor.block2', 'all', 'hazard', 'linear')
STEER_KEY = ('trained', f'predictor.block{STEER_LAYER}', 'all', 'hazard', 'linear')


def evaluation_ids(which, n, d_fp):
    first = (d_fp['plan_idx'] == 0)
    if which == 'unsafe':
        pool = d_fp['episode'][first & d_fp['hazard']]
        rng = np.random.default_rng(0)
    else:
        pool = d_fp['ep/id']
        rng = np.random.default_rng(1)
    return np.sort(rng.choice(pool, size=n, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['none', 'probe', 'random', 'geometry', 'steer', 'steer_random'], required=True)
    ap.add_argument('--lam', type=float, default=0.0, help='penalty weight (probe, random, geometry)')
    ap.add_argument('--alpha', type=float, default=0.0, help='steering scale in projection std units (steer modes)')
    ap.add_argument('--set', choices=['unsafe', 'random'], default='unsafe')
    ap.add_argument('--episodes', type=int, default=120)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--tag', default='main', help='subfolder of data/steering (keeps pilots apart from the main run)')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    strength = args.alpha if args.mode.startswith('steer') else args.lam
    path = out / f'{args.mode}_{strength:g}_{args.set}.npz'

    d_fp = D.load()
    ids = evaluation_ids(args.set, args.episodes, d_fp)
    starts, goals, _ = pairs_near_zone(ZONE, POOL_SIZE, seed=0, **PAIRS)
    frozen = Frozen()
    from joblib import load
    probes = load(C.ROOT / 'outputs' / 'figure1' / 'frozen_probes.joblib')['probes']
    if args.mode.startswith('steer'):
        acts = d_fp[f'act/predictor.block{STEER_LAYER}']  # (n, 5, D), for the projection std

    model = C.load_model()
    guided = GuidedCost(model, ZONE.center, ZONE.radius)
    guided.configure(args.mode, lam=args.lam, alpha=args.alpha)
    world = C.make_world()
    policy = C.make_policy(guided, C.action_scaler(ACTION_STATS['mean'], ACTION_STATS['std']))
    policy.set_env(world.envs)
    recorder = C.ActivationRecorder(C.predictor_layers(model))

    done = {}
    if path.exists():
        prev = dict(np.load(path))
        done = {int(e): i for i, e in enumerate(prev['id'])}
        rows = [{k: prev[k][i] for k in prev if k != 'meta'} for i in range(len(prev['id']))]
        print(f'resuming: {len(done)} episodes already in {path.name}', flush=True)
    else:
        rows = []
    t0, n_new = time.perf_counter(), 0
    for n, e in enumerate(ids, 1):
        if int(e) in done:
            continue
        fold = frozen.fold_of[int(e)]
        if args.mode in ('probe', 'random'):
            guided.set_probe(probes[PROBE_KEY]['models'][fold], random_seed=1234 if args.mode == 'random' else None)
        if args.mode.startswith('steer'):
            pipe = probes[STEER_KEY]['models'][fold]
            scale, coef = pipe[0].scale_, pipe[-1].coef_[0]
            dirs = (coef / scale).reshape(5, -1)
            dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
            if args.mode == 'steer_random':
                r = np.random.default_rng(1234).standard_normal(dirs.shape)
                dirs = r / np.linalg.norm(r, axis=1, keepdims=True)
            proj_std = np.array([(acts[:, k] @ dirs[k]).std() for k in range(5)])
            guided.set_direction(dirs, proj_std)
        seed = int(e)
        policy.solver.torch_gen.manual_seed(seed)
        ep = run_episode(model, world, policy, recorder, seed=seed, max_plans=2,
                         goal_pos=goals[e], start_pos=starts[e], stop_at_goal=False)
        pos = ep['positions']
        rows.append(dict(id=int(e), success=ep['success'], success_step=ep['success_step'] if ep['success'] else -1,
                         hazard=ZONE.entered(pos[1:]), hazard_first=ZONE.entered(pos[1:26]),
                         min_dist=float(ZONE.distance(pos[1:]).min()), n_steps=len(pos) - 1,
                         positions=np.pad(pos, ((0, 51 - len(pos)), (0, 0)), mode='edge'),
                         plan0=ep['plans'][0]))
        n_new += 1
        if n_new % 10 == 0 or n == len(ids):
            data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
            data['meta'] = np.array(json.dumps(vars(args)))
            tmp = path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, **data)
            tmp.replace(path)
            el = time.perf_counter() - t0
            print(f'{n}/{len(ids)} episodes, {el / n_new:.1f}s/episode, hazard {np.mean([r["hazard"] for r in rows]):.0%}, '
                  f'success {np.mean([r["success"] for r in rows]):.0%}', flush=True)
    print(f'wrote {path.relative_to(C.ROOT)}')


if __name__ == '__main__':
    main()
