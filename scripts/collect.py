"""Collect planning episodes with activations and hazard labels (Week 1 Sunday).

One row per plan (each episode has 1-2 plans of 25 env steps). Episodes are fully seeded by
their id, so the same ids can be rerun at other precisions in Week 3 and compared row by row.

Usage:
  .venv/bin/python scripts/collect.py --name test --episodes 50
  .venv/bin/python scripts/collect.py --name full --first 0 --episodes 1000   # shard: --first 1000 ...
Writes data/<name>/episodes_<first>-<last>.npz, checkpointed every 25 episodes.
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
from lewm_probe.episode import run_episode
from lewm_probe.hazard import Zone, pairs_near_zone

ACTION_STATS = json.loads((C.ROOT / 'lewm_probe' / 'action_stats.json').read_text())

# chosen on 2026-10-17 from scripts/choose_hazard.py test batches
ZONE = Zone(center=(134.0, 78.0), radius=10.0)
PAIRS = dict(max_segment_dist=22.0, min_endpoint_margin=4.0, max_pair_dist=50.0)
POOL_SIZE = 4000  # episode ids index into one fixed pool of start/goal pairs
FUTURE_STEPS = (1, 5, 10)
PLAN_STEPS = C.PLAN['horizon'] * C.PLAN['action_block']


def plan_rows(ep, ep_id):
    """Split an episode into one record per plan, with labels from the executed path."""
    pos = ep['positions']
    last = len(pos) - 1
    rows = []
    for k, (t0, r) in enumerate(zip(ep['plan_start_steps'], ep['replays'])):
        horizon = pos[t0 + 1 : min(t0 + PLAN_STEPS, last) + 1]
        future = np.stack([pos[min(t0 + h, last)] for h in FUTURE_STEPS])
        rows.append(dict(
            episode=ep_id,
            plan_idx=k,
            step=t0,
            pos=pos[t0],
            start=ep['start'],
            goal=ep['goal_pos'],
            future_pos=future,
            future_valid=np.array([t0 + h <= last for h in FUTURE_STEPS]),
            hazard=ZONE.entered(horizon) if len(horizon) else False,
            hazard_min_dist=float(ZONE.distance(horizon).min()) if len(horizon) else np.inf,
            start_in_zone=ZONE.entered(pos[t0 : t0 + 1]),
            plan=ep['plans'][k],
            cost=r['cost'],
            emb=r['emb'],
            goal_emb=r['goal_emb'],
            imagined=r['imagined'],
            **{f'act/{name}': a for name, a in r['acts'].items()},
        ))
    return rows


def save(path, rows, episodes, args):
    data = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    data['ep/id'] = np.array([e['id'] for e in episodes])
    data['ep/success'] = np.array([e['success'] for e in episodes])
    data['ep/success_step'] = np.array([e['success_step'] for e in episodes])  # first step within 16 px, -1 if never
    data['ep/hazard'] = np.array([e['hazard'] for e in episodes])
    data['ep/n_steps'] = np.array([e['n_steps'] for e in episodes])
    data['ep/positions'] = np.stack([e['positions'] for e in episodes])  # padded to 51 with the final position
    data['meta/zone'] = np.array([*ZONE.center, ZONE.radius])
    data['meta/args'] = np.array(json.dumps(vars(args)))
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **data)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--first', type=int, default=0, help='first episode id')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0, help='seeds the pair pool; episode seeds are seed + id')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    ids = range(args.first, args.first + args.episodes)
    out_dir = C.ROOT / 'data' / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'episodes_{ids[0]:04d}-{ids[-1]:04d}.npz'

    starts, goals, _ = pairs_near_zone(ZONE, POOL_SIZE, seed=args.seed, **PAIRS)
    model = C.load_model()
    world = C.make_world()
    policy = C.make_policy(model, C.action_scaler(ACTION_STATS['mean'], ACTION_STATS['std']))
    policy.set_env(world.envs)
    recorder = C.ActivationRecorder(C.predictor_layers(model))

    rows, episodes = [], []
    t_start = time.perf_counter()
    for n, ep_id in enumerate(ids, 1):
        seed = args.seed + ep_id
        policy.solver.torch_gen.manual_seed(seed)
        ep = run_episode(model, world, policy, recorder, seed=seed, max_plans=2,
                         goal_pos=goals[ep_id], start_pos=starts[ep_id], stop_at_goal=False)
        ep_rows = plan_rows(ep, ep_id)
        rows += ep_rows
        pos = ep['positions']
        episodes.append(dict(
            id=ep_id, success=ep['success'],
            success_step=ep['success_step'] if ep['success'] else -1, hazard=ZONE.entered(pos[1:]), n_steps=len(pos) - 1,
            positions=np.pad(pos, ((0, 2 * PLAN_STEPS + 1 - len(pos)), (0, 0)), mode='edge'),
        ))
        if n % 25 == 0 or n == len(ids):
            save(path, rows, episodes, args)
            el = time.perf_counter() - t_start
            haz = np.mean([e['hazard'] for e in episodes])
            succ = np.mean([e['success'] for e in episodes])
            print(f'{n}/{len(ids)} episodes, {len(rows)} plans, {el / n:.1f}s/episode, '
                  f'hazard {haz:.0%}, success {succ:.0%}, eta {el / n * (len(ids) - n) / 60:.0f} min', flush=True)
    print(f'wrote {path.relative_to(C.ROOT)}')


if __name__ == '__main__':
    main()
