"""Build minimal safe/unsafe plan pairs for activation patching (Figure 2).

For collected first plans whose start-goal line passes 7-22 px from the zone centre (where
geometry does not decide the route), re-run CEM from the same start and goal with new seeds
until a plan with the opposite hazard outcome appears. Both plans must reach the goal. The pair
then differs only in the plan: same start frame, same goal frame.

Usage: .venv/bin/python scripts/make_pairs.py --first 0 --count 250   # shard over candidate rows
Writes data/pairs/pairs_<first>.npz.
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
from lewm_probe.episode import goal_image
from lewm_probe.hazard import segment_distance

OUT = C.ROOT / 'data' / 'pairs'
BAND = (7.0, 22.0)
SEEDS = (10_001, 10_002, 10_003)
SUCCESS_PX = 16.0


def candidate_rows(d):
    line = segment_distance(d['pos'], d['goal'], D.zone(d).center)
    rows = np.nonzero((d['plan_idx'] == 0) & (line >= BAND[0]) & (line < BAND[1]))[0]
    return np.random.default_rng(0).permutation(rows)


class Planner:
    def __init__(self):
        st = json.loads((C.ROOT / 'lewm_probe' / 'action_stats.json').read_text())
        self.model = C.load_model()
        self.world = C.make_world()
        self.world.reset(seed=0)
        self.scaler = C.action_scaler(st['mean'], st['std'])
        self.policy = C.make_policy(self.model, self.scaler)
        self.policy.set_env(self.world.envs)
        self.env = self.world.envs.envs[0].unwrapped
        self.tf = C.img_transform()

    def info(self, start, goal):
        self.env._set_goal_state(goal)
        self.env._set_state(start)
        frame = torch.from_numpy(self.env.render().copy()).permute(2, 0, 1)
        goal_frame = torch.from_numpy(goal_image(self.env)[0, 0]).permute(2, 0, 1)
        return {'pixels': self.tf(frame)[None, None], 'goal': self.tf(goal_frame)[None, None],
                'action': torch.zeros(1, 1, 2)}

    def plan(self, start, goal, seed):
        self.policy.solver.torch_gen.manual_seed(seed)
        return self.policy.solver(self.info(start, goal))['actions'][0].numpy()

    def execute(self, start, goal, plan):
        """Executed path of a whole plan (25 steps) from start."""
        self.env._set_goal_state(goal)
        self.env._set_state(start)
        path = [np.asarray(start, dtype=np.float32)]
        for a in self.scaler.inverse_transform(plan.reshape(-1, 2)):
            self.world.envs.step(a[None].astype(np.float32))
            path.append(self.env.agent_position.numpy().copy())
        return np.array(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--count', type=int, default=250)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)

    d = D.load()
    zone = D.zone(d)
    rows = candidate_rows(d)[args.first : args.first + args.count]
    p = Planner()
    pairs, t0 = [], time.perf_counter()
    for n, r in enumerate(rows, 1):
        start, goal, orig = d['pos'][r], d['goal'][r], d['plan'][r]
        orig_path = p.execute(start, goal, orig)
        orig_ok = (np.linalg.norm(orig_path[1:] - goal, axis=1) < SUCCESS_PX).any()
        orig_haz = zone.entered(orig_path[1:])
        if orig_ok:
            for seed in SEEDS:
                alt = p.plan(start, goal, seed)
                alt_path = p.execute(start, goal, alt)
                alt_ok = (np.linalg.norm(alt_path[1:] - goal, axis=1) < SUCCESS_PX).any()
                if alt_ok and zone.entered(alt_path[1:]) != orig_haz:
                    unsafe, safe = (orig, alt) if orig_haz else (alt, orig)
                    up, sp = (orig_path, alt_path) if orig_haz else (alt_path, orig_path)
                    pairs.append(dict(row=r, episode=d['episode'][r], start=start, goal=goal, seed=seed,
                                      original_unsafe=orig_haz, plan_unsafe=unsafe, plan_safe=safe,
                                      path_unsafe=up, path_safe=sp,
                                      min_dist_unsafe=zone.distance(up[1:]).min(),
                                      min_dist_safe=zone.distance(sp[1:]).min()))
                    break
        if n % 10 == 0 or n == len(rows):
            if pairs:
                path = OUT / f'pairs_{args.first:04d}.npz'
                np.savez_compressed(path, **{k: np.array([q[k] for q in pairs]) for k in pairs[0]})
            el = time.perf_counter() - t0
            print(f'{n}/{len(rows)} rows, {len(pairs)} pairs, {el / n:.1f}s/row, '
                  f'eta {el / n * (len(rows) - n) / 60:.0f} min', flush=True)


if __name__ == '__main__':
    main()
