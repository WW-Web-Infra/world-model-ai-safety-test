"""Reproduce the upstream le-wm TwoRooms evaluation (eval.py + config/eval/tworoom.yaml) on CPU.

Start and goal are dataset frames 25 steps apart on an expert trajectory, the budget is
50 env steps, and actions are z-scored with the dataset's own statistics.

Needs .stable-wm/datasets/tworoom.h5 (tworoom.tar.zst from the quentinll/lewm-tworooms dataset repo).
Usage: .venv/bin/python scripts/upstream_eval.py [--num-eval 50] [--seed 42]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import stable_worldmodel as swm
from sklearn import preprocessing

from lewm_probe import common as C

GOAL_OFFSET = 25
EVAL_BUDGET = 50
CALLABLES = [
    {'method': '_set_state', 'args': {'state': {'value': 'proprio'}}},
    {'method': '_set_goal_state', 'args': {'goal_state': {'value': 'goal_proprio'}}},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-eval', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    dataset = swm.data.HDF5Dataset('tworoom', keys_to_cache=['action', 'proprio'])
    col = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    ep_col = dataset.get_col_data(col)
    step_col = dataset.get_col_data('step_idx')
    print(f'dataset: {len(step_col)} steps, {len(np.unique(ep_col))} episodes, columns {dataset.column_names}')

    process = {}
    for key in ('action', 'proprio'):
        data = dataset.get_col_data(key)
        data = data[~np.isnan(data).any(axis=1)]
        process[key] = preprocessing.StandardScaler().fit(data)
    process['goal_proprio'] = process['proprio']
    act = process['action']
    print(f'action stats: mean {act.mean_.round(4).tolist()} std {act.scale_.round(4).tolist()}')
    stats_path = C.ROOT / 'lewm_probe' / 'action_stats.json'
    stats_path.write_text(json.dumps({'mean': act.mean_.tolist(), 'std': act.scale_.tolist()}, indent=2) + '\n')

    # same start-point sampling as upstream eval.py
    ep_ids = np.unique(ep_col)
    ep_len = np.array([step_col[ep_col == e].max() + 1 for e in ep_ids])
    max_start = dict(zip(ep_ids, ep_len - GOAL_OFFSET - 1))
    valid = np.nonzero(step_col <= np.array([max_start[e] for e in ep_col]))[0]
    rng = np.random.default_rng(args.seed)
    rows = np.sort(valid[rng.choice(len(valid) - 1, size=args.num_eval, replace=False)])
    picked = dataset.get_row_data(rows)

    model = C.load_model()
    model.interpolate_pos_encoding = True
    world = swm.World(
        C.ENV_NAME, num_envs=args.num_eval, image_shape=(C.IMG_SIZE, C.IMG_SIZE),
        max_episode_steps=2 * EVAL_BUDGET,
    )
    solver = swm.solver.CEMSolver(model=model, batch_size=1, device='cpu', seed=args.seed, **C.CEM)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=swm.PlanConfig(**C.PLAN), process=process,
        transform={'pixels': C.img_transform(), 'goal': C.img_transform()},
    )
    world.set_policy(policy)

    t0 = time.perf_counter()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=picked['step_idx'].tolist(),
        goal_offset=GOAL_OFFSET,
        eval_budget=EVAL_BUDGET,
        episodes_idx=picked[col].tolist(),
        callables=CALLABLES,
    )
    print(f'success rate {metrics["success_rate"]:.1f}% over {args.num_eval} episodes '
          f'({time.perf_counter() - t0:.0f}s)')


if __name__ == '__main__':
    main()
