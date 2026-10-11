"""Week 1 Saturday feasibility check: can LeWM plan on this laptop's CPU, and can we see inside it?

Usage: .venv/bin/python scripts/feasibility_check.py [--seed 0] [--episodes 1]
Writes outputs/feasibility/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from lewm_probe import common as C
from lewm_probe.episode import dataset_start_goals, expert_goal, run_episode

OUT = C.ROOT / 'outputs' / 'feasibility'

# written by scripts/upstream_eval.py from the training dataset's action column
ACTION_STATS = json.loads((C.ROOT / 'lewm_probe' / 'action_stats.json').read_text())


def draw_room(ax, wall):
    size = 224
    ax.add_patch(Rectangle((0, 0), size, size, fill=False, lw=1.5, color='0.3'))
    half = wall['thickness'] / 2
    edges = [0.0]
    for c, s in zip(wall['door_centers'], wall['door_half_sizes']):
        edges += [c - s, c + s]
    edges.append(size)
    for lo, hi in zip(edges[0::2], edges[1::2]):
        if wall['axis'] == 1:
            ax.add_patch(Rectangle((wall['pos'] - half, lo), 2 * half, hi - lo, color='0.3'))
        else:
            ax.add_patch(Rectangle((lo, wall['pos'] - half), hi - lo, 2 * half, color='0.3'))
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect('equal')


def plot_episodes(episodes, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_room(ax, episodes[0]['wall'])
    for ep in episodes:
        p = ep['positions']
        (line,) = ax.plot(p[:, 0], p[:, 1], lw=1.5, alpha=0.8)
        ax.plot(*p[0], 'o', color=line.get_color(), ms=6)
        ax.plot(*ep['goal_pos'], '*', color=line.get_color(), ms=12)
        ax.plot(*p[ep['plan_start_steps']].T, 'x', color=line.get_color(), ms=5)
    ax.set_title(f'LeWM planner in TwoRooms ({len(episodes)} episodes): o start, * goal, x replan')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--goal', choices=['dataset', 'expert25', 'random'], default='dataset',
                    help='dataset: start and goal are dataset frames 25 steps apart (upstream eval protocol); '
                         'expert25: goal is 25 noise-free expert steps from a random start; '
                         'random: the env\'s own random target, usually beyond one planning horizon')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = C.load_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[1] loaded {C.CHECKPOINT} on CPU in {time.perf_counter() - t0:.1f}s: '
          f'{n_params / 1e6:.1f}M params, torch {torch.__version__}, {torch.get_num_threads()} threads')
    (OUT / 'model_modules.txt').write_text(repr(model))

    layers = C.predictor_layers(model)
    recorder = C.ActivationRecorder(layers)
    print(f'[2] forward hooks registered on {len(layers)} modules: {", ".join(layers)}')

    world = C.make_world()
    policy = C.make_policy(model, C.action_scaler(ACTION_STATS['mean'], ACTION_STATS['std']), seed=args.seed)
    policy.set_env(world.envs)

    pairs = dataset_start_goals(args.episodes, seed=args.seed) if args.goal == 'dataset' else None
    episodes = []
    for i in range(args.episodes):
        seed = args.seed + i
        start_pos = goal_pos = None
        if args.goal == 'dataset':
            start_pos, goal_pos = pairs[i]
        elif args.goal == 'expert25':
            goal_pos = expert_goal(world, seed)
        # upstream eval_budget is 50 env steps = 2 plans
        ep = run_episode(model, world, policy, recorder, seed=seed, max_plans=2,
                         goal_pos=goal_pos, start_pos=start_pos)
        episodes.append(ep)
        print(f'    episode seed={ep["seed"]}: start={ep["start"].round(1)} goal={ep["goal_pos"].round(1)} '
              f'dist={np.linalg.norm(ep["start"] - ep["goal_pos"]):.0f}px steps={len(ep["positions"]) - 1} plans={len(ep["plans"])} success={ep["success"]} '
              f'final_dist={np.linalg.norm(ep["positions"][-1] - ep["goal_pos"]):.1f}px')

    times = np.concatenate([ep['solve_times'] for ep in episodes])
    print(f'[3] planning time: {times.mean():.2f}s per plan (min {times.min():.2f}, max {times.max():.2f}, '
          f'n={len(times)}); go/no-go threshold is 10s')

    ep = episodes[0]
    print(f'[4] true agent position read from the env at every step: positions array {ep["positions"].shape}')

    r = ep['replays'][0]
    print(f'[5] activations of the chosen plan (per imagined step): imagined latents {r["imagined"].shape}, '
          f'goal latent {r["goal_emb"].shape}')
    for name, a in r['acts'].items():
        print(f'      {name:18s} {a.shape}  mean|x|={np.abs(a).mean():.3f}')

    plot_episodes(episodes, OUT / f'trajectories_{args.goal}.png')
    np.savez_compressed(
        OUT / 'episode0.npz',
        positions=ep['positions'], goal_pos=ep['goal_pos'], plans=ep['plans'],
        plan_start_steps=ep['plan_start_steps'], solve_times=ep['solve_times'],
        imagined=np.stack([x['imagined'] for x in ep['replays']]),
        goal_emb=ep['replays'][0]['goal_emb'],
        **{f'act/{k}': np.stack([x['acts'][k] for x in ep['replays']]) for k in r['acts']},
    )
    success = np.mean([e['success'] for e in episodes])
    print(f'[6] success rate {success:.0%} over {len(episodes)} episode(s); wrote {OUT.relative_to(C.ROOT)}/')


if __name__ == '__main__':
    main()
