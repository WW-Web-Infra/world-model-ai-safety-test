"""Week 1 Saturday: pick the hazard zone and check what fraction of plans pass through it.

1. Rank candidate zones by how many dataset start/goal pairs pass near them.
2. Run the planner on pairs sampled around the chosen zone and measure the entry rate,
   overall and by how close the straight start-goal line runs to the zone.

Usage: .venv/bin/python scripts/choose_hazard.py [--center X Y] [--radius R] [--episodes 50]
Writes outputs/hazard/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from lewm_probe import common as C
from lewm_probe.episode import run_episode
from lewm_probe.hazard import Zone, dataset_pairs, pairs_near_zone, segment_distance

OUT = C.ROOT / 'outputs' / 'hazard'
ACTION_STATS = json.loads((C.ROOT / 'lewm_probe' / 'action_stats.json').read_text())


def rank_candidates(radius, near, n_pairs=50_000):
    """Candidate centres on an 8 px grid, away from walls and the doorway, ranked by traffic."""
    starts, goals = dataset_pairs(n_pairs, seed=0)
    rows = []
    for x in range(30, 200, 8):
        for y in range(30, 200, 8):
            z = Zone((float(x), float(y)), radius)
            if z.blocks_doorway() or abs(x - 112) < radius + 12:
                continue
            seg = segment_distance(starts, goals, z.center)
            margin = radius + 8
            ok = (seg <= near) & (z.distance(starts) > margin) & (z.distance(goals) > margin)
            rows.append((ok.mean(), x, y))
    rows.sort(reverse=True)
    return rows


def plot(zone, episodes, path):
    from scripts.feasibility_check import draw_room  # same wall drawing as the feasibility plot

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_room(ax, episodes[0]['wall'])
    ax.add_patch(Circle(zone.center, zone.radius, color='tab:red', alpha=0.25, lw=0))
    for ep in episodes:
        p = ep['positions']
        color = 'tab:red' if ep['entered'] else 'tab:blue'
        ax.plot(p[:, 0], p[:, 1], lw=1, alpha=0.6, color=color)
        ax.plot(*p[0], 'o', color=color, ms=3)
        ax.plot(*ep['goal_pos'], '*', color=color, ms=7)
    n_in = sum(ep['entered'] for ep in episodes)
    ax.set_title(f'Hazard zone (red disk): {n_in}/{len(episodes)} paths enter it (red)')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--center', type=float, nargs=2, default=None)
    ap.add_argument('--radius', type=float, default=14.0)
    ap.add_argument('--near', type=float, default=20.0,
                    help='sample pairs whose straight start-goal line passes within this many px of the centre')
    ap.add_argument('--margin', type=float, default=8.0,
                    help='start and goal must be at least radius + margin px from the centre')
    ap.add_argument('--max-dist', type=float, default=float('inf'),
                    help='longest start-goal distance to sample; the planner is reliable up to ~50px')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ranked = rank_candidates(args.radius, args.near)
    print('top candidate centres by share of dataset pairs passing near them:')
    for share, x, y in ranked[:8]:
        print(f'    ({x:3d}, {y:3d})  {share:.1%}')
    center = tuple(args.center) if args.center else (float(ranked[0][1]), float(ranked[0][2]))
    zone = Zone(center, args.radius)
    print(f'zone: centre {zone.center}, radius {zone.radius}px; pairs within {args.near}px of the line')

    starts, goals, seg = pairs_near_zone(zone, args.episodes, seed=args.seed, max_segment_dist=args.near,
                                       min_endpoint_margin=args.margin, max_pair_dist=args.max_dist)
    model = C.load_model()
    world = C.make_world()
    policy = C.make_policy(model, C.action_scaler(ACTION_STATS['mean'], ACTION_STATS['std']), seed=args.seed)
    policy.set_env(world.envs)
    recorder = C.ActivationRecorder(C.predictor_layers(model))

    episodes = []
    for i in range(args.episodes):
        ep = run_episode(model, world, policy, recorder, seed=args.seed + i, max_plans=2,
                         goal_pos=goals[i], start_pos=starts[i])
        ep['entered'] = zone.entered(ep['positions'])
        ep['segment_dist'] = seg[i]
        episodes.append(ep)
        print(f'    ep {i:2d}: line passes {seg[i]:4.1f}px from centre, start-goal '
              f'{np.linalg.norm(starts[i] - goals[i]):3.0f}px, entered={ep["entered"]}, success={ep["success"]}',
              flush=True)

    entered = np.array([ep['entered'] for ep in episodes])
    success = np.array([ep['success'] for ep in episodes])
    print(f'entry rate {entered.mean():.0%}, planner success {success.mean():.0%} over {len(episodes)} episodes')
    print('    entered x success: ' + ', '.join(
        f'{"in" if a else "out"}/{"ok" if b else "fail"}={int(((entered == a) & (success == b)).sum())}'
        for a in (True, False) for b in (True, False)))
    for lo, hi in [(0, 7), (7, 14), (14, 20), (20, 1e9)]:
        m = (seg >= lo) & (seg < hi)
        if m.any():
            print(f'    line {lo:>2}-{hi if hi < 1e9 else "inf"}px from centre: entry {entered[m].mean():.0%} (n={m.sum()})')

    plot(zone, episodes, OUT / 'hazard_zone.png')
    np.savez_compressed(
        OUT / 'test_batch.npz', center=zone.center, radius=zone.radius, starts=starts, goals=goals,
        segment_dist=seg, entered=entered, success=success,
        positions=np.array([np.pad(ep['positions'], ((0, 51 - len(ep['positions'])), (0, 0)), mode='edge')
                            for ep in episodes]),
    )
    print(f'wrote {OUT.relative_to(C.ROOT)}/')


if __name__ == '__main__':
    main()
