"""Validate collected episode files before anything is trained on them.

Errors mean the data is wrong (labels disagree with the recorded path, hooks captured the
wrong tensors, physics was violated). Warnings mean the data is valid but the distribution
is off target (hazard rate, success rate). Exit code 1 if any error.

Usage: .venv/bin/python scripts/validate_data.py data/full/*.npz [--rerun 3]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from lewm_probe.hazard import pairs_near_zone, segment_distance
from scripts.collect import FUTURE_STEPS, PAIRS, PLAN_STEPS, POOL_SIZE, ZONE

LAYERS = [f'predictor.block{i}' for i in range(6)] + ['predictor.norm', 'pred_proj']
WALL_X = (107.0, 117.0)  # wall slab; the agent centre can only be here inside the doorway
DOOR_Y = (35.0, 63.0)
DOOR_MARGIN = 1.75  # env treats the doorway as this much wider when resolving collisions
# The env resolves a wall collision by moving the agent to the wall face plus radius plus 0.5,
# on the side of the wall centre it came from. From inside the doorway that can be a jump of
# up to ~13 px (TwoRoomEnv._apply_collisions). These snaps are real env physics, not data errors.
SNAP_X = (WALL_X[0] - 7.0 - 0.5, WALL_X[1] + 7.0 + 0.5)
ARENA = (14.0, 209.0)
MAX_STEP_PX = 5.0 * np.sqrt(2) + 1e-3  # speed 5 px, each action axis clamped to [-1, 1]
SUCCESS_PX = 16.0


class Report:
    def __init__(self):
        self.errors, self.warnings = 0, 0

    def check(self, ok, name, detail='', warn=False):
        ok = bool(ok)
        tag = 'ok  ' if ok else ('WARN' if warn else 'FAIL')
        print(f'  [{tag}] {name}' + (f': {detail}' if detail and not ok else ''))
        if not ok:
            if warn:
                self.warnings += 1
            else:
                self.errors += 1


def load(paths):
    parts = [dict(np.load(p)) for p in paths]
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if not k.startswith('meta/')}
    data['meta/zone'] = parts[0]['meta/zone']
    data['meta/args'] = parts[0]['meta/args']  # JSON string of the collection arguments
    return data


def validate(d, rep, pool):
    n = len(d['episode'])
    ep_ids, pos_all = d['ep/id'], d['ep/positions']
    ep_index = {e: i for i, e in enumerate(ep_ids)}
    rows_ep = np.array([ep_index.get(e, -1) for e in d['episode']])
    print(f'{n} plans from {len(ep_ids)} episodes, precision {json.loads(str(d["meta/args"])).get("precision", "fp32")}')

    print('structure')
    rep.check(np.allclose(d['meta/zone'], [*ZONE.center, ZONE.radius]), 'zone in file matches collect.py')
    rep.check(all(len(v) == n for k, v in d.items() if not k.startswith(('ep/', 'meta/'))), 'all per-plan fields have one row per plan')
    rep.check(len(np.unique(ep_ids)) == len(ep_ids), 'episode ids unique across files')
    rep.check((rows_ep >= 0).all(), 'every plan belongs to a recorded episode')
    counts = np.bincount(rows_ep, minlength=len(ep_ids))
    rep.check(((counts >= 1) & (counts <= 2)).all(), '1-2 plans per episode', f'counts {np.unique(counts)}')
    rep.check(np.array_equal(d['step'], d['plan_idx'] * PLAN_STEPS), 'plan k starts at env step 25k')
    bad = [k for k, v in d.items() if v.dtype.kind == 'f' and not np.isfinite(v[np.isfinite(v) | ~np.isinf(v)]).all()]
    finite = [k for k, v in d.items() if v.dtype.kind == 'f' and k != 'hazard_min_dist' and not np.isfinite(v).all()]
    rep.check(not finite and not bad, 'no NaN or inf values', ', '.join(finite + bad))

    print('episode setup')
    rep.check(np.allclose(d['start'], pool[0][d['episode']]) and np.allclose(d['goal'], pool[1][d['episode']]),
              'start and goal match the seeded pair pool for each episode id')
    first = d['plan_idx'] == 0
    rep.check(np.allclose(d['pos'][first], d['start'][first]), 'first plan starts at the episode start')
    seg = segment_distance(d['start'], d['goal'], ZONE.center)
    rep.check((seg <= PAIRS['max_segment_dist'] + 1e-6).all(), 'start-goal line passes within 22 px of the zone')
    margin = ZONE.radius + PAIRS['min_endpoint_margin']
    rep.check((ZONE.distance(d['start']) > margin).all() & (ZONE.distance(d['goal']) > margin).all(),
              'start and goal outside the zone plus margin')
    rep.check((np.linalg.norm(d['goal'] - d['start'], axis=1) <= PAIRS['max_pair_dist'] + 1e-6).all(), 'trips <= 50 px')
    rep.check(not d['start_in_zone'][first].any(), 'no episode starts inside the zone')

    print('physics')
    n_steps = d['ep/n_steps']
    long_steps, snaps, snap_eps = 0, 0, 0
    for p, k in zip(pos_all, n_steps):
        step = np.linalg.norm(np.diff(p[: k + 1], axis=0), axis=1)
        snapped = np.isclose(p[1 : k + 1, 0][:, None], SNAP_X, atol=0.05).any(1)
        long_steps += int(((step > MAX_STEP_PX) & ~snapped).sum())
        snaps += int((step > MAX_STEP_PX).sum())
        snap_eps += int((step > MAX_STEP_PX).any())
    rep.check(long_steps == 0, 'steps over 7.07 px only come from wall-collision snaps', f'{long_steps} unexplained')
    rep.check(True, f'wall-collision snaps: {snaps} steps in {snap_eps} episodes ({snap_eps / len(ep_ids):.0%})')
    xy = np.concatenate([p[: k + 1] for p, k in zip(pos_all, n_steps)])
    rep.check(((xy >= ARENA[0] - 1) & (xy <= ARENA[1] + 1)).all(), 'agent stays inside the arena')
    in_wall = (xy[:, 0] > WALL_X[0]) & (xy[:, 0] < WALL_X[1]) & (
        (xy[:, 1] < DOOR_Y[0] - DOOR_MARGIN) | (xy[:, 1] > DOOR_Y[1] + DOOR_MARGIN))
    rep.check(not in_wall.any(), 'agent centre only inside the wall slab within the doorway', f'{in_wall.sum()} positions')
    succ, succ_step = d['ep/success'], d['ep/success_step']
    first_hit = []
    for p, k, g in zip(pos_all, n_steps, pool[1][ep_ids]):
        hit = np.nonzero(np.linalg.norm(p[1 : k + 1] - g, axis=1) < SUCCESS_PX)[0]
        first_hit.append(hit[0] + 1 if len(hit) else -1)
    rep.check(np.array_equal(np.where(succ, succ_step, -1), first_hit),
              'success step == first step within 16 px of the goal (and failures never get there)')
    plans = np.bincount(rows_ep, minlength=len(ep_ids))
    rep.check(np.array_equal(n_steps, plans * PLAN_STEPS), 'every plan is executed for its full 25 steps')
    first_plan_reached = succ & (succ_step <= PLAN_STEPS)
    rep.check(np.array_equal(plans, np.where(first_plan_reached, 1, 2)),
              'a second plan is made exactly when the first did not reach the goal')

    print('labels recomputed from the recorded path')
    haz, mind, fut, valid = [], [], [], []
    for r in range(n):
        p, last, t0 = pos_all[rows_ep[r]], n_steps[rows_ep[r]], d['step'][r]
        window = p[t0 + 1 : min(t0 + PLAN_STEPS, last) + 1]
        haz.append(ZONE.entered(window) if len(window) else False)
        mind.append(ZONE.distance(window).min() if len(window) else np.inf)
        fut.append([p[min(t0 + h, last)] for h in FUTURE_STEPS])
        valid.append([t0 + h <= last for h in FUTURE_STEPS])
        if not np.allclose(d['pos'][r], p[t0]):
            rep.check(False, 'plan position matches the episode path', f'row {r}')
            break
    rep.check(np.array_equal(d['hazard'], haz), 'hazard label matches the path', f'{(d["hazard"] != haz).sum()} rows differ')
    rep.check(np.allclose(d['hazard_min_dist'], mind), 'closest approach matches the path')
    rep.check(np.array_equal(d['hazard'], d['hazard_min_dist'] <= ZONE.radius), 'hazard == closest approach <= radius')
    rep.check(np.allclose(d['future_pos'], fut) and np.array_equal(d['future_valid'], valid), 'future positions match the path')
    ep_haz = np.array([d['hazard'][rows_ep == i].any() for i in range(len(ep_ids))])
    rep.check(np.array_equal(ep_haz, d['ep/hazard']), 'episode hazard == any plan hazard')

    print('activations (hooks captured the right tensors)')
    rep.check(np.allclose(d['act/pred_proj'], d['imagined'][:, 1:], atol=1e-4),
              'pred_proj hook == imagined latents 1-5 (hooks aligned with imagined steps)')
    rep.check(np.allclose(d['emb'], d['imagined'][:, 0], atol=1e-5), 'current-frame latent == imagined latent 0')
    cost = ((d['imagined'][:, -1] - d['goal_emb']) ** 2).sum(-1)
    rep.check(np.allclose(cost, d['cost'], rtol=1e-3, atol=1e-2), 'stored cost == |final imagined - goal|^2')
    for name in LAYERS:
        a = d[f'act/{name}']
        flat = a.reshape(n, -1)
        dup = len(np.unique(flat.round(5), axis=0)) < n
        rep.check(not dup and a.std(axis=0).mean() > 1e-3, f'{name}: rows distinct and not constant')
    rep.check(not any(np.allclose(d[f'act/{a}'], d[f'act/{b}']) for a, b in zip(LAYERS, LAYERS[1:])),
              'consecutive layers differ (no hook registered twice)')

    print('distribution (warnings only)')
    rep.check(0.3 <= ep_haz.mean() <= 0.5, f'episode hazard rate {ep_haz.mean():.0%} in 30-50%', warn=True)
    rep.check(0.3 <= d['hazard'].mean() <= 0.5, f'plan hazard rate {d["hazard"].mean():.0%} in 30-50%', warn=True)
    rep.check(succ.mean() >= 0.85, f'planner success {succ.mean():.0%} >= 85%', warn=True)
    both = [(succ & ep_haz).sum(), (succ & ~ep_haz).sum(), (~succ & ep_haz).sum(), (~succ & ~ep_haz).sum()]
    rep.check(min(both[0], both[1]) > 0, f'hazard not tied to success (in/ok {both[0]}, out/ok {both[1]}, '
              f'in/fail {both[2]}, out/fail {both[3]})', warn=True)
    valid = np.array(valid)
    rep.check(valid.all(), 'future-position labels available for every plan', f'valid at +1/+5/+10: {valid.mean(0).round(2).tolist()}')


def rerun(d, k, rep):
    """Recollect k random episodes and require bit-identical plans and activations."""
    import subprocess
    import tempfile

    ids = np.random.default_rng(0).choice(d['ep/id'], size=k, replace=False)
    print(f'rerun of episodes {ids.tolist()}')
    root = Path(__file__).resolve().parent.parent
    for e in ids:
        with tempfile.TemporaryDirectory(dir=root / 'data') as tmp:
            name = Path(tmp).name
            meta = json.loads(str(d['meta/args']))
            precision, offset = meta.get('precision', 'fp32'), str(meta.get('cem_seed_offset', 0))
            subprocess.run([sys.executable, str(root / 'scripts' / 'collect.py'), '--name', name, '--precision', precision,
                            '--cem-seed-offset', offset,
                            '--first', str(e), '--episodes', '1'], check=True, capture_output=True)
            r = np.load(next(Path(tmp).glob('episodes_*.npz')))
            m = d['episode'] == e
            same = np.array_equal(r['plan'], d['plan'][m]) and all(
                np.array_equal(r[f'act/{x}'], d[f'act/{x}'][m]) for x in LAYERS)
            rep.check(same, f'episode {e} reproduces exactly')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--rerun', type=int, default=0, help='also recollect this many random episodes')
    args = ap.parse_args()
    rep = Report()
    d = load(args.files)
    pool = pairs_near_zone(ZONE, POOL_SIZE, seed=0, **PAIRS)
    validate(d, rep, pool)
    if args.rerun:
        rerun(d, args.rerun, rep)
    print(f'{rep.errors} errors, {rep.warnings} warnings')
    sys.exit(1 if rep.errors else 0)


if __name__ == '__main__':
    main()
