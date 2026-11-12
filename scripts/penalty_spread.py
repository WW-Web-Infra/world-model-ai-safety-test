"""RQ4 control check: how much do the monitor penalty and the random-direction penalty vary across
the candidates CEM compares?

Runs the unguided planner on the first N unsafe evaluation episodes (one plan each) and, at every CEM
cost evaluation, computes both P_probe and P_random (same weight norm, seed 1234, as in steering.py)
for the same 300 candidates, plus the goal cost C. Nothing is written to disk.

Usage: .venv/bin/python -W ignore scripts/penalty_spread.py 40     (about 4 minutes)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from lewm_probe import common as C
from lewm_probe import dataset as D
from lewm_probe.episode import run_episode
from lewm_probe.frozen import load_probes
from lewm_probe.hazard import pairs_near_zone
from lewm_probe.steering import GuidedCost, linear_parts
from scripts.collect import ACTION_STATS, PAIRS, POOL_SIZE, ZONE
from scripts.figure3 import Frozen
from scripts.steering import PROBE_KEY, evaluation_ids

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
torch.set_num_threads(4)

class Both(GuidedCost):
    rows = []
    def get_cost(self, info, candidates):
        self._acts, self._call, self._active = [], 0, True
        try:
            cost = self.model.get_cost(info, candidates)
        finally:
            self._active = False
        h = torch.stack(self._acts, dim=1).flatten(1)
        out = {'C': cost.flatten().numpy()}
        for name, (mean, scale, coef, b) in (('probe', self.p_probe), ('random', self.p_random)):
            out[name] = torch.sigmoid(((h - mean) / scale) @ coef + b).numpy()
        Both.rows.append(out)
        self._acts = []
        return cost

d_fp = D.load()
ids = evaluation_ids('unsafe', 240, d_fp)[:N]
starts, goals, _ = pairs_near_zone(ZONE, POOL_SIZE, seed=0, **PAIRS)
frozen = Frozen()
probes = load_probes(C.ROOT / 'outputs' / 'figure1' / 'frozen_probes.npz')['probes']
model = C.load_model()
g = Both(model, ZONE.center, ZONE.radius)
g.configure('probe', lam=0.0)          # reads block-2 activations, adds nothing to the cost
world = C.make_world()
policy = C.make_policy(g, C.action_scaler(ACTION_STATS['mean'], ACTION_STATS['std']))
policy.set_env(world.envs)
recorder = C.ActivationRecorder(C.predictor_layers(model))
t0 = time.time()
for e in ids:
    fold = frozen.fold_of[int(e)]
    pm = probes[PROBE_KEY]['models'][fold]
    g.set_probe(pm); g.p_probe = g.probe
    g.set_probe(pm, random_seed=1234); g.p_random = g.probe
    policy.solver.torch_gen.manual_seed(int(e))
    run_episode(model, world, policy, recorder, seed=int(e), max_plans=1, goal_pos=goals[e], start_pos=starts[e], stop_at_goal=False)
print(f'{N} episodes, {len(Both.rows)} cost evaluations, {time.time()-t0:.0f}s')
R = Both.rows
def summ(key, f):
    v = np.array([f(r[key]) for r in R]); return np.median(v), np.percentile(v, 25), np.percentile(v, 75)
for key in ('probe', 'random'):
    print(f'{key:7s} sd of P across candidates: median {summ(key, np.std)[0]:.4f} (IQR {summ(key, np.std)[1]:.4f}-{summ(key, np.std)[2]:.4f}); '
          f'range median {summ(key, np.ptp)[0]:.3f}; mean P median {summ(key, np.mean)[0]:.3f}')
print(f'goal cost C sd across candidates: median {summ("C", np.std)[0]:.2f} (IQR {summ("C", np.std)[1]:.2f}-{summ("C", np.std)[2]:.2f})')
first = [r for i, r in enumerate(R) if i % 30 == 0]; last = [r for i, r in enumerate(R) if i % 30 == 29]
for tag, rr in (('first iteration', first), ('last iteration', last)):
    print(tag, ' '.join(f"{k} sd {np.median([np.std(r[k]) for r in rr]):.4f}" for k in ('probe', 'random', 'C')))
