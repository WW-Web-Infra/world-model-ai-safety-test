"""Week 4 (RQ4): can the readable intent be used to steer the planner away from the hazard?

LeWM's planner is CEM over action sequences, scored by the latent distance from each candidate's
imagined end state to the goal. The hazard is not in that cost. `GuidedCost` wraps the model's
`get_cost` so CEM sees one of:

  'probe'     cost + lam * P(hazard), from the frozen Figure 1 probe on each candidate's block-2
              activations (the monitor turned into a constraint)
  'random'    cost + lam * sigmoid(random direction of the same norm): any penalty, not the monitor
  'geometry'  cost + lam * sigmoid((r - d) / tau), d = closest approach of the decoded imagined path
              to the zone centre: a hand-built penalty that knows where the zone is
  'steer'     the plan's literal version: during every candidate evaluation, subtract alpha x the
              hazard-probe direction from block 4's output (where patching found the route)
  'steer_random'  the same with a random direction

The probe is applied out-of-fold: for an episode, use the probe from the fold that did not train on it.
"""

import numpy as np
import torch
from torch import nn

from . import position_decoder as PD

PROBE_LAYER = 2   # headline monitor in Figures 1 and 3
STEER_LAYER = 4   # Figure 2: the route is fully carried by block 4's output
N_CALLS = 5


def linear_parts(pipeline):
    """(mean, scale, coef, intercept) of a StandardScaler + logistic pipeline, as float tensors."""
    sc, lr = pipeline[0], pipeline[-1]
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))
    return t(sc.mean_), t(sc.scale_), t(lr.coef_[0]), t(lr.intercept_)


class GuidedCost(nn.Module):
    def __init__(self, model, zone_center, zone_radius):
        super().__init__()
        self.model = model
        self.mode, self.lam, self.alpha = 'none', 0.0, 0.0
        self.center = np.asarray(zone_center, dtype=np.float64)
        self.radius = float(zone_radius)
        self.tau = 2.0
        self.decoder = None
        self._acts, self._call, self._active = [], 0, False
        blocks = model.predictor.transformer.layers
        self._handles = [blocks[PROBE_LAYER].register_forward_hook(self._read),
                         blocks[STEER_LAYER].register_forward_hook(self._steer),
                         model.predictor.register_forward_pre_hook(self._count)]
        self.probe = None   # (mean, scale, coef, intercept) for the penalty
        self.direction = None  # (N_CALLS, D) steering vectors, already scaled

    # -- configuration ------------------------------------------------------------------------
    def set_probe(self, pipeline, random_seed=None):
        mean, scale, coef, b = linear_parts(pipeline)
        if random_seed is not None:
            g = torch.Generator().manual_seed(random_seed)
            r = torch.randn(coef.shape, generator=g)
            coef = r / r.norm() * coef.norm()
        self.probe = (mean, scale, coef, b)

    def set_direction(self, unit_dirs, proj_std):
        """Steering vectors per predictor call: unit directions (N_CALLS, D) scaled by the data std
        of the activations' projection on each, so alpha is in standard deviations."""
        d = torch.as_tensor(np.asarray(unit_dirs, dtype=np.float32))
        self.direction = d * torch.as_tensor(np.asarray(proj_std, dtype=np.float32))[:, None]

    def configure(self, mode, lam=0.0, alpha=0.0):
        self.mode, self.lam, self.alpha = mode, float(lam), float(alpha)
        if mode == 'geometry' and self.decoder is None:
            self.decoder = PD.load()['model']

    # -- hooks --------------------------------------------------------------------------------
    # hooks act only inside GuidedCost.get_cost, not when the raw model is used elsewhere
    def _count(self, _mod, _args):
        if self._active:
            self._call += 1

    def _read(self, _mod, _args, out):
        if self._active and self.mode in ('probe', 'random'):
            self._acts.append(out[:, -1].detach())

    def _steer(self, _mod, _args, out):
        if self._active and self.mode in ('steer', 'steer_random') and self.alpha:
            out = out.clone()
            out[:, -1] -= self.alpha * self.direction[self._call - 1].to(out.dtype)
        return out

    # -- the cost CEM sees --------------------------------------------------------------------
    def get_cost(self, info, candidates):
        self._acts, self._call, self._active = [], 0, True
        try:
            cost = self.model.get_cost(info, candidates)
        finally:
            self._active = False
        if self.mode in ('probe', 'random') and self.lam:
            h = torch.stack(self._acts, dim=1).flatten(1)  # (B*S, 5*D), the probe's feature layout
            mean, scale, coef, b = self.probe
            p = torch.sigmoid(((h - mean) / scale) @ coef + b)
            cost = cost + self.lam * p.view_as(cost)
        elif self.mode == 'geometry' and self.lam:
            z = info['predicted_emb']  # (B, S, 6, D)
            pts = self.decoder.predict(z.reshape(-1, z.shape[-1]).numpy()).reshape(*z.shape[:3], 2)
            dist = closest_approach(pts, self.center)
            p = torch.as_tensor(1 / (1 + np.exp(-(self.radius - dist) / self.tau)), dtype=cost.dtype)
            cost = cost + self.lam * p
        self._acts = []
        return cost

    def remove(self):
        for h in self._handles:
            h.remove()


def closest_approach(points, center):
    """Distance from `center` to the polyline through points[..., k, :], for every candidate."""
    a, b = points[..., :-1, :], points[..., 1:, :]
    d = b - a
    t = np.clip(((center - a) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(a + t[..., None] * d - center, axis=-1).min(-1)
