"""Shared setup for loading LeWM and driving its planner in TwoRooms on CPU."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# keep checkpoints inside the project instead of ~/.stable-wm; must be set before importing swm
os.environ.setdefault('STABLEWM_HOME', str(ROOT / '.stable-wm'))

import numpy as np
import stable_worldmodel as swm
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

CHECKPOINT = 'quentinll/lewm-tworooms'
ENV_NAME = 'swm/TwoRoom-v1'
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# planner settings from le-wm config/eval/tworoom.yaml and config/eval/solver/cem.yaml
PLAN = dict(horizon=5, receding_horizon=5, action_block=5)
CEM = dict(num_samples=300, n_steps=30, topk=30, var_scale=1.0)


def load_model():
    """The trained checkpoint, from a pinned and hash-verified revision (see checkpoint.py)."""
    from . import checkpoint

    return checkpoint.load()


def img_transform():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.Resize(size=IMG_SIZE),
        ]
    )


def make_world(max_episode_steps=100):
    return swm.World(
        ENV_NAME,
        num_envs=1,
        image_shape=(IMG_SIZE, IMG_SIZE),
        max_episode_steps=max_episode_steps,
    )


def action_scaler(mean, std):
    """StandardScaler with given per-dimension action mean and std.

    The upstream eval fits this on the training dataset's action column.
    """
    scaler = preprocessing.StandardScaler()
    scaler.mean_ = np.asarray(mean, dtype=np.float64)
    scaler.scale_ = np.asarray(std, dtype=np.float64)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


def make_policy(model, scaler, seed=42, callbacks=None):
    solver = swm.solver.CEMSolver(
        model=model, batch_size=1, device='cpu', seed=seed, callbacks=callbacks, **CEM
    )
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(**PLAN),
        process={'action': scaler},
        transform={'pixels': img_transform(), 'goal': img_transform()},
    )


def predictor_layers(model):
    """Named modules to hook: the six predictor blocks and the latent in/out heads."""
    layers = {f'predictor.block{i}': blk for i, blk in enumerate(model.predictor.transformer.layers)}
    layers['predictor.norm'] = model.predictor.transformer.norm
    layers['pred_proj'] = model.pred_proj
    layers['projector'] = model.projector
    return layers


class ActivationRecorder:
    """Forward hooks that keep every output of the hooked modules while `active` is True.

    Off by default so the 300 x 30 CEM candidate evaluations are not recorded.
    """

    def __init__(self, modules):
        self.active = False
        self.records = {name: [] for name in modules}
        self._handles = [
            mod.register_forward_hook(self._hook(name)) for name, mod in modules.items()
        ]

    def _hook(self, name):
        def fn(_module, _inputs, output):
            if self.active:
                self.records[name].append(output.detach().cpu())

        return fn

    def clear(self):
        for v in self.records.values():
            v.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
