"""Replay every collected plan through control models to get their activations.

- trained: the real checkpoint; must reproduce the stored activations (checks the replay path)
- random_init: same architecture, default initialisation (the predictor's adaLN-zero gates
  start at 0, so its blocks pass the input through and ignore the actions)
- shuffled: the trained weights with each parameter tensor's entries randomly permuted
  (keeps every tensor's value distribution, destroys what was learned)

Frames are re-rendered from the stored positions; dataset frames equal env renders pixel for pixel.
Usage: .venv/bin/python scripts/replay_controls.py
Writes data/full/replay_<model>.npz.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from lewm_probe import common as C
from lewm_probe import dataset as D
from lewm_probe.episode import goal_image, replay_plan

OUT = C.ROOT / 'data' / 'full'


def random_init_model(seed=0):
    from lewm_probe import checkpoint

    return checkpoint.build_untrained(seed)


def shuffled_model(seed=0):
    model = C.load_model()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            flat = p.view(-1)
            flat.copy_(flat[torch.randperm(flat.numel(), generator=g)])
    return model


def replay_all(model, d, world):
    env = world.envs.envs[0].unwrapped
    tf = C.img_transform()
    recorder = C.ActivationRecorder(C.predictor_layers(model))
    out = {k: [] for k in ['emb', 'goal_emb', 'imagined', 'cost'] + [f'act/{l}' for l in D.LAYERS]}
    for i in range(len(d['episode'])):
        env._set_goal_state(d['goal'][i])
        env._set_state(d['pos'][i])
        pixels = torch.from_numpy(env.render().copy()).permute(2, 0, 1)
        goal = torch.from_numpy(goal_image(env)[0, 0]).permute(2, 0, 1)
        info = {
            'pixels': tf(pixels)[None, None],
            'goal': tf(goal)[None, None],
            'action': torch.zeros(1, 1, 2),  # overwritten by the plan inside rollout
        }
        r = replay_plan(model, info, torch.from_numpy(d['plan'][i])[None], recorder)
        for k in ('emb', 'goal_emb', 'imagined', 'cost'):
            out[k].append(r[k])
        for l in D.LAYERS:
            out[f'act/{l}'].append(r['acts'][l])
        if (i + 1) % 500 == 0:
            print(f'    {i + 1}/{len(d["episode"])}', flush=True)
    recorder.remove()
    return {k: np.array(v) for k, v in out.items()}


def main():
    d = D.load()
    world = C.make_world()
    world.reset(seed=0)
    for name, make in [('trained', C.load_model), ('random_init', random_init_model), ('shuffled', shuffled_model)]:
        print(f'{name}:', flush=True)
        acts = replay_all(make(), d, world)
        if name == 'trained':
            diffs = {k: float(np.abs(acts[k] - d[k]).max()) for k in acts if k != 'cost'}
            worst = max(diffs, key=diffs.get)
            print(f'    max |replay - stored| = {diffs[worst]:.2e} ({worst})')
            if diffs[worst] > 1e-3:
                sys.exit('replay does not reproduce the stored activations; control replays would not be comparable')
        np.savez_compressed(OUT / f'replay_{name}.npz', **acts)
    print(f'wrote {OUT.relative_to(C.ROOT)}/replay_*.npz')


if __name__ == '__main__':
    main()
