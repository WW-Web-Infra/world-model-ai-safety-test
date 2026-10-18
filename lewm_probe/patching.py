"""Activation patching inside LeWM's predictor rollout.

The predictor is called once per imagined step (5 calls for a 5-block plan). Each of its six
ConditionalBlocks takes (x, c): x is the residual stream (one token per context frame) and c is
the action embedding, which enters only through the block's adaLN modulation. Two kinds of patch:

  ('res', layer, step)  replace the residual stream: layer 'input' is the input to block 0,
                        layer i is the output of block i
  ('cond', i, step)     replace the action conditioning that block i receives

A patch takes the tensor recorded in a source run at the same (layer, step), so source and target
must share start and goal frames and differ only in the plan.
"""

import torch

INPUT = 'input'


class Patcher:
    def __init__(self, model):
        self.model = model
        self.blocks = list(model.predictor.transformer.layers)
        self.layers = [INPUT] + list(range(len(self.blocks)))
        self.step = -1
        self.record = None  # dict to fill during a clean run
        self.patches = {}  # (kind, layer, step) -> tensor
        self._handles = [model.predictor.register_forward_pre_hook(self._count)]
        for i, blk in enumerate(self.blocks):
            self._handles.append(blk.register_forward_pre_hook(self._pre(i)))
            self._handles.append(blk.register_forward_hook(self._post(i)))

    def _count(self, _mod, _args):
        self.step += 1

    def _pre(self, i):
        def fn(_mod, args):
            x, c = args
            if self.record is not None:
                if i == 0:
                    self.record[('res', INPUT, self.step)] = x.detach().clone()
                self.record[('cond', i, self.step)] = c.detach().clone()
            if i == 0 and ('res', INPUT, self.step) in self.patches:
                x = self.patches[('res', INPUT, self.step)]
            if ('cond', i, self.step) in self.patches:
                c = self.patches[('cond', i, self.step)]
            return (x, c)

        return fn

    def _post(self, i):
        def fn(_mod, _args, out):
            if self.record is not None:
                self.record[('res', i, self.step)] = out.detach().clone()
            return self.patches.get(('res', i, self.step), out)

        return fn

    def run(self, info, plan, patches=None, record=False):
        """Roll out `plan` (1, horizon, action_dim) from `info`; returns imagined latents (6, D)
        and, if record=True, every residual/conditioning tensor keyed (kind, layer, step)."""
        single = {k: v.unsqueeze(1) for k, v in info.items() if torch.is_tensor(v)}
        self.step = -1
        self.patches = patches or {}
        self.record = {} if record else None
        try:
            with torch.inference_mode():
                self.model.get_cost(single, plan.unsqueeze(1))
        finally:
            rec, self.record, self.patches = self.record, None, {}
        return single['predicted_emb'][0, 0].numpy(), rec

    def remove(self):
        for h in self._handles:
            h.remove()


def steps(record):
    return sorted({k[2] for k in record})


def residual_patch(source, layer, which_steps):
    return {('res', layer, t): source[('res', layer, t)] for t in which_steps}


def conditioning_patch(source, blocks, which_steps):
    return {('cond', b, t): source[('cond', b, t)] for b in blocks for t in which_steps}
