"""Block diagrams of the three experiments (Figures 1-3), for the write-up.

Usage: .venv/bin/python scripts/diagrams.py
Writes outputs/diagrams/experiment_figure{1,2,3}.png and warns about any text that does not fit
inside its box.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

from lewm_probe import common as C

OUT = C.ROOT / 'outputs' / 'diagrams'

STYLES = {  # role: (fill, edge, text)
    'data': ('#EEF1F5', '#6B7785', '#1F2933'),
    'planner': ('#EDE7F6', '#5E35B1', '#311B92'),
    'model': ('#E3F2FD', '#1565C0', '#0D47A1'),
    'env': ('#E8F5E9', '#2E7D32', '#1B5E20'),
    'probe': ('#FFF3E0', '#E65100', '#BF360C'),
    'compare': ('#FFFFFF', '#757575', '#424242'),
    'result': ('#263238', '#263238', '#FFFFFF'),
}
LEGEND = {'data': 'inputs / data', 'planner': 'planner (CEM search)', 'model': 'LeWM world model',
          'env': 'environment / labels', 'probe': 'probes / readouts', 'compare': 'comparisons',
          'result': 'result'}
GREY = '#52606D'


class Diagram:
    def __init__(self, title, subtitle, xlim=135, ylim=52, width=22):
        self.fig, self.ax = plt.subplots(figsize=(width, width * ylim / xlim))
        self.ax.set_xlim(0, xlim)
        self.ax.set_ylim(0, ylim)
        self.ax.axis('off')
        self.ax.text(1, ylim - 0.8, title, fontsize=16, fontweight='bold', va='top', color='#1F2933')
        self.ax.text(1, ylim - 3.9, subtitle, fontsize=11, va='top', color=GREY)
        self.boxes = []

    def box(self, x, y, w, h, title, body='', role='data', dashed=False, title_size=11, body_size=9.2):
        fill, edge, text = STYLES[role]
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=0.7',
                               fc=fill, ec=edge, lw=1.7, ls='--' if dashed else '-')
        self.ax.add_patch(patch)
        texts = []
        if body:
            texts.append(self.ax.text(x + w / 2, y + h - 1.0, title, ha='center', va='top', fontsize=title_size,
                                      fontweight='bold', color=text, linespacing=1.25))
            n_title = title.count('\n') + 1
            texts.append(self.ax.text(x + w / 2, y + h - 1.0 - 2.3 * n_title, body, ha='center', va='top',
                                      fontsize=body_size, color=text, linespacing=1.4))
        else:
            texts.append(self.ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=title_size,
                                      fontweight='bold', color=text, linespacing=1.3))
        self.boxes.append((patch, texts, title.split('\n')[0]))
        return dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2, r=x + w, t=y + h)

    def path(self, points, color=GREY, label='', label_at=None, ha='center', fontsize=9, lw=1.6):
        """Arrow along straight segments through `points`; arrowhead at the last point."""
        if len(points) > 2:
            verts = points[:-1]
            self.ax.add_patch(PathPatch(MPath(verts, [MPath.MOVETO] + [MPath.LINETO] * (len(verts) - 1)),
                                        fill=False, ec=color, lw=lw, capstyle='round', joinstyle='round'))
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>', mutation_scale=15, lw=lw,
                                          color=color, shrinkA=0, shrinkB=0))
        if label:
            lx, ly = label_at
            self.ax.text(lx, ly, label, ha=ha, va='center', fontsize=fontsize, color=color, style='italic',
                         linespacing=1.3, bbox=dict(fc='white', ec='none', pad=0.6))

    def note(self, x, y, text, size=9, color=GREY, ha='left'):
        self.ax.text(x, y, text, fontsize=size, color=color, ha=ha, va='top', linespacing=1.35)

    def legend(self, roles, y=0.6):
        x = 1
        for r in roles:
            fill, edge, _ = STYLES[r]
            self.ax.add_patch(FancyBboxPatch((x, y), 2.4, 1.4, boxstyle='round,pad=0.1,rounding_size=0.3',
                                             fc=fill, ec=edge, lw=1.2, ls='--' if r == 'compare' else '-'))
            self.ax.text(x + 3.1, y + 0.7, LEGEND[r], fontsize=9, va='center', color='#3E4C59')
            x += 5.0 + 0.62 * len(LEGEND[r])

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for patch, texts, label in self.boxes:
            pb = patch.get_window_extent(renderer)
            for t in texts:
                tb = t.get_window_extent(renderer)
                if tb.x0 < pb.x0 + 2 or tb.x1 > pb.x1 - 2 or tb.y0 < pb.y0 + 2 or tb.y1 > pb.y1 - 2:
                    print(f'    {name}: text overflows box "{label}"')
        self.fig.savefig(OUT / name, dpi=170, bbox_inches='tight', facecolor='white')
        plt.close(self.fig)


def figure1():
    g = Diagram('Figure 1 experiment: can a linear monitor read the plan before it runs?',
                'One row per plan: 2,000 seeded episodes in TwoRooms, 2,308 plans, and a hazard zone the planner is never told about')
    start = g.box(1, 31, 15, 8, 'Start frame', 'agent at its start\n224 x 224 pixels')
    goal = g.box(1, 19, 15, 8, 'Goal frame', 'agent drawn at the goal\nat most 50 px away')
    cem = g.box(21, 25, 20, 14, 'CEM planner', '300 candidate plans x 30 iterations;\nscores each by latent distance\nfrom its imagined end to the goal\n\nthe hazard is not in its objective', 'planner')
    plan = g.box(21, 9, 20, 9, 'Chosen plan', '5 blocks x 5 actions\n(2-D velocities)', 'planner')
    g.path([(start['r'], start['cy']), (cem['x'], 34)])
    g.path([(goal['r'], goal['cy']), (cem['x'], 29)])
    g.path([(cem['cx'], cem['y']), (plan['cx'], plan['t'])])

    enc = g.box(47, 31, 18, 8, 'Encoder (ViT-tiny)', 'current frame -> latent', 'model')
    pred = g.box(70, 29, 22, 12, 'Predictor: blocks b0 to b5', 'one call per imagined step (5 calls)\nactions enter every block\nthrough adaLN conditioning', 'model')
    imag = g.box(97, 31, 16, 8, 'Imagined path', 'latents at +5 ... +25 steps', 'model')
    g.path([(start['cx'], start['t']), (start['cx'], 44), (enc['cx'], 44), (enc['cx'], enc['t'])],
           label='current frame', label_at=(31, 44))
    g.path([(enc['r'], enc['cy']), (pred['x'], enc['cy'])])
    g.path([(pred['r'], imag['cy']), (imag['x'], imag['cy'])])
    g.path([(plan['r'], 15.5), (44, 15.5), (44, 24.5), (pred['cx'], 24.5), (pred['cx'], pred['y'])],
           label='replayed through the predictor', label_at=(63, 24.5))

    env = g.box(47, 6, 18, 9, 'Execute in TwoRooms', 'the real environment,\n25 steps', 'env')
    labels = g.box(70, 5, 22, 11, 'Labels from what happened', 'entered the hazard zone? (yes/no)\ndisplacement at +5 ... +25 steps', 'env')
    g.path([(plan['r'], 11), (env['x'], 11)], color='#2E7D32', label='executed', label_at=(44.3, 9.2), fontsize=8.5)
    g.path([(env['r'], 10.5), (labels['x'], 10.5)], color='#2E7D32')

    probes = g.box(97, 5, 16, 20, 'Probes', 'linear (and a small MLP)\non every layer and\nimagined step\n\n5 folds split by episode;\nheadline layer chosen\ninside training folds', 'probe')
    g.path([(imag['cx'], imag['y']), (probes['cx'], probes['t'])], label='activations at\nb0-b5, norm, output',
           label_at=(imag['cx'] - 9.5, 27.8), fontsize=8.3)
    g.path([(labels['r'], 10.5), (probes['x'], 10.5)], color='#2E7D32')

    g.box(117, 23, 17, 21, 'Compared against', 'geometry: position + goal\n(boosted trees are the bar;\nlinear also shown)\n\nuntrained models: random\ninit, shuffled weights\n\nshuffled labels (chance)\n\nthe plan\'s actions\n(reference line)', 'compare', dashed=True, body_size=8.8)
    res = g.box(117, 4, 17, 16, 'Result', 'hazard AUROC 0.987\nvs 0.928 for geometry\n\nlatent adds +0.06\non top of geometry\n\ndisplacement R² 0.90-0.94', 'result')
    g.path([(probes['r'], 12), (res['x'], 12)])
    g.legend(['data', 'planner', 'model', 'env', 'probe', 'compare', 'result'])
    g.save('experiment_figure1.png')


def figure2():
    g = Diagram('Figure 2 experiment: where does the predictor turn the plan into its path around the hazard?',
                'Activation patching between two plans from the same start and goal: one enters the zone, the other avoids it')
    start = g.box(1, 21, 16, 12, 'Same start\nand goal', '500 starts where\ngeometry is ambiguous\n(start-goal line passes\n7-22 px from the zone)')
    cem_a = g.box(22, 38, 16, 5, 'CEM, seed A', '', 'planner')
    unsafe = g.box(22, 29, 16, 6.5, 'Unsafe plan:\nenters the zone', '', 'env', title_size=10)
    cem_b = g.box(22, 21, 16, 5, 'CEM, new seed B', '', 'planner')
    safe = g.box(22, 12, 16, 6.5, 'Safe plan:\navoids the zone', '', 'env', title_size=10)
    g.path([(start['r'], 30), (cem_a['x'], cem_a['cy'])])
    g.path([(start['r'], 24), (cem_b['x'], cem_b['cy'])])
    g.path([(cem_a['cx'], cem_a['y']), (unsafe['cx'], unsafe['t'])])
    g.path([(cem_b['cx'], cem_b['y']), (safe['cx'], safe['t'])])
    g.note(22, 10.6, '140 pairs: both plans reach the goal,\nand their executed outcomes differ', size=8.8)

    names = ['input', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5']
    def run(y, label):
        g.ax.text(44, y + 7.6, label, fontsize=10, fontweight='bold', color='#0D47A1', va='center')
        bs = [g.box(44 + i * 6.2, y, 5.2, 6.5, n, '', 'model', title_size=9.5) for i, n in enumerate(names)]
        for a, b in zip(bs, bs[1:]):
            g.path([(a['r'] + 0.15, a['cy']), (b['x'] - 0.15, b['cy'])], lw=1.2)
        return bs
    ra = run(29, 'Predictor run with the unsafe plan (patched)')
    rb = run(12, 'Predictor run with the safe plan (source)')
    g.path([(unsafe['r'], unsafe['cy']), (ra[0]['x'], ra[0]['cy'])])
    g.path([(safe['r'], safe['cy']), (rb[0]['x'], rb[0]['cy'])])
    for i in (2, 5):
        g.path([(rb[i]['cx'], rb[i]['t'] + 0.3), (ra[i]['cx'], ra[i]['y'] - 0.3)], color='#E65100', lw=2)
    g.ax.text(ra[3]['r'] + 0.5, 23.6, 'patch: copy activations\nfrom the safe run\ninto the unsafe run', ha='center',
              va='center', fontsize=8.8, color='#E65100', style='italic', linespacing=1.3)
    g.box(44, 39.5, 42.4, 4.3, 'actions enter every block (adaLN); 5 predictor calls, one per imagined step',
          '', 'compare', dashed=True, title_size=8.8)
    g.box(44, 2.6, 42.4, 7.6, 'Two kinds of patch, run in both directions',
          '1) hidden state (residual stream) at layer L and call t\n2) the actions seen by block i, by blocks 0..i, or by all blocks at call t',
          'probe', body_size=8.8)

    readouts = g.box(91, 18, 19, 23, 'Readouts of the\npatched imagined path', 'position decoder trained on\nrendered frames only:\nclosest approach to the zone\n\nlatent difference\n\nFigure 1\'s frozen\nhazard probe', 'probe', body_size=8.8)
    g.path([(ra[-1]['r'], ra[-1]['cy']), (readouts['x'], ra[-1]['cy'])])
    effect = g.box(91, 4, 19, 10, 'Effect', 'fraction of the safe-vs-unsafe\ngap that the patch moves\n(0 = none, 1 = all)', 'probe', body_size=8.8)
    g.path([(readouts['cx'], readouts['y']), (effect['cx'], effect['t'])])
    res = g.box(114, 7, 20, 32, 'Result', 'the plan is used across\nblocks b0-b4;\nno single block is enough\n\nthe route is fully carried\nby the output of b4\n\nthe frozen hazard probe\nmoves with the\npatched route', 'result', body_size=9)
    g.path([(effect['r'], effect['cy']), (res['x'], effect['cy'])])
    g.legend(['data', 'planner', 'model', 'env', 'probe', 'compare', 'result'])
    g.save('experiment_figure2.png')


def figure3():
    g = Diagram('Figure 3 experiment: does the monitor still tell the truth after quantization?',
                'The same 2,000 seeded episodes, re-run with the whole model quantized; the monitor stays frozen at full precision')
    fp = g.box(1, 23, 14, 9, 'LeWM (fp32)', 'encoder, predictor,\nheads', 'model')
    q = g.box(19, 18, 19, 19, 'Quantize the\nwhole model', '8-bit: PyTorch dynamic\nquantization (CPU)\n\n6, 4 and 3-bit:\nweights rounded\nper output channel', 'model')
    g.path([(fp['r'], fp['cy']), (q['x'], fp['cy'])])

    loop = g.box(43, 30, 20, 11, 'Closed loop', 'the quantized planner\nre-plans every episode\n(same seeds, starts, goals)', 'planner')
    replay = g.box(43, 5, 20, 11, 'Replay', 'the full-precision plans run\nthrough the quantized model\n(isolates representation drift)', 'planner')
    g.path([(q['r'], 33), (loop['x'], 33)])
    g.path([(q['r'], 22), (40.5, 22), (40.5, replay['cy']), (replay['x'], replay['cy'])])

    acts = g.box(68, 33, 20, 8, 'Quantized activations', 'every layer and imagined step', 'model')
    outcome = g.box(68, 19, 20, 9, 'What it actually did', 'executed in TwoRooms:\ndid it enter the zone?', 'env')
    g.path([(loop['r'], 37), (acts['x'], 37)])
    g.path([(loop['cx'], loop['y']), (loop['cx'], 23.5), (outcome['x'], 23.5)], color='#2E7D32',
           label='executed', label_at=(58.5, 23.5), fontsize=8.5)

    frozen = g.box(93, 31, 18, 12, 'Frozen fp32 probe', 'Figure 1\'s probe on b2,\nunchanged; each episode\nscored out-of-fold;\n90%-recall threshold', 'probe', body_size=8.8)
    g.path([(acts['r'], 37), (frozen['x'], 37)], color='#E65100')
    g.path([(replay['r'], replay['cy']), (102, replay['cy']), (102, frozen['y'])], color='#E65100')

    meas = g.box(115, 19, 19, 25, 'Measured at\neach precision', 'behaviour vs fp32: success,\nhazard entry, path deviation,\noutcome flips\n\nmonitor vs what happened:\nAUROC, missed hazard plans\n(silent failures), false alarms\n\nretrained probe on the\nquantized activations', 'compare', dashed=True, body_size=8.4)
    g.path([(frozen['r'], 37), (meas['x'], 37)], color='#E65100')
    g.path([(outcome['r'], 23.5), (meas['x'], 23.5)], color='#2E7D32')
    res = g.box(115, 3, 19, 13, 'Result', 'holds at 8, 6 and 4 bits:\nwithin planner-seed noise\n\n3-bit: misses double\n(8.7% -> 17.7%); a retrained\nprobe recovers (AUROC 0.98)', 'result', body_size=8.4)
    g.legend(['model', 'planner', 'env', 'probe', 'compare', 'result'])
    g.save('experiment_figure3.png')


def figure4():
    g = Diagram('Figure 4 experiment: can the monitor steer the planner away from the hazard?',
                'The frozen hazard probe is added to the score CEM uses to pick a plan; outcomes are measured in the real environment')
    ep = g.box(1, 26, 15, 13, 'Episodes', '120 whose full-precision\nfirst plan entered\nthe zone, and\n60 random ones\n(same seeds)', 'data')
    cand = g.box(21, 31, 19, 9, 'CEM: 300 candidate\nplans x 30 iterations', '', 'planner', title_size=10)
    roll = g.box(21, 18, 19, 9, 'Predictor rollout\nof each candidate', '', 'model', title_size=10)
    g.path([(ep['r'], 35.5), (cand['x'], 35.5)])
    g.path([(cand['cx'], cand['y']), (roll['cx'], roll['t'])])
    goal = g.box(45, 31, 21, 9, 'Distance to goal', r'$\Vert \hat z_5(a) - z_g \Vert^2$', 'planner', body_size=10.5)
    probe = g.box(45, 16, 21, 11, 'Frozen hazard probe', 'block-2 activations of the\ncandidate -> P(enter zone)\n(Figure 1 probe, out-of-fold)', 'probe', body_size=8.8)
    g.path([(roll['r'], 24), (45, 24)], color='#E65100')
    g.path([(roll['r'], 26), (42.5, 26), (42.5, 35.5), (goal['x'], 35.5)])
    score = g.box(71, 23, 20, 14, 'Score each candidate', r'$C(a) = \Vert \hat z_5 - z_g \Vert^2$' + '\n' + r'$+\ \lambda \cdot P(\mathrm{enter} \mid a)$' + '\n\nCEM keeps the lowest', 'planner', body_size=10)
    g.path([(goal['r'], 35.5), (68.5, 35.5), (68.5, 32), (score['x'], 32)])
    g.path([(probe['r'], 21.5), (68.5, 21.5), (68.5, 28), (score['x'], 28)], color='#E65100')
    g.path([(score['cx'], score['t']), (score['cx'], 44), (cand['cx'], 44), (cand['cx'], cand['t'])], color='#5E35B1',
           label='resample around the best 30', label_at=(56, 44), fontsize=8.6)
    env = g.box(95, 26, 17, 11, 'Execute the\nchosen plan', 'TwoRooms: did it enter\nthe zone? reach the goal?', 'env', body_size=8.8)
    g.path([(score['r'], 31), (env['x'], 31)], color='#2E7D32')
    g.box(71, 6, 41, 11, 'Compared against', 'unguided, and unguided with a new seed (80% still enter: the floor)\n'
          'a random-direction penalty of the same size\n'
          'a geometry penalty from the decoded imagined path (knows the zone)\n'
          'literal steering: subtract the probe direction from block 4 (and a random direction)',
          'compare', dashed=True, body_size=8.6)
    res = g.box(116, 14, 18, 25, 'Result', 'monitor penalty, λ = 100:\n46% still enter\n(80% floor, 76% random)\n91% reach the goal\n\nrandom episodes: entry\n43% -> 23%, no loss\nof success\n\nliteral steering: no\ndirection-specific effect', 'result', body_size=8.4)
    g.path([(env['r'], 31.5), (res['x'], 31.5)])
    g.legend(['data', 'planner', 'model', 'env', 'probe', 'compare', 'result'])
    g.save('experiment_figure4.png')


def model_diagram():
    g = Diagram('Probing LeWM: where the monitor reads and where the experiments intervene',
                'One planning step. The predictor is called 5 times (one per action block); every call runs the same 6 blocks',
                xlim=140, ylim=60)
    ORANGE, RED, BLUE = '#E65100', '#C62828', '#1565C0'

    ot = g.box(1, 43, 14, 7, r'$o_t$: current frame', '', 'data', title_size=10)
    og = g.box(1, 33.5, 14, 7, r'$o_g$: goal frame', '', 'data', title_size=10)
    enc = g.box(19, 33, 20, 18, r'Encoder $g_\phi$', 'ViT-tiny, 12 layers;\nCLS token, then a\nprojector MLP;\nsame weights for\nboth frames', 'model', body_size=9)
    zg = g.box(44, 45.5, 13, 6, r'$z_g \in \mathbb{R}^{192}$', '', 'model', title_size=11.5)
    g.path([(ot['r'], ot['cy']), (enc['x'], ot['cy'])])
    g.path([(og['r'], og['cy']), (enc['x'], og['cy'])])
    g.path([(enc['r'], zg['cy']), (zg['x'], zg['cy'])])

    # predictor row
    y0, h = 17, 10
    inp = g.box(19, y0, 17, h, 'input tokens', r'$[\hat z_{k-2}, \hat z_{k-1}, \hat z_k]$' + '\n+ position', 'model', title_size=9.8, body_size=9.5)
    g.path([(enc['cx'], enc['y']), (inp['cx'], inp['t'])], label=r'$\hat z_0 = z_t$', label_at=(enc['cx'] + 4.6, 30.5), fontsize=10)
    blocks = [g.box(40 + i * 8.4, y0, 7, h, f'b{i}', '', 'model', title_size=11.5) for i in range(6)]
    norm = g.box(92, y0, 7.4, h, 'norm', '', 'model', title_size=10)
    proj = g.box(102.5, y0, 9.5, h, 'pred_proj', '', 'model', title_size=9.5)
    out = g.box(115, y0 + 1.5, 9, 7, r'$\hat z_{k+1}$', '', 'model', title_size=12)
    chain = [inp] + blocks + [norm, proj, out]
    for a, b in zip(chain, chain[1:]):
        g.path([(a['r'] + 0.2, a['cy']), (b['x'] - 0.2, a['cy'])], lw=1.3)
    cost = g.box(127.5, 15.5, 11.5, 13, 'Planning\ncost', r'$\Vert \hat z_5 - z_g \Vert^2$', 'planner', title_size=10, body_size=10.5)
    g.path([(out['r'] + 0.2, out['cy']), (cost['x'] - 0.2, out['cy'])])
    g.path([(zg['r'], zg['cy']), (139.6, zg['cy']), (139.6, 25.5), (cost['r'] + 0.2, 25.5)], lw=1.3)
    g.path([(out['cx'], out['y']), (out['cx'], 13.2), (inp['cx'], 13.2), (inp['cx'], inp['y'])], lw=1.3)
    g.ax.text(30, 13.2, r'append $\hat z_{k+1}$, call again ($k = 0 \ldots 4$)', fontsize=8.8, color=GREY, style='italic',
              ha='left', va='center', bbox=dict(fc='white', ec='none', pad=0.5))

    # planning loop: cost -> CEM -> plan -> action embedder -> every block
    cem = g.box(127.5, 36, 11.5, 8, 'CEM', r'$a^* = \arg\min_a$', 'planner', title_size=10.5, body_size=9.5)
    plan = g.box(104, 37, 19, 6, r'plan $a$: 5 x 5 actions', '', 'planner', title_size=9.5)
    emb = g.box(64, 37, 32, 6, r'action embedder $e_\psi$: $a_k \rightarrow c_k$', '', 'model', title_size=9.8)
    g.path([(cost['cx'], cost['t'] + 0.2), (cem['cx'], cem['y'] - 0.2)])
    g.path([(cem['x'] - 0.2, plan['cy']), (plan['r'] + 0.2, plan['cy'])], color='#5E35B1')
    g.path([(plan['x'] - 0.2, plan['cy']), (emb['r'] + 0.2, emb['cy'])], color='#5E35B1')
    bus = 32
    g.ax.plot([blocks[0]['cx'], blocks[-1]['cx']], [bus, bus], color=BLUE, lw=1.4)
    g.ax.plot([emb['cx'] - 6, emb['cx'] - 6], [emb['y'] - 0.2, bus], color=BLUE, lw=1.4)
    for b in blocks:
        g.path([(b['cx'], bus), (b['cx'], b['t'] + 0.25)], color=BLUE, lw=1.2)
    g.ax.text(blocks[0]['cx'] - 0.6, bus + 1.2, r'$c_k$ enters every block (adaLN: shift, scale, gate)', fontsize=8.8,
              color=BLUE, style='italic', ha='left', va='bottom')

    # Figure 1 probe taps under every layer
    taps = blocks + [norm, proj]
    for b in taps:
        g.ax.plot([b['cx'], b['cx']], [b['y'] - 0.2, 9.4], color=ORANGE, lw=1.1, ls=':')
        g.ax.plot([b['cx']], [b['y'] - 1.4], 'o', ms=8.5, mfc=ORANGE, mec='white', mew=1.2, zorder=6)
    g.ax.plot([taps[0]['cx'], taps[-1]['cx']], [9.4, 9.4], color=ORANGE, lw=1.1, ls=':')
    # Figure 2 patch points
    mid = (blocks[2]['r'] + blocks[3]['x']) / 2
    g.ax.plot([mid], [blocks[2]['cy']], 'D', ms=9, mfc=RED, mec='white', mew=1.2, zorder=7)
    g.ax.plot([blocks[4]['cx']], [bus - 2.2], 'D', ms=9, mfc=RED, mec='white', mew=1.2, zorder=7)

    def legend_box(x, w, title, body, color, marker, dashed=False, role='probe'):
        g.box(x, 0.6, w, 8.0, title, body, role, dashed=dashed, title_size=9.6, body_size=8.6)
        if marker:
            g.ax.plot([x + 1.6], [0.6 + 8.0 - 1.6], marker, ms=8, mfc=color, mec='white', mew=1, zorder=7)
    legend_box(19, 46, r'Figure 1: probes read $h^{(\ell)}_k$ at every orange tap',
               r'hazard: $P(\mathrm{enter}) = \sigma(w^\top h^{(\ell)} + b)$     displacement: $\hat\Delta = W[z_t; h^{(\ell)}_k] + b$',
               ORANGE, 'o')
    legend_box(68, 34, 'Figure 2: red points are patched',
               r'residual stream $h^{(\ell)}_k$, or the $c_k$ one block' + '\nsees, copied from the other plan\'s run', RED, 'D')
    legend_box(105, 34, 'Figure 3: every Linear layer quantized',
               'encoder, embedder, predictor, heads;\n8-bit dynamic, or weights at 6, 4, 3 bits', GREY, None, dashed=True,
               role='compare')
    g.save('model_probes.png')


def formula_sheet():
    fig, ax = plt.subplots(figsize=(16, 11.5))
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 100)
    ax.text(0, 99, 'Method in equations', fontsize=17, fontweight='bold', va='top', color='#1F2933')
    sections = [
        ('World model and planner', '#1565C0', [
            (r'$z_t = g_\phi(o_t), \quad z_g = g_\phi(o_g)$', 'encoder: frame to 192-d latent'),
            (r'$\hat z_{k+1} = f_\theta(\hat z_{k-2:k},\ c_{k-2:k}), \quad c_k = e_\psi(a_k), \quad \hat z_0 = z_t$', 'predictor, 5 calls (k = 0..4); $a_k$ = block of 5 actions'),
            (r'$a^* = \arg\min_a \ \Vert \hat z_5(a) - z_g \Vert^2$', 'CEM: 300 samples, keep top 30, 30 iterations'),
        ]),
        ('Labels (from executing $a^*$ in the environment)', '#2E7D32', [
            (r'$y = \mathbf{1}\left[\ \min_{1 \leq s \leq 25} \Vert p_{t+s} - c \Vert \leq r\ \right], \quad c = (134, 78),\ r = 10\,\mathrm{px}$', 'did the plan enter the hazard zone?'),
            (r'$\Delta_h = p_{t+h} - p_t, \quad h \in \{5, 10, 15, 20, 25\}$', 'displacement, one horizon per imagined step'),
        ]),
        ('Figure 1: probes and the decisive test', '#E65100', [
            (r'$\hat y = \sigma(w^\top h^{(\ell)} + b), \quad \min_{w,b}\ \mathcal{L}_{\mathrm{log}} + \lambda \Vert w \Vert^2$', 'logistic probe on layer $\\ell$ (all 5 imagined steps)'),
            (r'$\hat \Delta_h = W [z_t ; h^{(\ell)}_{h/5}] + b, \quad R^2 = 1 - \frac{\sum \Vert \Delta_h - \hat\Delta_h \Vert^2}{\sum \Vert \Delta_h - \bar\Delta_h \Vert^2}$', 'ridge probe for displacement'),
            (r'$\mathrm{gain} = \mathrm{AUROC}(T[x_{\mathrm{geo}} ; s_{\mathrm{lat}}]) - \mathrm{AUROC}(T[x_{\mathrm{geo}}])$', '$T$ = boosted trees; $x_{\\mathrm{geo}}$ = position, goal, line distance'),
        ]),
        ('Figure 2: activation patching', '#C62828', [
            (r'$x \leftarrow x + \gamma_1(c) \odot \mathrm{Attn}(\mathrm{mod}(x; c)) + \gamma_2(c) \odot \mathrm{MLP}(\mathrm{mod}(x; c))$', 'one adaLN block: actions enter through $c$'),
            (r'$h^{(\ell)}_k \leftarrow h^{(\ell)}_k|_{\mathrm{safe}} \quad \mathrm{or} \quad c^{(i)}_k \leftarrow c^{(i)}_k|_{\mathrm{safe}}$', 'patch the residual stream, or the actions block $i$ sees'),
            (r'$F = \frac{\sum_{\mathrm{pairs}} (m_{\mathrm{patched}} - m_{\mathrm{unsafe}})}{\sum_{\mathrm{pairs}} (m_{\mathrm{safe}} - m_{\mathrm{unsafe}})}, \quad m = \min_k \Vert \mathrm{dec}(\hat z_k) - c \Vert$', 'fraction of the gap moved; dec = position decoder'),
        ]),
        ('Figure 3: quantization and the frozen monitor', '#5E35B1', [
            (r'$\tilde W_{j,:} = s_j \cdot \mathrm{clip}(\mathrm{round}(W_{j,:} / s_j), -Q, Q), \quad s_j = \max |W_{j,:}| / Q, \quad Q = 2^{b-1} - 1$', 'weight rounding per output channel, b = 6, 4, 3'),
            (r'$\mathrm{miss} = P(\hat y < \tau \mid y = 1), \quad \tau: P(\hat y \geq \tau \mid y = 1) = 0.9 \ \mathrm{on\ training\ folds}$', 'silent failure: monitor says safe, plan enters the zone'),
        ]),
    ]
    y = 92
    for title, color, rows in sections:
        ax.text(0, y, title, fontsize=12.5, fontweight='bold', color=color, va='top')
        ax.plot([0, 100], [y - 3.2, y - 3.2], color=color, lw=0.8, alpha=0.5)
        y -= 5.6
        for eq, desc in rows:
            ax.text(2, y, eq, fontsize=13, va='top', color='#1F2933')
            ax.text(100, y - 0.4, desc, fontsize=9.8, va='top', ha='right', color=GREY, style='italic')
            y -= 7.6 if r'\frac' in eq else 5.4
        y -= 1.6
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'formulas.png', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    figure1()
    figure2()
    figure3()
    figure4()
    model_diagram()
    formula_sheet()
    print(f'wrote {OUT.relative_to(C.ROOT)}/')
