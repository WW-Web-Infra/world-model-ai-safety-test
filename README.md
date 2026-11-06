# world-model-ai-safety-test

Can we read a latent planner's intent before it acts? Interpreting a JEPA world model, and testing whether the monitor survives quantization

The model is [LeWorldModel](https://arxiv.org/abs/2603.19312) (LeWM), a JEPA world model that plans in latent space with the cross-entropy method (CEM), in its TwoRooms navigation task. A hazard zone is added that the planner is never told about: it is not drawn in the frames and not in the planner's objective. Everything runs on a laptop CPU.

## Why this matters

Capable agents increasingly plan in a learned latent space instead of in text. A language model's chain of thought can be read before it acts ([Korbak et al., 2025](https://arxiv.org/abs/2507.11473)); a plan computed as a sequence of latent vectors cannot, unless we can decode it. Robots also run compressed models, so a monitor validated on the full model could keep reporting "safe" after quantization has changed what the planner does.

Probing studies of world models ask what a model knows about its world ([Zhang, 2026](https://arxiv.org/abs/2603.21546); [Joseph et al., 2026](https://arxiv.org/abs/2602.07050)). They do not test whether an overseer can read the plan a planner has chosen, including a property it was never trained on, before the plan runs. Studies of whether activation monitors survive model updates or quantization have used language models only ([Do Activation Monitors Survive Model Updates?, 2026](https://arxiv.org/abs/2606.15980); [Duan, 2026](https://arxiv.org/abs/2606.03002)). This project tests both on a planner that reasons only in latent space: if a cheap linear monitor can read the plan, an overseer can check it before the agent acts, and the check has to keep working on the compressed model that actually runs.

![Why this question matters](figures/why_it_matters.gif)

## Results

A linear monitor can read where the planner is about to go, including whether its plan will enter the hazard zone, before the plan runs. It keeps working after 8-, 6- and 4-bit quantization, degrades at 3 bits where a retrained probe recovers most of the loss, and can be used as a penalty in the planner's cost to steer plans away from the hazard.

![Summary of the four experiments](figures/summary.png)

| | Question | Result |
| --- | --- | --- |
| Figure 1 | Can a linear probe read the plan? | Hazard-entry AUROC 0.987 from the predictor's activations, against 0.928 for boosted trees on start, goal and line-to-zone distance. Adding the activations to that baseline gains 0.054 to 0.059. Displacement 5 to 25 steps ahead: R² 0.90 to 0.94. |
| Figure 2 | Where is the route computed? | Activation patching between same-start safe and unsafe plans: each of predictor blocks 0 to 4 uses the plan's actions (13% to 28% of the effect each, block 5 least), blocks 0 to 4 together give 96%, and the residual stream after block 4 carries the whole route. |
| Figure 3 | Does the monitor survive quantization? | At 8, 6 and 4 bits the frozen probe's miss rate (7.8% to 9.0%) stays at the full-precision level (8.7%); its AUROC is 0.985 to 0.988 at 8 and 6 bits and dips to 0.981 at 4 bits (0.987 at full precision). Plan changes at these precisions are smaller than a new planner seed causes. At 3 bits the frozen AUROC falls to 0.970 and the miss rate doubles to 17.7%; a probe retrained on 3-bit activations recovers most of the loss (0.982). |
| Figure 4 | Can the monitor steer the planner? | Adding 100 x the probe's hazard probability to CEM's cost lowers hazard entry on previously unsafe episodes to 46% (80% with a new seed alone, 76% with a random-direction penalty) while 91% still reach the goal. Literal activation steering has no effect specific to the hazard direction: a random direction lowers hazard entry as much or more. |

## The setup in pictures

How the model plans and acts, and how the monitor is trained:

![Setup at a glance](figures/setup.gif)

What the model sees, and where the hazard is (the model never sees the pink disk):

![Start frame, goal frame and an annotated episode](figures/example_frames.png)

Where the monitor reads the model and where the experiments intervene:

![LeWM architecture with probe taps and patch points](figures/model_probes.png)

The method in equations:

![Method in equations](figures/formulas.png)

## Experiments

### Figure 1: can a linear monitor read the plan before it runs?

![Figure 1 experiment design](figures/experiment_figure1.png)

![Figure 1 results](figures/results_figure1.png)

### Figure 2: where does the predictor turn the plan into its path?

![Figure 2 experiment design](figures/experiment_figure2.png)

![Figure 2 results](figures/results_figure2.png)

### Figure 3: does the monitor survive quantization?

![Figure 3 experiment design](figures/experiment_figure3.png)

![Figure 3 results](figures/results_figure3.png)

### Figure 4: can the monitor steer the planner?

![Figure 4 experiment design](figures/experiment_figure4.png)

![Figure 4 results](figures/results_figure4.png)

## Method in brief

- **Episodes:** 2,000 seeded start-goal pairs from the LeWM dataset, at most 50 px apart, whose straight line passes within 22 px of the zone (radius 10 px at (134, 78)); start and goal more than 14 px from its centre.
- **Recording:** the chosen plan is replayed through the predictor with hooks on every block; labels come from executing the plan in the environment.
- **Evaluation:** 5-fold cross-validation split by episode, regularisation and the headline layer chosen inside training folds, 95% intervals from resampling episodes, and a planner-seed control as the noise floor for behaviour changes.
- **Controls:** flexible geometry baselines, a randomly initialised and a weight-shuffled model, shuffled labels, and the plan's own actions as a reference.

## Setup

Python 3.13, CPU only. Install the locked, hash-checked dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements.lock
```

`requirements.txt` lists the top-level packages at the versions the results were produced with; `requirements.lock` pins every package with its SHA-256 hashes. `transformers` must stay below 5: the checkpoint uses the 4.x ViT layer names.

The TwoRooms checkpoint downloads on first use into `.stable-wm/`, from a pinned revision of `quentinll/lewm-tworooms` on Hugging Face, and is checked against pinned SHA-256 hashes on every load (`lewm_probe/checkpoint.py`).

The dataset is only needed for start-goal pairs and action statistics. Download `tworoom.tar.zst` (3.4 GB, no account needed) from the pinned dataset revision, check its hash, and extract `tworoom.h5` to `.stable-wm/datasets/`:

```bash
curl -L -o tworoom.tar.zst https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/6903a2de048b13819d812da0b4dd661290bc01e4/tworoom.tar.zst
```

```bash
echo "494b1a02f0765cd9a0d9daf1786c419ced1009977fc45d01e3158932f8d080ca  tworoom.tar.zst" | shasum -a 256 -c
```

```bash
mkdir -p .stable-wm/datasets && tar --zstd -xf tworoom.tar.zst -C .stable-wm/datasets tworoom.h5
```

## Security notes

- The checkpoint config is turned into a model by Hydra, which imports whatever each `_target_` names. `lewm_probe/checkpoint.py` loads only from a pinned revision, verifies SHA-256 hashes on every load, accepts only the six classes this checkpoint uses, and loads weights with `torch.load(weights_only=True)`.
- Saved probes and the position decoder are plain `.npz` arrays read with `allow_pickle=False` (`lewm_probe/frozen.py`); nothing in this repo loads pickles.
- Dependencies are locked with hashes in `requirements.lock`.
- `transformers` stays on 4.57.6 because 5.x renames the ViT's modules and can no longer load this checkpoint. The open advisories against 4.57.6 cover remote code, `Trainer`, generation, `save_pretrained` and LightGlue loading; this repo only builds a ViT from a config (`ViTModel(ViTConfig(...))`) and downloads nothing through `transformers`.

## Reproducing

Run from the repo root with `.venv/bin/python`. Data goes to `data/` and figures to `outputs/` (both git-ignored); the copies shown in this README are in `figures/`.

| Step | Command | Time on an M4 laptop |
| --- | --- | --- |
| Check the install, reproduce the upstream eval (90% success) | `scripts/feasibility_check.py`, `scripts/upstream_eval.py` | minutes |
| Pick and test the hazard zone | `scripts/choose_hazard.py` | ~5 min |
| Collect 2,000 seeded episodes (two shards) | `scripts/collect.py --name full --first 0 --episodes 1000` and `--first 1000` | ~3 h |
| Validate the data (recomputes every label, checks hooks, reruns episodes) | `scripts/validate_data.py data/full/*.npz --rerun 5` | minutes |
| Replay plans through untrained controls | `scripts/replay_controls.py` | ~5 min |
| Figure 1 | `scripts/figure1.py` | ~8 min on 8 cores |
| Safe/unsafe plan pairs, then Figure 2 | `scripts/make_pairs.py --first 0 --count 250` (and `--first 250`), then `scripts/figure2.py` | ~1.5 h, then minutes |
| Quantized reruns and the planner-seed control | `scripts/run_quantization_sweep.sh`, then `scripts/run_reseed_control.sh` | ~14 h, then ~3.5 h |
| Figure 3 | `scripts/figure3.py` | ~15 min |
| Steering runs, then Figure 4 | `scripts/run_steering.sh`, then `scripts/figure4.py` | ~3 h, then seconds |
| Experiment diagrams, the summary card and the animated GIFs | `scripts/diagrams.py` | seconds |
| Plain-language charts for the blog post | `scripts/blog_figures.py` | seconds |

Collection is seeded by episode id and resumes from its last checkpoint if interrupted, so reruns give identical data. Run long jobs one process at a time on a laptop charger that cannot supply full load.

## Code

- `lewm_probe/`: model loading and planner setup (`common.py`), episode runner and activation replay (`episode.py`), hazard zone and pair sampling (`hazard.py`), dataset loading and targets (`dataset.py`), probes (`probes.py`), patching hooks (`patching.py`), position decoder (`position_decoder.py`), quantization (`quantize.py`), guided planning and steering (`steering.py`).
- `scripts/`: one script per step in the table above.

## Limitations

TwoRooms is a 2-D toy task with one zone placement, so the results show the method works on a real latent planner, not that it scales to robots. The monitor reads one extra predictor pass on the chosen plan. The probes were not tested against a model trained to evade them. The 6-, 4- and 3-bit models use rounded weights computed in floating point; only the 8-bit model uses integer kernels.

## License

The code in this repository is released under the [MIT License](LICENSE). The LeWM checkpoint and the TwoRooms dataset are not part of this repository; they are downloaded from Hugging Face (`quentinll/lewm-tworooms`) and remain under their authors' terms.
