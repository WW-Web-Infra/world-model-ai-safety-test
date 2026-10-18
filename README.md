# world-model-ai-safety-test

Can we read a latent planner's intent before it acts? Interpreting a JEPA world model, and testing whether the monitor survives quantization

The model is [LeWorldModel](https://arxiv.org/abs/2603.19312) (LeWM), a ~15M-parameter JEPA world model that plans in latent space with CEM, in its TwoRooms navigation task. A hazard zone is added that the planner is never told about. Everything runs on a laptop CPU.

## Results so far

- **Figure 1: the plan is readable.** A linear probe on the predictor's activations predicts whether a plan will enter the hazard zone with AUROC 0.987, against 0.928 for boosted trees on position and goal, and adds +0.06 on top of that geometry baseline. Displacement 5 to 25 steps ahead is readable with R² 0.90 to 0.94.
- **Figure 2: where the route is computed.** Activation patching between same-start safe and unsafe plans shows the plan's actions are used across predictor blocks 0 to 4, with no single block sufficient; the route is fully carried by block 4's output.
- **Figure 3: the monitor survives realistic quantization.** At 8, 6 and 4 bits the frozen probe's AUROC and miss rate stay within the noise of re-running the planner with a different seed. At 3 bits the planner reaches the goal only 79% of the time and the miss rate doubles (8.7% to 17.7%); a probe retrained on 3-bit activations recovers AUROC 0.982.

## Setup

Python 3.13, CPU only.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The TwoRooms checkpoint (`quentinll/lewm-tworooms`) downloads on first use into `.stable-wm/`. The dataset is only needed for start/goal pairs and action statistics: download `tworoom.tar.zst` (3.4 GB, no account needed) from the `quentinll/lewm-tworooms` dataset repo on Hugging Face, extract it, and place `tworoom.h5` at `.stable-wm/datasets/tworoom.h5`.

## Reproducing

Run from the repo root with `.venv/bin/python`. Data goes to `data/` and figures to `outputs/` (both git-ignored).

| Step | Command | Time on an M4 laptop |
| --- | --- | --- |
| Check the install, reproduce the upstream eval (90% success) | `scripts/feasibility_check.py`, `scripts/upstream_eval.py` | minutes |
| Pick and test the hazard zone | `scripts/choose_hazard.py` | ~5 min |
| Collect 2,000 seeded episodes (two shards) | `scripts/collect.py --name full --first 0 --episodes 1000` and `--first 1000` | ~3 h |
| Validate the data (recomputes every label, checks hooks, reruns episodes) | `scripts/validate_data.py data/full/*.npz --rerun 5` | minutes |
| Replay plans through untrained controls | `scripts/replay_controls.py` | ~5 min |
| Figure 1 | `scripts/figure1.py` | ~8 min on 8 cores |
| Safe/unsafe plan pairs, then Figure 2 | `scripts/make_pairs.py --first 0 --count 250` (and `--first 250`), then `scripts/figure2.py` | ~1.5 h, then minutes |
| Quantized reruns and the planner-seed control | `scripts/run_quantization_sweep.sh`, then `scripts/run_reseed_control.sh` | ~16 h |
| Figure 3 | `scripts/figure3.py` | ~15 min |
| Experiment diagrams | `scripts/diagrams.py` | seconds |

Collection is seeded by episode id and resumes from its last checkpoint if interrupted, so reruns give identical data. `transformers` must stay below 5: the checkpoint uses the 4.x ViT layer names.

## Code

- `lewm_probe/`: model loading and planner setup (`common.py`), episode runner and activation replay (`episode.py`), hazard zone and pair sampling (`hazard.py`), dataset loading and targets (`dataset.py`), probes (`probes.py`), patching hooks (`patching.py`), position decoder (`position_decoder.py`), quantization (`quantize.py`).
- `scripts/`: one script per step in the table above.
