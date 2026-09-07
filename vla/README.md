# Vision-Language-Action demos

This directory contains small, runnable examples using the
[`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero)
checkpoint with the [LIBERO](https://huggingface.co/docs/lerobot/libero)
MuJoCo benchmark. The checkpoint predicts 7-dimensional robot actions from
camera images, robot state, and a natural-language instruction.

The RWFT example intentionally optimizes an artificial “move right” reward.
It is for demonstrating how fine-tuning alters policy behavior, not for
learning the original LIBERO manipulation task.

## Setup

From the repository root, run:

```bash
./vla/setup.sh
```

The script creates or reuses `.venv`, installs LeRobot with SmolVLA, dataset,
LIBERO, and MuJoCo dependencies, initializes LIBERO non-interactively, and
downloads the default checkpoint and dataset.

To install dependencies and download only the model:

```bash
./vla/setup.sh --model-only
```

The first simulator rollout also downloads LIBERO assets to the local cache.
For a headless Linux system, the scripts default to `MUJOCO_GL=egl`. Override
that environment variable if your system needs a different MuJoCo backend.

## Scripts

### `vla_demo_onesample.py`

Runs the model on one recorded `lerobot/libero` sample and prints a 10-step
action chunk. It does not start a simulator.

```bash
./.venv/bin/python vla/vla_demo_onesample.py
```

Useful options:

```bash
--episode 0 --frame-offset 0 --chunk-steps 10
--model path-or-hugging-face-model-id
--dataset hugging-face-dataset-id
```

### `vla_demo_viz.py`

Runs a closed-loop LIBERO rollout: each predicted action is executed in MuJoCo,
which produces the next observation. By default it executes up to 50 steps,
then saves:

- `vla_outputs/closed_loop_summary.png`
- `vla_outputs/closed_loop_execution.gif`

```bash
./.venv/bin/python vla/vla_demo_viz.py --steps 50
```

Select another suite/task or save elsewhere:

```bash
./.venv/bin/python vla/vla_demo_viz.py \
  --suite libero_spatial --task-id 0 --seed 0 \
  --output-dir vla_outputs/baseline_rollout
```

### `vla_simple_rwft.py`

Demonstrates iterative reward-weighted fine-tuning (RWFT):

1. Collect fresh on-policy 50-step simulator rollouts.
2. Score each rollout by net positive-X end-effector displacement.
3. Weight action-chunk imitation losses by those rewards.
4. Update only the small action-output projection.
5. Repeat the collection/training outer loop.

The default run uses three iterations, eight rollouts per iteration, and 20
updates per collection batch:

```bash
./.venv/bin/python vla/vla_simple_rwft.py
```

For a fast smoke test:

```bash
./.venv/bin/python vla/vla_simple_rwft.py \
  --iterations 1 --rollouts 2 --updates 1
```

Checkpoints are saved every 10 iterations by default:

```text
vla_outputs/move_right_rwft/checkpoint-010/
```

Set `--checkpoint-freq` to change that cadence. The final policy and
`rwft_metrics.json` are always written to `--output-dir`.

## Running a fine-tuned checkpoint

Use a saved checkpoint in either inference demo:

```bash
./.venv/bin/python vla/vla_demo_onesample.py \
  --model vla_outputs/move_right_rwft/checkpoint-010

./.venv/bin/python vla/vla_demo_viz.py \
  --model vla_outputs/move_right_rwft/checkpoint-010 \
  --steps 50 \
  --output-dir vla_outputs/move_right_rollout
```

Compare it with the original checkpoint under the same `--suite`, `--task-id`,
and `--seed` to observe the learned rightward bias.
