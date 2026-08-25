# AlphaZero Connect-4

This directory contains a Connect-4 implementation with classic UCT-MCTS,
AlphaZero self-play training, and a Gradio interface for playing against either
agent. 

## Usage

Run the following commands from the repository root.

### Setup

```bash
.venv/bin/pip install -r model_based_rl/requirements.txt
```

The interactive search-tree view also needs the Graphviz system binary:

```bash
sudo apt install graphviz
```

### Play against classic MCTS

Start the UI without a checkpoint:

```bash
.venv/bin/python model_based_rl/connect4_alphazero/play_gui.py
```

Open the URL printed by Gradio (normally `http://127.0.0.1:7860`). The UI lets
you configure the board, choose who moves first, and adjust the MCTS
simulation budget and exploration constant.

### Train an AlphaZero model

Training alternates between MCTS-guided self-play and neural-network updates.
For a small CPU smoke test:

```bash
.venv/bin/python model_based_rl/connect4_alphazero/train.py \
  --device cpu \
  --iters 2 \
  --games 4 \
  --sims 25 \
  --train-steps 20 \
  --checkpoint model_based_rl/connect4_alphazero/checkpoints/connect4_az.pt
```

For faster training on GPU, use `model_based_rl/connect4_alphazero/train_parallel.py`

For a longer run, increase `--iters`, `--games`, `--sims`, and
`--train-steps`; use `--device cuda` when CUDA is available. A checkpoint and
its replay buffer are saved after each evaluation (`--eval-every`, default 5)
and at the final iteration.

Resume a previous run with:

```bash
.venv/bin/python model_based_rl/connect4_alphazero/train.py \
  --resume model_based_rl/connect4_alphazero/checkpoints/connect4_az.pt \
  --iters 100
```

### Play against a trained model

Pass a training checkpoint to run PUCT-MCTS guided by the model's policy and
value heads:

```bash
.venv/bin/python model_based_rl/connect4_alphazero/play_gui.py \
  --checkpoint model_based_rl/connect4_alphazero/checkpoints/connect4_az.pt
```

The checkpoint fixes the board dimensions and connect target. The UI still
lets you set the computer's search budget and PUCT exploration constant. To
force a device, append `--device cpu` or `--device cuda`.

### Compute Elo ratings

`compute_elo.py` runs round-robin matches among the players returned by
`get_players()` and fits relative Elo ratings from their pairwise results.

1. Edit `MODEL_CKPT` and the list of player configurations in
   `compute_elo.py:get_players()` to select the checkpoints and MCTS budgets
   you want to compare.
2. Set each player's `device` appropriately (`cpu` or `cuda`).
3. Run:

```bash
.venv/bin/python model_based_rl/connect4_alphazero/compute_elo.py
```

The script prints the ratings and writes `elo_ratings.json` in the current
working directory. Ratings are relative: the lowest rated configured player is
assigned Elo 0.

To visualize Elo ratings, run `plot_elo.ipynb` and point to the `elo_ratings.json`.

----------------------------------------------------------------------------------------

## Important Code Pointers

### Pure MCTS

Classic UCT search is implemented in `mcts.py`. Start with the `Node` class
and `mcts_search()` for selection, expansion, random-rollout evaluation, and
backpropagation. `MCTSPlayer` wraps that search for automated matches, while
`play_gui.py` calls it when launched without a checkpoint. Shared game state,
legal-move, and win-condition logic lives in `core.py`.

### AlphaZero MCTS Inference

`alpha_zero.py` contains the learned inference stack. `AlphaZeroNet` defines
the policy-value network, `net_predict()` evaluates a board, and `AZMCTS`
implements PUCT search using network priors and values instead of random
rollouts. `load_az_checkpoint()` restores a trained network; `AZMCTSPlayer`
and `PlainAlphaZeroPlayer` expose the searched and direct-network variants for
evaluation and Elo matches. The checkpoint-enabled path in `play_gui.py`
constructs and uses `AZMCTS`.

### AlphaZero Training

The self-play training loop is in `train.py`. `play_game()` generates
policy/value targets with `AZMCTS`, `ReplayBuffer` retains those samples, and
`train_steps()` optimizes `AlphaZeroNet` using `alphazero_loss()`. The
module's `main()` function orchestrates self-play, training, evaluation,
checkpointing, and replay-buffer persistence. `train_parallel.py` contains
the GPU-oriented parallel training entry point.