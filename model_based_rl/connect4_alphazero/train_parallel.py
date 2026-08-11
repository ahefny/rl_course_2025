"""Parallel AlphaZero Connect-4 trainer with batched GPU inference.

Self-play runs many games in lockstep. At each MCTS simulation round, every
active game contributes at most one leaf; those leaves are evaluated in a
single batched forward pass. Tree select / expand / backup stay on the CPU.

Imports training helpers from ``train.py``

Quick sanity check:
    python model_based_rl/connect4_alphazero/train_parallel.py --iters 2 --games 8 \\
        --sims 32 --parallel-games 8 --train-steps 50

Fuller run (GPU recommended):
    python model_based_rl/connect4_alphazero/train_parallel.py --iters 50 --games 64 \\
        --sims 100 --parallel-games 32
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import Connect4State
from alpha_zero import (
    AZNode,
    AlphaZeroNet,
    encode_state,
    outcome_for,
)
from train import (
    ReplayBuffer,
    Sample,
    evaluate_vs_pure_mcts,
    evaluate_vs_random,
    load_replay_buffer,    
    replay_buffer_path,
    save_checkpoint,
    train_steps,
)

# ===========================================================================
# Batched network evaluation
# ===========================================================================


@torch.no_grad()
def net_predict_batch(
    net: AlphaZeroNet,
    states: Sequence[Connect4State],
    device: torch.device,
) -> list[tuple[np.ndarray, float]]:
    """Masked policy probs + value for each state, one forward for the batch."""
    if not states:
        return []
    net.eval()
    cols = states[0].cols
    x = np.stack([encode_state(s) for s in states]).astype(np.float32)
    logits_t, values_t = net(torch.from_numpy(x).to(device))
    logits = logits_t.detach().cpu().numpy()
    values = values_t.detach().cpu().numpy()

    out: list[tuple[np.ndarray, float]] = []
    for i, state in enumerate(states):
        row = logits[i].astype(np.float32, copy=True)
        mask = np.full(cols, -np.inf, dtype=np.float32)
        mask[state.get_moves()] = 0.0
        row = row + mask
        row = row - row.max()
        probs = np.exp(row)
        probs /= probs.sum()
        out.append((probs.astype(np.float32), float(values[i])))
    return out


def _select(node: AZNode, c_puct: float) -> tuple[int, AZNode]:
    move, child = max(
        node.children.items(),
        key=lambda kv: kv[1].puct(c_puct),
    )
    return move, child


def _backup(node: AZNode | None, value: float) -> None:
    while node is not None:
        node.N += 1
        node.W += value
        node.Q = node.W / node.N
        value = -value
        node = node.parent


def _expand_with_priors(
    node: AZNode, state: Connect4State, priors: np.ndarray,
) -> None:
    for move in state.get_moves():
        node.children[move] = AZNode(
            parent=node, move=move, prior=float(priors[move]),
        )


def _add_dirichlet_noise(
    root: AZNode, alpha: float, eps: float, rng: np.random.Generator,
) -> None:
    moves = list(root.children.keys())
    if not moves:
        return
    noise = rng.dirichlet([alpha] * len(moves))
    for m, n in zip(moves, noise):
        child = root.children[m]
        child.prior = (1.0 - eps) * child.prior + eps * float(n)


def _visit_policy(root: AZNode, cols: int) -> np.ndarray:
    policy = np.zeros(cols, dtype=np.float32)
    for move, child in root.children.items():
        policy[move] = child.N
    total = policy.sum()
    if total > 0:
        policy /= total
    return policy


# ===========================================================================
# Parallel self-play (many games, batched leaves)
# ===========================================================================


@dataclass
class _LiveGame:
    state: Connect4State
    history: list[tuple[np.ndarray, np.ndarray, int]] = field(default_factory=list)
    move_idx: int = 0
    root: AZNode | None = None
    root_state: Connect4State | None = None
    sims_done: int = 0


def _new_game(rows: int, cols: int, connect: int) -> _LiveGame:
    return _LiveGame(state=Connect4State(rows, cols, connect))


def _finalize(game: _LiveGame) -> list[Sample]:
    return [
        Sample(planes=planes, policy=pi, value=outcome_for(game.state, player))
        for planes, pi, player in game.history
    ]


def play_games_batched(
    net: AlphaZeroNet,
    device: torch.device,
    n_games: int,
    rows: int,
    cols: int,
    connect: int,
    n_sims: int,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    temp_moves: int = 10,
    parallel_games: int = 8,
    seed: int | None = None,
) -> list[Sample]:
    """Generate `n_games` self-play games with batched leaf evaluation.

    Up to `parallel_games` matches run concurrently. Each MCTS simulation
    round evaluates one leaf per active search in a single GPU/CPU batch.
    """
    rng = np.random.default_rng(seed)
    parallel_games = max(1, min(parallel_games, n_games))
    pending = n_games
    active: list[_LiveGame] = []
    finished: list[Sample] = []

    def _refill() -> None:
        nonlocal pending
        while pending > 0 and len(active) < parallel_games:
            active.append(_new_game(rows, cols, connect))
            pending -= 1

    _refill()

    while active:
        # --- start a fresh search on every game that needs a move ---
        need_root: list[_LiveGame] = []
        for g in active:
            if g.root is None:
                g.root = AZNode(parent=None, move=None, prior=1.0)
                g.root_state = g.state.clone()
                g.sims_done = 0
                need_root.append(g)

        if need_root:
            root_states = [g.root_state for g in need_root]
            assert all(s is not None for s in root_states)
            preds = net_predict_batch(
                net, root_states, device,  # type: ignore[arg-type]
            )
            for g, (priors, _value) in zip(need_root, preds):
                assert g.root is not None and g.root_state is not None
                _expand_with_priors(g.root, g.root_state, priors)
                if dirichlet_eps > 0.0:
                    _add_dirichlet_noise(
                        g.root, dirichlet_alpha, dirichlet_eps, rng,
                    )
            # Root network value is discarded (same as AZMCTS.run).

        # --- n_sims batched simulation rounds ---
        for _ in range(n_sims):
            leaves_g: list[_LiveGame] = []
            leaves_node: list[AZNode] = []
            leaves_state: list[Connect4State] = []
            # Terminal leaves: backup immediately (no NN).
            for g in active:
                assert g.root is not None and g.root_state is not None
                node = g.root
                state = g.root_state.clone()
                while node.is_expanded() and not state.is_terminal():
                    move, node = _select(node, c_puct)
                    state.do_move(move)
                if state.is_terminal():
                    value = outcome_for(state, state.player_just_moved)
                    _backup(node, value)
                    g.sims_done += 1
                else:
                    leaves_g.append(g)
                    leaves_node.append(node)
                    leaves_state.append(state)

            if leaves_state:
                preds = net_predict_batch(net, leaves_state, device)
                for g, node, state, (priors, value) in zip(
                    leaves_g, leaves_node, leaves_state, preds,
                ):
                    _expand_with_priors(node, state, priors)
                    # value is to-play's; backup wants mover-into-node view.
                    _backup(node, -value)
                    g.sims_done += 1

        # --- act with the visit policy; retire finished games ---
        still_active: list[_LiveGame] = []
        for g in active:
            assert g.root is not None
            policy = _visit_policy(g.root, cols)
            g.history.append(
                (encode_state(g.state), policy.copy(), g.state.to_play),
            )
            tau = 1.0 if g.move_idx < temp_moves else 0.0
            # Prefer numpy RNG for reproducibility when seed is set; fall back
            # to the module helper (which uses global np.random) for tau=0.
            if tau <= 1e-3:
                move = int(np.argmax(policy))
            else:
                logits = np.log(np.maximum(policy, 1e-12)) / tau
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                move = int(rng.choice(len(policy), p=probs))
            g.state.do_move(move)
            g.move_idx += 1
            g.root = None
            g.root_state = None
            g.sims_done = 0
            if g.state.is_terminal():
                finished.extend(_finalize(g))
            else:
                still_active.append(g)

        active = still_active
        _refill()

    return finished


# ===========================================================================
# CLI + training loop
# ===========================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AlphaZero Connect-4 trainer (parallel self-play, batched inference)",
    )
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=7)
    p.add_argument("--connect", type=int, default=4)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--iters", type=int, default=30,
                   help="Outer training iterations (self-play + update)")
    p.add_argument("--games", type=int, default=20,
                   help="Self-play games per iteration")
    p.add_argument("--parallel-games", type=int, default=8,
                   help="Games searched concurrently (NN batch size per sim round)")
    p.add_argument("--sims", type=int, default=100,
                   help="MCTS simulations per move")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--temp-moves", type=int, default=10,
                   help="Use tau=1 for the first this-many moves, then greedy")
    p.add_argument("--buffer", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-steps", type=int, default=200,
                   help="Gradient steps per iteration")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--eval-azmcts-sims", type=int, default=-1,
                   help="AZ-MCTS simulations per move during eval "
                        "(-1 = use --sims)")
    p.add_argument("--eval-mcts-sims", type=int, default=400,
                   help="Pure UCT-MCTS simulations per move for the eval opponent")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str,
                   default="model_based_rl/connect4_alphazero/checkpoints/connect4_az_parallel.pt")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(
        f"device={device}  board={args.rows}x{args.cols}  "
        f"sims={args.sims}  games/iter={args.games}  "
        f"parallel_games={args.parallel_games}"
    )

    net = AlphaZeroNet(
        rows=args.rows,
        cols=args.cols,
        hidden_dim=args.hidden_dim,
        blocks=args.blocks,
    ).to(device)
    opt = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buf = ReplayBuffer(args.buffer)
    start_iter = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_iter = int(ckpt.get("iteration", 0)) + 1
        loaded = load_replay_buffer(args.resume, args.buffer)
        if loaded is not None:
            buf = loaded
            print(
                f"resumed from {args.resume} at iteration {start_iter}  "
                f"buf={len(buf)} ({replay_buffer_path(args.resume)})"
            )
        else:
            print(
                f"resumed from {args.resume} at iteration {start_iter}  "
                f"no replay file {replay_buffer_path(args.resume)}; "
                f"starting with empty buffer"
            )

    for it in range(start_iter, args.iters + 1):
        t0 = time.time()
        samples = play_games_batched(
            net, device,
            n_games=args.games,
            rows=args.rows,
            cols=args.cols,
            connect=args.connect,
            n_sims=args.sims,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_eps=args.dirichlet_eps,
            temp_moves=args.temp_moves,
            parallel_games=args.parallel_games,
            seed=args.seed + it,
        )
        buf.extend(samples)
        n_samples = len(samples)
        play_s = time.time() - t0

        t1 = time.time()
        stats = train_steps(
            net, opt, buf, device, args.train_steps, args.batch_size,
        )
        train_s = time.time() - t1

        print(
            f"iter {it:3d}/{args.iters}  "
            f"samples+={n_samples:4d}  buf={len(buf):6d}  "
            f"loss={stats['loss']:.3f} (p={stats['p']:.3f} v={stats['v']:.3f})  "
            f"play={play_s:.1f}s train={train_s:.1f}s"
        )

        if it % args.eval_every == 0 or it == args.iters:
            eval_az_sims = (
                args.sims if args.eval_azmcts_sims < 0 else args.eval_azmcts_sims
            )
            ev = evaluate_vs_random(
                net, device, args.rows, args.cols, args.connect,
                n_games=args.eval_games, sims=eval_az_sims,
            )
            print(
                f"  eval vs random: "
                f"W/D/L={ev['wins']}/{ev['draws']}/{ev['losses']}  "
                f"win_rate={ev['win_rate']:.2f} over {ev['n_games']} games "
                f"(AZ sims={eval_az_sims})"
            )
            t_eval = time.time()
            ev_mcts = evaluate_vs_pure_mcts(
                net, device, args.rows, args.cols, args.connect,
                n_games=args.eval_games,
                net_sims=eval_az_sims,
                opp_sims=args.eval_mcts_sims,
            )
            print(
                f"  eval vs pure MCTS ({args.eval_mcts_sims} sims): "
                f"W/D/L={ev_mcts['wins']}/{ev_mcts['draws']}/{ev_mcts['losses']}  "
                f"win_rate={ev_mcts['win_rate']:.2f} over {ev_mcts['n_games']} games "
                f"(AZ sims={eval_az_sims}, {time.time() - t_eval:.1f}s)"
            )
            save_checkpoint(args.checkpoint, net, opt, it, args, buf=buf)
            print(
                f"  saved {args.checkpoint}  "
                f"and {replay_buffer_path(args.checkpoint)} (buf={len(buf)})"
            )

    print("done.")


if __name__ == "__main__":
    main()
