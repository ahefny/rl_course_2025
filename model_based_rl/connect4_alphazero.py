"""Train a policy+value network to play Connect-4 with the AlphaZero algorithm.

Self-play games are generated with PUCT-MCTS guided by the network (no random
rollouts). Each position is stored with the MCTS visit distribution (policy
target) and the eventual game outcome (value target). The network is then
trained on a replay buffer of those triples.

Reuse of game rules:
    from connect4_mcts import Connect4State, EMPTY, P1, P2

Run a quick sanity check (CPU/GPU, ~1 min):
    python model_based_rl/connect4_alphazero.py --iters 3 --games 4 --sims 32 \\
        --train-steps 50 --batch-size 64 --hidden-dim 512 --blocks 4

Fuller training (GPU recommended):
    python model_based_rl/connect4_alphazero.py --iters 50 --games 25 --sims 100

Layout of this file:
    1. Board encoding
    2. Neural network (policy + value heads)
    3. PUCT MCTS guided by the network
    4. Self-play + replay buffer
    5. Training loop + CLI
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

# Allow `python model_based_rl/connect4_alphazero.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from connect4_mcts import P1, P2, Connect4State, best_move, mcts_search

# ===========================================================================
# 1. BOARD ENCODING
# ===========================================================================
# Two binary planes, always from the perspective of the player about to move:
#   plane 0 = current player's discs
#   plane 1 = opponent's discs
# Value targets are likewise from the current player's point of view:
#   +1 win, 0 draw, -1 loss.


def encode_state(state: Connect4State) -> np.ndarray:
    """Return a float32 array of size 2 * rows * cols."""
    me = state.to_play
    opp = P1 if me == P2 else P2
    planes = np.zeros((2, state.rows, state.cols), dtype=np.float32)
    planes[0] = state.board == me
    planes[1] = state.board == opp
    return planes.flatten()


def outcome_for(state: Connect4State, player: int) -> float:
    """Terminal value for `player`: +1 / 0 / -1."""
    if state.winner == 0:
        return 0.0
    return 1.0 if state.winner == player else -1.0


# ===========================================================================
# 2. NEURAL NETWORK
# ===========================================================================


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.mlp(x))


def _lex_less(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise lexicographic `a < b` for 2D tensors of shape `(B, D)`."""
    ne = a != b
    # First differing index per row (`cumsum == 1`); equal rows stay all-False.
    first_diff = ne & (ne.to(dtype=a.dtype).cumsum(dim=-1) == 1)
    return ((a < b) & first_diff).any(dim=-1)


class AlphaZeroNet(nn.Module):
    """Small residual tower with a policy head (over columns) and a value head.

    Horizontal mirrors are canonicalized inside `forward`: the lexicographically
    smaller of `x` and its left–right flip is fed to the trunk; policy logits
    are flipped back when the mirror was chosen so outputs stay in board order.
    """

    def __init__(self, rows: int = 6, cols: int = 7, hidden_dim: int = 512, blocks: int = 4):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self._input_dim = 2 * rows * cols
        self._hidden_dim = hidden_dim
        self._blocks = blocks

        self.projection = nn.Linear(self._input_dim, self._hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(self._hidden_dim) for _ in range(self._blocks)])

        self.policy_head = nn.Sequential(
            nn.Linear(self._hidden_dim, self.cols),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self._hidden_dim, 1),
            nn.Tanh(),
        )

    def _get_canonical_representation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the canonical state representation:
        A state is representated by a vector of size 2 * rows * cols.
        To increase data efficiency, we ensure the two states that are mirror images of each other
        are encoded the same way to the network.
        The canonical representation of a state x is the lexicographically smaller representationof x and its mirror image.

        Args:
            x: (B, 2 * rows * cols) in board coordinates.

        Returns:
            x_canon: (B, 2 * rows * cols) in canonical representation.
            mirror: (B,) boolean tensor indicating whether the state is mirrored.
        """
        planes = x.view(x.shape[0], 2, self.rows, self.cols)
        x_mirror = planes.flip(-1).reshape(x.shape[0], -1)
        mirror = _lex_less(x_mirror, x)
        x_canon = torch.where(mirror.unsqueeze(-1), x_mirror, x)
        return x_canon, mirror

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_canon, mirror = self._get_canonical_representation(x)

        h = self.projection(x_canon)
        h = self.blocks(h)
        logits = self.policy_head(h)
        value = self.value_head(h)

        # Map policy logits back to the original board's column order.
        logits_out = torch.where(mirror.unsqueeze(-1), logits.flip(-1), logits)
        return logits_out, value.squeeze(-1)


@torch.no_grad()
def net_predict(net: AlphaZeroNet, state: Connect4State,
                device: torch.device) -> tuple[np.ndarray, float]:
    """Masked policy probabilities over columns + scalar value for `state`."""
    net.eval()
    x = torch.from_numpy(encode_state(state)[None]).to(device)
    logits, value = net(x)
    logits = logits[0].cpu().numpy()
    legal = state.get_moves()
    mask = np.full(state.cols, -np.inf, dtype=np.float32)
    mask[legal] = 0.0
    logits = logits + mask
    # Softmax in a numerically stable way.
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return probs.astype(np.float32), float(value[0].item())


# ===========================================================================
# 3. PUCT MCTS
# ===========================================================================
# Classic AlphaZero tree search: select by PUCT, expand with a single network
# evaluation (which also provides the leaf value), then back up that value.
# No random rollouts.


class AZNode:
    __slots__ = ("N", "Q", "W", "children", "move", "parent", "prior")

    def __init__(self, parent: AZNode | None, move: int | None,
                 prior: float):
        self.parent = parent
        self.move = move          # column played to reach this node (None at root)
        self.prior = prior
        self.children: dict[int, AZNode] = {}
        self.N = 0                # visit count
        self.W = 0.0              # total value (from the mover-to-this-node view)
        self.Q = 0.0              # mean value = W / N

    def is_expanded(self) -> bool:
        return bool(self.children)

    def puct(self, c_puct: float) -> float:
        # Parent visits: use max(1, ...) so the root's children still explore.
        parent_N = self.parent.N if self.parent is not None else 1
        u = c_puct * self.prior * math.sqrt(parent_N) / (1 + self.N)
        return self.Q + u


class AZMCTS:
    def __init__(self, net: AlphaZeroNet, device: torch.device,
                 n_sims: int = 100, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25):
        self.net = net
        self.device = device
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def run(self, root_state: Connect4State,
            add_noise: bool = True) -> tuple[np.ndarray, AZNode]:
        """Run `n_sims` simulations; return (visit_policy, root_node)."""
        root = AZNode(parent=None, move=None, prior=1.0)
        self._expand(root, root_state)
        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(self.n_sims):
            node = root
            state = root_state.clone()
            # Selection.
            while node.is_expanded() and not state.is_terminal():
                move, node = self._select(node)
                state.do_move(move)
            # Evaluate / expand leaf.
            if state.is_terminal():
                # Value from the perspective of the player who just moved into
                # this terminal state: +1 if they won, etc. Backup will flip.
                value = outcome_for(state, state.player_just_moved)
            else:
                value = self._expand(node, state)
            # Backup: value is for the player who just moved into `node`.
            self._backup(node, value)

        policy = np.zeros(root_state.cols, dtype=np.float32)
        for move, child in root.children.items():
            policy[move] = child.N
        total = policy.sum()
        if total > 0:
            policy /= total
        return policy, root

    def _select(self, node: AZNode) -> tuple[int, AZNode]:
        move, child = max(
            node.children.items(),
            key=lambda kv: kv[1].puct(self.c_puct),
        )
        return move, child

    def _expand(self, node: AZNode, state: Connect4State) -> float:
        """Expand `node` with network priors; return value for player-to-move.

        The returned value is immediately re-interpreted as the value for the
        player who just moved (i.e. negated) so that backup is uniform.
        """
        priors, value = net_predict(self.net, state, self.device)
        for move in state.get_moves():
            node.children[move] = AZNode(parent=node, move=move,
                                         prior=float(priors[move]))
        # `value` is from the current (to-play) player's view. The backup loop
        # expects the value of the player who moved into this node, so negate.
        return -value

    def _backup(self, node: AZNode | None, value: float) -> None:
        # `value` is for the player who just moved into `node`. Walking up,
        # each parent sees the opposite outcome.
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value
            node = node.parent

    def _add_dirichlet_noise(self, root: AZNode) -> None:
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        for m, n in zip(moves, noise):
            child = root.children[m]
            child.prior = ((1 - self.dirichlet_eps) * child.prior
                           + self.dirichlet_eps * float(n))


def sample_move(policy: np.ndarray, temperature: float) -> int:
    """Sample a column from `policy` with temperature; tau->0 = argmax."""
    if temperature <= 1e-3:
        return int(np.argmax(policy))
    logits = np.log(np.maximum(policy, 1e-12)) / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(np.random.choice(len(policy), p=probs))


# ===========================================================================
# 4. SELF-PLAY + REPLAY BUFFER
# ===========================================================================


@dataclass
class Sample:
    planes: np.ndarray          # (2, rows, cols)
    policy: np.ndarray          # (cols,)
    value: float                # +1 / 0 / -1 for the player to move


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: deque[Sample] = deque(maxlen=capacity)

    def extend(self, samples: list[Sample]) -> None:
        self.buf.extend(samples)

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        x = torch.from_numpy(np.stack([s.planes for s in batch]))
        pi = torch.from_numpy(np.stack([s.policy for s in batch]))
        z = torch.tensor([s.value for s in batch], dtype=torch.float32)
        return x, pi, z


def play_game(mcts: AZMCTS, rows: int, cols: int, connect: int,
              temp_moves: int = 10) -> list[Sample]:
    """One self-play game; returns training samples with filled-in values."""
    state = Connect4State(rows, cols, connect)
    history: list[tuple[np.ndarray, np.ndarray, int]] = []  # planes, pi, to_play

    move_idx = 0
    while not state.is_terminal():
        policy, _ = mcts.run(state, add_noise=True)
        history.append((encode_state(state), policy.copy(), state.to_play))
        tau = 1.0 if move_idx < temp_moves else 0.0
        state.do_move(sample_move(policy, tau))
        move_idx += 1

    samples = [
        Sample(planes=planes, policy=pi, value=outcome_for(state, player))
        for planes, pi, player in history
    ]
    return samples


# ===========================================================================
# 5. TRAINING LOOP
# ===========================================================================


def alphazero_loss(logits: torch.Tensor, value: torch.Tensor,
                   target_pi: torch.Tensor, target_z: torch.Tensor) -> tuple[
                       torch.Tensor, torch.Tensor, torch.Tensor]:
    """Policy cross-entropy + value MSE. Returns (total, policy_loss, value_loss)."""
    log_probs = F.log_softmax(logits, dim=-1)
    # Mask: positions with zero target mass (illegal) contribute nothing.
    policy_loss = -(target_pi * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(value, target_z)
    return policy_loss + value_loss, policy_loss, value_loss


def train_steps(net: AlphaZeroNet, opt: torch.optim.Optimizer, buf: ReplayBuffer,
                device: torch.device, n_steps: int, batch_size: int) -> dict:
    if len(buf) < batch_size:
        return {"loss": float("nan"), "p": float("nan"), "v": float("nan")}
    net.train()
    total = p_tot = v_tot = 0.0
    for _ in range(n_steps):
        x, pi, z = buf.sample(batch_size)
        x, pi, z = x.to(device), pi.to(device), z.to(device)
        logits, value = net(x)
        loss, p_loss, v_loss = alphazero_loss(logits, value, pi, z)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        total += loss.item()
        p_tot += p_loss.item()
        v_tot += v_loss.item()
    return {"loss": total / n_steps, "p": p_tot / n_steps, "v": v_tot / n_steps}


@torch.no_grad()
def evaluate_vs_random(net: AlphaZeroNet, device: torch.device,
                       rows: int, cols: int, connect: int,
                       n_games: int = 20, sims: int = 50) -> dict:
    """W/L/D of the net (as P1 and P2) against a uniform random opponent."""
    mcts = AZMCTS(net, device, n_sims=sims, dirichlet_eps=0.0)
    wins = losses = draws = 0
    for g in range(n_games):
        state = Connect4State(rows, cols, connect)
        net_is_p1 = (g % 2 == 0)
        while not state.is_terminal():
            if (state.to_play == P1) == net_is_p1:
                policy, _ = mcts.run(state, add_noise=False)
                state.do_move(int(np.argmax(policy)))
            else:
                state.do_move(random.choice(state.get_moves()))
        if state.winner == 0:
            draws += 1
            continue
        net_player = P1 if net_is_p1 else P2
        if state.winner == net_player:
            wins += 1
        else:
            losses += 1
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "n_games": n_games,
        "win_rate": wins / n_games,
    }


@torch.no_grad()
def evaluate_vs_pure_mcts(net: AlphaZeroNet, device: torch.device,
                          rows: int, cols: int, connect: int,
                          n_games: int = 20, net_sims: int = 100,
                          opp_sims: int = 400, opp_C: float = 1.41) -> dict:
    """W/L/D of AZ-MCTS(net) vs classic UCT-MCTS with random rollouts.

    Colors alternate each game. `win_rate` is wins / n_games (draws are not wins).
    """
    az = AZMCTS(net, device, n_sims=net_sims, dirichlet_eps=0.0)
    wins = losses = draws = 0
    for g in range(n_games):
        state = Connect4State(rows, cols, connect)
        net_is_p1 = (g % 2 == 0)
        while not state.is_terminal():
            if (state.to_play == P1) == net_is_p1:
                policy, _ = az.run(state, add_noise=False)
                state.do_move(int(np.argmax(policy)))
            else:
                root = mcts_search(state, opp_sims, opp_C)
                state.do_move(best_move(root))
        if state.winner == 0:
            draws += 1
            continue
        net_player = P1 if net_is_p1 else P2
        if state.winner == net_player:
            wins += 1
        else:
            losses += 1
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "n_games": n_games,
        "win_rate": wins / n_games,
    }


def save_checkpoint(path: str, net: AlphaZeroNet, opt: torch.optim.Optimizer,
                    iteration: int, args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "iteration": iteration,
        "model": net.state_dict(),
        "optimizer": opt.state_dict(),
        "args": vars(args),
    }, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlphaZero Connect-4 trainer")
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=7)
    p.add_argument("--connect", type=int, default=4)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--iters", type=int, default=30,
                   help="Outer training iterations (self-play + update)")
    p.add_argument("--games", type=int, default=20,
                   help="Self-play games per iteration")
    p.add_argument("--sims", type=int, default=100,
                   help="MCTS simulations per move")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
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
    p.add_argument("--eval-mcts-sims", type=int, default=400,
                   help="Pure UCT-MCTS simulations per move for the eval opponent")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str,
                   default="model_based_rl/checkpoints/connect4_az.pt")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"device={device}  board={args.rows}x{args.cols}  "
          f"sims={args.sims}  games/iter={args.games}")

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
        print(f"resumed from {args.resume} at iteration {start_iter}")

    mcts = AZMCTS(
        net, device,
        n_sims=args.sims,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
    )

    for it in range(start_iter, args.iters + 1):
        t0 = time.time()
        n_samples = 0
        for g in range(args.games):
            samples = play_game(
                mcts, args.rows, args.cols, args.connect, args.temp_moves,
            )
            buf.extend(samples)
            n_samples += len(samples)
        play_s = time.time() - t0

        t1 = time.time()
        stats = train_steps(net, opt, buf, device, args.train_steps, args.batch_size)
        train_s = time.time() - t1

        print(
            f"iter {it:3d}/{args.iters}  "
            f"samples+={n_samples:4d}  buf={len(buf):6d}  "
            f"loss={stats['loss']:.3f} (p={stats['p']:.3f} v={stats['v']:.3f})  "
            f"play={play_s:.1f}s train={train_s:.1f}s"
        )

        if it % args.eval_every == 0 or it == args.iters:
            ev = evaluate_vs_random(
                net, device, args.rows, args.cols, args.connect,
                n_games=args.eval_games, sims=max(32, args.sims // 2),
            )
            print(
                f"  eval vs random: "
                f"W/L/D={ev['wins']}/{ev['losses']}/{ev['draws']}  "
                f"win_rate={ev['win_rate']:.2f} over {ev['n_games']} games"
            )
            t_eval = time.time()
            ev_mcts = evaluate_vs_pure_mcts(
                net, device, args.rows, args.cols, args.connect,
                n_games=args.eval_games,
                net_sims=args.sims,
                opp_sims=args.eval_mcts_sims,
            )
            print(
                f"  eval vs pure MCTS ({args.eval_mcts_sims} sims): "
                f"W/L/D={ev_mcts['wins']}/{ev_mcts['losses']}/{ev_mcts['draws']}  "
                f"win_rate={ev_mcts['win_rate']:.2f} over {ev_mcts['n_games']} games "
                f"({time.time() - t_eval:.1f}s)"
            )
            save_checkpoint(args.checkpoint, net, opt, it, args)
            print(f"  saved {args.checkpoint}")

    print("done.")


if __name__ == "__main__":
    main()
