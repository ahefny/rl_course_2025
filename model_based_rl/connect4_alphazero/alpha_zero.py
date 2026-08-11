from __future__ import annotations
from dataclasses import dataclass
from core import GameRulesConfig, PlayerConfig, GameConfig, Player
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from core import Connect4State, P1, P2

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
        self.blocks = nn.Sequential(
            *[ResidualBlock(self._hidden_dim) for _ in range(self._blocks)],
        )

        self.policy_head = nn.Sequential(
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self._hidden_dim, self.cols),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.ReLU(inplace=True),
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


def load_az_checkpoint(path: str, device: torch.device) -> tuple[AlphaZeroNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]
    rows = int(args.get("rows", 6))
    cols = int(args.get("cols", 7))
    connect = int(args.get("connect", 4))
    hidden_dim = int(args["hidden_dim"])
    blocks = int(args["blocks"])
    net = AlphaZeroNet(
        rows=rows, cols=cols, hidden_dim=hidden_dim, blocks=blocks,
    ).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    meta = {
        "rows": rows,
        "cols": cols,
        "connect": connect,
        "iteration": ckpt.get("iteration"),
        "path": path,
    }
    return net, meta


@dataclass
class AZMCTSConfig(PlayerConfig):
    model_key: str    
    num_simulations: int
    uct_constant: float

    dirichlet_alpha: float
    dirichlet_epsilon: float
    num_hightemperature_turns: int


class AZMCTSPlayer(Player):
    def __init__(self, config: AZMCTSConfig):
        super().__init__(config)
        self.config = config
        


