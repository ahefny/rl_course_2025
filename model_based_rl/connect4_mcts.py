"""Connect-4 game rules and classic UCT-MCTS with random rollouts.

Pure logic — no Gradio / Graphviz. For the interactive UI, run:

    python model_based_rl/connect4_play.py

Layout of this file:
    1. Core game logic      -- Connect4State
    2. MCTS                  -- Node, mcts_search, best_move, helpers
"""

from __future__ import annotations

import enum
import math
import random

import numpy as np

# ===========================================================================
# 1. CORE GAME LOGIC
# ===========================================================================
# A Connect-K state on a rows x cols board. Discs are dropped into columns and
# fall to the lowest empty cell. `board[0]` is the TOP row, `board[rows-1]` the
# bottom. Cells hold EMPTY, P1 or P2.

EMPTY, P1, P2 = 0, 1, 2


class Connect4State:
    """Immutable-by-convention Connect-K game state.

    Mutating methods (`do_move`) change the object in place; use `clone()`
    before mutating when you need to keep the original (as MCTS does).
    """

    def __init__(self, rows: int = 6, cols: int = 7, connect: int = 4):
        self.rows = rows
        self.cols = cols
        self.connect = connect
        self.board = np.zeros((rows, cols), dtype=np.int8)
        # Number of discs already in each column (0 == empty column).
        self.heights = np.zeros(cols, dtype=np.int8)
        # The player who made the most recent move. Init so that P1 moves first.
        self.player_just_moved = P2
        self.winner = 0  # 0 == no winner yet / draw; otherwise P1 or P2.
        self.last_move: tuple | None = None  # (row, col) of last placed disc.

    # -- basic accessors ----------------------------------------------------
    @property
    def to_play(self) -> int:
        """Player about to move."""
        return P1 if self.player_just_moved == P2 else P2

    def clone(self) -> Connect4State:
        s = Connect4State.__new__(Connect4State)
        s.rows, s.cols, s.connect = self.rows, self.cols, self.connect
        s.board = self.board.copy()
        s.heights = self.heights.copy()
        s.player_just_moved = self.player_just_moved
        s.winner = self.winner
        s.last_move = self.last_move
        return s

    def get_moves(self) -> list[int]:
        """Legal moves == columns that are not full (empty if game is over)."""
        if self.winner != 0:
            return []
        return [c for c in range(self.cols) if self.heights[c] < self.rows]

    def is_terminal(self) -> bool:
        return self.winner != 0 or not any(
            self.heights[c] < self.rows for c in range(self.cols)
        )

    # -- transitions --------------------------------------------------------
    def do_move(self, col: int) -> None:
        """Drop a disc for the player to move into `col` (mutates self)."""
        player = self.to_play
        row_from_bottom = int(self.heights[col])
        r = self.rows - 1 - row_from_bottom  # convert to top-indexed row
        self.board[r, col] = player
        self.heights[col] += 1
        self.player_just_moved = player
        self.last_move = (r, col)
        if self._wins_from(r, col, player):
            self.winner = player

    def _wins_from(self, r: int, c: int, player: int) -> bool:
        """Did placing `player` at (r, c) complete a line of `connect`?"""
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                rr, cc = r + dr * sign, c + dc * sign
                while (
                    0 <= rr < self.rows
                    and 0 <= cc < self.cols
                    and self.board[rr, cc] == player
                ):
                    count += 1
                    rr += dr * sign
                    cc += dc * sign
            if count >= self.connect:
                return True
        return False

    def get_result(self, player: int) -> float:
        """Terminal reward for `player`: 1.0 win, 0.0 loss, 0.5 draw."""
        if self.winner == 0:
            return 0.5
        return 1.0 if self.winner == player else 0.0


# ===========================================================================
# 2. MCTS
# ===========================================================================
# Classic UCT (Kocsis & Szepesvari) with random rollouts. Each node stores the
# player that moved to reach it, so backups use the correct perspective.


class Node:
    def __init__(self, state: Connect4State, move: int | None = None,
                 parent: Node | None = None):
        self.move = move              # column played to reach this node (None at root)
        self.parent = parent
        self.children: list[Node] = []
        self.untried: list[int] = state.get_moves()
        self.player_just_moved = state.player_just_moved
        self.visits = 0
        self.wins = 0.0               # summed reward from player_just_moved's view

    @property
    def q(self) -> float:
        return self.wins / self.visits if self.visits else 0.0

    def uct_score(self, C: float) -> float:
        exploit = self.wins / self.visits
        explore = C * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def uct_select_child(self, C: float) -> Node:
        return max(self.children, key=lambda ch: ch.uct_score(C))

    def add_child(self, move: int, state: Connect4State) -> Node:
        child = Node(state=state, move=move, parent=self)
        self.untried.remove(move)
        self.children.append(child)
        return child

    def update(self, result: float) -> None:
        self.visits += 1
        self.wins += result


class MCTSReuseTree(enum.Enum):
    """Enum for the different tree reuse strategies.
    
    Given an MCTS tree rooted at state s, we can reuse the node visits and wins when planning from a state s' that is a child of s.
    """

    # Do not reuse the parent tree. Each MCTS starts from a fresh tree with 0 visits.
    NO_REUSE = 0
    # Reuse the parent tree. add N simulations to existing nodes.
    # By combining the sim results from the parent tree and the new simulations, 
    # we can get a more accurate value estimates with the same simulation budget.
    REUSE_KEEP_SIM = 1
    # Reuse the parent tree. Instead of running N new simulations, run N-M new simulations,
    # where M is the number of simulations already run on the parent tree. This reduces the number of simulations
    # needed to reach the same level of accuracy.
    REUSE_REDUCE_SIM = 2

def mcts_search(
        root_state: Connect4State, n_iter: int, C: float,
        seed: int | None = None,
        reuse_tree: MCTSReuseTree = MCTSReuseTree.NO_REUSE,
        reuse_root: Node | None = None) -> Node:
    """Run `n_iter` UCT simulations from `root_state`; return the root node.

    If `reuse_root` is given and reuse_tree is not NO_REUSE, it is re-rooted (its parent link cut) and
    the new simulations are added on top of its existing statistics; otherwise a
    fresh 0-visit tree is created.
    """
    
    rng = random.Random(seed)
    root = reuse_root
    if root is None:
        assert reuse_tree == MCTSReuseTree.NO_REUSE, "Reuse options other than NO_REUSE require a reuse_root"

    if reuse_tree == MCTSReuseTree.NO_REUSE:
        root = Node(state=root_state)
    else:
        root.parent = None  # re-root: detach from the previous turn's tree

    if reuse_tree == MCTSReuseTree.REUSE_REDUCE_SIM:
        n_iter = max(0, n_iter - root.visits)

    for _ in range(n_iter):
        node = root
        state = root_state.clone()

        # -- Selection: descend fully-expanded nodes by UCT.
        while not node.untried and node.children:
            node = node.uct_select_child(C)
            state.do_move(node.move)

        # -- Expansion: add one child for an untried move.
        if node.untried:
            m = rng.choice(node.untried)
            state.do_move(m)
            node = node.add_child(m, state)

        # -- Rollout: play random moves to a terminal state.
        moves = state.get_moves()
        while moves:
            state.do_move(rng.choice(moves))
            moves = state.get_moves()

        # -- Backpropagation: update every node on the path.
        while node is not None:
            node.update(state.get_result(node.player_just_moved))
            node = node.parent

    return root


def best_move(root: Node) -> int:
    """Robust choice: the most-visited child."""
    return max(root.children, key=lambda ch: ch.visits).move


def child_by_move(node: Node, move: int) -> Node | None:
    """The child reached by playing `move`, or None if it was never expanded."""
    for ch in node.children:
        if ch.move == move:
            return ch
    return None


def count_nodes(node: Node) -> int:
    return 1 + sum(count_nodes(ch) for ch in node.children)
