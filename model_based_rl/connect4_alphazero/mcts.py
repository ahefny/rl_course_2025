from __future__ import annotations
import math
import random
import enum
from dataclasses import dataclass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import PlayerConfig, Connect4State, Player

# ===========================================================================
# MCTS
# ===========================================================================
# Classic UCT (Kocsis & Szepesvari) with random rollouts. Each node stores the
# player that moved to reach it, so backups use the correct perspective.


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


@dataclass
class MCTSPlayerConfig(PlayerConfig):
    num_simulations: int
    uct_constant: float
    reuse_tree: MCTSReuseTree

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


class MCTSPlayer(Player):
    def __init__(self, config: MCTSPlayerConfig):
        super().__init__(config)


    def get_move(self, state: Connect4State) -> int:
        root = mcts_search(
            state=state,
            n_iter=self.config.num_simulations,
            C=self.config.uct_constant,
            reuse_tree=self.config.reuse_tree,
            reuse_root=None,
        )
        return best_move(root)