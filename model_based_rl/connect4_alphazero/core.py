"""Connect-4 game rules and player config dataclasses.

Classic UCT-MCTS lives in ``mcts.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GameRulesConfig:
    rows: int
    cols: int
    connect: int


@dataclass
class PlayerConfig:
    name: str


@dataclass
class HumanPlayerConfig(PlayerConfig):
    pass


@dataclass
class RandomPlayerConfig(PlayerConfig):
    pass


@dataclass
class AZMCTSConfig(PlayerConfig):
    model_key: str    
    num_simulations: int
    uct_constant: float

    dirichlet_alpha: float
    dirichlet_epsilon: float
    num_hightemperature_turns: int


@dataclass
class GameConfig:
    game_rules: GameRulesConfig
    player1: PlayerConfig
    player2: PlayerConfig

    def swap_players(self) -> None:
        self.player1, self.player2 = self.player2, self.player1


# ===========================================================================
# CORE GAME LOGIC
# ===========================================================================
# A Connect-K state on a rows x cols board. Discs are dropped into columns and
# fall to the lowest empty cell. `board[0]` is the TOP row, `board[rows-1]` the
# bottom. Cells hold EMPTY, P1 or P2.

EMPTY, P1, P2 = 0, 1, 2


class Connect4State:
    
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


class Player:

    def __init__(self, config: PlayerConfig):
        self.config = config
        self.name = config.name


    def init_game(self, game_config: GameConfig, is_player1: bool) -> None:
        self.game_config = game_config
        self.is_player1 = is_player1

    def get_move(self, state: Connect4State) -> int:
        raise NotImplementedError
