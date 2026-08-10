import os
import sys
from multiprocessing import Pool
import time
from dataclasses import dataclass
from copy import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    GameConfig,
    GameRulesConfig,
    PlayerConfig,
    Connect4State,
    Player,
    P1,
    P2,
    RandomPlayerConfig,
    RandomPlayer,
)
from mcts import MCTSPlayerConfig, MCTSReuseTree
from mcts import MCTSPlayer

def create_player(config: PlayerConfig) -> Player:
    if isinstance(config, MCTSPlayerConfig):
        return MCTSPlayer(config)
    elif isinstance(config, RandomPlayerConfig):
        return RandomPlayer(config)
    else:
        raise ValueError(f"Unknown player config: {config}")


def run_game(game_config: GameConfig) -> float:
    game_rules_config = game_config.game_rules
    state = Connect4State(game_rules_config.rows, game_rules_config.cols, game_rules_config.connect)

    player1 = create_player(game_config.player1)
    player2 = create_player(game_config.player2)
    player1.init_game(game_config, True)
    player2.init_game(game_config, False)

    while not state.is_terminal():
        move, _ = player1.get_move(state)
        state.do_move(move)

        if state.is_terminal():
            break

        move, _ = player2.get_move(state)
        state.do_move(move)

    return state.get_result(P1)


def _run_game_worker(game_config: GameConfig) -> float:
    """Top-level worker so multiprocessing can pickle the target."""
    return run_game(game_config)


def run_n_games(
    game_config: GameConfig,
    n_games: int,
    n_processes: int,
) -> list[float]:
    """Play `n_games` independently, optionally in parallel.

    Results are in game-index order (same as a serial loop).
    """
    
    if n_games == 0:
        return []
    if n_processes == 1:
        return [run_game(game_config) for _ in range(n_games)]

    with Pool(processes=n_processes) as pool:
        return pool.map(_run_game_worker, [game_config] * n_games)


@dataclass
class GameStats:
    wins: int
    losses: int
    draws: int
    # Average score: defined as: (W + 0.5 * D) / (W + L + D)
    score: float

def run_game_stats(
    game_config: GameConfig,
    n_games_per_turn: int,
    n_processes: int) -> GameStats:

    results = run_n_games(game_config, n_games_per_turn, n_processes)
    game_config = copy(game_config)
    game_config.swap_players()
    results_swapped = run_n_games(game_config, n_games_per_turn, n_processes)
    results += [1.0 - result for result in results_swapped]

    return GameStats(
        wins=results.count(1),
        losses=results.count(0),
        draws=results.count(0.5),
        score=sum(results) / len(results),
    )

if __name__ == "__main__":
    game_config = GameConfig(
        game_rules=GameRulesConfig(
            rows=6,
            cols=7,
            connect=4,
        ),
        player1=MCTSPlayerConfig(
            name="player1",
            num_simulations=200,
            uct_constant=1.0,
            reuse_tree=MCTSReuseTree.NO_REUSE,
        ),
        player2=RandomPlayerConfig(
            name="player2",
        ),
    )

    start_time = time.time()
    stats = run_game_stats(game_config, 32, 16)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Stats: {stats}")