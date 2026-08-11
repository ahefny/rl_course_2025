"""Compute Elo ratings for a list of players based on pairwise matches.

Specify the player configs in the `get_players` function.
"""

import os
import sys
from multiprocessing import Pool
import time
from dataclasses import dataclass, asdict
from copy import copy
import numpy as np
from scipy.optimize import minimize
import json
import tqdm

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
from mcts import MCTSPlayerConfig, MCTSReuseTree, MCTSPlayer
from alpha_zero import AZMCTSConfig, AZMCTSPlayer, PlainAlphaZeroConfig, PlainAlphaZeroPlayer

def create_player(config: PlayerConfig) -> Player:
    if isinstance(config, MCTSPlayerConfig):
        return MCTSPlayer(config)
    elif isinstance(config, RandomPlayerConfig):
        return RandomPlayer(config)
    elif isinstance(config, AZMCTSConfig):
        return AZMCTSPlayer(config)
    elif isinstance(config, PlainAlphaZeroConfig):
        return PlainAlphaZeroPlayer(config)
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


@dataclass(frozen=True)
class EloRatings:
    """Elo ratings for a list of players based on pairwise matches."""

    # Number of games playes between each pair of players.
    num_games: int

    # Scores for each pair of players.
    # scores[i][j] is the average score of player i against player j (average score over N games).
    # where 1 means player i wins, 0 means player j wins, and 0.5 means draw.
    scores: dict[str, dict[str, float]]

    # Elo ratings for each player
    ratings: dict[str, float]


def compute_elo_ratings(
    game_rules_config: GameRulesConfig,
    players: list[PlayerConfig],
    n_games_per_turn: int,
    n_processes: int,
) -> EloRatings:

    pairwise_scores = np.zeros((len(players), len(players)))

    num_games = len(players) * (len(players) - 1) // 2

    pbar = tqdm.tqdm(total=num_games, desc="Running games")
    for i in range(len(players)):
        for j in range(0, i):
            game_config = GameConfig(
                game_rules=game_rules_config,
                player1=players[i],
                player2=players[j],
            )
            stats = run_game_stats(game_config, n_games_per_turn, n_processes)
            pairwise_scores[i, j] = stats.score
            pairwise_scores[j, i] = 1.0 - stats.score
            pbar.update(1)
    pbar.close()

    raw_ratings = np.zeros(len(players))

    if len(players) > 1:
        # Compute raw rating for each player by minimizing the cross entropy loss
        # between the expected pairwise scores from Bradley-Terry model and the actual scores.
        # Observation: pairwise_score[i, j]
        # Prediction: sigmoid(ratings[i] - ratings[j])        

        player_i, player_j = np.tril_indices(len(players), k=-1)
        scores = pairwise_scores[player_i, player_j]

        def loss_and_gradient(ratings: np.ndarray) -> tuple[float, np.ndarray]:
            differences = ratings[player_i] - ratings[player_j]
            probabilities = 1.0 / (1.0 + np.exp(-differences))

            # loss_ij = -score_ij * log sigmoid(diff_ij) - (1 - score_ij) * log(1 - sigmoid(diff_ij))
            #         = log(1 + exp(diff_ij)) - score_ij * diff_ij
            loss = np.sum(np.logaddexp(0.0, differences) - scores * differences)

            # \partial{loss_ij} / \partial{ratings[i]} = sigmoid(diff_ij) - score_ij
            gradient = np.zeros_like(ratings)
            errors = probabilities - scores
            np.add.at(gradient, player_i, errors)
            np.add.at(gradient, player_j, -errors)
            return float(loss), gradient

        raw_ratings = minimize(
            loss_and_gradient,
            raw_ratings,
            jac=True,
            method="BFGS",
        ).x

    # Ratings are relative, so we use the minimum rating as the baseline.
    raw_ratings -= np.min(raw_ratings)

    # Convert raw ratings to Elo scale.
    elo_scale = 400.0 / np.log(10.0)    

    elo_ratings = {
        player.name: float(rating * elo_scale)
        for player, rating in zip(players, raw_ratings)
    }

    return EloRatings(
        scores={
            players[i].name: {
                players[j].name: pairwise_scores[i, j].item()
                for j in range(i)
            }
            for i in range(len(players))
        },
        ratings=elo_ratings,
        num_games=n_games_per_turn * 2,
    )


def get_players() -> list[PlayerConfig]:
    MODEL_CKPT = "model_based_rl/connect4_alphazero/checkpoints/connect4_az_b8h512s100_260810_2326.pt"

    return [
        RandomPlayerConfig(name="random"),        
    ] + [
        PlainAlphaZeroConfig(
            name=f"plain_az",
            model_path=MODEL_CKPT,
            device="cuda",
        )        
    ] + [
        AZMCTSConfig(
            name=f"az_mcts_{i}",
            model_path=MODEL_CKPT,
            device="cuda",
            num_simulations=i,
            uct_constant=1.0,
            dirichlet_alpha=0.1,
            dirichlet_epsilon=0.0,
            num_hightemperature_turns=0,
        )
        for i in [10, 50, 100, 200]
    ] + [
        MCTSPlayerConfig(name=f"mcts_{i}", num_simulations=i, uct_constant=1.0, reuse_tree=MCTSReuseTree.REUSE_REDUCE_SIM)
        for i in [10, 50, 100, 200, 500, 1000, 2000]
    ]


if __name__ == "__main__":
    game_rules_config = GameRulesConfig(
        rows=6,
        cols=7,
        connect=4,
    )
    players = get_players()

    elo_ratings = compute_elo_ratings(game_rules_config, players, 64, 16)
    print(elo_ratings)

    
    with open("elo_ratings.json", "w") as f:
        json.dump(asdict(elo_ratings), f)
