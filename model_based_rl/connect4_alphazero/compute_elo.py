"""Compute Elo ratings for a list of players based on pairwise matches.

Specify the player configs in the `get_players` function.
"""


import os
import sys
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import minimize
import json
import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game_run_utils import run_game_stats
from core import GameConfig, GameRulesConfig, PlayerConfig, RandomPlayerConfig
from alpha_zero import PlainAlphaZeroConfig, AZMCTSConfig
from mcts import MCTSPlayerConfig, MCTSReuseTree


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
