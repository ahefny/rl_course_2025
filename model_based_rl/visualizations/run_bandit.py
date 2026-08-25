"""Visualize UCB1 learning on a Bernoulli multi-armed bandit.

Example:
    python run_bandit.py --probabilities 0.5 0.1 0.8 --iterations 200
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

_DEFAULT_PROBABILITIES = [0.5, 0.1, 0.8]


@dataclass(frozen=True)
class BanditState:
    """The statistics displayed after a number of UCB1 iterations."""

    iteration: int
    visits: np.ndarray
    empirical_means: np.ndarray
    ucb_values: np.ndarray


def ucb1_values(visits: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """Return UCB1 values for selecting the next arm.

    An arm that has not been sampled has an infinite UCB value so UCB1 samples
    every arm before comparing confidence bounds.
    """
    total_visits = int(visits.sum())
    values = np.full(len(visits), np.inf)
    sampled = visits > 0
    values[sampled] = rewards[sampled] / visits[sampled] + np.sqrt(
        2 * np.log(total_visits) / visits[sampled]
    )
    return values


def run_ucb1(
    probabilities: list[float],
    total_iterations: int,
    iterations_per_frame: int,
    seed: int | None,
) -> list[BanditState]:
    """Run UCB1 and retain a state after every requested frame interval."""
    rng = np.random.default_rng(seed)
    arm_count = len(probabilities)
    visits = np.zeros(arm_count, dtype=int)
    rewards = np.zeros(arm_count, dtype=float)
    states: list[BanditState] = []

    for iteration in range(1, total_iterations + 1):
        bounds = ucb1_values(visits, rewards) if iteration > 1 else np.full(arm_count, np.inf)
        arm = int(np.argmax(bounds))
        reward = float(rng.random() < probabilities[arm])
        visits[arm] += 1
        rewards[arm] += reward

        if iteration % iterations_per_frame == 0 or iteration == total_iterations:
            states.append(
                BanditState(
                    iteration=iteration,
                    visits=visits.copy(),
                    empirical_means=np.divide(
                        rewards,
                        visits,
                        out=np.zeros(arm_count, dtype=float),
                        where=visits > 0,
                    ),
                    ucb_values=ucb1_values(visits, rewards),
                )
            )
    return states


def draw_frame(
    axis: plt.Axes,
    state: BanditState,
    probabilities: list[float],
) -> None:
    """Draw one frame of the bandit visualization."""
    axis.clear()
    actions = np.arange(len(probabilities))
    finite_ucbs = state.ucb_values[np.isfinite(state.ucb_values)]
    maximum_value = max(
        2.0,
        float(finite_ucbs.max()) if len(finite_ucbs) else 1.0,
    )
    top = maximum_value * 1.12

    for action in actions:
        axis.vlines(action, 0, 2, color="#424242", linewidth=1.5, zorder=1)

    axis.scatter(
        actions,
        probabilities,
        marker="_",
        s=420,
        linewidths=3,
        color="red",
        label="True winning probability",
        zorder=3,
    )
    axis.scatter(
        actions,
        state.empirical_means,
        marker="_",
        s=420,
        linewidths=3,
        color="blue",
        label="Empirical average",
        zorder=4,
    )
    display_ucbs = np.minimum(state.ucb_values, top)
    axis.scatter(
        actions,
        display_ucbs,
        marker="_",
        s=420,
        linewidths=3,
        color="gold",
        label="UCB value",
        zorder=5,
    )
    top_ucb_action = int(np.argmax(state.ucb_values))
    axis.scatter(
        top_ucb_action,
        display_ucbs[top_ucb_action],
        marker="_",
        s=420,
        linewidths=3,
        color="orange",
        label="_nolegend_",
        zorder=6,
    )

    highest_visits = state.visits.max()
    for action, count in enumerate(state.visits):
        axis.text(
            action,
            -0.12,
            str(count),
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontweight="bold" if count == highest_visits else "normal",
            fontsize=12,
            clip_on=False,
        )

    axis.set_xlim(-0.5, len(probabilities) - 0.5)
    axis.set_ylim(0, top)
    axis.set_xticks(actions, [str(action) for action in actions])
    axis.set_xlabel("action", labelpad=28)
    axis.set_ylabel("")
    axis.set_title(f"UCB1 after {state.iteration} iterations")
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a GIF showing UCB1 learning a Bernoulli bandit."
    )
    parser.add_argument(
        "--probabilities",
        "-p",
        nargs="+",
        type=float,
        default=_DEFAULT_PROBABILITIES,
        help="Winning probability for each arm (default: 0.5 0.1 0.8).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Total UCB1 iterations (default: 100).",
    )
    parser.add_argument(
        "--iterations-per-frame",
        type=int,
        default=1,
        help="Iterations represented by each GIF frame (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ucb1_bandit.gif"),
        help="GIF output path (default: ucb1_bandit.gif).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--fps", type=int, default=2, help="GIF frames per second.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.probabilities:
        raise ValueError("Provide at least one arm probability.")
    if any(not 0 <= probability <= 1 for probability in args.probabilities):
        raise ValueError("Arm probabilities must be between 0 and 1.")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")
    if args.iterations_per_frame < 1:
        raise ValueError("--iterations-per-frame must be at least 1.")
    if args.fps < 1:
        raise ValueError("--fps must be at least 1.")


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    states = run_ucb1(
        args.probabilities,
        args.iterations,
        args.iterations_per_frame,
        args.seed,
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    figure.subplots_adjust(bottom=0.24)
    animation = FuncAnimation(
        figure,
        lambda frame: draw_frame(axis, states[frame], args.probabilities),
        frames=len(states),
        repeat=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=args.fps))
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

