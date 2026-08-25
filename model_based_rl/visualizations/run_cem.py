"""Visualization of CEM planning for a 1-D vehicle (longitudinal planning).

The scenario is a vehicle approaching an intersection
where a perpendicular vehicle is predicted to enter the intersection.

The CEM algorithm is used to plan the acceleration sequence that minimizes the
cost function. The cost function is a weighted sum of the distance to the intersection,
the acceleration, the jerk, and the collision with the perpendicular vehicle.

The script generate two GIF animations:
- CEM planning process. Shows the sampled trajectories, the elite trajectories, and the current mean plan across iterations.
- Open-loop execution of the final highest-scoring plan.
Shows how the plan is executed as the ego vehicle and the perpendicular vehicle move in an intersection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class CEMState:
    """One CEM update and the trajectories used to make it."""

    iteration: int
    sampled_positions: np.ndarray
    elite_positions: np.ndarray
    mean_positions: np.ndarray
    best_positions: np.ndarray
    best_reward: float
    mean_reward: float


def rollout(accelerations: np.ndarray, dt: float, initial_velocity: float) -> np.ndarray:
    """Integrate batches of acceleration sequences into position trajectories."""
    velocities = initial_velocity + np.cumsum(accelerations, axis=-1) * dt
    distances = np.cumsum(velocities, axis=-1) * dt
    return np.concatenate(
        (np.zeros((*distances.shape[:-1], 1), dtype=float), distances), axis=-1
    )


def trajectory_rewards(
    accelerations: np.ndarray,
    positions: np.ndarray,
    dt: float,
    distance_weight: float,
    acceleration_weight: float,
    jerk_weight: float,
    collision_weight: float,
    collision_time: tuple[float, float],
    collision_distance: tuple[float, float],
) -> np.ndarray:
    """Return distance reward minus acceleration, jerk, and collision penalties."""
    times = np.arange(positions.shape[-1]) * dt
    in_time = (collision_time[0] <= times) & (times <= collision_time[1])
    in_distance = (collision_distance[0] <= positions) & (
        positions <= collision_distance[1]
    )
    collides = np.any(in_distance & in_time, axis=-1)
    distance_reward = distance_weight * positions[..., -1]
    acceleration_cost = acceleration_weight * np.sum(np.abs(accelerations) * (np.abs(accelerations) > 2) / 2, axis=-1)
    jerk_cost = jerk_weight * np.sum(np.abs(np.diff(accelerations, axis=-1)), axis=-1)
    return distance_reward - acceleration_cost - jerk_cost - collision_weight * collides


def run_cem(args: argparse.Namespace) -> list[CEMState]:
    """Fit a diagonal Gaussian over acceleration sequences with CEM."""
    rng = np.random.default_rng(args.seed)
    mean = np.full(args.horizon, args.initial_acceleration, dtype=float)
    std = np.full(args.horizon, args.initial_std, dtype=float)
    states: list[CEMState] = []
    collision_time = (args.collision_t0, args.collision_t1)
    collision_distance = (args.collision_x0, args.collision_x1)

    for iteration in range(1, args.iterations + 1):
        accelerations = rng.normal(mean, std, size=(args.samples, args.horizon))
        accelerations = np.clip(accelerations, -args.max_acceleration, args.max_acceleration)
        positions = rollout(accelerations, args.dt, args.initial_velocity)
        rewards = trajectory_rewards(
            accelerations,
            positions,
            args.dt,
            args.distance_weight,
            args.acceleration_weight,
            args.jerk_weight,
            args.collision_weight,
            collision_time,
            collision_distance,
        )
        elite_indices = np.argpartition(rewards, -args.elites)[-args.elites :]
        elite_accelerations = accelerations[elite_indices]
        mean = elite_accelerations.mean(axis=0)
        std = np.maximum(elite_accelerations.std(axis=0), args.min_std)
        best_index = int(np.argmax(rewards))

        states.append(
            CEMState(
                iteration=iteration,
                sampled_positions=positions,
                elite_positions=positions[elite_indices],
                mean_positions=rollout(
                    mean[np.newaxis, :], args.dt, args.initial_velocity
                )[0],
                best_positions=positions[best_index],
                best_reward=float(rewards[best_index]),
                mean_reward=float(rewards.mean()),
            )
        )
    return states


def draw_frame(axis: plt.Axes, state: CEMState, args: argparse.Namespace) -> None:
    """Draw the sample population, elite trajectories, and current mean plan."""
    axis.clear()
    times = np.arange(args.horizon + 1) * args.dt
    collision = Rectangle(
        (args.collision_t0, args.collision_x0),
        args.collision_t1 - args.collision_t0,
        args.collision_x1 - args.collision_x0,
        facecolor="tomato",
        edgecolor="firebrick",
        alpha=0.28,
        label="Collision region",
        zorder=0,
    )
    axis.add_patch(collision)
    axis.plot(
        times,
        state.sampled_positions.T,
        color="tab:blue",
        linewidth=0.6,
        alpha=0.25,        
        zorder=1,
    )
    axis.plot(
        times,
        state.elite_positions.T,
        color="tab:blue",
        linewidth=2.3,
        alpha=0.9,        
        zorder=2,
    )
    axis.plot(
        times,
        state.mean_positions,
        color="black",
        linestyle=":",
        linewidth=3.2,
        label="Mean acceleration trajectory",
        zorder=3,
    )
    ymax = max(
        args.plot_max_distance,
        float(state.sampled_positions.max()),
        float(state.mean_positions.max()),
        1.0,
    )
    axis.set_xlim(0, times[-1])
    axis.set_ylim(min(0.0, args.collision_x0) - 0.05 * ymax, 1.1 * ymax)
    axis.set_xlabel("Time")
    axis.set_ylabel("Distance")
    axis.set_title(
        f"CEM longitudinal planning — iteration {state.iteration}/{args.iterations}"
        f"\nBest reward: {state.best_reward:.1f}; population mean: {state.mean_reward:.1f}"
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")


def perpendicular_vehicle_center(time: float, args: argparse.Namespace) -> float:
    """Return the red vehicle's horizontal center at ``time``.

    Its front reaches the intersection at collision_t0 and its rear clears it
    at collision_t1, so it occupies the intersection throughout that interval.
    """
    street_width = 3.0
    safety_buffer_length = 7.0
    vehicle_length = safety_buffer_length
    entry_center = -(street_width + vehicle_length) / 2
    exit_center = (street_width + vehicle_length) / 2
    speed = (exit_center - entry_center) / (args.collision_t1 - args.collision_t0)
    return entry_center + speed * (time - args.collision_t0)


def draw_plan_frame(
    axis: plt.Axes, frame: int, best_positions: np.ndarray, args: argparse.Namespace
) -> None:
    """Draw the final highest-scoring plan as moving vehicles in an intersection."""
    ego_length = 4.0
    ego_width = 1.8
    street_width = 3.0
    other_vehicle_length = 4.0
    safety_buffer_length = 7.0
    cross_street_width = args.collision_x1 - args.collision_x0 - ego_length
    intersection_start = args.collision_x0 + ego_length / 2
    time = frame * args.dt
    ego_center = best_positions[frame]
    scene_top = max(float(best_positions.max()) + ego_length, args.collision_x1 + ego_length)
    scene_bottom = min(-ego_length, intersection_start - ego_length)

    axis.clear()
    axis.add_patch(
        Rectangle(
            (-street_width / 2, scene_bottom),
            street_width,
            scene_top - scene_bottom,
            facecolor="#9e9e9e",
            edgecolor="none",
            zorder=0,
        )
    )
    axis.add_patch(
        Rectangle(
            (-10, intersection_start),
            20,
            cross_street_width,
            facecolor="#9e9e9e",
            edgecolor="none",
            zorder=0,
        )
    )
    axis.add_patch(
        Rectangle(
            (-ego_width / 2, ego_center - ego_length / 2),
            ego_width,
            ego_length,
            facecolor="tab:blue",
            edgecolor="navy",
            label="Ego vehicle",
            zorder=3,
        )
    )
    red_center = perpendicular_vehicle_center(time, args)
    axis.add_patch(
        Rectangle(
            (
                red_center - safety_buffer_length / 2,
                intersection_start + (cross_street_width - 1.8) / 2,
            ),
            safety_buffer_length,
            1.8,
            fill=False,
            edgecolor="tab:red",
            linewidth=2,
            label="Safety buffer",
            zorder=2,
        )
    )
    axis.add_patch(
        Rectangle(
            (
                red_center - other_vehicle_length / 2,
                intersection_start + (cross_street_width - 1.8) / 2,
            ),
            other_vehicle_length,
            1.8,
            facecolor="tab:red",
            edgecolor="maroon",
            label="Perpendicular vehicle with buffer",
            zorder=2,
        )
    )
    axis.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.65)
    axis.text(
        1.8,
        0,
        "Ego center origin",
        va="center",
        fontsize=9,
        color="#333333",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-10, 10)
    axis.set_ylim(scene_bottom, scene_top)
    axis.set_xlabel("Lateral position (m)")
    axis.set_ylabel("Longitudinal distance (m)")
    axis.set_title(f"Highest-scoring CEM plan — time {time:.1f} s")
    axis.legend(loc="upper right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=30, help="Planning steps.")
    parser.add_argument("--dt", type=float, default=0.2, help="Seconds per step.")
    parser.add_argument("--iterations", type=int, default=25, help="CEM updates.")
    parser.add_argument("--samples", type=int, default=100, help="Samples per update.")
    parser.add_argument("--elites", type=int, default=10, help="Top samples retained.")
    parser.add_argument(
        "--v0",
        "--initial-velocity",
        dest="initial_velocity",
        type=float,
        default=2.0,
        help="Initial ego velocity in m/s (default: 2.0).",
    )
    parser.add_argument("--initial-acceleration", type=float, default=0.0)
    parser.add_argument("--initial-std", type=float, default=2.0)
    parser.add_argument("--min-std", type=float, default=0.08)
    parser.add_argument("--max-acceleration", type=float, default=4.0)
    parser.add_argument("--distance-weight", "-a", type=float, default=1.0)
    parser.add_argument("--acceleration-weight", "-b", type=float, default=0.15)
    parser.add_argument(
        "--jerk-weight",
        type=float,
        default=0.1,
        help="Penalty weight for absolute changes between accelerations.",
    )
    parser.add_argument("--collision-weight", "-c", type=float, default=100.0)
    parser.add_argument("--collision-t0", type=float, default=2.0)
    parser.add_argument("--collision-t1", type=float, default=3.5)
    parser.add_argument("--collision-x0", type=float, default=3.0)
    parser.add_argument("--collision-x1", type=float, default=9.0)
    parser.add_argument("--plot-max-distance", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cem_longitudinal_planning.gif"),
        help="GIF output path.",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=Path("cem_best_plan.gif"),
        help="Animation output path for the final highest-scoring plan.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.horizon < 1 or args.iterations < 1 or args.samples < 1:
        raise ValueError("--horizon, --iterations, and --samples must be positive.")
    if not 0 < args.elites <= args.samples:
        raise ValueError("--elites must be between 1 and --samples.")
    if args.dt <= 0 or args.initial_std <= 0 or args.min_std <= 0 or args.fps < 1:
        raise ValueError("--dt, standard deviations, and --fps must be positive.")
    if args.collision_t1 <= args.collision_t0 or args.collision_x1 <= args.collision_x0:
        raise ValueError("Collision region upper bounds must exceed lower bounds.")
    if args.collision_x1 - args.collision_x0 <= 4.0:
        raise ValueError("Collision distance range must exceed the 4 m ego length.")


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    states = run_cem(args)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    animation = FuncAnimation(
        figure,
        lambda frame: draw_frame(axis, states[frame], args),
        frames=len(states),
        repeat=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=args.fps))
    plt.close(figure)
    print(f"Wrote {args.output}")

    plan_figure, plan_axis = plt.subplots(figsize=(6, 8))
    plan_animation = FuncAnimation(
        plan_figure,
        lambda frame: draw_plan_frame(plan_axis, frame, states[-1].best_positions, args),
        frames=args.horizon + 1,
        repeat=False,
    )
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    plan_animation.save(args.plan_output, writer=PillowWriter(fps=args.fps))
    plt.close(plan_figure)
    print(f"Wrote {args.plan_output}")


if __name__ == "__main__":
    main()
