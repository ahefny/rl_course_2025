#!/usr/bin/env python3
"""Print one 10-step SmolVLA action chunk for a recorded LeRobot sample."""

from __future__ import annotations

import argparse

import torch


DEFAULT_MODEL = "lerobot/smolvla_libero"
DEFAULT_DATASET = "lerobot/libero"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local checkpoint.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="LeRobot dataset repository ID.")
    parser.add_argument("--episode", type=int, default=0, help="Episode from which to read a frame.")
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Frame offset within the selected episode (default: first frame).",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=10,
        help="Number of predicted future actions to print (default: 10).",
    )
    parser.add_argument("--device", default=None, help="Torch device; defaults to CUDA when available.")
    return parser.parse_args()


def adapt_observation_to_policy(frame: dict, input_features: dict) -> dict:
    """Map dataset observations to the checkpoint's required input features.

    This fallback permits inspection of datasets with differently named camera
    streams. Low-dimensional observations remain untouched because their
    dimension is checkpoint-specific and is normalized by the checkpoint's
    preprocessor.
    """
    observation = {"task": frame["task"]} if "task" in frame else {}
    expected_images = [key for key in input_features if key.startswith("observation.images.")]
    available_images = [
        key for key in frame if key.startswith("observation.images.") and key not in expected_images
    ]

    for index, key in enumerate(expected_images):
        if key in frame:
            observation[key] = frame[key]
        elif available_images:
            observation[key] = frame[available_images[index % len(available_images)]]
        else:
            raise ValueError(f"Dataset frame has no images to supply required feature {key!r}.")

    for key in input_features:
        if key.startswith("observation.images."):
            continue
        if key not in frame:
            raise ValueError(f"Dataset frame is missing required feature {key!r}.")
        observation[key] = frame[key]

    return observation


def main() -> None:
    args = parse_args()
    if args.chunk_steps < 1:
        raise ValueError("--chunk-steps must be positive.")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as error:
        raise SystemExit(
            "A required LeRobot component is unavailable. Run ./vla/setup.sh again.\n"
            f"Original error: {error}"
        ) from error

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading SmolVLA from {args.model!r} on {device}...")
    policy = SmolVLAPolicy.from_pretrained(args.model).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args.model,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print(f"Loading LeRobot dataset {args.dataset!r}...")
    dataset = LeRobotDataset(args.dataset)
    if not 0 <= args.episode < len(dataset.meta.episodes):
        raise ValueError(f"--episode must be in [0, {len(dataset.meta.episodes) - 1}].")

    start = dataset.meta.episodes["dataset_from_index"][args.episode]
    end = dataset.meta.episodes["dataset_to_index"][args.episode]
    frame_index = start + args.frame_offset
    if not start <= frame_index < end:
        raise ValueError(
            f"--frame-offset must be in [0, {end - start - 1}] for episode {args.episode}."
        )

    frame = dict(dataset[frame_index])
    task = frame.get("task", "<no task annotation>")
    print(f"Running frame {frame_index} from episode {args.episode}: {task}")
    policy_frame = adapt_observation_to_policy(frame, policy.config.input_features)

    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(preprocess(policy_frame))
        action_chunk = postprocess(action_chunk)

    action_chunk = action_chunk.detach().cpu().squeeze(0)
    action_chunk = action_chunk[: args.chunk_steps]
    print(f"Predicted action chunk ({len(action_chunk)} steps × {action_chunk.shape[-1]} values):")
    for step, action in enumerate(action_chunk):
        print(f"  step {step}: {action.tolist()}")


if __name__ == "__main__":
    main()
