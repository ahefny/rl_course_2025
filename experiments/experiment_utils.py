import numpy as np
import gymnasium as gym

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Video

class EpisodeEvalCallback(BaseCallback):
    """Periodically runs a test episode using the current greedy policy.

    Logs the reward evey ``eval_every_episodes`` episodes.
    Records a video every ``record_every_episodes`` episodes.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        eval_every_episodes: int = 1000,
        record_every_episodes: int = 1000,
        max_steps: int = 500,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_every_episodes = eval_every_episodes
        self.record_every_episodes = record_every_episodes
        self.max_steps = max_steps
        self._episode_count = 0
        self._last_eval_at = 0
        self._last_record_at = 0

    def _log_eval_run(self, reward: float, episode_length: int, frames: list[np.ndarray]) -> None:
        self.logger.record("eval/episode_reward", reward)
        self.logger.record("eval/episode_length", episode_length)
        
        if frames:
            video = np.array(frames, dtype=np.uint8).transpose(0, 3, 1, 2)[None]
            self.logger.record(
                "eval/video",
                Video(video, fps=30),
                exclude=("stdout", "log", "json", "csv"),
            )

    def _run_episode(self, record: bool) -> None:
        obs, _ = self.eval_env.reset()
        frames: list[np.ndarray] = []
        total_reward = 0.0
        episode_length = 0

        for _ in range(self.max_steps):
            if record:
                frame = self.eval_env.render()
                if frame is not None:
                    frames.append(frame)

            episode_length += 1
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.eval_env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        self._log_eval_run(total_reward, episode_length, frames)

    def _on_step(self) -> bool:
        # Count how many training env episodes ended on this step
        dones = self.locals.get("dones")
        if dones is None:
            return True

        self._episode_count += int(np.sum(dones))

        should_eval = (
            self._episode_count - self._last_eval_at
            >= self.eval_every_episodes
        )

        should_record = (
            self.record_every_episodes > 0 and
            self._episode_count - self._last_record_at >= self.record_every_episodes
        )

        if should_eval or should_record:
            if self.verbose:
                print(
                    f"Running eval at episode {self._episode_count:,} "
                    f"(t={self.num_timesteps:,})"
                )
            self._run_episode(record=should_record)
            self._last_eval_at = self._episode_count

        if should_record:
            self._last_record_at = self._episode_count

        return True