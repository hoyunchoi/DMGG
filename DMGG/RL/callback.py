from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class TrackerCallback(BaseCallback):
    """
    Records scalar values at each step, and at the end of each episode,
    - logs a line plot of the episode sequence,
    - logs sequence statistics (mean/std/slope),
    - logs a histogram,
    to TensorBoard.
    """

    def __init__(self):
        super().__init__()
        self.writer: SummaryWriter | None = None

        # Monotonic episode id across all environments
        self._global_episode_count = 0

        # Store the last completed episode to report the range of completed episodes in current report cycle
        self._last_completed_episode = 0

        # Store the success flag of each episode only for current report cycle
        self.is_success: list[bool] = []


    def _on_training_start(self) -> None:
        if SummaryWriter is None:
            return
        log_dir = self.logger.get_dir()
        if log_dir is None:
            return

        self.writer = SummaryWriter(log_dir=log_dir)


    def _on_step(self) -> bool:
        if self.writer is None:
            # If no writer is set (e.g., tensorboard), do nothing
            return True

        # self.locals contains variables in the context of rollout collection
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)

        if infos is None or dones is None:
            return True

        for env_id in range(len(infos)):
            # Collect scalar for each environment
            info, done = infos[env_id], dones[env_id]

            # If episode is done, record the success flag
            if done:
                self._global_episode_count += 1
                self.is_success.append(not info["TimeLimit.truncated"])

        return True

    def _on_rollout_end(self) -> None:
        if self.writer is None:
            # If no writer is set (e.g., tensorboard), do nothing
            return

        # (start, end] episodes are complete in this report cycle
        start = self._last_completed_episode
        end = self._global_episode_count
        num_episodes = end - start

        # Success rate in this report cycle
        success_rate = (
            sum(self.is_success) / num_episodes if num_episodes > 0 else 0.0
        )

        self.logger.record(
            "rollout/episodes_range", f"{start+1}-{end}" if end > start else "—"
        )
        self.logger.record("rollout/num_episodes", num_episodes)
        self.logger.record("rollout/success_rate", success_rate)

        # Update the last completed episode
        self._last_completed_episode = end
        self.is_success.clear()

    def _on_training_end(self) -> None:
        if self.writer is None:
            return

        self.writer.flush()
        self.writer.close()


    def save(self, result_dir: Path) -> None:
        with open(result_dir / "global_episode_count.txt", "w") as f:
            f.write(str(self._global_episode_count))

    def load(self, result_dir: Path) -> None:
        with open(result_dir / "global_episode_count.txt", "r") as f:
            self._global_episode_count = int(f.read())
            self._last_completed_episode = self._global_episode_count
