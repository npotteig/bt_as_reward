from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class RewardThresholdCallback(BaseCallback):
    def __init__(self, threshold=1.0, num_envs=1, verbose=0):
        super().__init__(verbose)
        self.threshold = threshold
        self.n_episodes = 0
        self.num_envs = num_envs
        self.rewards = []
        self.current_rewards = np.zeros(num_envs)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        # Update cumulative rewards per environment
        self.current_rewards += np.array(rewards)

        for i in range(self.num_envs):
            if dones[i]:
                if infos[i].get("success", False):
                    self.rewards.append(1)
                else:
                    self.rewards.append(0)
                self.current_rewards[i] = 0
                self.n_episodes += 1
        return True

    def _on_rollout_end(self) -> None:
        # Collect rewards and episode starts from the rollout buffer
        if len(self.rewards) == 0:
            return
        # Iterate through the rewards and aggregate them per episode
        success_rate = (
            np.sum(np.array(self.rewards) >= self.threshold) / self.n_episodes
        )

        # Log the percentage to TensorBoard
        self.logger.record("rollout/success_rate", success_rate)

        # Reset the rewards and episode count
        self.rewards = []
        self.n_episodes = 0
