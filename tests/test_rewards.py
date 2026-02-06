import gymnasium as gym
import numpy as np
import minigrid  # noqa: F401

from llm_functions.doorkey.subtask_functions_gpt5 import (
    subtask_1_complete,
    subtask_2_complete,
    subtask_3_complete,
    subtask_1_object,
    subtask_2_object,
    subtask_3_object,
)
from bt_as_reward.rewards.bt import BehaviourTreeConfig
from bt_as_reward.training.wrappers import MinigridSubtaskWrapper
from bt_as_reward.utils import minigrid_create_distance_function, minigrid_object_to_str


def test_rewards():
    bt_config = BehaviourTreeConfig(
        subtask_names=["Acquire Key", "Open Door", "Reach Goal"],
        subtask_functions=[subtask_1_complete, subtask_2_complete, subtask_3_complete],
        object_functions=[subtask_1_object, subtask_2_object, subtask_3_object],
        object_to_str=minigrid_object_to_str,
        create_distance_function=minigrid_create_distance_function,
        distance_threshold=1.0,
        use_memory=False,
    )

    env = gym.make("MiniGrid-DoorKey-6x6-v0")
    env = MinigridSubtaskWrapper(env, mode="bt_as_reward", bt_config=bt_config)
    _, _ = env.reset(seed=42)

    actions = [1, 1, 2, 3, 4, 3, 2, 1, 1, 2, 0, 5, 5, 5, 2, 2, 4, 3, 1, 2, 2, 0, 2]

    rewards = np.zeros(len(actions))
    for i, action in enumerate(actions):
        _, reward, _, _, _ = env.step(action)
        rewards[i] = reward
    assert np.sum(rewards) == 4.0, "BT Rewards: Total reward should be 4.0"
    assert np.any(rewards < 0), "BT Rewards: There should be some negative rewards"

    env.mode = "proc_as_reward"
    env.bt_config.use_memory = True
    _, _ = env.reset(seed=42)
    rewards = np.zeros(len(actions))
    for i, action in enumerate(actions):
        _, reward, _, _, _ = env.step(action)
        rewards[i] = reward
    assert np.sum(rewards) == 4.0, (
        "Procedural Rewards: Total reward should be 4.0"
    )
    assert np.all(rewards >= 0), (
        "Procedural Rewards: There should be no negative rewards"
    )

    env.mode = "environment"
    env.bt_config = None
    _, _ = env.reset(seed=42)
    rewards = np.zeros(len(actions))
    for i, action in enumerate(actions):
        _, reward, _, _, _ = env.step(action)
        rewards[i] = reward
    assert np.sum(rewards) == 1, (
        "Environment Rewards: Total reward should be 1.0"
    )
    assert np.all(rewards >= 0), (
        "Environment Rewards: There should be no negative rewards"
    )
