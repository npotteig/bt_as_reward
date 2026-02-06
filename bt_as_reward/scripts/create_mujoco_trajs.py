import argparse
import os
import numpy as np
import gymnasium as gym
from tqdm import tqdm

import bt_as_reward.envs

gym.register_envs(bt_as_reward.envs)

NUM_BLOCKS = 2


def create_mujoco_random_traj(env: gym.Env):
    obs, _ = env.reset()

    state_arr = [obs]
    action_arr = []

    terminated, truncated = False, False
    while not terminated and not truncated:
        action = env.action_space.sample()  # Random action
        obs, _, terminated, truncated, _ = env.step(action)
        state_arr.append(obs)
        action_arr.append(action)

    return state_arr, action_arr


def create_pick_and_place_expert_traj(env: gym.Env):
    obs, info = env.reset()
    goal_id = env.unwrapped.goal_index

    state_arr = [obs]
    action_arr = []

    while np.abs(obs["observation"][6]) > 0.02:
        if obs["observation"][6] >= 0:
            action = 0
        else:
            action = 1
        obs, _, _, _, info = env.step(action)
        state_arr.append(obs)
        action_arr.append(action)

    while np.abs(obs["observation"][7]) > 0.02:
        if obs["observation"][7] >= 0:
            action = 2
        else:
            action = 3
        obs, _, _, _, info = env.step(action)
        state_arr.append(obs)
        action_arr.append(action)

    while obs["observation"][8] < -0.01:
        action = 5
        obs, _, _, _, info = env.step(action)
        state_arr.append(obs)
        action_arr.append(action)

    for _ in range(2):
        action = 7
        obs, _, _, _, info = env.step(action)
        state_arr.append(obs)
        action_arr.append(action)

    early_stop = False
    while (not early_stop) and np.abs(
        obs["desired_goal"][3 * goal_id + 2] - obs["observation"][2]
    ) > 0.025:
        if (
            obs["desired_goal"][3 * goal_id + 2] - obs["observation"][3 * goal_id + 2]
            >= 0
        ):
            action = 4
        else:
            action = 5
        obs, _, _, _, info = env.step(action)
        state_arr.append(obs)
        action_arr.append(action)
        if info["is_success"]:
            early_stop = True

    while (not early_stop) and np.abs(
        obs["desired_goal"][3 * goal_id] - obs["observation"][0]
    ) > 0.025:
        if obs["desired_goal"][3 * goal_id] - obs["observation"][3 * goal_id] >= 0:
            action = 0
        else:
            action = 1
        obs, _, _, _, info = env.step(action)
        state_arr.append(obs)
        action_arr.append(action)
        if info["is_success"]:
            early_stop = True

    while (not early_stop) and np.abs(
        obs["desired_goal"][3 * goal_id + 1] - obs["observation"][1]
    ) > 0.025:
        if (
            obs["desired_goal"][3 * goal_id + 1] - obs["observation"][3 * goal_id + 1]
            >= 0
        ):
            action = 2
        else:
            action = 3
        obs, _, _, _, info = env.step(action)
        state_arr.append(obs)
        action_arr.append(action)
        if info["is_success"]:
            early_stop = True

    return state_arr, action_arr, info["is_success"]


if __name__ == "__main__":
    # num trajs as argument
    parser = argparse.ArgumentParser(
        description="Create Mujoco trajectories for Fetch task."
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="FetchPickAndPlace-v5",
        help="Name of the MiniGrid environment.",
    )
    parser.add_argument(
        "--num_expert_trajs",
        type=int,
        default=100,
        help="Number of expert trajectories to create.",
    )
    parser.add_argument(
        "--num_random_trajs",
        type=int,
        default=100,
        help="Number of random trajectories to create.",
    )
    args = parser.parse_args()

    if not os.path.exists("trajs/pickandplace"):
        os.makedirs("trajs/pickandplace", exist_ok=True)
        print("Directory 'trajs/pickandplace' created.")

    if not os.path.exists("trajs/pickandplace2"):
        os.makedirs("trajs/pickandplace2", exist_ok=True)
        print("Directory 'trajs/pickandplace2' created.")

    env = gym.make(args.env_name, max_episode_steps=500)
    create_expert_traj = create_pick_and_place_expert_traj
    env.reset(seed=1)

    # Collect expert trajectories
    expert_episodes = [None] * args.num_expert_trajs
    for i in tqdm(range(args.num_expert_trajs)):
        is_success = False
        while not is_success:
            state_arr, action_arr, is_success = create_expert_traj(env)
        expert_episodes[i] = {"states": state_arr, "actions": action_arr}
    if "PickAndPlace2" in args.env_name:
        np.savez("trajs/pickandplace2/expert.npz", episodes=expert_episodes)
    else:
        np.savez("trajs/pickandplace/expert.npz", episodes=expert_episodes)

    # Collect random trajectories
    random_episodes = [None] * args.num_random_trajs
    for i in tqdm(range(args.num_random_trajs)):
        state_arr, action_arr = create_mujoco_random_traj(env)
        random_episodes[i] = {"states": state_arr, "actions": action_arr}
    if "PickAndPlace2" in args.env_name:
        np.savez("trajs/pickandplace2/random.npz", episodes=random_episodes)
    else:
        np.savez("trajs/pickandplace/random.npz", episodes=random_episodes)
