import os
import argparse
import gymnasium as gym
import numpy as np
from minigrid.wrappers import FullyObsWrapper
from bt_as_reward.constants import OBJECT_TO_IDX
from bt_as_reward.path_planning.astar import astar


def construct_freespace_func(door_coord):
    def is_freespace(coord):
        x, y = coord
        if (x, y) != door_coord and y == door_coord[1]:
            return False
        return True

    return is_freespace


def heuristic(a, b):
    # Example: Using Manhattan distance as heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def create_doorkey_random_traj(env: FullyObsWrapper):
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


def create_doorkey_expert_traj(env: FullyObsWrapper):
    obs, _ = env.reset()

    state_arr = [obs]
    action_arr = []

    # find x, y coordinates of agent and door
    door_coord = tuple(np.argwhere(obs["image"][:, :, 0] == OBJECT_TO_IDX["door"])[0])
    key_coord = tuple(np.argwhere(obs["image"][:, :, 0] == OBJECT_TO_IDX["key"])[0])
    goal_coord = tuple(np.argwhere(obs["image"][:, :, 0] == OBJECT_TO_IDX["goal"])[0])

    is_freespace = construct_freespace_func((door_coord[1], door_coord[0]))

    actions, _ = astar(
        is_freespace=is_freespace,
        heuristic=heuristic,
        start=(
            env.unwrapped.agent_pos[1],
            env.unwrapped.agent_pos[0],
            env.unwrapped.agent_dir,
        ),  # (x, y, direction)
        goal=(key_coord[1], key_coord[0]),  # (x, y)
    )
    action_arr.extend(actions[1:-1] + [3])  # Add a pickup action at the end

    for action in actions[1:-1] + [3]:  # Add a pickup action at the end
        obs, _, terminated, _, _ = env.step(action)
        state_arr.append(obs)

    actions, _ = astar(
        is_freespace=is_freespace,
        heuristic=heuristic,
        start=(
            env.unwrapped.agent_pos[1],
            env.unwrapped.agent_pos[0],
            env.unwrapped.agent_dir,
        ),  # (x, y, direction)
        goal=(door_coord[1], door_coord[0]),  # (x, y)
    )
    action_arr.extend(
        actions[1:-1] + [5, 2, 2]
    )  # Add an interact action + move through doorway at the end

    for action in actions[1:-1] + [
        5,
        2,
        2,
    ]:  # Add an interact action + move through doorway at the end
        obs, _, terminated, _, _ = env.step(action)
        state_arr.append(obs)

    actions, _ = astar(
        is_freespace=is_freespace,
        heuristic=heuristic,
        start=(
            env.unwrapped.agent_pos[1],
            env.unwrapped.agent_pos[0],
            env.unwrapped.agent_dir,
        ),  # (x, y, direction)
        goal=(goal_coord[1], goal_coord[0]),  # (x, y)
    )
    action_arr.extend(actions[1:])  # Add the remaining actions to reach the goal

    for action in actions[1:]:
        obs, _, terminated, _, _ = env.step(action)
        state_arr.append(obs)

    assert terminated, "The episode should terminate after reaching the goal."
    return state_arr, action_arr


if __name__ == "__main__":
    # num trajs as argument
    parser = argparse.ArgumentParser(
        description="Create MiniGrid trajectories for DoorKey task."
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

    # make trajs/doorkey if it doesn't exist
    if not os.path.exists("trajs/doorkey"):
        os.makedirs("trajs/doorkey", exist_ok=True)
        print("Directory 'trajs/doorkey' created.")

    env = FullyObsWrapper(gym.make("MiniGrid-DoorKey-6x6-v0"))

    # Collect expert trajectories
    expert_episodes = [None] * args.num_expert_trajs
    for i in range(args.num_expert_trajs):
        state_arr, action_arr = create_doorkey_expert_traj(env)
        expert_episodes[i] = {"states": state_arr, "actions": action_arr}
    np.savez("trajs/doorkey/expert.npz", episodes=expert_episodes)

    # Collect random trajectories
    random_episodes = [None] * args.num_random_trajs
    for i in range(args.num_random_trajs):
        state_arr, action_arr = create_doorkey_random_traj(env)
        random_episodes[i] = {"states": state_arr, "actions": action_arr}
    np.savez("trajs/doorkey/random.npz", episodes=random_episodes)
