import os
import argparse
import gymnasium as gym
import numpy as np
from minigrid.wrappers import FullyObsWrapper
from tqdm import tqdm
from bt_as_reward.constants import (
    MINIGRID_OBJECT_TO_IDX,
    MINIGRID_COLOR_TO_IDX,
    MINIGRID_STATE_TO_IDX,
)
from bt_as_reward.path_planning.astar import astar
import bt_as_reward.envs  # noqa: F401

gym.register_envs(bt_as_reward.envs)


def construct_freespace_func(door_coord, room_coord=None):
    def is_freespace(coord):
        x, y = coord

        if room_coord is not None:
            # Door and room on same side
            if door_coord[1] == room_coord[1]:
                if (x, y) != door_coord and (x, y) != room_coord and y == door_coord[1]:
                    return False
            else:
                if ((x, y) != room_coord and y == room_coord[1]) or (
                    (x, y) != door_coord and y == door_coord[1]
                ):
                    return False
        elif (x, y) != door_coord and y == door_coord[1]:
            return False
        return True

    return is_freespace


def construct_drone_firefighter_freespace_func(obstacles, target_coords=[]):
    def is_freespace(coord):
        x, y = coord
        for obstacle in obstacles:
            if (x, y) == tuple(obstacle):
                return False
        if target_coords is not None and (x, y) in target_coords:
            return False
        return True

    return is_freespace


def heuristic(a, b):
    # Example: Using Manhattan distance as heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def create_minigrid_random_traj(env: FullyObsWrapper):
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
    door_coord = tuple(
        np.argwhere(obs["image"][:, :, 0] == MINIGRID_OBJECT_TO_IDX["door"])[0]
    )
    key_coord = tuple(
        np.argwhere(obs["image"][:, :, 0] == MINIGRID_OBJECT_TO_IDX["key"])[0]
    )
    goal_coord = tuple(
        np.argwhere(obs["image"][:, :, 0] == MINIGRID_OBJECT_TO_IDX["goal"])[0]
    )

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


def create_lockedroom_expert_traj(env: FullyObsWrapper):
    obs, _ = env.reset()
    mission = obs["mission"]
    # print(env.unwrapped.pprint_grid())

    state_arr = [obs]
    action_arr = []
    mission_words = mission.split()
    room_color = mission_words[mission_words.index("room,") - 1]
    key_color = mission_words[mission_words.index("key") - 1]
    door_color = mission_words[mission_words.index("door") - 1]

    door_coord = tuple(
        np.argwhere(
            (obs["image"][:, :, 0] == MINIGRID_OBJECT_TO_IDX["door"])
            & (obs["image"][:, :, 1] == MINIGRID_COLOR_TO_IDX[door_color])
        )[0]
    )
    key_coord = tuple(
        np.argwhere(
            (obs["image"][:, :, 0] == MINIGRID_OBJECT_TO_IDX["key"])
            & (obs["image"][:, :, 1] == MINIGRID_COLOR_TO_IDX[key_color])
        )[0]
    )
    room_coord = tuple(
        np.argwhere(
            (obs["image"][:, :, 0] == MINIGRID_OBJECT_TO_IDX["door"])
            & (obs["image"][:, :, 1] == MINIGRID_COLOR_TO_IDX[room_color])
        )[0]
    )
    goal_coord = tuple(
        np.argwhere(obs["image"][:, :, 0] == MINIGRID_OBJECT_TO_IDX["goal"])[0]
    )

    is_freespace = construct_freespace_func(
        (door_coord[1], door_coord[0]), (room_coord[1], room_coord[0])
    )

    # Room
    actions, _ = astar(
        is_freespace=is_freespace,
        heuristic=heuristic,
        start=(
            env.unwrapped.agent_pos[1],
            env.unwrapped.agent_pos[0],
            env.unwrapped.agent_dir,
        ),  # (x, y, direction)
        goal=(room_coord[1], room_coord[0]),  # (x, y)
    )
    action_arr.extend(actions[1:-1] + [5, 2])
    for action in actions[1:-1] + [
        5,
        2,
    ]:  # Add an interact action + move through doorway at the end
        obs, _, terminated, _, _ = env.step(action)
        # print(env.unwrapped.pprint_grid())
        state_arr.append(obs)

    # Key
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

    if len(actions[1:-1]) == 0:
        # If key was immediately behind the door, no need to backtrack to room
        actions = [1, 1, 2]
        action_arr.extend(actions)
        for action in actions:
            obs, _, terminated, _, _ = env.step(action)
            state_arr.append(obs)
    else:
        # Backtrack to Room
        actions, _ = astar(
            is_freespace=is_freespace,
            heuristic=heuristic,
            start=(
                env.unwrapped.agent_pos[1],
                env.unwrapped.agent_pos[0],
                env.unwrapped.agent_dir,
            ),  # (x, y, direction)
            goal=(room_coord[1], room_coord[0]),  # (x, y)
        )
        action_arr.extend(actions[1:-1] + [2, 2])
        for action in actions[1:-1] + [
            2,
            2,
        ]:  # Move through doorway at the end
            obs, _, terminated, _, _ = env.step(action)
            state_arr.append(obs)

    # Locked Door
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
    action_arr.extend(actions[1:-1] + [5, 2])
    for action in actions[1:-1] + [
        5,
        2,
    ]:  # Add an interact action + move through doorway at the end
        obs, _, terminated, _, _ = env.step(action)
        state_arr.append(obs)

    # Goal
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

    assert terminated, (
        f"The episode should terminate after reaching the goal.\n{env.unwrapped.pprint_grid()}"
    )
    return state_arr, action_arr


def create_drone_supplier_expert_traj(env: FullyObsWrapper):
    obs, _ = env.reset()

    state_arr = [obs]
    action_arr = []

    mission = obs["mission"]
    # get word before "box"
    mission_words = mission.split()
    region_color = mission_words[mission_words.index("box,") - 1]

    # find x, y coordinates of agent and door
    door_coord = tuple(
        np.argwhere(
            (obs["image"][:, :, 0] == MINIGRID_OBJECT_TO_IDX["door"])
            & (obs["image"][:, :, 2] == MINIGRID_STATE_TO_IDX["locked"])
        )[0]
    )
    box_coord = tuple(
        np.argwhere(
            (obs["image"][:, :, 0] == MINIGRID_OBJECT_TO_IDX["box"])
            & (obs["image"][:, :, 1] == MINIGRID_COLOR_TO_IDX[region_color])
        )[0]
    )
    other_door_coords = np.argwhere(
        (obs["image"][:, :, 0] == MINIGRID_OBJECT_TO_IDX["door"])
        & (obs["image"][:, :, 2] != MINIGRID_STATE_TO_IDX["locked"])
    )
    wall_obstacles = np.argwhere(
        obs["image"][:, :, 0] == MINIGRID_OBJECT_TO_IDX["wall"]
    )
    other_box_coords = np.argwhere(
        (obs["image"][:, :, 0] == MINIGRID_OBJECT_TO_IDX["box"])
        & (obs["image"][:, :, 1] != MINIGRID_COLOR_TO_IDX[region_color])
    )
    obstacles = np.vstack([wall_obstacles, other_door_coords, other_box_coords])[
        :, ::-1
    ]

    is_freespace_second = construct_drone_firefighter_freespace_func(
        obstacles, [(door_coord[1], door_coord[0])]
    )

    is_freespace_third = construct_drone_firefighter_freespace_func(
        obstacles, [(box_coord[1], box_coord[0])]
    )

    actions, _ = astar(
        is_freespace=is_freespace_second,
        heuristic=heuristic,
        start=(
            env.unwrapped.agent_pos[1],
            env.unwrapped.agent_pos[0],
            env.unwrapped.agent_dir,
        ),  # (x, y, direction)
        goal=(box_coord[1], box_coord[0]),  # (x, y)
    )
    action_arr.extend(actions[1:-1] + [5, 3])  # Add an interact action

    for action in actions[1:-1] + [5, 3]:  # Add an interact action
        obs, _, terminated, _, _ = env.step(action)
        state_arr.append(obs)

    actions, _ = astar(
        is_freespace=is_freespace_third,
        heuristic=heuristic,
        start=(
            env.unwrapped.agent_pos[1],
            env.unwrapped.agent_pos[0],
            env.unwrapped.agent_dir,
        ),  # (x, y, direction)
        goal=(door_coord[1], door_coord[0]),  # (x, y)
    )
    action_arr.extend(actions[1:-1] + [5])  # Add an interact action

    for action in actions[1:-1] + [
        5,
    ]:  # Add an interact action
        obs, _, terminated, _, _ = env.step(action)
        state_arr.append(obs)

    assert terminated, "The episode should terminate after opening the door."
    return state_arr, action_arr


if __name__ == "__main__":
    # num trajs as argument
    parser = argparse.ArgumentParser(
        description="Create MiniGrid trajectories for MiniGrid task."
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="MiniGrid-DoorKey-6x6-v0",
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

    # make trajs/doorkey if it doesn't exist
    if not os.path.exists("trajs/doorkey"):
        os.makedirs("trajs/doorkey", exist_ok=True)
        print("Directory 'trajs/doorkey' created.")

    if not os.path.exists("trajs/lockedroom"):
        os.makedirs("trajs/lockedroom", exist_ok=True)
        print("Directory 'trajs/lockedroom' created.")

    if not os.path.exists("trajs/dronesupplier"):
        os.makedirs("trajs/dronesupplier", exist_ok=True)
        print("Directory 'trajs/dronesupplier' created.")

    env = FullyObsWrapper(gym.make(args.env_name, max_steps=500))
    env.reset(seed=1)

    if "DoorKey" in args.env_name:
        create_expert_traj = create_doorkey_expert_traj
    elif "LockedRoom" in args.env_name:
        create_expert_traj = create_lockedroom_expert_traj
    elif "DroneSupplier" in args.env_name:
        create_expert_traj = create_drone_supplier_expert_traj
    else:
        raise ValueError("Environment not supported for expert trajectory generation.")

    # Collect expert trajectories
    expert_episodes = [None] * args.num_expert_trajs
    for i in tqdm(range(args.num_expert_trajs)):
        state_arr, action_arr = create_expert_traj(env)
        expert_episodes[i] = {"states": state_arr, "actions": action_arr}
    if "DoorKey" in args.env_name:
        np.savez("trajs/doorkey/expert.npz", episodes=expert_episodes)
    elif "LockedRoom" in args.env_name:
        np.savez("trajs/lockedroom/expert.npz", episodes=expert_episodes)
    else:
        np.savez("trajs/dronesupplier/expert.npz", episodes=expert_episodes)

    # Collect random trajectories
    random_episodes = [None] * args.num_random_trajs
    for i in tqdm(range(args.num_random_trajs)):
        state_arr, action_arr = create_minigrid_random_traj(env)
        random_episodes[i] = {"states": state_arr, "actions": action_arr}
    if "DoorKey" in args.env_name:
        np.savez("trajs/doorkey/random.npz", episodes=random_episodes)
    elif "LockedRoom" in args.env_name:
        np.savez("trajs/lockedroom/random.npz", episodes=random_episodes)
    else:
        np.savez("trajs/dronesupplier/random.npz", episodes=random_episodes)
