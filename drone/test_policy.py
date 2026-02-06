import argparse
import math
import json
import copy
import airsim
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import torch
import imageio
from stable_baselines3 import PPO
from sb3_contrib.ppo_mask import MaskablePPO
import minigrid  # noqa: F401
from minigrid.wrappers import DictObservationSpaceWrapper

from bt_as_reward.training.wrappers import MinigridSubtaskWrapper
from bt_as_reward.rewards.bt import BehaviourTreeConfig
from bt_as_reward.utils import (
    load_functions_from_file,
    minigrid_create_distance_function,
    minigrid_object_to_str,
)
import bt_as_reward.envs  # noqa: F401

gym.register_envs(bt_as_reward.envs)

AIRSIM_FORWARD = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}

def airsim_step(action, client: airsim.MultirotorClient, env, num_actions=1, direction=0):
    state_synced = False
    match action:
        case (0, 1): # left, right
            angle = direction * 90
            client.moveByVelocityZAsync(0, 0, -5, 1, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=angle)).join()
            state_synced = True
            ori = client.simGetGroundTruthKinematics().orientation
            q = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])
            yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]*q[2] + q[3]*q[3]))
            
            yaw_degrees = math.degrees(yaw) % 360
            if abs(yaw_degrees - angle) <= 5:
                state_synced = True
            else:
                print(f"Desync detected: Drone yaw {yaw_degrees}, Env yaw {angle}")
        case 2: # forward
            pos_delta = AIRSIM_FORWARD[direction]
            kinematics = client.simGetGroundTruthKinematics()
            current_pos = kinematics.position
            target_x = round(current_pos.x_val - 0.5) + num_actions*pos_delta[0] + 0.5
            target_y = round(current_pos.y_val - 0.5) + num_actions*pos_delta[1] + 0.5
            
            client.moveToPositionAsync(target_x, target_y, -8, 1, timeout_sec=60).join()
            client.moveByVelocityZAsync(0, 0, -5, 0.8).join()
            next_kinematics = client.simGetGroundTruthKinematics()
            next_pos = next_kinematics.position
            agent_position = env.unwrapped.agent_pos
            if (round(next_pos.x_val - 0.5) + 0.5, round(next_pos.y_val - 0.5) + 0.5) == (agent_position[0] - 11.5, agent_position[1] - 11.5):
                state_synced = True
            else:
                print("Current position:", (current_pos.x_val), (current_pos.y_val))
                print("Target position:", (target_x, target_y))
                print(f"Desync detected: Drone at ({round(next_pos.x_val - 0.5) + 0.5}, {round(next_pos.y_val - 0.5) + 0.5}), Env at ({agent_position[0]-11.5}, {agent_position[1]-11.5})")
                print(env.unwrapped.pprint_grid())
        case _:
            state_synced = True  # Interact actions
    return state_synced
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO with GPT Subtasks")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained model",
        required=True,
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        help="Reward mode to use",
        choices=["environment", "proc_as_reward", "bt_as_reward"],
        default="environment",
    )
    parser.add_argument(
        "--use_airsim",
        type=str,
        default="true",
        help="Whether to use AirSim for drone simulation",
    )
    parser.add_argument(
        "--drop_key",
        type=str,
        default="false",
        help="Whether to drop the key randomly in the environment",
    )
    parser.add_argument(
        "--object_function_file",
        type=str,
        required=True,
        help="Path to the file containing the object detection function.",
    )
    parser.add_argument(
        "--object_function_names",
        type=str,
        required=True,
        help="Names of the object detection functions to verify.",
    )
    parser.add_argument(
        "--subtask_function_file",
        type=str,
        required=True,
        help="Path to the file containing the subtask function.",
    )
    parser.add_argument(
        "--subtask_names",
        type=str,
        required=True,
        help="Names of the subtask functions to verify.",
    )
    parser.add_argument(
        "--subtask_function_names",
        type=str,
        required=True,
        help="Names of the subtask function to verify.",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=1.0,
        help="Distance threshold for proximity check.",
    )
    parser.add_argument(
        "--action_mask_file",
        type=str,
        default="none",
        help="Path to the action mask file.",
    )
    args = parser.parse_args()
    drop_key = args.drop_key.lower() == "true"
    use_airsim = args.use_airsim.lower() == "true"

    if args.action_mask_file != "none":
        with open(args.action_mask_file, "r") as f:
            action_mask_dict = json.load(f)

    object_functions = load_functions_from_file(
        args.object_function_file, args.object_function_names.strip().split(", ")
    )
    subtask_functions = load_functions_from_file(
        args.subtask_function_file, args.subtask_function_names.strip().split(", ")
    )

    bt_config = BehaviourTreeConfig(
        subtask_names=args.subtask_names.strip().split(", "),
        subtask_functions=subtask_functions,
        object_functions=object_functions,
        object_to_str=minigrid_object_to_str,
        create_distance_function=minigrid_create_distance_function,
        distance_threshold=args.distance_threshold,
        use_memory=False if args.reward_mode == "bt_as_reward" else True,
    )

    env = gym.make("DroneSupplier-v0")
    env = MinigridSubtaskWrapper(
        env,
        drop_key=drop_key,
        drop_prob=0.05,
        mode=args.reward_mode,
        action_mask_dict=action_mask_dict if args.action_mask_file != "none" else None,
        bt_config=bt_config,
    )
    env = DictObservationSpaceWrapper(env, max_words_in_mission=16)
    env.reset(seed=0)

    if args.action_mask_file != "none":
        model_class = MaskablePPO
    else:
        model_class = PPO
    
    model = model_class.load(
        args.model_path,
        device=torch.device("cpu")
    )
    
    if use_airsim:
        client = airsim.MultirotorClient()
        client.confirmConnection()
    
    terminated, truncated = False, False
    n_success = 0
    n_synced = 0
    n_episodes = 1
    episode_lengths = []
    for episode in tqdm(range(n_episodes)):
        if use_airsim:
            client.reset()
            client.enableApiControl(True)
            client.armDisarm(True)
            client.takeoffAsync().join()
            # Move to 5 meters altitude
            client.moveToPositionAsync(0.5, 0.5, -5, velocity=5).join()
        obs, info = env.reset()
        images = [env.unwrapped.get_frame()]
        state_synced = False
        step = 0
        num_forward_actions = 0
        while not (terminated or truncated):
            action_masks = env.env.action_masks()
            if args.action_mask_file != "none":
                action, _ = model.predict(
                    obs,
                    action_masks=action_masks
                )
            else:
                action, _ = model.predict(
                    obs
                )
            prev_agent_pos = copy.deepcopy(env.unwrapped.agent_pos)
            prev_agent_dir = env.unwrapped.agent_dir
            obs, reward, terminated, truncated, info = env.step(action)
            images.append(env.unwrapped.get_frame())
            agent_pos = env.unwrapped.agent_pos
            agent_dir = env.unwrapped.agent_dir
            print(f"Step {step}")
            # print("Predicted action:", action)
            if (agent_pos != prev_agent_pos or agent_dir != prev_agent_dir) and use_airsim:
                if action == 2:  # forward
                    num_forward_actions += 1
                else:
                    if num_forward_actions > 0:
                        state_synced = airsim_step(2, client, env, num_actions=num_forward_actions, direction=prev_agent_dir)
                        if not state_synced:
                            break
                    state_synced = airsim_step(action, client, env, direction=agent_dir)
                    if not state_synced:
                        break
                    num_forward_actions = 0
                drone_pos = client.simGetGroundTruthKinematics().position
                # print("Drone position (rounded):", (round(drone_pos.x_val - 0.5) + 0.5, round(drone_pos.y_val - 0.5) + 0.5))
                # print("Env agent pos (adjusted):", (env.unwrapped.agent_pos[0]-11.5, env.unwrapped.agent_pos[1]-11.5))
            step += 1
            
        if state_synced:
            n_synced += 1
        if info.get("success", False):
            # print("Success")
            n_success += 1
        episode_lengths.append(step)
        terminated, truncated = False, False
    # save video using imageio
    video_filename = "plots/test_policy_output.mp4"
    imageio.mimwrite(video_filename, images, fps=10)
    print(f"Video saved to {video_filename}")
    print(f"State synced rate over {n_episodes} episodes: {n_synced/n_episodes:.2f}")
    print(f"State desynced rate over {np.sum(episode_lengths)} steps: {(n_episodes - n_synced)/np.sum(episode_lengths):.2f}")
    print(f"Success rate over {n_episodes} episodes: {n_success/n_episodes:.2f}")
    print("Max episode length:", max(episode_lengths) if episode_lengths else "N/A")
    