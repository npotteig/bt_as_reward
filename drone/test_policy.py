import argparse
import math
import json
import copy
import time
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
    minigrid_state_mission_to_z3
)
import bt_as_reward.envs  # noqa: F401

gym.register_envs(bt_as_reward.envs)

AIRSIM_FORWARD = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}

def write_variable_time_video(path, images, durations, fps=30):

    if len(images) != len(durations):
        raise ValueError("images and durations must have the same length")

    expanded_frames = []
    for img, dt in zip(images, durations):
        # Number of video frames the step should last
        count = max(1, math.ceil(dt * fps))
        expanded_frames.extend([img] * count)

    # Write the video
    imageio.mimwrite(path, expanded_frames, fps=fps)

def airsim_step(action, client: airsim.MultirotorClient, env, num_actions=1, direction=0):
    agent_dir_mapping = [3, 0, 1, 2]  
    match action:
        case 0 | 1: # left, right
            angle = (direction % 4) * 90
            client.moveByVelocityZAsync(0, 0, -5, 2, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=angle)).join()
            kinematics = client.simGetGroundTruthKinematics()
            ori = kinematics.orientation
            current_pos = kinematics.position
            q = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])
            yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]*q[2] + q[3]*q[3]))
            
            yaw_degrees = math.degrees(yaw) % 360
            agent_dir = agent_dir_mapping[round(yaw_degrees / 90) % 4]
            agent_position = (round(current_pos.y_val - 0.5) + 12, -round(current_pos.x_val - 0.5) + 12)
            return agent_position, agent_dir
            # if abs(yaw_degrees - angle) <= 5 or (angle == 0 and yaw_degrees >= 355):
            #     return True
            # else:
            #     print(f"Desync detected: Drone yaw {yaw_degrees}, Env yaw {angle}")
            #     return False
        case 2: # forward
            pos_delta = AIRSIM_FORWARD[direction % 4]
            kinematics = client.simGetGroundTruthKinematics()
            current_pos = kinematics.position
            # print("Num forward actions:", num_actions)
            target_x = round(current_pos.x_val - 0.5) + num_actions*pos_delta[0] + 0.5
            target_y = round(current_pos.y_val - 0.5) + num_actions*pos_delta[1] + 0.5
            
            client.moveToPositionAsync(target_x, target_y, -8, 1, timeout_sec=60).join()
            client.moveByVelocityZAsync(0, 0, -5, 0.8).join()
            next_kinematics = client.simGetGroundTruthKinematics()
            next_pos = next_kinematics.position
            agent_position = (round(next_pos.y_val - 0.5) + 12, -round(next_pos.x_val - 0.5) + 12)
            ori = next_kinematics.orientation
            q = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])
            yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]*q[2] + q[3]*q[3]))
            
            yaw_degrees = math.degrees(yaw) % 360
            agent_dir = agent_dir_mapping[round(yaw_degrees / 90) % 4]
            return agent_position, agent_dir
            # agent_position = env.unwrapped.agent_pos
            # if (round(next_pos.x_val - 0.5) + 0.5, round(next_pos.y_val - 0.5) + 0.5) == (-(agent_position[1] - 12) + 0.5, agent_position[0] - 11.5):
            #     return True
            # else:
            #     print("Current position:", (current_pos.x_val), (current_pos.y_val))
            #     print("Target position:", (target_x, target_y))
            #     # Env Position Rotated By 90 Degrees CCW
            #     print(f"Desync detected: Drone at ({round(next_pos.x_val - 0.5) + 0.5}, {round(next_pos.y_val - 0.5) + 0.5}), Env at ({-(agent_position[1]-12)+0.5}, {agent_position[0]-11.5})")
            #     print(env.unwrapped.pprint_grid())
            #     return False
        case _:
            next_kinematics = client.simGetGroundTruthKinematics()
            next_pos = next_kinematics.position
            agent_position = (round(next_pos.y_val - 0.5) + 12, -round(next_pos.x_val - 0.5) + 12)
            ori = next_kinematics.orientation
            q = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])
            yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]*q[2] + q[3]*q[3]))
            
            yaw_degrees = math.degrees(yaw) % 360
            agent_dir = agent_dir_mapping[round(yaw_degrees / 90) % 4]
            return agent_position, agent_dir  # Interact actions
    return None, None
    
# Environment Drone starts at (0.5, 0.5) facing East (yaw 90 degrees)
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
        state_mission_to_z3=minigrid_state_mission_to_z3
    )
    env = DictObservationSpaceWrapper(env, max_words_in_mission=17)
    env.reset(seed=0)
    env.reset()
    
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
    n_episodes = 100
    episode_lengths = []
    for episode in tqdm(range(n_episodes)):
        if use_airsim:
            client.reset()
            client.enableApiControl(True)
            client.armDisarm(True)
            client.takeoffAsync().join()
            # Move to 5 meters altitude and face East (yaw 90 degrees)
            client.moveToPositionAsync(0.5, 0.5, -5, velocity=2, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=90)).join()

        obs, info = env.reset()
        frames = [env.unwrapped.get_frame()]
        time.sleep(1.0)
        durations = [1.0]
        step = 0
        num_forward_actions = 0
        agent_position = (np.int16(12), np.int16(12))
        agent_dir = 0
        # print(env.unwrapped.pprint_grid())
        while not (terminated or truncated):
            start_time = time.time()
            action_masks = env.env.action_masks()
            
            if args.action_mask_file != "none":
                del obs["mission"][8]
                action, _ = model.predict(
                    obs,
                    action_masks=action_masks
                )
            else:
                del obs["mission"][8]
                action, _ = model.predict(
                    obs
                )
            # frames.append(env.unwrapped.get_frame())

            print(f"Step {step}")
            # print(f"Action taken: {action}")
            if use_airsim:
                if action == 2:  # forward
                    # Assume forward actions are deterministic until simulated when agent policy is considering next action
                    prev_position = env.unwrapped.agent_pos
                    obs, reward, terminated, truncated, info = env.step(action)
                    after_position = env.unwrapped.agent_pos
                    # Prevents drone from moving onto MiniGrid objects (walls, doors, keys, etc.)
                    if prev_position[0] != after_position[0] or prev_position[1] != after_position[1]:
                        num_forward_actions += 1

                    time.sleep(0.1)
                    durations.append(0.1)
                else:
                    if num_forward_actions > 0:
                        agent_position, agent_dir = airsim_step(2, client, env, num_actions=num_forward_actions, direction=agent_dir+1)

                    match action:
                        case 0:  # left
                            agent_dir = (agent_dir + 3) % 4
                            agent_position, agent_dir = airsim_step(action, client, env, direction=agent_dir+1)
                            env.unwrapped.agent_pos = agent_position
                            env.unwrapped.agent_dir = agent_dir
                            obs, reward, terminated, truncated, info = env.step(6) # No-op
                        case 1:  # right
                            agent_dir = (agent_dir + 1) % 4
                            agent_position, agent_dir = airsim_step(action, client, env, direction=agent_dir+1)
                            env.unwrapped.agent_pos = agent_position
                            env.unwrapped.agent_dir = agent_dir
                            obs, reward, terminated, truncated, info = env.step(6) # No-op
                        case _:
                            time.sleep(0.1)
                            agent_position, agent_dir = airsim_step(action, client, env, direction=agent_dir+1)
                            env.unwrapped.agent_pos = agent_position
                            env.unwrapped.agent_dir = agent_dir
                            obs, reward, terminated, truncated, info = env.step(action)
                    

                    num_forward_actions = 0
                    durations.append(time.time() - start_time)
                # drone_pos = client.simGetGroundTruthKinematics().position
                
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                durations.append(time.time() - start_time)
            step += 1
            
        if info.get("success", False):
            n_success += 1
        episode_lengths.append(step)
        terminated, truncated = False, False
    # save video using imageio
    # if n_episodes == 1:
    #     write_variable_time_video("test_video.mp4", frames, durations, fps=30)
    print(f"Success rate over {n_episodes} episodes: {n_success/n_episodes:.2f}")
    print("Max episode length (successful episodes):", max(episode_lengths) if episode_lengths else "N/A")
    