import argparse
import json
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO

from bt_as_reward.training.callbacks import RewardThresholdCallback
from bt_as_reward.training.models import MuJoCoCombinedExtractor
from bt_as_reward.training.wrappers import (
    MuJoCoSubtaskWrapper,
    MuJoCoDictObservationSpaceWrapper,
)
from bt_as_reward.rewards.bt import BehaviourTreeConfig
from bt_as_reward.utils import (
    load_functions_from_file,
    mujoco_create_distance_function,
    mujoco_object_to_str,
    mujoco_state_mission_to_z3
)
import bt_as_reward.envs

gym.register_envs(bt_as_reward.envs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO with GPT Subtasks")
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for the environment"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./results/checkpoints/",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--grip_fail",
        type=str,
        default="false",
        help="Whether to drop the key randomly in the environment",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./results/tensorboard/",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        help="Reward mode to use",
        choices=["environment", "proc_as_reward", "bt_as_reward"],
        default="environment",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1e8,
        help="Number of timesteps to train the model",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="FetchPickAndPlace-v5",
        help="Name of the MuJoCo environment",
    )
    parser.add_argument(
        "--max_mission_words",
        type=int,
        default=15,
        help="Maximum number of words in the mission",
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
        default=0.15,
        help="Distance threshold for proximity check.",
    )
    parser.add_argument(
        "--action_mask_file",
        type=str,
        default="none",
        help="Path to the action mask file.",
    )
    parser.add_argument(
        "--use_z3", action='store_true',
        help="Whether to use Z3 for subtask logic.",
    )
    args = parser.parse_args()
    grip_fail = args.grip_fail.lower() == "true"
    
    os.makedirs(args.checkpoint, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.action_mask_file != "none":
        with open(args.action_mask_file, "r") as f:
            action_mask_dict = json.load(f)

    object_functions = load_functions_from_file(
        args.object_function_file, args.object_function_names.strip().split(", ")
    )
    subtask_functions = load_functions_from_file(
        args.subtask_function_file, args.subtask_function_names.strip().split(", ")
    )

    policy_kwargs = dict(
        features_extractor_class=MuJoCoCombinedExtractor,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        optimizer_kwargs=dict(eps=1e-5),
    )

    bt_config = BehaviourTreeConfig(
        subtask_names=args.subtask_names.strip().split(", "),
        subtask_functions=subtask_functions,
        object_functions=object_functions,
        object_to_str=mujoco_object_to_str,
        create_distance_function=mujoco_create_distance_function,
        distance_threshold=args.distance_threshold,
        use_memory=False if args.reward_mode == "bt_as_reward" else True,
    )

    env = gym.make(args.env_name, max_episode_steps=100)
    env = MuJoCoSubtaskWrapper(
        env,
        mode=args.reward_mode,
        grip_fail=grip_fail,
        grip_fail_prob=0.05,
        action_mask_dict=action_mask_dict if args.action_mask_file != "none" else None,
        bt_config=bt_config,
        state_mission_to_z3=mujoco_state_mission_to_z3 if args.use_z3 else None,
    )
    env = MuJoCoDictObservationSpaceWrapper(
        env, max_words_in_mission=args.max_mission_words
    )
    obs, _ = env.reset(seed=args.seed)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.num_timesteps,
        save_path=args.checkpoint
        + f"{args.env_name}_{args.reward_mode}"
        + f"{'_mask' if args.action_mask_file != 'none' else ''}"
        + f"{'_uncertainty' if grip_fail else ''}"
        + "/",
        name_prefix="ppo_mlp",
    )

    if args.action_mask_file != "none":
        model_class = MaskablePPO
        ac_policy = MaskableMultiInputActorCriticPolicy
    else:
        model_class = PPO
        ac_policy = "MultiInputPolicy"

    model = model_class(
        ac_policy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=args.log_dir
        + f"{args.env_name}_{args.reward_mode}"
        + f"{'_mask' if args.action_mask_file != 'none' else ''}"
        + f"{'_uncertainty' if grip_fail else ''}"
        + "/",
        learning_rate=3e-4,
        batch_size=64,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    model.learn(
        args.num_timesteps,
        callback=CallbackList(
            [checkpoint_callback, RewardThresholdCallback(threshold=5e-3)]
        ),
    )
