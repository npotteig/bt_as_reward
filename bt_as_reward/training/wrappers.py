from typing import Callable, Optional, Any, Tuple, Dict, List
import gymnasium as gym
from gymnasium.core import ObsType
import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, DIR_TO_VEC
import z3

from bt_as_reward.rewards.bt import BehaviourTreeReward, BehaviourTreeConfig


class MinigridSubtaskWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        mode: str = "environment",
        drop_key: bool = False,
        drop_prob: float = 0.05,
        bt_config: Optional[BehaviourTreeConfig] = None,
        action_mask_dict: Optional[Dict[str, List[int]]] = None,
        state_mission_to_z3: Optional[Callable[[np.ndarray, str], Tuple[Tuple[z3.ExprRef, ...], Tuple[z3.ExprRef, ...]]]] = None,
    ):
        super().__init__(env)
        self.mode = mode
        self.drop_key = drop_key
        self.drop_prob = drop_prob
        self.action_mask_dict = action_mask_dict
        self.state_mission_to_z3 = state_mission_to_z3

        match self.mode:
            case "environment":
                self.bt_config = None
            case "proc_as_reward":
                assert bt_config is not None, (
                    "BehaviourTreeConfig must be provided for proc_as_reward mode"
                )
                assert bt_config.use_memory, (
                    "Memory must be enabled for proc_as_reward mode"
                )
                self.bt_config = bt_config
            case "bt_as_reward":
                assert bt_config is not None, (
                    "BehaviourTreeConfig must be provided for bt_as_reward mode"
                )
                self.bt_config = bt_config
            case _:
                raise ValueError(f"Invalid mode: {self.mode}")

    def _full_observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        return {**obs, "image": full_grid}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if self.bt_config is not None:
            mission_str = obs["mission"]
            self.bt = BehaviourTreeReward.create_bt(
                mission_str=mission_str,
                bt_config=self.bt_config,
                action_masks=self.action_mask_dict,
                state_mission_to_z3=self.state_mission_to_z3,
            )
            full_obs = self._full_observation(obs)
            _, self.action_mask = self.bt.step_reward(full_obs["image"], mission_str)
        info["success"] = False
        return obs, info

    def action_masks(self) -> np.ndarray:
        if self.action_mask_dict is not None and self.action_mask is not None:
            return self.action_mask
        return np.ones(self.env.action_space.n, dtype=np.int8)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = 1 if reward > 0 else 0
        info["success"] = True if reward > 0 else False

        # Drop key if the agent is carrying it and the drop probability is met
        # Don't drop key if action is toggle to avoid race conditions with door opening/closing and key dropping
        if (
            action != 5
            and self.drop_key
            and self.env.unwrapped.carrying
            and self.env.unwrapped._rand_float(0, 1) < self.drop_prob
        ):
            agent_pos = self.env.unwrapped.agent_pos
            adj_pos = [agent_pos + DIR_TO_VEC[i] for i in range(4)]
            for pos in adj_pos:
                if not self.env.unwrapped.grid.get(*pos):
                    self.env.unwrapped.grid.set(
                        pos[0], pos[1], self.env.unwrapped.carrying
                    )
                    self.env.unwrapped.carrying.cur_pos = pos
                    self.env.unwrapped.carrying = None
                    # print("Dropped key at position:", pos)
                    break
            obs = self.env.unwrapped.gen_obs()

        if self.bt_config is not None:
            full_obs = self._full_observation(obs)
            r, self.action_mask = self.bt.step_reward(
                full_obs["image"], full_obs["mission"]
            )
            reward += r
        return obs, reward, terminated, truncated, info


class MuJoCoSubtaskWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        mode: str = "environment",
        grip_fail: bool = False,
        grip_fail_prob: float = 0.05,
        bt_config: Optional[BehaviourTreeConfig] = None,
        action_mask_dict: Optional[Dict[str, List[int]]] = None,
        state_mission_to_z3: Optional[Callable[[np.ndarray, str], Tuple[Tuple[z3.ExprRef, ...], Tuple[z3.ExprRef, ...]]]] = None,
    ):
        super().__init__(env)
        self.mode = mode
        self.action_mask_dict = action_mask_dict
        self.grip_fail = grip_fail
        self.grip_fail_prob = grip_fail_prob
        self.state_mission_to_z3 = state_mission_to_z3

        match self.mode:
            case "environment":
                self.bt_config = None
            case "proc_as_reward":
                assert bt_config is not None, (
                    "BehaviourTreeConfig must be provided for proc_as_reward mode"
                )
                assert bt_config.use_memory, (
                    "Memory must be enabled for proc_as_reward mode"
                )
                self.bt_config = bt_config
            case "bt_as_reward":
                assert bt_config is not None, (
                    "BehaviourTreeConfig must be provided for bt_as_reward mode"
                )
                self.bt_config = bt_config
            case _:
                raise ValueError(f"Invalid mode: {self.mode}")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if self.bt_config is not None:
            mission_str = obs["mission"]
            self.bt = BehaviourTreeReward.create_bt(
                mission_str=mission_str,
                bt_config=self.bt_config,
                action_masks=self.action_mask_dict,
                state_mission_to_z3=self.state_mission_to_z3,
            )
            _, self.action_mask = self.bt.step_reward(obs, mission_str)
        info["success"] = False
        return obs, info

    def action_masks(self) -> np.ndarray:
        if self.action_mask_dict is not None and self.action_mask is not None:
            return self.action_mask
        return np.ones(self.env.action_space.n, dtype=np.int8)

    def step(self, action):
        if (
            self.grip_fail
            and not self.env.unwrapped.open_gripper
            and self.env.unwrapped._rand_float(0, 1) < self.grip_fail_prob
        ):
            action = 6  # Open gripper action

        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = 0 if reward < 0 else 1
        terminated = reward > 0
        info["success"] = info["is_success"]

        if self.bt_config is not None:
            r, self.action_mask = self.bt.step_reward(obs, obs["mission"])
            reward += r
        return obs, reward, terminated, truncated, info


class MuJoCoDictObservationSpaceWrapper(gym.ObservationWrapper):
    def __init__(self, env, max_words_in_mission=50, word_dict=None):
        """
        max_words_in_mission is the length of the array to represent a mission, value 0 for missing words
        word_dict is a dictionary of words to use (keys=words, values=indices from 1 to < max_words_in_mission),
                  if None, use the MuJoCo language
        """
        super().__init__(env)

        if word_dict is None:
            word_dict = self.get_mujoco_words()

        self.max_words_in_mission = max_words_in_mission
        self.word_dict = word_dict

        self.observation_space = gym.spaces.Dict(
            {
                **self.env.observation_space.spaces,
                "mission": gym.spaces.MultiDiscrete(
                    [len(self.word_dict.keys()) + 1] * max_words_in_mission
                ),
            }
        )

    @staticmethod
    def get_mujoco_words():
        colors = ["green", "yellow", "lightblue", "magenta", "darkblue", "red"]
        objects = [
            "block",
            "target",
            "agent",
        ]

        verbs = ["pick", "move"]

        extra_words = ["the", "to", "location", "and", "up"]

        all_words = colors + objects + verbs + extra_words
        assert len(all_words) == len(set(all_words))
        return {word: i for i, word in enumerate(all_words)}

    def string_to_indices(self, string, offset=1):
        """
        Convert a string to a list of indices.
        """
        indices = []
        # adding space before and after commas
        string = string.replace(",", " , ")
        for word in string.split():
            if word in self.word_dict.keys():
                indices.append(self.word_dict[word] + offset)
            else:
                raise ValueError(f"Unknown word: {word}")
        return indices

    def observation(self, obs):
        obs["mission"] = self.string_to_indices(obs["mission"])
        assert len(obs["mission"]) < self.max_words_in_mission
        obs["mission"] += [0] * (self.max_words_in_mission - len(obs["mission"]))

        return obs
