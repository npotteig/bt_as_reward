"""
Environment derived from
https://github.com/CDMCH/gym-fetch-stack
https://github.com/Farama-Foundation/Gymnasium-Robotics
"""

import os
from typing import Optional

import numpy as np

from minigrid.core.mission import MissionSpace
from gymnasium import spaces
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv, BaseRobotEnv
from gymnasium_robotics.utils import rotations

from bt_as_reward.constants import MUJOCO_IDX_TO_COLOR

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}


class MujocoFetchPickAndPlaceEnv(MujocoRobotEnv):
    def __init__(
        self,
        num_blocks: int = 1,
        num_goals: int = 1,
        n_substeps: int = 20,
        gripper_extra_height: float = 0.2,
        target_in_the_air: bool = True,
        target_offset: float = 0.0,
        obj_range: float = 0.15,
        target_range: float = 0.15,
        distance_threshold: float = 0.05,
        reward_type: str = "sparse",
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        randomize_stack: bool = False,
        **kwargs,
    ):
        self.num_blocks = num_blocks
        self.gripper_extra_height = gripper_extra_height
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        model_path = os.path.join(
            os.path.dirname(__file__), "assets", "fetch", f"stack{num_blocks}.xml"
        )

        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        for i in range(num_blocks):
            initial_qpos[f"object{i}:joint"] = [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0]

        self.object_names = ["object{}".format(i) for i in range(self.num_blocks)]
        self.goal_index = 0
        self.n_goals = num_goals
        self.randomize_stack = randomize_stack
        mission_space = MissionSpace(
            self._gen_new_mission,
            ordered_placeholders=[list(MUJOCO_IDX_TO_COLOR.values())[: self.n_goals]],
        )
        self.mission = self._gen_new_mission(MUJOCO_IDX_TO_COLOR[self.goal_index])
        super().__init__(
            n_actions=4,
            model_path=model_path,
            default_camera_config=default_camera_config,
            n_substeps=n_substeps,
            initial_qpos=initial_qpos,
            **kwargs,
        )
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "mission": mission_space}
        )
        self.observation_space.spaces["desired_goal"] = spaces.Box(
            -np.inf, np.inf, shape=(3 * self.n_goals,), dtype="float64"
        )

    @staticmethod
    def _gen_new_mission(color: str):
        return f"pick up and move the block to the {color} target location"

    def _rand_float(self, low: float, high: float) -> float:
        """
        Generate random float in [low,high[
        """
        return self.np_random.uniform(low, high)

    def step(self, action):
        assert self.action_space.contains(action), f"{action} invalid"
        mj_action = np.zeros(4)
        match action:
            case 0:
                mj_action[0] = 0.5  # move right
            case 1:
                mj_action[0] = -0.5  # move left
            case 2:
                mj_action[1] = 0.5  # move forward
            case 3:
                mj_action[1] = -0.5  # move backward
            case 4:
                mj_action[2] = 0.5  # move up
            case 5:
                mj_action[2] = -0.5  # move down
            case 6:
                self.open_gripper = True  # open gripper
            case 7:
                self.open_gripper = False  # close gripper
            case _:
                raise ValueError("Invalid action")
        if self.open_gripper:
            mj_action[3] = 0.5  # open
        else:
            mj_action[3] = -0.5  # close

        self._set_action(mj_action)

        self._mujoco_step(mj_action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()

        info = {
            "is_success": self._is_success(
                obs["achieved_goal"],
                self.goal[3 * self.goal_index : 3 * (self.goal_index + 1)],
            ),
        }

        terminated = self.compute_terminated(
            obs["achieved_goal"],
            self.goal[3 * self.goal_index : 3 * (self.goal_index + 1)],
            info,
        )
        truncated = self.compute_truncated(
            obs["achieved_goal"],
            self.goal[3 * self.goal_index : 3 * (self.goal_index + 1)],
            info,
        )

        reward = self.compute_reward(
            obs["achieved_goal"],
            self.goal[3 * self.goal_index : 3 * (self.goal_index + 1)],
            info,
        )

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super(BaseRobotEnv, self).reset(seed=seed)
        did_reset_sim = False
        if self.randomize_stack:
            self.object_indices = self.np_random.permutation(self.num_blocks)
        self.goal_index = self.np_random.integers(0, self.n_goals)
        self.mission = self._gen_new_mission(MUJOCO_IDX_TO_COLOR[self.goal_index])
        self.goal = self._sample_goal().copy()
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        obs = self._get_obs()
        self.open_gripper = True
        if self.render_mode == "human":
            self.render()

        return obs, {}

    # GoalEnv methods
    # ----------------------------

    def subgoal_goal_distances(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        for i in range(self.num_blocks):
            assert (
                goal_a[..., 3 * i : 3 * (i + 1)].shape
                == goal_b[..., 3 * i : 3 * (i + 1)].shape
            )
        return np.array(
            [
                np.linalg.norm(
                    goal_a[..., 3 * i : 3 * (i + 1)] - goal_b[..., 3 * i : 3 * (i + 1)],
                    axis=-1,
                )
                for i in range(self.num_blocks)
            ]
        )

    def gripper_pos_far_from_goals(self, goal):
        distances = [
            np.linalg.norm(self.grip_pos - goal[i * 3 : (i + 1) * 3], axis=-1)
            for i in range(self.num_blocks)
        ]
        return np.all([d > self.distance_threshold * 2 for d in distances], axis=0)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        distances = self.subgoal_goal_distances(achieved_goal, goal)
        if self.reward_type == "sparse":
            reward = np.min(
                [-(d > self.distance_threshold).astype(np.float32) for d in distances],
                axis=0,
            )
            reward = np.asarray(reward)
            # np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(goal))
            return np.float64(reward)
        else:
            if (
                min(
                    [
                        -(d > self.distance_threshold).astype(np.float32)
                        for d in distances
                    ]
                )
                == 0
            ):
                return 0
            return np.float64(-np.sum(distances))

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)

    def _get_obs(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()

        achieved_goal = np.squeeze(object_pos.flatten())
        self.grip_pos = grip_pos.copy()

        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
            "mission": self.mission,
        }

    def _sample_goal(self):
        goals = []
        for _ in range(self.n_goals):
            goal0 = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal0 += self.target_offset
            goal0[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal0[2] += self.np_random.uniform(0, 0.45)
            goals.append(goal0)
        return np.concatenate(np.array(goals))

    def _is_success(self, achieved_goal, desired_goal):
        if np.abs(achieved_goal[0] - desired_goal[0]) <= 0.05 and np.abs(
            achieved_goal[1] - desired_goal[1]
        ) <= 0.05 and np.abs(achieved_goal[2] - desired_goal[2]) <= 0.05:
            return True
        return False
        # distances = self.subgoal_goal_distances(achieved_goal, desired_goal)
        # if (
        #     sum([-(d > self.distance_threshold).astype(np.float32) for d in distances])
        #     == 0
        # ):
        #     return True
        # else:
        #     return False

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )

        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        object_pos = np.array(
            [
                self._utils.get_site_xpos(self.model, self.data, "object{}".format(i))
                for i in range(self.num_blocks)
            ]
        )
        # rotations
        object_rot = np.array(
            [
                rotations.mat2euler(
                    self._utils.get_site_xmat(
                        self.model, self.data, "object{}".format(i)
                    )
                )
                for i in range(self.num_blocks)
            ]
        )
        # velocities
        object_velp = np.array(
            [
                self._utils.get_site_xvelp(self.model, self.data, "object{}".format(i))
                * dt
                for i in range(self.num_blocks)
            ]
        )
        object_velr = np.array(
            [
                self._utils.get_site_xvelr(self.model, self.data, "object{}".format(i))
                * dt
                for i in range(self.num_blocks)
            ]
        )
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        )

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["robot0:gripper_link"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        for i in range(self.n_goals):
            site_id = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target{}".format(i)
            )
            self.model.site_pos[site_id] = (
                self.goal[3 * i : 3 * (i + 1)] - sites_offset[0]
            )
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        # Reset buffers for joint states, actuators, warm-start, control buffers etc.
        self._mujoco.mj_resetData(self.model, self.data)

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object(s).
        # prev_x_positions = [self.goal[:2]]
        prev_x_positions = [self.goal[3 * i : 3 * i + 2] for i in range(self.n_goals)]
        for i, obj_name in enumerate(self.object_names):
            # object_qpos = self._utils.get_joint_qpos(
            #     self.model, self.data, f"{obj_name}:joint"
            # )
            object_qpos = np.array([0, 0, 0.425, 1, 0, 0, 0])
            assert object_qpos.shape == (7,)
            object_qpos[2] = 0.425
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(
                object_xpos - self.initial_gripper_xpos[:2]
            ) < 0.1 or np.any(
                [
                    np.linalg.norm(object_xpos - other_xpos) < 0.1
                    for other_xpos in prev_x_positions
                ]
            ):
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            object_qpos[:2] = object_xpos

            prev_x_positions.append(object_qpos[:2])
            self._utils.set_joint_qpos(
                self.model, self.data, f"{obj_name}:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", gripper_rotation
        )
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()
        self.height_offset = 0.425
