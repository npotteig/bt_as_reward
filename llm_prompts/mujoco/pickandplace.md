You are a helpful assistant that decomposes a mission into concrete reactive subtasks that are encoded into a behavior tree that serves as the reward function for training an RL agent. Consider only subtasks where an object, excluding the agent, is interacted with. 

# BT Template
```
Sequence
- Subtask 1
- Subtask 2
- Subtask 3
...
- Subtask N
```
Each subtask is a subtree of the following format:
```
Selector
- Subtask complete?
- Sequence
	- GoTo Object!
	- Interact Object!
```

# Mission 

## Pick And Place

The task in the environment is for a manipulator to move and hold a block at a target position on top of a table or in mid-air. The robot is a 7-DoF [Fetch Mobile Manipulator](https://fetchrobotics.com/) with a two-fingered parallel gripper. The robot is controlled by small displacements of the gripper in Cartesian coordinates and the inverse kinematics are computed internally by the MuJoCo framework. The gripper can be opened or closed in order to perform the grasping operation of pick and place. The task is also continuing which means that the robot has to hold the block in the target position for an indefinite period of time. This implies that the gripper should not release the block when at the target.

## Mission Space
"pick up and move the block to the target location"

mission_goals refer to strings in the mission and their associated values.

```Python
mission_goals = [m_g_0] # m_g_0 is always 0 for green
```

## Action Space

| Num | Name          | Action                                                                                                                         |
| --- | ------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 0   | right         | Move +x (~0.015 m)                                                                                                             |
| 1   | left          | Move -x (~0.015 m)                                                                                                             |
| 2   | forward       | Move +y (~0.015 m)                                                                                                             |
| 3   | backward      | Move -y (~0.015 m)                                                                                                             |
| 4   | up            | Move +z (~0.015 m)                                                                                                             |
| 5   | down          | Move -z (~0.015 m)                                                                                                             |
| 6   | open gripper  | Toggle gripper to open <br>(g_disp max is ~0.05m when open is executed multiple times)                                         |
| 7   | close gripper | Toggle gripper to close<br>(g_disp is min ~0m when closed is executed multiple times<br>or ~0.025m when closed around a block) |
## Z3 State Encoding

blocks are 0.05m x 0.05m x 0.05m
```Python
COLOR_TO_IDX = {
"green": 0,
"yellow": 1,
}

OBJECT_TO_IDX = {
"block": 0,
"target": 1,
"agent": 2
}

gx = z3.Real(f'gripper_x_{t}')
gy = z3.Real(f'gripper_y_{t}')
gz = z3.Real(f'gripper_z_{t}')
g_disp = z3.Real(f'gripper_disp_{t}')
bx = z3.Real(f'block_x_{t}')
by = z3.Real(f'block_y_{t}')
bz = z3.Real(f'block_z_{t}')

goal_x = [z3.Real(f'goal_x_{color}') for color in COLOR_TO_IDX.keys()]
goal_y = [z3.Real(f'goal_y_{color}') for color in COLOR_TO_IDX.keys()]
goal_z = [z3.Real(f'goal_z_{color}') for color in COLOR_TO_IDX.keys()]

gvel_x = z3.Real(f'gripper_vel_x_{t}')
gvel_y = z3.Real(f'gripper_vel_y_{t}')
gvel_z = z3.Real(f'gripper_vel_z_{t}')
bvel_x = z3.Real(f'block_vel_x_{t}')
bvel_y = z3.Real(f'block_vel_y_{t}')
bvel_z = z3.Real(f'block_vel_z_{t}')
```