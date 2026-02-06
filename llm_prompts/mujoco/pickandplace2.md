You are a helpful assistant that decomposes a mission into concrete reactive subtasks that are encoded into a behavior tree that serves as the reward function for training an RL agent. Consider only subtasks where an object, excluding the agent, is interacted with. For each subtask, specify code to classify objects of interest and evaluate the success of the subtask. Your output will be checked to ensure every subtask has a well-defined reward, conditions are unambiguous, and the overall tree is logically consistent and executable. 

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

## Pick And Place 2

The task in the environment is for a manipulator to pick up and move a block to one of two target positions in mid-air. The robot is a 7-DoF [Fetch Mobile Manipulator](https://fetchrobotics.com/) with a two-fingered parallel gripper. The robot is controlled by small displacements of the gripper in Cartesian coordinates and the inverse kinematics are computed internally by the MuJoCo framework. The gripper can be opened or closed in order to perform the grasping operation of pick and move. The gripper should not be opened at the target, only moved to the target to complete the mission.

## Mission Space
"pick up and move the block to the {color} target location"

## Action Space

| Num | Name          | Action                  |
| --- | ------------- | ----------------------- |
| 0   | right         | Move +x (m)             |
| 1   | left          | Move -x (m)             |
| 2   | forward       | Move +y (m)             |
| 3   | backward      | Move -y (m)             |
| 4   | up            | Move +z (m)             |
| 5   | down          | Move -z (m)             |
| 6   | open gripper  | Toggle gripper to open  |
| 7   | close gripper | Toggle gripper to close |
## State Encoding
The state is a `goal-aware observation space`. It consists of a dictionary with information about the robot’s end effector state and goal. The kinematics observations are derived from Mujoco bodies known as [sites](https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=site#body-site) attached to the body of interest such as the block or the end effector. Only the observations from the gripper fingers are derived from joints. Also to take into account the temporal influence of the step time, velocity values are multiplied by the step time dt=number_of_sub_steps*sub_step_time. The dictionary consists of the following 3 keys:

- `observation`: its value is an `ndarray` of shape `(25,)`. It consists of kinematic information of the block object and gripper. The elements of the array correspond to the following:
    
|Num|Observation|Min|Max|Site Name (in corresponding XML file)|Joint Name (in corresponding XML file)|Joint Type|Unit|
|---|---|---|---|---|---|---|---|
|0|End effector x position in global coordinates|-Inf|Inf|robot0:grip|-|-|position (m)|
|1|End effector y position in global coordinates|-Inf|Inf|robot0:grip|-|-|position (m)|
|2|End effector z position in global coordinates|-Inf|Inf|robot0:grip|-|-|position (m)|
|3|Block x position in global coordinates|-Inf|Inf|object0|-|-|position (m)|
|4|Block y position in global coordinates|-Inf|Inf|object0|-|-|position (m)|
|5|Block z position in global coordinates|-Inf|Inf|object0|-|-|position (m)|
|6|Relative block x position with respect to gripper x position in global coordinates. Equals to xblock - xgripper|-Inf|Inf|object0|-|-|position (m)|
|7|Relative block y position with respect to gripper y position in global coordinates. Equals to yblock - ygripper|-Inf|Inf|object0|-|-|position (m)|
|8|Relative block z position with respect to gripper z position in global coordinates. Equals to zblock - zgripper|-Inf|Inf|object0|-|-|position (m)|
|9|Joint displacement of the right gripper finger|-Inf|Inf|-|robot0:r_gripper_finger_joint|hinge|position (m)|
|10|Joint displacement of the left gripper finger|-Inf|Inf|-|robot0:l_gripper_finger_joint|hinge|position (m)|
|11|Global x rotation of the block in a XYZ Euler frame rotation|-Inf|Inf|object0|-|-|angle (rad)|
|12|Global y rotation of the block in a XYZ Euler frame rotation|-Inf|Inf|object0|-|-|angle (rad)|
|13|Global z rotation of the block in a XYZ Euler frame rotation|-Inf|Inf|object0|-|-|angle (rad)|
|14|Relative block linear velocity in x direction with respect to the gripper|-Inf|Inf|object0|-|-|velocity (m/s)|
|15|Relative block linear velocity in y direction with respect to the gripper|-Inf|Inf|object0|-|-|velocity (m/s)|
|16|Relative block linear velocity in z direction|-Inf|Inf|object0|-|-|velocity (m/s)|
|17|Block angular velocity along the x axis|-Inf|Inf|object0|-|-|angular velocity (rad/s)|
|18|Block angular velocity along the y axis|-Inf|Inf|object0|-|-|angular velocity (rad/s)|
|19|Block angular velocity along the z axis|-Inf|Inf|object0|-|-|angular velocity (rad/s)|
|20|End effector linear velocity x direction|-Inf|Inf|robot0:grip|-|-|velocity (m/s)|
|21|End effector linear velocity y direction|-Inf|Inf|robot0:grip|-|-|velocity (m/s)|
|22|End effector linear velocity z direction|-Inf|Inf|robot0:grip|-|-|velocity (m/s)|
|23|Right gripper finger linear velocity|-Inf|Inf|-|robot0:r_gripper_finger_joint|hinge|velocity (m/s)|
|24|Left gripper finger linear velocity|-Inf|Inf|-|robot0:l_gripper_finger_joint|hinge|velocity (m/s)|

- `desired_goal`: this key represents the two of the final goals, only one of which is chosen as the actual final goal based on the mission. In this environment it is a 3-dimensional `ndarray`, `(3,)`, that consists of the three cartesian coordinates of the desired final block position `[x,y,z]`. In order for the robot to perform a pick and place trajectory, the goal position can be elevated over the table or on top of the table. The elements of the array are the following:
    
| Num | Observation                                          | Min  | Max | Site Name (in corresponding XML file) | Unit         |
| --- | ---------------------------------------------------- | ---- | --- | ------------------------------------- | ------------ |
| 0   | Green final goal block position in the x coordinate  | -Inf | Inf | target0                               | position (m) |
| 1   | Green final goal block position in the y coordinate  | -Inf | Inf | target0                               | position (m) |
| 2   | Green final goal block position in the z coordinate  | -Inf | Inf | target0                               | position (m) |
| 3   | Yellow final goal block position in the x coordinate | -Inf | Inf | target1                               | position (m) |
| 4   | Yellow final goal block position in the y coordinate | -Inf | Inf | target1                               | position (m) |
| 5   | Yellow final goal block position in the z coordinate | -Inf | Inf | target1                               | position (m) |

- `achieved_goal`: this key represents the current state of the block, as if it would have achieved a goal. The value is an `ndarray` with shape `(3,)`. The elements of the array are the following:
    
|Num|Observation|Min|Max|Site Name (in corresponding XML file)|Unit|
|---|---|---|---|---|---|
|0|Current block position in the x coordinate|-Inf|Inf|object0|position (m)|
|1|Current block position in the y coordinate|-Inf|Inf|object0|position (m)|
|2|Current block position in the z coordinate|-Inf|Inf|object0|position (m)|

```
COLOR_TO_IDX = {
"green": 0,
"yellow": 1,
"lightblue": 2,
"magenta": 3,
"darkblue": 4,
"red": 5,
}

OBJECT_TO_IDX = {
"block": 0,
"target": 1,
"agent": 2
}
```