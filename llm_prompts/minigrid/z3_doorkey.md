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

## DoorKey

### Description
This environment has a key that the agent must pick up in order to unlock a door and then get to the green goal square.
### Mission Space
“use the key to open the door and then get to the goal”

mission_keys, mission_doors, and mission_boxes refer to strings in the mission and their associated values.

```
# m_k_0 is always 4 (i.e. COLOR_TO_IDX["yellow"])
mission_keys = [m_k_0]
# m_d_0 is always 4 (i.e. COLOR_TO_IDX["yellow"])
mission_doors = [m_d_0]
```  

### Action Space

| Num | Name    | Action                    |
| --- | ------- | ------------------------- |
| 0   | left    | Turn left                 |
| 1   | right   | Turn right                |
| 2   | forward | Move forward              |
| 3   | pickup  | Pick up an object         |
| 4   | drop    | Unused                    |
| 5   | toggle  | Toggle/activate an object |
| 6   | done    | Unused                    |

### Z3 State Encoding

The agent cannot step on keys, boxes, or locked/closed doors.
  
```Python
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}

OBJECT_TO_IDX = {
"unseen": 0,
"empty": 1,
"wall": 2,
"floor": 3,
"door": 4,
"key": 5,
"ball": 6,
"box": 7,
"goal": 8,
"lava": 9,
"agent": 10,
}

STATE_TO_IDX = {
"open": 0,
"closed": 1,
"locked": 2,
}

# Symbolic State
x = z3.Int(f"x_{t}")
y = z3.Int(f"y_{t}")
# dir is 0-3 (East, South, West, North)
dir = z3.Int(f"dir_{t}")

# For every color of door, we can have a separate door variable
# door_x, door_y, and door_state are -1 if occluded or not present.
# Occlusion is when the agent is on top of the door
door_x_colors = [z3.Int(f"door_x_{color}") for color in COLOR_TO_IDX.keys()]
door_y_colors = [z3.Int(f"door_y_{color}") for color in COLOR_TO_IDX.keys()]
door_state_colors = [z3.Int(f"door_state_{color}_{t}") for color in COLOR_TO_IDX.keys()]

# key_x, key_y -1 if not present or picked up
key_x_colors = [z3.Int(f"key_x_{color}_{t}") for color in COLOR_TO_IDX.keys()]
key_y_colors = [z3.Int(f"key_y_{color}_{t}") for color in COLOR_TO_IDX.keys()]

# Boxes can contain the color of the key they hold else -1 for empty
# box_x, box_y -1 if not present or opened
# box_contains -1 if empty or color idx of key
box_x_colors = [z3.Int(f"box_x_{color}_{t}") for color in COLOR_TO_IDX.keys()]
box_y_colors = [z3.Int(f"box_y_{color}_{t}") for color in COLOR_TO_IDX.keys()]
box_contains = [z3.Int(f"box_contains_{color}") for color in COLOR_TO_IDX.keys()]

# goal_x, goal_y are -1 if occluded or not present
# Occlusion is when the agent is on top of the goal
goal_x = z3.Int(f"goal_x")
goal_y = z3.Int(f"goal_y")
```