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

## Unlock

The agent must open a box to pick up the key. The agent must then open a locked door. Once a box is opened, it's tile is replaced with the item it contains else none.
### Mission Space

“open the {box_color} box, pick up the key, then open the {door_color} door”

{door_color} and {box_color} can be “red”, “green”, “blue”, “purple”, “yellow” or “grey”.

### Action Space

|Num|Name|Action|
|---|---|---|
|0|left|Turn left|
|1|right|Turn right|
|2|forward|Move forward|
|3|pickup|Pick up an object|
|4|drop|Unused|
|5|toggle|Toggle/activate an object|
|6|done|Unused|

### State Encoding
- Each tile is encoded as a 3 dimensional tuple: `(OBJECT_IDX, COLOR_IDX, STATE)`
- `STATE` refers to the door state with 0=open, 1=closed and 2=locked
- The state input is a 3D array
- If the agent is on top of a tile then the agent is visible and the tile is invisible in the state

Type alias for a 3D array of shape (N, M, 3)
ArrayNxMx3 = Annotated[npt.NDArray[np.float64], (None, None, 3)]

```
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
```
