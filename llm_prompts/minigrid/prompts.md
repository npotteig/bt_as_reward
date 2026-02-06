#### Prompt 1
Given the mission, provide the list of subtask names. Each subtask should have a different object. Please do not include traversal tasks unless it has to do with a target or goal.
Place the subtasks in a JSON markdown block as a list of strings.
```json
``` 

#### Prompt 2
For subtask 1, implement a function that inputs the symbolic state and symbolic mission and returns a z3 expression to check if subtask 1 has been completed. If you need to use distance, consider using a simpler function to check for proximity than manhattan/euclidean distance as this raises verification time significantly. Do not include actual coordinate values in the function to remain agnostic to environment size. Do not introduce new free symbolic variables.
Place the function code in a Python markdown block. 
```Python 
from typing import Tuple
import z3
def subtask_1_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
	x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
```

#### Prompt 3
For subtask 1, implement a function that inputs the symbolic state and symbolic mission and returns a z3 expression to check if the agent is adjacent/near the object of interest for subtask 1. Do not include actual coordinate values in the function to remain agnostic to environment size. Consider using a simpler function to check for proximity than manhattan/euclidean distance as this raises verification time significantly. Do not introduce new free symbolic variables. Remember this should be a comparison between a new object and the agent, not one that was compared against in previous subtasks. Place the function code in a Python markdown block. 
```Python 
from typing import Tuple
import z3
def subtask_1_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
	x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
```

#### Prompt 4
Below is a set of action masks (1 if action is available, 0 otherwise) for each subtask. If there are more actions available for specific subtask(s) based on your knowledge of the environment and mission, then update the dictionary accordingly. GoTo is the subtask where an agent must travel to be adjacent/near the subtask object where the agent can be in any pose or orientation. Once, adjacent/near the object, interact is when the agent must orient itself and perform an interaction, manipulation, or traversal onto the subtask object. Think about only the necessary actions to complete each subtask (i.e. don't add traversal if the subtask is already complete). Provide a concise explanation of your updates. Output the updated action mask dictionary in a JSON markdown block.