#### Prompt 1
Given the mission, provide the list of subtask names. Each subtask should have a different object. Please do not include traversal tasks unless it has to do with a target or goal.
Place the subtasks in a JSON markdown block as a list of strings.
```json
``` 

#### Prompt 2
For subtask 1, implement a function that inputs the symbolic state and symbolic mission and returns a z3 expression to check if subtask 1 has been completed. Do not include actual coordinate values in the function to remain agnostic to environment size. If you need to use distance, consider using a simpler function to check for proximity than manhattan/euclidean distance as this raises verification time significantly. Please do not use equality to check for Reals as in the simulator this is often never True. Use an approximate range instead. Do not introduce new free symbolic variables.  Place the function code in a Python markdown block. 
```Python 
from typing import Tuple
import z3
def subtask_1_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
	gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
	mission_goals = mission
```

#### Prompt 3
For subtask 1, implement a function that inputs the symbolic state and symbolic mission and returns a z3 expression to check if the agent is adjacent/near the object of interest for subtask 1. Check proximity only in the xy-plane. Do not include actual coordinate values in the function to remain agnostic to environment size. Consider using a simpler function to check for proximity than manhattan/euclidean distance as this raises verification time significantly. Please do not use equality to check for Reals as in the simulator this is often never True. Use an approximate range instead. Do not introduce new free symbolic variables. Remember this should be a comparison between a new object and the agent, not one that was compared against in previous subtasks. Place the function code in a Python markdown block. 
```Python 
from typing import Tuple
import z3
def subtask_1_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
	gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
	mission_goals = mission
```

#### Prompt 4
Below is a set of action masks (1 if action is available, 0 otherwise) for each subtask. If there are more actions available for specific subtask(s) based on your knowledge of the environment and mission, then update the dictionary accordingly. GoTo is the subtask where an agent must travel to be adjacent/near the subtask object where the agent can be in any pose or orientation. Once, adjacent/near the object, interact is when the agent must orient itself and perform an interaction, manipulation, or traversal onto the subtask object. Think about only the necessary actions to complete each subtask (i.e. don't add traversal if the subtask is already complete). Consider enabling traversal actions or other interaction actions if there is a possibility in the real simulator performing an interaction is not guaranteed to be successful and may require small displacements. Provide a concise explanation of your updates. Output the updated action mask dictionary in a JSON markdown block.