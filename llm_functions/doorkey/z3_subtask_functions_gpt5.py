from typing import Tuple
import z3

def subtask_1_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    # Unpack state and mission tuples
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # The mission specifies the color index of the key to pick up: mission_keys[0]
    # Completion condition per Z3 state encoding:
    # "key_x, key_y -1 if not present or picked up"
    # We check that for the mission key color, both key_x and key_y are -1.
    # Since key_x_colors/key_y_colors are Python lists (not Z3 arrays), we construct
    # a disjunction over all colors that matches the mission key color index.
    return z3.Or(
        z3.And(mission_keys[0] == 0, key_x_colors[0] == -1, key_y_colors[0] == -1),
        z3.And(mission_keys[0] == 1, key_x_colors[1] == -1, key_y_colors[1] == -1),
        z3.And(mission_keys[0] == 2, key_x_colors[2] == -1, key_y_colors[2] == -1),
        z3.And(mission_keys[0] == 3, key_x_colors[3] == -1, key_y_colors[3] == -1),
        z3.And(mission_keys[0] == 4, key_x_colors[4] == -1, key_y_colors[4] == -1),
        z3.And(mission_keys[0] == 5, key_x_colors[5] == -1, key_y_colors[5] == -1),
    )

from typing import Tuple
import z3

def subtask_1_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    def adjacent_to_key_color(idx: int) -> z3.ExprRef:
        kx = key_x_colors[idx]
        ky = key_y_colors[idx]
        return z3.And(
            kx != -1,
            ky != -1,
            z3.Or(
                z3.And(x == kx, y == ky + 1),
                z3.And(x == kx, y == ky - 1),
                z3.And(y == ky, x == kx + 1),
                z3.And(y == ky, x == kx - 1),
            ),
        )

    return z3.Or(
        z3.And(mission_keys[0] == 0, adjacent_to_key_color(0)),
        z3.And(mission_keys[0] == 1, adjacent_to_key_color(1)),
        z3.And(mission_keys[0] == 2, adjacent_to_key_color(2)),
        z3.And(mission_keys[0] == 3, adjacent_to_key_color(3)),
        z3.And(mission_keys[0] == 4, adjacent_to_key_color(4)),
        z3.And(mission_keys[0] == 5, adjacent_to_key_color(5)),
    )

from typing import Tuple
import z3

def subtask_2_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    def door_open_or_occluded(idx: int) -> z3.ExprRef:
        ds = door_state_colors[idx]
        # Persistence fix: treat occluded (-1) as completed after open,
        # since occlusion happens when the agent stands on the door.
        # See: "door_state are -1 if occluded or not present. Occlusion is when the agent is on top of the door"
        # [Z3 State Encoding, Paragraph 9–11]
        return z3.Or(ds == 0, ds == -1)

    return z3.Or(
        z3.And(mission_doors[0] == 0, door_open_or_occluded(0)),
        z3.And(mission_doors[0] == 1, door_open_or_occluded(1)),
        z3.And(mission_doors[0] == 2, door_open_or_occluded(2)),
        z3.And(mission_doors[0] == 3, door_open_or_occluded(3)),
        z3.And(mission_doors[0] == 4, door_open_or_occluded(4)),
        z3.And(mission_doors[0] == 5, door_open_or_occluded(5)),
    )
    
from typing import Tuple
import z3

def subtask_2_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    def adjacent_to_door_color(idx: int) -> z3.ExprRef:
        dx = door_x_colors[idx]
        dy = door_y_colors[idx]
        # Ensure the door is present and not occluded (-1 indicates occluded or not present)
        return z3.And(
            dx != -1,
            dy != -1,
            z3.Or(
                z3.And(x == dx, y == dy + 1),
                z3.And(x == dx, y == dy - 1),
                z3.And(y == dy, x == dx + 1),
                z3.And(y == dy, x == dx - 1),
            ),
        )

    return z3.Or(
        z3.And(mission_doors[0] == 0, adjacent_to_door_color(0)),
        z3.And(mission_doors[0] == 1, adjacent_to_door_color(1)),
        z3.And(mission_doors[0] == 2, adjacent_to_door_color(2)),
        z3.And(mission_doors[0] == 3, adjacent_to_door_color(3)),
        z3.And(mission_doors[0] == 4, adjacent_to_door_color(4)),
        z3.And(mission_doors[0] == 5, adjacent_to_door_color(5)),
    )

from typing import Tuple
import z3

def subtask_3_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    # Completion condition leverages occlusion semantics:
    # "goal_x, goal_y are -1 if occluded or not present
    #  Occlusion is when the agent is on top of the goal"
    # [Z3 State Encoding › Python block › goal variables]
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Subtask 3 (reach the green goal square) is complete when the goal is occluded,
    # which happens exactly when the agent is standing on the goal tile.
    return z3.And(goal_x == -1, goal_y == -1)

from typing import Tuple
import z3

def subtask_3_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Goal is present and not occluded (-1 indicates occluded or not present),
    # and the agent is in one of the four adjacent cells.
    return z3.And(
        goal_x != -1,
        goal_y != -1,
        z3.Or(
            z3.And(x == goal_x, y == goal_y + 1),
            z3.And(x == goal_x, y == goal_y - 1),
            z3.And(y == goal_y, x == goal_x + 1),
            z3.And(y == goal_y, x == goal_x - 1),
        ),
    )