from typing import Tuple
import z3

def subtask_1_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Check that the mission-specified key (color index) has been picked up,
    # which is encoded as its coordinates being -1
    mk0 = mission_keys[0]
    clauses = []
    for i in range(len(key_x_colors)):
        clauses.append(z3.And(
            mk0 == z3.IntVal(i),
            key_x_colors[i] == z3.IntVal(-1),
            key_y_colors[i] == z3.IntVal(-1)
        ))
    return z3.Or(*clauses)

from typing import Tuple
import z3

def subtask_1_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    mk0 = mission_keys[0]
    clauses = []
    for i in range(len(key_x_colors)):
        kx = key_x_colors[i]
        ky = key_y_colors[i]
        clauses.append(z3.And(
            mk0 == z3.IntVal(i),
            kx != z3.IntVal(-1),
            ky != z3.IntVal(-1),
            z3.Or(
                z3.And(kx == x + z3.IntVal(1), ky == y),
                z3.And(kx == x - z3.IntVal(1), ky == y),
                z3.And(ky == y + z3.IntVal(1), kx == x),
                z3.And(ky == y - z3.IntVal(1), kx == x),
            )
        ))
    return z3.Or(*clauses)

from typing import Tuple
import z3

def subtask_2_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    md0 = mission_doors[0]
    clauses = []
    for i in range(len(door_state_colors)):
        # Subtask 2 (Unlock the door): door is not locked (state != 2)
        clauses.append(z3.And(
            md0 == z3.IntVal(i),
            door_state_colors[i] != z3.IntVal(2)
        ))
    return z3.Or(*clauses)

from typing import Tuple
import z3

def subtask_2_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    md0 = mission_doors[0]
    clauses = []
    for i in range(len(door_x_colors)):
        dx = door_x_colors[i]
        dy = door_y_colors[i]
        dstate = door_state_colors[i]
        clauses.append(z3.And(
            md0 == z3.IntVal(i),
            z3.Or(
                # Door is visible (not occluded) and agent is adjacent (4-neighborhood)
                z3.And(
                    dx != z3.IntVal(-1),
                    dy != z3.IntVal(-1),
                    z3.Or(
                        z3.And(dx == x + z3.IntVal(1), dy == y),
                        z3.And(dx == x - z3.IntVal(1), dy == y),
                        z3.And(dy == y + z3.IntVal(1), dx == x),
                        z3.And(dy == y - z3.IntVal(1), dx == x),
                    )
                ),
                # Door is occluded because agent is on top of an open door (treated as "near")
                z3.And(
                    dx == z3.IntVal(-1),
                    dy == z3.IntVal(-1),
                    dstate == z3.IntVal(0)  # open
                )
            )
        ))
    return z3.Or(*clauses)

from typing import Tuple
import z3

def subtask_3_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Completion for reaching the goal: agent is on top of the goal,
    # which is encoded as the goal coordinates being -1 (occluded).
    return z3.And(goal_x == z3.IntVal(-1), goal_y == z3.IntVal(-1))

from typing import Tuple
import z3

def subtask_3_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    return z3.Or(
        # Goal is visible and agent is adjacent (4-neighborhood)
        z3.And(
            goal_x != z3.IntVal(-1),
            goal_y != z3.IntVal(-1),
            z3.Or(
                z3.And(goal_x == x + z3.IntVal(1), goal_y == y),
                z3.And(goal_x == x - z3.IntVal(1), goal_y == y),
                z3.And(goal_y == y + z3.IntVal(1), goal_x == x),
                z3.And(goal_y == y - z3.IntVal(1), goal_x == x),
            )
        ),
        # Agent is on top of the goal (occluded), treated as near
        z3.And(goal_x == z3.IntVal(-1), goal_y == z3.IntVal(-1))
    )

