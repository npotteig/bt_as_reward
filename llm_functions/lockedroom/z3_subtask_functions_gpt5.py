from typing import Tuple
import z3

def subtask_1_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Helper to select an element from a fixed-size array of length 6 using a symbolic color index (0..5)
    def select_by_color(arr, idx):
        return z3.If(idx == 0, arr[0],
            z3.If(idx == 1, arr[1],
            z3.If(idx == 2, arr[2],
            z3.If(idx == 3, arr[3],
            z3.If(idx == 4, arr[4],
            arr[5])))))

    # Subtask 1: Open the first mission door (m_d_0). Treat occlusion (-1s) as complete to avoid false reversions.
    md0_color_idx = mission_doors[0]
    md0_state = select_by_color(door_state_colors, md0_color_idx)
    md0_x = select_by_color(door_x_colors, md0_color_idx)
    md0_y = select_by_color(door_y_colors, md0_color_idx)

    is_open = (md0_state == z3.IntVal(0))
    occluded = z3.And(md0_x == z3.IntVal(-1), md0_y == z3.IntVal(-1), md0_state == z3.IntVal(-1))

    return z3.Or(is_open, occluded)

def subtask_1_complete_failure(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Helper to select an element from a fixed-size array of length 6 using a symbolic color index (0..5)
    def select_by_color(arr, idx):
        return z3.If(idx == 0, arr[0],
            z3.If(idx == 1, arr[1],
            z3.If(idx == 2, arr[2],
            z3.If(idx == 3, arr[3],
            z3.If(idx == 4, arr[4],
            arr[5])))))

    # Subtask 1: Open the first mission door (m_d_0). Treat occlusion (-1s) as complete to avoid false reversions.
    md0_color_idx = mission_doors[0]
    md0_state = select_by_color(door_state_colors, md0_color_idx)
    md0_x = select_by_color(door_x_colors, md0_color_idx)
    md0_y = select_by_color(door_y_colors, md0_color_idx)

    is_open = (md0_state == z3.IntVal(0))

    return is_open

from typing import Tuple
import z3
def subtask_1_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Helper to select an element from a fixed-size array of length 6 using a symbolic color index (0..5)
    def select_by_color(arr, idx):
        return z3.If(idx == 0, arr[0],
            z3.If(idx == 1, arr[1],
            z3.If(idx == 2, arr[2],
            z3.If(idx == 3, arr[3],
            z3.If(idx == 4, arr[4],
            arr[5])))))

    # Subtask 1 object: the first mission door (m_d_0)
    md0_color_idx = mission_doors[0]
    door_x = select_by_color(door_x_colors, md0_color_idx)
    door_y = select_by_color(door_y_colors, md0_color_idx)

    # If the door is occluded (coordinates -1), the agent is on top of the door -> consider near.
    occluded = z3.And(door_x == -1, door_y == -1)

    # Adjacent if same column and one row apart OR same row and one column apart.
    adjacent = z3.Or(
        z3.And(x == door_x, z3.Or(y == door_y + 1, y == door_y - 1)),
        z3.And(y == door_y, z3.Or(x == door_x + 1, x == door_x - 1))
    )

    return z3.Or(occluded, adjacent)

from typing import Tuple
import z3

def subtask_2_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Helper to select an element from a fixed-size array of length 6 using a symbolic color index (0..5)
    def select_by_color(arr, idx):
        return z3.If(idx == 0, arr[0],
            z3.If(idx == 1, arr[1],
            z3.If(idx == 2, arr[2],
            z3.If(idx == 3, arr[3],
            z3.If(idx == 4, arr[4],
            arr[5])))))

    # Subtask 2: Get the mission key (m_k_0).
    mk0_color_idx = mission_keys[0]
    kx = select_by_color(key_x_colors, mk0_color_idx)
    ky = select_by_color(key_y_colors, mk0_color_idx)

    # Completion when the mission key has been picked up (coordinates are -1).
    return z3.And(kx == z3.IntVal(-1), ky == z3.IntVal(-1))

from typing import Tuple
import z3

def subtask_2_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Helper to select an element from a fixed-size array of length 6 using a symbolic color index (0..5)
    def select_by_color(arr, idx):
        return z3.If(idx == 0, arr[0],
            z3.If(idx == 1, arr[1],
            z3.If(idx == 2, arr[2],
            z3.If(idx == 3, arr[3],
            z3.If(idx == 4, arr[4],
            arr[5])))))

    # Subtask 2 object: the mission key (m_k_0)
    mk0_color_idx = mission_keys[0]
    kx = select_by_color(key_x_colors, mk0_color_idx)
    ky = select_by_color(key_y_colors, mk0_color_idx)

    # Only consider proximity when the key is present (not picked up / not -1)
    key_present = z3.And(kx != z3.IntVal(-1), ky != z3.IntVal(-1))

    # Near if orthogonally adjacent by one cell (cannot be on the same tile as a key)
    adjacent = z3.Or(
        z3.And(x == kx, z3.Or(y == ky + 1, y == ky - 1)),
        z3.And(y == ky, z3.Or(x == kx + 1, x == kx - 1))
    )

    return z3.And(key_present, adjacent)

from typing import Tuple
import z3

def subtask_3_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Helper to select an element from a fixed-size array of length 6 using a symbolic color index (0..5)
    def select_by_color(arr, idx):
        return z3.If(idx == 0, arr[0],
            z3.If(idx == 1, arr[1],
            z3.If(idx == 2, arr[2],
            z3.If(idx == 3, arr[3],
            z3.If(idx == 4, arr[4],
            arr[5])))))

    # Subtask 3: Unlock/open the second mission door (m_d_1).
    md1_color_idx = mission_doors[1]
    md1_state = select_by_color(door_state_colors, md1_color_idx)
    md1_x = select_by_color(door_x_colors, md1_color_idx)
    md1_y = select_by_color(door_y_colors, md1_color_idx)

    # Completion when the door is open (state == 0).
    is_open = (md1_state == z3.IntVal(0))

    # Treat occlusion (-1s) as complete to avoid false reversions when the agent stands on the door.
    # "door_x, door_y, and door_state are -1 if occluded or not present. Occlusion is when the agent is on top of the door"
    # [Z3 State Encoding, code block under door_x_colors/door_y_colors/door_state_colors]
    occluded = z3.And(md1_x == z3.IntVal(-1), md1_y == z3.IntVal(-1), md1_state == z3.IntVal(-1))

    return z3.Or(is_open, occluded)

from typing import Tuple
import z3

def subtask_3_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Helper to select an element from a fixed-size array of length 6 using a symbolic color index (0..5)
    def select_by_color(arr, idx):
        return z3.If(idx == 0, arr[0],
            z3.If(idx == 1, arr[1],
            z3.If(idx == 2, arr[2],
            z3.If(idx == 3, arr[3],
            z3.If(idx == 4, arr[4],
            arr[5])))))

    # Subtask 3 object: the second mission door (m_d_1)
    md1_color_idx = mission_doors[1]
    door_x = select_by_color(door_x_colors, md1_color_idx)
    door_y = select_by_color(door_y_colors, md1_color_idx)

    # Near if the door is occluded (agent on top of door) or orthogonally adjacent by one cell.
    occluded = z3.And(door_x == z3.IntVal(-1), door_y == z3.IntVal(-1))
    adjacent = z3.Or(
        z3.And(x == door_x, z3.Or(y == door_y + 1, y == door_y - 1)),
        z3.And(y == door_y, z3.Or(x == door_x + 1, x == door_x - 1))
    )

    return z3.Or(occluded, adjacent)

from typing import Tuple
import z3

def subtask_4_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Completion when the agent is on the goal. We treat occlusion (-1, -1) as completed,
    # and also consider equality when the goal is visible.
    occluded = z3.And(goal_x == z3.IntVal(-1), goal_y == z3.IntVal(-1))
    on_goal = z3.And(x == goal_x, y == goal_y)
    return z3.Or(occluded, on_goal)

from typing import Tuple
import z3

def subtask_4_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Near the goal if:
    # - The goal is occluded (agent is on top of the goal), or
    # - The agent is on the same tile as a visible goal, or
    # - The agent is orthogonally adjacent by one cell.
    occluded = z3.And(goal_x == z3.IntVal(-1), goal_y == z3.IntVal(-1))
    same_tile = z3.And(x == goal_x, y == goal_y)
    adjacent = z3.Or(
        z3.And(x == goal_x, z3.Or(y == goal_y + 1, y == goal_y - 1)),
        z3.And(y == goal_y, z3.Or(x == goal_x + 1, x == goal_x - 1))
    )

    return z3.Or(occluded, same_tile, adjacent)