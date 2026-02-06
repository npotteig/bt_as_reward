from typing import Tuple
import z3

def subtask_1_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Target box color index from mission (m_b_0)
    m_b_0 = mission_boxes[0]

    # Subtask 1: "Open the {box_color} box"
    # A box is considered opened when its coordinates are -1 (opened or not present).
    # Build a disjunction over all color indices to select the mission box color symbolically.
    color_count = len(box_x_colors)
    conds = [
        z3.And(m_b_0 == idx, box_x_colors[idx] == -1, box_y_colors[idx] == -1)
        for idx in range(color_count)
    ]

    return z3.Or(*conds)

from typing import Tuple
import z3

def subtask_1_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Target box color index from mission (m_b_0)
    m_b_0 = mission_boxes[0]

    # Adjacent if agent is one cell away in cardinal directions and the box exists (not opened/not absent)
    color_count = len(box_x_colors)
    conds = []
    one = z3.IntVal(1)
    neg_one = z3.IntVal(-1)

    for idx in range(color_count):
        bx = box_x_colors[idx]
        by = box_y_colors[idx]
        adjacent = z3.Or(
            z3.And(x == bx, z3.Or(y == by + one, y == by - one)),
            z3.And(y == by, z3.Or(x == bx + one, x == bx - one)),
        )
        conds.append(z3.And(m_b_0 == idx, bx != neg_one, by != neg_one, adjacent))

    return z3.Or(*conds)

from typing import Tuple
import z3

def subtask_2_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Target mission indices (symbolic)
    m_k_0 = mission_keys[0]
    m_b_0 = mission_boxes[0]

    neg_one = z3.IntVal(-1)
    color_count = len(box_x_colors)

    # Subtask 2 complete iff:
    # - The mission box is the chosen color and is opened (box_x, box_y == -1).
    # - That box contains the mission key color (box_contains == mission key idx).
    # - The mission key is no longer present on the grid (key_x, key_y == -1).
    # Enumerate over colors to avoid symbolic list indexing.
    conds = []
    for bidx in range(color_count):
        bx = box_x_colors[bidx]
        by = box_y_colors[bidx]
        bc = box_contains[bidx]
        for kidx in range(color_count):
            kx = key_x_colors[kidx]
            ky = key_y_colors[kidx]
            conds.append(
                z3.And(
                    m_b_0 == bidx,          # mission box color selection
                    m_k_0 == kidx,          # mission key color selection
                    bx == neg_one,          # box opened (or not present, but tied via contents)
                    by == neg_one,
                    bc == kidx,             # box contained this key color (non-empty and matched)
                    kx == neg_one,          # key now picked up/not present
                    ky == neg_one
                )
            )

    return z3.Or(*conds)

from typing import Tuple
import z3

def subtask_2_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Target key color index from mission
    m_k_0 = mission_keys[0]

    one = z3.IntVal(1)
    neg_one = z3.IntVal(-1)
    color_count = len(key_x_colors)

    # Adjacent if agent is one cell away in cardinal directions and the key exists (not -1, -1)
    conds = []
    for kidx in range(color_count):
        kx = key_x_colors[kidx]
        ky = key_y_colors[kidx]
        adjacent = z3.Or(
            z3.And(x == kx, z3.Or(y == ky + one, y == ky - one)),
            z3.And(y == ky, z3.Or(x == kx + one, x == kx - one)),
        )
        conds.append(z3.And(m_k_0 == kidx, kx != neg_one, ky != neg_one, adjacent))

    return z3.Or(*conds)

from typing import Tuple
import z3

def subtask_3_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Target door color index from mission
    m_d_0 = mission_doors[0]

    # Door state indices: open=0, closed=1, locked=2 (see STATE_TO_IDX in provided encoding)
    OPEN = z3.IntVal(0)
    NEG_ONE = z3.IntVal(-1)

    color_count = len(door_state_colors)

    # Subtask 3 is complete iff the mission door is present (not occluded/not absent) and its state is open.
    # We explicitly require door_x and door_y != -1 to avoid treating "not present" or "occluded" as completion.
    conds = [
        z3.And(
            m_d_0 == didx,
            door_state_colors[didx] == OPEN,
            door_x_colors[didx] != NEG_ONE,
            door_y_colors[didx] != NEG_ONE
        )
        for didx in range(color_count)
    ]

    return z3.Or(*conds)

from typing import Tuple
import z3

def subtask_3_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission

    # Target door color index from mission
    m_d_0 = mission_doors[0]

    one = z3.IntVal(1)
    neg_one = z3.IntVal(-1)

    color_count = len(door_x_colors)
    conds = []
    for didx in range(color_count):
        dx = door_x_colors[didx]
        dy = door_y_colors[didx]

        # Adjacent in cardinal directions when door position is known (not occluded/not absent)
        adjacent_present = z3.And(
            dx != neg_one, dy != neg_one,
            z3.Or(
                z3.And(x == dx, z3.Or(y == dy + one, y == dy - one)),
                z3.And(y == dy, z3.Or(x == dx + one, x == dx - one))
            )
        )

        # Treat occlusion (agent on top of the door tile) as near per encoding:
        # "door_x, door_y, and door_state are -1 if occluded or not present.
        #  Occlusion is when the agent is on top of the door" [Z3 State Encoding]
        occluded_on_tile = z3.And(dx == neg_one, dy == neg_one)

        conds.append(
            z3.And(
                m_d_0 == didx,
                z3.Or(adjacent_present, occluded_on_tile)
            )
        )

    return z3.Or(*conds)