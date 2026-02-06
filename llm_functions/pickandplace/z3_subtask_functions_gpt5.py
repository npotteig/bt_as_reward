from typing import Tuple
import z3

def subtask_1_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    """
    Subtask 1: Grasp Block
    Completion criteria (range-based, environment-agnostic):
    - Gripper is near the block in all axes (axis-aligned proximity using block-size-derived tolerance).
    - Gripper displacement indicates a closed-on-block state (around ~0.025m, using a range).
    - Relative velocity between gripper and block is small (range-based), consistent with a grasp.

    References:
    - "pick up and move the block to the target location" [Mission Space, Paragraph 1]
    - "Toggle gripper to close ... or ~0.025m when closed around a block" [Action Space, Row 7]
    - "blocks are 0.05m x 0.05m x 0.05m" [Z3 State Encoding, blocks spec]
    """
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
    mission_goals = mission  # not needed for grasp completion

    def abs_expr(e: z3.ExprRef) -> z3.ExprRef:
        return z3.If(e >= 0, e, -e)

    # Tolerances derived from provided specs; avoid exact equality on Reals.
    block_size = z3.RealVal(0.05)       # from spec
    prox_tol = block_size               # proximity within one block-length per axis
    grip_min = z3.RealVal(0.015)        # around closed-on-block lower bound
    grip_max = z3.RealVal(0.035)        # around closed-on-block upper bound
    vel_tol = z3.RealVal(0.02)          # small relative velocity tolerance

    proximity = z3.And(
        abs_expr(gx - bx) <= prox_tol,
        abs_expr(gy - by) <= prox_tol,
        abs_expr(gz - bz) <= prox_tol
    )

    grip_closed_on_block = z3.And(
        g_disp >= grip_min,
        g_disp <= grip_max
    )

    relative_velocity_small = z3.And(
        abs_expr(gvel_x - bvel_x) <= vel_tol,
        abs_expr(gvel_y - bvel_y) <= vel_tol,
        abs_expr(gvel_z - bvel_z) <= vel_tol
    )

    return z3.And(proximity, grip_closed_on_block, relative_velocity_small)

from typing import Tuple
import z3

def subtask_1_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    """
    Subtask 1 object: Block
    Check if the gripper (agent) is adjacent/near the block in the XY-plane using
    axis-aligned proximity with a range tolerance (no exact equality).
    """
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
    mission_goals = mission  # not needed for XY proximity to the block

    def abs_expr(e: z3.ExprRef) -> z3.ExprRef:
        return z3.If(e >= 0, e, -e)

    # Use block dimension as a reasonable XY proximity tolerance (range-based).
    block_size = z3.RealVal(0.05)

    return z3.And(
        abs_expr(gx - bx) <= block_size,
        abs_expr(gy - by) <= block_size
    )

from typing import Tuple
import z3

def subtask_2_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    """
    Subtask 2: Place at Target (for the mission-specified color; green index = 0)
    Completion criteria (range-based, environment-agnostic):
    - Block is near the target position in all axes using axis-aligned tolerances (no exact equality).
    - Gripper remains closed around the block (range around ~0.025m) to reflect the continuing/hold requirement.
    """
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
    mission_goals = mission  # mission_goals = [m_g_0], with m_g_0 always 0 for green

    def abs_expr(e: z3.ExprRef) -> z3.ExprRef:
        return z3.If(e >= 0, e, -e)

    # Tolerances based on block dimensions; avoid hard-coded coordinates and exact equality.
    block_size = z3.RealVal(0.05)  # block edge length
    tol = block_size               # use one block-length as positional tolerance
    grip_min = z3.RealVal(0.015)   # lower bound for "closed around block"
    grip_max = z3.RealVal(0.035)   # upper bound for "closed around block"

    # Mission color index: per spec, m_g_0 is always 0 for green.
    # Use the green goal (index 0) without introducing new symbolic vars or equality over Reals.
    target_x = goal_x[0]
    target_y = goal_y[0]
    target_z = goal_z[0]

    block_near_target = z3.And(
        abs_expr(bx - target_x) <= tol,
        abs_expr(by - target_y) <= tol,
        abs_expr(bz - target_z) <= tol
    )

    gripper_holding = z3.And(
        g_disp >= grip_min,
        g_disp <= grip_max
    )

    return z3.And(block_near_target, gripper_holding)

from typing import Tuple
import z3

def subtask_2_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    """
    Subtask 2 object: Target (green index = 0)
    Check if the gripper (agent) is adjacent/near the target in the XY-plane using
    axis-aligned proximity with a range tolerance (no exact equality).
    Tolerance is set to block_size + 2 * action_step to account for discrete motion.
    """
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
    mission_goals = mission  # m_g_0 is always 0 for green

    def abs_expr(e: z3.ExprRef) -> z3.ExprRef:
        return z3.If(e >= 0, e, -e)

    # Environment scale constants (not absolute coordinates)
    block_size = z3.RealVal(0.05)    # "blocks are 0.05m x 0.05m x 0.05m" [Z3 State Encoding]
    action_step = z3.RealVal(0.015)  # "Move +x (~0.015 m)" [Action Space, Row 0]
    tol = block_size + action_step + action_step  # block_size + 2 * action_step

    target_x = goal_x[0]  # "m_g_0 is always 0 for green" [Mission Space, Code Snippet]
    target_y = goal_y[0]

    return z3.And(
        abs_expr(gx - target_x) <= tol,
        abs_expr(gy - target_y) <= tol
    )