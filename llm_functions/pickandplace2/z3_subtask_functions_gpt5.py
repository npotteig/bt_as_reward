from typing import Tuple
import z3

def subtask_1_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
    mission_goals = mission  # not used for grasping

    # Tolerances (use ranges; avoid equality on Reals)
    # Position tolerance: within one block-length in each axis (axis-aligned proximity)
    tol_pos = z3.RealVal(0.05)

    # Gripper displacement band indicating it's closed around a block (not fully open, not fully zero)
    # Based on: g_disp ~0.025m when closed around a block; open ~0.05m; fully closed ~0m.
    grip_band_low = z3.RealVal(0.01)
    grip_band_high = z3.RealVal(0.04)

    # Optional velocity coupling tolerance: block velocity close to gripper velocity
    tol_vel = z3.RealVal(0.05)

    # Axis-aligned proximity between gripper and block
    pos_near = z3.And(
        bx >= gx - tol_pos, bx <= gx + tol_pos,
        by >= gy - tol_pos, by <= gy + tol_pos,
        bz >= gz - tol_pos, bz <= gz + tol_pos
    )

    # Gripper closed around a block (displacement within band)
    gripper_closed_around_block = z3.And(
        g_disp >= grip_band_low,
        g_disp <= grip_band_high
    )

    # Velocities approximately coupled (simple range, not Euclidean/Manhattan)
    vel_coupled = z3.And(
        bvel_x >= gvel_x - tol_vel, bvel_x <= gvel_x + tol_vel,
        bvel_y >= gvel_y - tol_vel, bvel_y <= gvel_y + tol_vel,
        bvel_z >= gvel_z - tol_vel, bvel_z <= gvel_z + tol_vel
    )

    # Subtask complete if gripper is near the block, closed around it, and moving with it
    return z3.And(pos_near, gripper_closed_around_block, vel_coupled)

from typing import Tuple
import z3

def subtask_1_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
    mission_goals = mission  # not needed for adjacency to the block in subtask 1

    # Axis-aligned proximity in the xy-plane using a tolerance derived from block size.
    tol_xy = z3.RealVal(0.05)  # approximate range; avoids equality on Reals

    near_in_xy = z3.And(
        bx >= gx - tol_xy, bx <= gx + tol_xy,
        by >= gy - tol_xy, by <= gy + tol_xy
    )

    return near_in_xy

from typing import Tuple
import z3

def subtask_2_complete(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
    mission_goals = mission

    # Select the target coordinates based on mission color index m_g_0 in {0,1}
    m_g_0 = mission_goals[0]
    target_x = z3.If(m_g_0 == z3.IntVal(0), goal_x[0], goal_x[1])
    target_y = z3.If(m_g_0 == z3.IntVal(0), goal_y[0], goal_y[1])
    target_z = z3.If(m_g_0 == z3.IntVal(0), goal_z[0], goal_z[1])

    # Tolerances (axis-aligned ranges; avoid equality on Reals)
    tol_pos = z3.RealVal(0.05)  # use block-length scale for proximity
    grip_band_low = z3.RealVal(0.01)
    grip_band_high = z3.RealVal(0.04)
    tol_vel = z3.RealVal(0.05)

    # Block near the selected target (axis-aligned range in x,y,z)
    block_near_target = z3.And(
        bx >= target_x - tol_pos, bx <= target_x + tol_pos,
        by >= target_y - tol_pos, by <= target_y + tol_pos,
        bz >= target_z - tol_pos, bz <= target_z + tol_pos
    )

    # Gripper closed around the block (holding, not released)
    gripper_holding = z3.And(
        g_disp >= grip_band_low,
        g_disp <= grip_band_high
    )

    # Ensure the gripper remains near the block while holding (axis-aligned range)
    gripper_near_block = z3.And(
        bx >= gx - tol_pos, bx <= gx + tol_pos,
        by >= gy - tol_pos, by <= gy + tol_pos,
        bz >= gz - tol_pos, bz <= gz + tol_pos
    )

    # Optional "hold" criterion: velocities approximately zero (simple band)
    block_vel_small = z3.And(
        bvel_x >= -tol_vel, bvel_x <= tol_vel,
        bvel_y >= -tol_vel, bvel_y <= tol_vel,
        bvel_z >= -tol_vel, bvel_z <= tol_vel
    )
    gripper_vel_small = z3.And(
        gvel_x >= -tol_vel, gvel_x <= tol_vel,
        gvel_y >= -tol_vel, gvel_y <= tol_vel,
        gvel_z >= -tol_vel, gvel_z <= tol_vel
    )

    return z3.And(
        block_near_target,
        gripper_holding,
        gripper_near_block,
        block_vel_small,
        gripper_vel_small
    )

from typing import Tuple
import z3

def subtask_2_object(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
    mission_goals = mission

    # Select target coordinates by mission color index m_g_0 in {0,1}
    # "mission_goals = [m_g_0] # m_g_0 is 0 or 1" [Mission Space, Code block]
    m_g_0 = mission_goals[0]
    target_x = z3.If(m_g_0 == z3.IntVal(0), goal_x[0], goal_x[1])
    target_y = z3.If(m_g_0 == z3.IntVal(0), goal_y[0], goal_y[1])

    # Axis-aligned proximity in the xy-plane using an approximate range; avoid equality on Reals.
    # Tolerance set to block size + two action steps:
    # "blocks are 0.05m x 0.05m x 0.05m" [Z3 State Encoding, Code block]
    # "Move +/-x, +/-y (~0.015 m)" [Action Space, Rows 0–3]
    tol_xy = z3.RealVal(0.08)

    near_in_xy = z3.And(
        gx >= target_x - tol_xy, gx <= target_x + tol_xy,
        gy >= target_y - tol_xy, gy <= target_y + tol_xy
    )

    return near_in_xy

def subtask_2_object_failure(state: Tuple[z3.ExprRef], mission: Tuple[z3.ExprRef]) -> z3.ExprRef:
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
    mission_goals = mission

    # Select target coordinates by mission color index m_g_0 in {0,1}
    # "mission_goals = [m_g_0] # m_g_0 is 0 or 1" [Mission Space, Code block]
    m_g_0 = mission_goals[0]
    target_x = z3.If(m_g_0 == z3.IntVal(0), goal_x[0], goal_x[1])
    target_y = z3.If(m_g_0 == z3.IntVal(0), goal_y[0], goal_y[1])

    # Axis-aligned proximity in the xy-plane using an approximate range; avoid equality on Reals.
    # Tolerance set to block size + two action steps:
    # "blocks are 0.05m x 0.05m x 0.05m" [Z3 State Encoding, Code block]
    # "Move +/-x, +/-y (~0.015 m)" [Action Space, Rows 0–3]
    tol_xy = z3.RealVal(0.05)

    near_in_xy = z3.And(
        gx >= target_x - tol_xy, gx <= target_x + tol_xy,
        gy >= target_y - tol_xy, gy <= target_y + tol_xy
    )

    return near_in_xy