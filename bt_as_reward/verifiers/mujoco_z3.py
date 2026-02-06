import z3
from typing import List, Tuple
from bt_as_reward.constants import MUJOCO_COLOR_TO_IDX, MUJOCO_IDX_TO_COLOR

NUM_GOALS = 2
SPEED = z3.RealVal("0.015")

def _env_constraints(env_name, 
                     state_vars, 
                     s: z3.Solver, 
                     initial_goal_pos=None,
                     initial_block_pos=None,
                     initial_mission_goals=None,
                     T=25,
                     negative_check=False):
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, block_held, goal_x, goal_y, goal_z, mission_goals = state_vars
    
    if initial_block_pos is not None:
        s.add(bx[0] == z3.RealVal(f"{initial_block_pos[0]:.4f}"))
        s.add(by[0] == z3.RealVal(f"{initial_block_pos[1]:.4f}"))
        s.add(bz[0] == z3.RealVal(f"{initial_block_pos[2]:.4f}"))

    match env_name:
        case "FetchPickAndPlace-v5":
            if initial_goal_pos is not None:
                s.add(goal_x[0] == z3.RealVal(f"{initial_goal_pos[0][0]:.4f}"))
                s.add(goal_y[0] == z3.RealVal(f"{initial_goal_pos[0][1]:.4f}"))
                s.add(goal_z[0] == z3.RealVal(f"{initial_goal_pos[0][2]:.4f}"))
            
            s.add(mission_goals[0] == MUJOCO_COLOR_TO_IDX["green"])
            if not negative_check:
                s.add(z3.And(
                    z3.Abs(goal_x[MUJOCO_COLOR_TO_IDX["green"]] - bx[T-1]) <= z3.RealVal("0.05"),
                    z3.Abs(goal_y[MUJOCO_COLOR_TO_IDX["green"]] - by[T-1]) <= z3.RealVal("0.05"),
                    z3.Abs(goal_z[MUJOCO_COLOR_TO_IDX["green"]] - bz[T-1]) <= z3.RealVal("0.05"),
                ))
            else:
                s.add(z3.Or(
                    z3.Abs(goal_x[MUJOCO_COLOR_TO_IDX["green"]] - bx[T-1]) > z3.RealVal("0.05"),
                    z3.Abs(goal_y[MUJOCO_COLOR_TO_IDX["green"]] - by[T-1]) > z3.RealVal("0.05"),
                    z3.Abs(goal_z[MUJOCO_COLOR_TO_IDX["green"]] - bz[T-1]) > z3.RealVal("0.05"),
                )
                )
        case "FetchPickAndPlace2-v1":
            if initial_goal_pos is not None:
                for i in range(2):
                    s.add(goal_x[i] == z3.RealVal(f"{initial_goal_pos[i][0]:.4f}"))
                    s.add(goal_y[i] == z3.RealVal(f"{initial_goal_pos[i][1]:.4f}"))
                    s.add(goal_z[i] == z3.RealVal(f"{initial_goal_pos[i][2]:.4f}"))
            if initial_mission_goals is not None:
                s.add(mission_goals[0] == initial_mission_goals[0])
            for i in range(2):
                if not negative_check:
                    s.add(z3.Implies(
                        mission_goals[0] == i,
                        z3.And(
                            z3.Abs(goal_x[i] - bx[T-1]) <= z3.RealVal("0.05"),
                            z3.Abs(goal_y[i] - by[T-1]) <= z3.RealVal("0.05"),
                            z3.Abs(goal_z[i] - bz[T-1]) <= z3.RealVal("0.05"),
                        )
                    ))

def verify_spec(spec_type: str,
                subtask_complete_func=None,
                near_object_func=None,
                initial_goal_state=None,
                initial_block_state=None,
                initial_mission_goal=None,
                env_name: str = "FetchPickAndPlace-v5",
                num_mission_goals: int = 1,
                T: int = 25,
                restricted_initial_states: List[Tuple] = None,
                negative_check: bool = False,
                timeout: int = 900000):
    gx = [z3.Real(f'gripper_x_{t}') for t in range(T)]
    gy = [z3.Real(f'gripper_y_{t}') for t in range(T)]
    gz = [z3.Real(f'gripper_z_{t}') for t in range(T)]
    g_disp = [z3.Real(f'gripper_disp_{t}') for t in range(T)]
    bx = [z3.Real(f'block_x_{t}') for t in range(T)]
    by = [z3.Real(f'block_y_{t}') for t in range(T)]
    bz = [z3.Real(f'block_z_{t}') for t in range(T)]
    block_held = [z3.Bool(f'block_held_{t}') for t in range(T)]
    
    goal_x = [z3.Real(f'goal_x_{MUJOCO_IDX_TO_COLOR[color_idx]}') for color_idx in range(NUM_GOALS)]
    goal_y = [z3.Real(f'goal_y_{MUJOCO_IDX_TO_COLOR[color_idx]}') for color_idx in range(NUM_GOALS)]
    goal_z = [z3.Real(f'goal_z_{MUJOCO_IDX_TO_COLOR[color_idx]}') for color_idx in range(NUM_GOALS)]
    
    gvel_x = [z3.Real(f'gripper_vel_x_{t}') for t in range(T)]
    gvel_y = [z3.Real(f'gripper_vel_y_{t}') for t in range(T)]
    gvel_z = [z3.Real(f'gripper_vel_z_{t}') for t in range(T)]
    bvel_x = [z3.Real(f'block_vel_x_{t}') for t in range(T)]
    bvel_y = [z3.Real(f'block_vel_y_{t}') for t in range(T)]
    bvel_z = [z3.Real(f'block_vel_z_{t}') for t in range(T)]
    
    mission_goals = [z3.Int(f"mission_goal_{i}") for i in range(num_mission_goals)]

    s = z3.Solver()
    s.set(timeout=timeout)
    if restricted_initial_states is not None:
        for state in restricted_initial_states:
            init_gx, init_gy, init_gz, init_gdisp, init_bx, init_by, init_bz, init_gvel_x, init_gvel_y, init_gvel_z, init_bvel_x, init_bvel_y, init_bvel_z, init_goal_x, init_goal_y, init_goal_z, init_mission_goal = state
            s.add(z3.Or(
                gx[0] != init_gx,
                gy[0] != init_gy,
                gz[0] != init_gz,
                g_disp[0] != init_gdisp,
                bx[0] != init_bx,
                by[0] != init_by,
                bz[0] != init_bz,
                gvel_x[0] != init_gvel_x,
                gvel_y[0] != init_gvel_y,
                gvel_z[0] != init_gvel_z,
                bvel_x[0] != init_bvel_x,
                bvel_y[0] != init_bvel_y,
                bvel_z[0] != init_bvel_z,
                *[goal_x[i] != init_goal_x[i] for i in range(NUM_GOALS)],
                *[goal_y[i] != init_goal_y[i] for i in range(NUM_GOALS)],
                *[goal_z[i] != init_goal_z[i] for i in range(NUM_GOALS)],
                mission_goals[0] != init_mission_goal
            ))
    # Set initial conditions
    s.add(gx[0] == z3.RealVal("1.34"), gy[0] == z3.RealVal("0.75"), gz[0] == z3.RealVal("0.53"), g_disp[0] == z3.RealVal("0")) # Gripper initial pos and closed
    s.add(bx[0] >= z3.RealVal("1.19"), bx[0] <= z3.RealVal("1.49"), by[0] >= z3.RealVal("0.6"), by[0] <= z3.RealVal("0.9"), bz[0] == z3.RealVal("0.425"))
    s.add(
        z3.And([z3.Or(
            z3.Abs(bx[0] - goal_x[i]) >= z3.RealVal("0.1"),
            z3.Abs(by[0] - goal_y[i]) >= z3.RealVal("0.1"),
            z3.Abs(bz[0] - goal_z[i]) >= z3.RealVal("0.1"),
        ) for i in range(NUM_GOALS)])
    )
    s.add(gvel_x[0] == z3.RealVal("0"), gvel_y[0] == z3.RealVal("0"), gvel_z[0] == z3.RealVal("0")) # Gripper initial vel
    s.add(bvel_x[0] == z3.RealVal("0"), bvel_y[0] == z3.RealVal("0"), bvel_z[0] == z3.RealVal("0")) # Block initial vel
    s.add([z3.And(mission_goals[i] >= 0, mission_goals[i] < NUM_GOALS) for i in range(num_mission_goals)])
    s.add(block_held[0] == False)
    
    # Set goal conditions
    for i in range(NUM_GOALS):
        s.add(goal_x[i] >= z3.RealVal("1.19"), goal_x[i] <= z3.RealVal("1.49"), goal_y[i] >= z3.RealVal("0.6"), goal_y[i] <= z3.RealVal("0.9"), goal_z[i] >= z3.RealVal("0.425"), goal_z[i] <= z3.RealVal("0.875"))
    
    state_vars = (gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, block_held, goal_x, goal_y, goal_z, mission_goals)
    _env_constraints(env_name, state_vars, s, initial_goal_pos=initial_goal_state,
                     initial_block_pos=initial_block_state,
                     initial_mission_goals=initial_mission_goal,
                     T=T,
                     negative_check=negative_check or (spec_type=="near_object"))
    
    grip_offsets = [z3.RealVal("0.0135"), z3.RealVal("0.014"), z3.RealVal("0.0385")]
    block_radius = z3.RealVal("0.025")
    # State constraints
    for t in range(T):
        # Gripper constraints
        s.add(gx[t] >= z3.RealVal("1.05"), gx[t] <= z3.RealVal("1.55"),
              gy[t] >= z3.RealVal("0.4"), gy[t] <= z3.RealVal("1.1"),
              gz[t] >= z3.RealVal("0.4385"), gz[t] <= z3.RealVal("0.9"), # 0.4 + 0.0385 (half gripper finger height)
              g_disp[t] >= z3.RealVal("0"), g_disp[t] <= z3.RealVal("0.05")) 

        r_grip_y = gy[t] + g_disp[t]
        l_grip_y = gy[t] - g_disp[t]
        
        # Block constraints
        s.add(bx[t] >= z3.RealVal("1.05"), bx[t] <= z3.RealVal("1.55"),
              by[t] >= z3.RealVal("0.4"), by[t] <= z3.RealVal("1.1"),
              bz[t] >= z3.RealVal("0.425"), bz[t] <= z3.RealVal("0.9"))
        
        left_clear = z3.Or(
            by[t] + block_radius <= l_grip_y - grip_offsets[1],
            by[t] - block_radius >= l_grip_y
        )
        
        right_clear = z3.Or(
            by[t] + block_radius <= r_grip_y,
            by[t] - block_radius >= r_grip_y + grip_offsets[1]
        )
        
        y_clear = z3.And(left_clear, right_clear)
        
        s.add(
            z3.Or(
                bx[t] + block_radius <= gx[t] - grip_offsets[0],
                bx[t] - block_radius >= gx[t] + grip_offsets[0],
                y_clear,
                bz[t] + block_radius <= gz[t] - grip_offsets[2],
            )
        )
        s.add(bz[t] <= gz[t] + grip_offsets[2])  # Block can't be higher than gripper
    
    
    for t in range(T-1):
        s.add(
            z3.Or(
                # -------- Right --------
                z3.And(
                    gx[t+1] == gx[t] + SPEED,
                    gy[t+1] == gy[t],
                    gz[t+1] == gz[t],
                    gvel_x[t+1] == SPEED,
                    gvel_y[t+1] == 0,
                    gvel_z[t+1] == 0,
                    g_disp[t+1] == g_disp[t],
                    block_held[t+1] == block_held[t],
                    z3.If(block_held[t], 
                          z3.And(
                                bx[t+1] == bx[t] + SPEED,
                                by[t+1] == by[t],
                                bz[t+1] == bz[t],
                                bvel_x[t+1] == SPEED,
                                bvel_y[t+1] == z3.RealVal("0"),
                                bvel_z[t+1] == z3.RealVal("0")
                            ),
                            z3.And(
                                bx[t+1] == bx[t],
                                by[t+1] == by[t],
                                bz[t+1] == bz[t],
                                bvel_x[t+1] == z3.RealVal("0"),
                                bvel_y[t+1] == z3.RealVal("0"),
                                bvel_z[t+1] == z3.RealVal("0")
                            )
                    )
                ),
                # -------- Left --------
                z3.And(
                    gx[t+1] == gx[t] - SPEED,
                    gy[t+1] == gy[t],
                    gz[t+1] == gz[t],
                    gvel_x[t+1] == -SPEED,
                    gvel_y[t+1] == z3.RealVal("0"),
                    gvel_z[t+1] == z3.RealVal("0"),
                    g_disp[t+1] == g_disp[t],
                    block_held[t+1] == block_held[t],
                    z3.If(block_held[t], 
                          z3.And(
                                bx[t+1] == bx[t] - SPEED,
                                by[t+1] == by[t],
                                bz[t+1] == bz[t],
                                bvel_x[t+1] == -SPEED,
                                bvel_y[t+1] == z3.RealVal("0"),
                                bvel_z[t+1] == z3.RealVal("0")
                            ),
                            z3.And(
                                bx[t+1] == bx[t],
                                by[t+1] == by[t],
                                bz[t+1] == bz[t],
                                bvel_x[t+1] == z3.RealVal("0"),
                                bvel_y[t+1] == z3.RealVal("0"),
                                bvel_z[t+1] == z3.RealVal("0")
                            )
                    )
                ),
                # -------- Forward --------
                z3.And(
                    gx[t+1] == gx[t],
                    gy[t+1] == gy[t] + SPEED,
                    gz[t+1] == gz[t],
                    gvel_x[t+1] == z3.RealVal("0"),
                    gvel_y[t+1] == SPEED,
                    gvel_z[t+1] == z3.RealVal("0"),
                    g_disp[t+1] == g_disp[t],
                    block_held[t+1] == block_held[t],
                    z3.If(block_held[t], 
                          z3.And(
                                bx[t+1] == bx[t],
                                by[t+1] == by[t] + SPEED,
                                bz[t+1] == bz[t],
                                bvel_x[t+1] == z3.RealVal("0"),
                                bvel_y[t+1] == SPEED,
                                bvel_z[t+1] == z3.RealVal("0")
                            ),
                            z3.And(
                                bx[t+1] == bx[t],
                                by[t+1] == by[t],
                                bz[t+1] == bz[t],
                                bvel_x[t+1] == z3.RealVal("0"),
                                bvel_y[t+1] == z3.RealVal("0"),
                                bvel_z[t+1] == z3.RealVal("0")
                            )
                    )
                ),
                # -------- Backward --------
                z3.And(
                    gx[t+1] == gx[t],
                    gy[t+1] == gy[t] - SPEED,
                    gz[t+1] == gz[t],
                    gvel_x[t+1] == z3.RealVal("0"),
                    gvel_y[t+1] == -SPEED,
                    gvel_z[t+1] == z3.RealVal("0"),
                    g_disp[t+1] == g_disp[t],
                    block_held[t+1] == block_held[t],
                    z3.If(block_held[t], 
                          z3.And(
                                bx[t+1] == bx[t],
                                by[t+1] == by[t] - SPEED,
                                bz[t+1] == bz[t],
                                bvel_x[t+1] == z3.RealVal("0"),
                                bvel_y[t+1] == -SPEED,
                                bvel_z[t+1] == z3.RealVal("0")
                            ),
                            z3.And(
                                bx[t+1] == bx[t],
                                by[t+1] == by[t],
                                bz[t+1] == bz[t],
                                bvel_x[t+1] == z3.RealVal("0"),
                                bvel_y[t+1] == z3.RealVal("0"),
                                bvel_z[t+1] == z3.RealVal("0")
                            )
                    )
                ),
                # -------- Up --------
                z3.And(
                    gx[t+1] == gx[t],
                    gy[t+1] == gy[t],
                    gz[t+1] == gz[t] + SPEED,
                    gvel_x[t+1] == z3.RealVal("0"),
                    gvel_y[t+1] == z3.RealVal("0"),
                    gvel_z[t+1] == SPEED,
                    g_disp[t+1] == g_disp[t],
                    block_held[t+1] == block_held[t],
                    z3.If(block_held[t], 
                          z3.And(
                                bx[t+1] == bx[t],
                                by[t+1] == by[t],
                                bz[t+1] == bz[t] + SPEED,
                                bvel_x[t+1] == z3.RealVal("0"),
                                bvel_y[t+1] == z3.RealVal("0"),
                                bvel_z[t+1] == SPEED
                            ),
                            z3.And(
                                bx[t+1] == bx[t],
                                by[t+1] == by[t],
                                bz[t+1] == bz[t],
                                bvel_x[t+1] == z3.RealVal("0"),
                                bvel_y[t+1] == z3.RealVal("0"),
                                bvel_z[t+1] == z3.RealVal("0")
                            )
                    )
                ),
                # -------- Down --------
                z3.And(
                    gx[t+1] == gx[t],
                    gy[t+1] == gy[t],
                    gz[t+1] == gz[t] - SPEED,
                    gvel_x[t+1] == z3.RealVal("0"),
                    gvel_y[t+1] == z3.RealVal("0"),
                    gvel_z[t+1] == -SPEED,
                    g_disp[t+1] == g_disp[t],
                    block_held[t+1] == block_held[t],
                    z3.If(block_held[t], 
                          z3.And(
                                bx[t+1] == bx[t],
                                by[t+1] == by[t],
                                bz[t+1] == bz[t] - SPEED,
                                bvel_x[t+1] == z3.RealVal("0"),
                                bvel_y[t+1] == z3.RealVal("0"),
                                bvel_z[t+1] == -SPEED
                            ),
                            z3.And(
                                bx[t+1] == bx[t],
                                by[t+1] == by[t],
                                bz[t+1] == bz[t],
                                bvel_x[t+1] == z3.RealVal("0"),
                                bvel_y[t+1] == z3.RealVal("0"),
                                bvel_z[t+1] == z3.RealVal("0")
                            )
                    )
                ),
                # -------- Open Gripper --------
                z3.And(
                    gx[t+1] == gx[t],
                    gy[t+1] == gy[t],
                    gz[t+1] == gz[t],
                    gvel_x[t+1] == z3.RealVal("0"),
                    gvel_y[t+1] == z3.RealVal("0"),
                    gvel_z[t+1] == z3.RealVal("0"),
                    g_disp[t+1] == z3.RealVal("0.05"),
                    block_held[t+1] == False,
                    bx[t+1] == bx[t],
                    by[t+1] == by[t],
                    bz[t+1] == bz[t],
                    bvel_x[t+1] == z3.RealVal("0"),
                    bvel_y[t+1] == z3.RealVal("0"),
                    bvel_z[t+1] == z3.RealVal("0")
                ),
                # -------- Close Gripper --------
                z3.And(
                    z3.If(
                        z3.And(
                            by[t] + block_radius <= gy[t] + g_disp[t],
                            by[t] - block_radius >= gy[t] - g_disp[t],
                            gx[t] + grip_offsets[0] <= bx[t] + block_radius,
                            gx[t] - grip_offsets[0] >= bx[t] - block_radius,
                            gz[t] - z3.RealVal("0.015") <= bz[t] + block_radius,
                        ),
                        z3.And(
                            g_disp[t+1] == block_radius,
                            block_held[t+1] == True,
                            bx[t+1] == bx[t],
                            by[t+1] == gy[t],
                            bz[t+1] == bz[t]
                        ),
                        z3.And(
                            g_disp[t+1] == 0,
                            block_held[t+1] == block_held[t],
                            bx[t+1] == bx[t],
                            by[t+1] == by[t],
                            bz[t+1] == bz[t]
                        )
                    ),
                    gx[t+1] == gx[t],
                    gy[t+1] == gy[t],
                    gz[t+1] == gz[t],
                    gvel_x[t+1] == z3.RealVal("0"),
                    gvel_y[t+1] == z3.RealVal("0"),
                    gvel_z[t+1] == z3.RealVal("0"),
                    bvel_x[t+1] == z3.RealVal("0"),
                    bvel_y[t+1] == z3.RealVal("0"),
                    bvel_z[t+1] == z3.RealVal("0")
                )
            )
        )
    
    
    if spec_type == "subtask_completion":
        if subtask_complete_func is None:
            raise ValueError("subtask_complete_func must be provided for subtask_completion spec type")
        # Check negation of spec: f is false at all timesteps before reaching goal
        s.add(z3.And([z3.Not(subtask_complete_func((gx[t], gy[t], gz[t], g_disp[t], bx[t], by[t], bz[t], gvel_x[t], gvel_y[t], gvel_z[t], bvel_x[t], bvel_y[t], bvel_z[t], goal_x, goal_y, goal_z), (mission_goals))) for t in range(T)]))
    elif spec_type == "near_object":
        if near_object_func is None:
            raise ValueError("near_object_func must be provided for near_object spec type")
        if subtask_complete_func is None:
            raise ValueError("subtask_complete_func must be provided for near_object spec type")
        if not negative_check:
            s.add(
                z3.Or([
                    z3.And(
                        subtask_complete_func((gx[t], gy[t], gz[t], g_disp[t], bx[t], by[t], bz[t], gvel_x[t], gvel_y[t], gvel_z[t], bvel_x[t], bvel_y[t], bvel_z[t], goal_x, goal_y, goal_z), (mission_goals)),
                        z3.Not(subtask_complete_func((gx[t-1], gy[t-1], gz[t-1], g_disp[t-1], bx[t-1], by[t-1], bz[t-1], gvel_x[t-1], gvel_y[t-1], gvel_z[t-1], bvel_x[t-1], bvel_y[t-1], bvel_z[t-1], goal_x, goal_y, goal_z), (mission_goals))),
                        z3.Not(near_object_func((gx[t-1], gy[t-1], gz[t-1], g_disp[t-1], bx[t-1], by[t-1], bz[t-1], gvel_x[t-1], gvel_y[t-1], gvel_z[t-1], bvel_x[t-1], bvel_y[t-1], bvel_z[t-1], goal_x, goal_y, goal_z), (mission_goals)))
                    )
                    for t in range(1, T)
                ])
            )
        else:
            # Negative check: Same as completion find a trajectory where near_object is never true
            s.add(z3.And([z3.Not(near_object_func((gx[t], gy[t], gz[t], g_disp[t], bx[t], by[t], bz[t], gvel_x[t], gvel_y[t], gvel_z[t], bvel_x[t], bvel_y[t], bvel_z[t], goal_x, goal_y, goal_z), (mission_goals))) for t in range(T)]))
    elif spec_type == "persistence":
        if subtask_complete_func is None:
            raise ValueError("subtask_complete_func must be provided for persistence spec type")
        if not isinstance(subtask_complete_func, list):
            raise ValueError("subtask_complete_func must be a list of functions for persistence spec type")
        # Check for a trajectory when f becomes true it remains true until goal is reached
        for t in range(T - 1):
            for i in range(len(subtask_complete_func)):
                s.add(
                    z3.Implies(
                        subtask_complete_func[i]((gx[t], gy[t], gz[t], g_disp[t], bx[t], by[t], bz[t], gvel_x[t], gvel_y[t], gvel_z[t], bvel_x[t], bvel_y[t], bvel_z[t], goal_x, goal_y, goal_z), (mission_goals)),
                        subtask_complete_func[i]((gx[t+1], gy[t+1], gz[t+1], g_disp[t+1], bx[t+1], by[t+1], bz[t+1], gvel_x[t+1], gvel_y[t+1], gvel_z[t+1], bvel_x[t+1], bvel_y[t+1], bvel_z[t+1], goal_x, goal_y, goal_z), (mission_goals))
                    )
                )
    
    if s.check() == z3.sat:
        m = s.model()
        initial_state = (m.evaluate(gx[0]), m.evaluate(gy[0]), m.evaluate(gz[0]), m.evaluate(g_disp[0]),
                             m.evaluate(bx[0]), m.evaluate(by[0]), m.evaluate(bz[0]),
                             m.evaluate(gvel_x[0]), m.evaluate(gvel_y[0]), m.evaluate(gvel_z[0]),
                             m.evaluate(bvel_x[0]), m.evaluate(bvel_y[0]), m.evaluate(bvel_z[0]),
                             [m.evaluate(goal_x[i]) for i in range(NUM_GOALS)],
                             [m.evaluate(goal_y[i]) for i in range(NUM_GOALS)],
                             [m.evaluate(goal_z[i]) for i in range(NUM_GOALS)],
                             m.evaluate(mission_goals[0]))
        if spec_type == "persistence" or negative_check:
            return "", True, initial_state
        out = ""
        out += f"Counterexample trajectory found for {spec_type} with subtask_complete_func ({subtask_complete_func.__name__ if subtask_complete_func else 'N/A'}) and near_object_func ({near_object_func.__name__ if near_object_func else 'N/A'}):\n"
        out += f"Mission Goal: {MUJOCO_IDX_TO_COLOR[m.evaluate(mission_goals[0]).as_long()]}\n"
        for t in range(T):
            out += ("-"*20) + "\n"
            out += f"t={t}, Gripper Pos: ({m.evaluate(gx[t]).as_decimal(5)}, {m.evaluate(gy[t]).as_decimal(5)}, {m.evaluate(gz[t]).as_decimal(5)}), Gripper Disp: {m.evaluate(g_disp[t]).as_decimal(5)}\n"
            for i in range(NUM_GOALS):
                out += f"Goal {MUJOCO_IDX_TO_COLOR[i]} Pos: ({m.evaluate(goal_x[i]).as_decimal(5)}, {m.evaluate(goal_y[i]).as_decimal(5)}, {m.evaluate(goal_z[i]).as_decimal(5)})\n"
            out += f"Gripper Vel: ({m.evaluate(gvel_x[t]).as_decimal(5)}, {m.evaluate(gvel_y[t]).as_decimal(5)}, {m.evaluate(gvel_z[t]).as_decimal(5)})\n"
            out += f"Block Pos: ({m.evaluate(bx[t]).as_decimal(5)}, {m.evaluate(by[t]).as_decimal(5)}, {m.evaluate(bz[t]).as_decimal(5)})\n"
            out += f"Block Vel: ({m.evaluate(bvel_x[t]).as_decimal(5)}, {m.evaluate(bvel_y[t]).as_decimal(5)}, {m.evaluate(bvel_z[t]).as_decimal(5)})\n"
            if subtask_complete_func:
                out += f"subtask_complete_func={m.evaluate(subtask_complete_func((gx[t], gy[t], gz[t], g_disp[t], bx[t], by[t], bz[t], gvel_x[t], gvel_y[t], gvel_z[t], bvel_x[t], bvel_y[t], bvel_z[t], goal_x, goal_y, goal_z), (mission_goals)))}\n"
            if near_object_func:
                out += f"near_object_func={m.evaluate(near_object_func((gx[t], gy[t], gz[t], g_disp[t], bx[t], by[t], bz[t], gvel_x[t], gvel_y[t], gvel_z[t], bvel_x[t], bvel_y[t], bvel_z[t], goal_x, goal_y, goal_z), (mission_goals)))}\n"
        return out, False, initial_state
    else:
        if spec_type == "persistence" or negative_check:
            return "", False, ()
        out = ""
        print(f"No counterexample trajectory exists; {spec_type} is true for subtask_complete_func ({subtask_complete_func.__name__ if subtask_complete_func else 'N/A'}) and near_object_func ({near_object_func.__name__ if near_object_func else 'N/A'}).")
        return out, True, ()
    