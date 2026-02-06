import z3
from bt_as_reward.constants import MUJOCO_IDX_TO_COLOR
from bt_as_reward.verifiers.mujoco_z3 import verify_spec

NUM_GOALS = 2
N = 10

def subtask_block_condition(state, mission):
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
    mission_goals = mission
    
    return z3.And(g_disp < z3.RealVal("0.05"), 
                  g_disp > z3.RealVal("0.01"),
                  z3.Abs(gx - bx) < z3.RealVal("0.05"),
                  z3.Abs(gy - by) < z3.RealVal("0.05"),
                  z3.Abs(gz - bz) < z3.RealVal("0.05"),
                  z3.Abs(gvel_x - bvel_x) < z3.RealVal("0.01"),
                  z3.Abs(gvel_y - bvel_y) < z3.RealVal("0.01"),
                  z3.Abs(gvel_z - bvel_z) < z3.RealVal("0.01"),
    )

def near_block_condition(state, mission):
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
    mission_goals = mission
    
    return z3.And(
        z3.Abs(gx - bx) < z3.RealVal("0.05"),
        z3.Abs(gy - by) < z3.RealVal("0.05"),
        z3.Abs(gz - bz) < z3.RealVal("0.05"),
    )

def subtask_goal_condition(state, mission):
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
    mission_goals = mission
    
    conds = []
    for i in range(NUM_GOALS):
        conds.append(
            z3.Implies(
                mission_goals[0] == i,
                z3.And(
                    z3.Abs(bx - goal_x[i]) <= z3.RealVal("0.05"),
                    z3.Abs(by - goal_y[i]) <= z3.RealVal("0.05"),
                    z3.Abs(bz - goal_z[i]) <= z3.RealVal("0.05"),
                )
            )
        )
    
    return z3.And(conds)

def near_goal_condition(state, mission):
    gx, gy, gz, g_disp, bx, by, bz, gvel_x, gvel_y, gvel_z, bvel_x, bvel_y, bvel_z, goal_x, goal_y, goal_z = state
    mission_goals = mission
    
    conds = []
    for i in range(NUM_GOALS):
        conds.append(
            z3.Implies(
                mission_goals[0] == i,
                z3.And(
                    z3.Abs(bx - goal_x[i]) <= z3.RealVal("0.1"),
                    z3.Abs(by - goal_y[i]) <= z3.RealVal("0.1"),
                    z3.Abs(bz - goal_z[i]) <= z3.RealVal("0.1"),
                )
            )
        )
    
    return z3.And(conds)

if __name__ == "__main__":
    import time
    start_time = time.time()
    # out, res, _ = verify_spec(
    #     "subtask_completion",
    #     subtask_complete_func=subtask_block_condition,
    # )
    # print(out)
    # Negative Subtask Completion Check
    # restricted_init_states = []
    # for i in range(N):
    #     print("Test ", i)
    #     _, res, init_state = verify_spec(
    #         "subtask_completion",
    #         subtask_complete_func=subtask_block_condition,
    #         restricted_initial_states=restricted_init_states,
    #         negative_check=True
    #     )
    #     if not res:
    #         print("  Negative subtask completion failed.")
    #         break
    #     restricted_init_states.append(init_state)
    #     print("  Negative subtask completion holds.")
    
    # out, res, _ = verify_spec(
    #     "near_object",
    #     subtask_complete_func=subtask_block_condition,
    #     near_object_func=near_block_condition
    # )
    # print(out)
    # Negative Near Object Check
    restricted_init_states = []
    for i in range(N):
        print("Test ", i)
        _, res, init_state = verify_spec(
            "near_object",
            subtask_complete_func=subtask_block_condition,
            near_object_func=near_block_condition,
            restricted_initial_states=restricted_init_states,
            negative_check=True
        )
        if not res:
            print("  Negative near object failed.")
            break
        restricted_init_states.append(init_state)
        print("  Negative near object holds.")
    
    # out, res, _ = verify_spec(
    #     "subtask_completion",
    #     subtask_complete_func=subtask_goal_condition,
    # )
    # print(out)
    
    # out, res, _ = verify_spec(
    #     "near_object",
    #     subtask_complete_func=subtask_goal_condition,
    #     near_object_func=near_goal_condition
    # )
    # print(out)
    
    # out, res, _ = verify_spec(
    #     "subtask_completion",
    #     subtask_complete_func=subtask_goal_condition,
    #     env_name="FetchPickAndPlace2-v1",
    # )
    # print(out)
    
    # out, res, _ = verify_spec(
    #     "near_object",
    #     subtask_complete_func=subtask_goal_condition,
    #     near_object_func=near_goal_condition,
    #     env_name="FetchPickAndPlace2-v1",
    # )
    # print(out)
    
    # restricted_init_states = []
    # for i in range(N):
    #     print("Test ", i)
    #     out, res, init_state = verify_spec(
    #         "persistence",
    #         subtask_complete_func=[subtask_block_condition, subtask_goal_condition],
    #         T=25,
    #         restricted_initial_states=restricted_init_states
    #     )
    #     if not res:
    #         print("  Persistence failed.")
    #         for func in [subtask_block_condition, subtask_goal_condition]:
    #             out_func, res_func, _ = verify_spec(
    #                 "persistence",
    #                 subtask_complete_func=[func],
    #                 T=25,
    #                 restricted_initial_states=restricted_init_states
    #             )
    #             if not res_func:
    #                 print(f"  Persistence failed for function: {func.__name__}")
    #                 break
    #         break
    #     restricted_init_states.append(init_state)
    #     print("  Persistence holds.")
    #     print(f"  Mission goal: {MUJOCO_IDX_TO_COLOR[init_state[16].as_long()]}")
    #     print(f"  Block init state: {init_state[4].as_decimal(4)}, {init_state[5].as_decimal(4)}, {init_state[6].as_decimal(4)}")
    #     for i in range(NUM_GOALS):
    #         print(f"  Goal {i} position: {init_state[13][i].as_decimal(4)}, {init_state[14][i].as_decimal(4)}, {init_state[15][i].as_decimal(4)}")
    
    
    # restricted_init_states = []
    # for i in range(N):
    #     print("Test ", i)
    #     out, res, init_state = verify_spec(
    #         "persistence",
    #         subtask_complete_func=[subtask_block_condition, subtask_goal_condition],
    #         T=25,
    #         env_name="FetchPickAndPlace2-v1",
    #         restricted_initial_states=restricted_init_states
    #     )
    #     if not res:
    #         print("  Persistence failed.")
    #         for func in [subtask_block_condition, subtask_goal_condition]:
    #             out_func, res_func, _ = verify_spec(
    #                 "persistence",
    #                 subtask_complete_func=[func],
    #                 T=25,
    #                 env_name="FetchPickAndPlace2-v1",
    #                 restricted_initial_states=restricted_init_states
    #             )
    #             if not res_func:
    #                 print(f"  Persistence failed for function: {func.__name__}")
    #                 break
    #         break
    #     restricted_init_states.append(init_state)
    #     print("  Persistence holds.")
    #     print(f"  Mission goal: {MUJOCO_IDX_TO_COLOR[init_state[16].as_long()]}")
    #     print(f"  Block init state: {init_state[4].as_decimal(4)}, {init_state[5].as_decimal(4)}, {init_state[6].as_decimal(4)}")
    #     for i in range(NUM_GOALS):
    #         print(f"  Goal {i} position: {init_state[13][i].as_decimal(4)}, {init_state[14][i].as_decimal(4)}, {init_state[15][i].as_decimal(4)}")
        
    
    print(f"Total time: {time.time() - start_time} seconds")
        



