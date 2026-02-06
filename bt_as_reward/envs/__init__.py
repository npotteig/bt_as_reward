from gymnasium import register

pick_and_place_2_kwargs = {
    "reward_type": "sparse",
    "num_blocks": 1,
    "num_goals": 2,
    "target_in_the_air": True,
}

pick_and_place_kwargs = {
    "reward_type": "sparse",
    "num_blocks": 1,
    "num_goals": 1,
    "target_in_the_air": True,
}

register(
    id="FetchPickAndPlace2-v1",
    entry_point="bt_as_reward.envs.fetch_pickandplace_env:MujocoFetchPickAndPlaceEnv",
    kwargs=pick_and_place_2_kwargs,
    max_episode_steps=50,
)

register(
    id="FetchPickAndPlace-v5",
    entry_point="bt_as_reward.envs.fetch_pickandplace_env:MujocoFetchPickAndPlaceEnv",
    kwargs=pick_and_place_kwargs,
    max_episode_steps=50,
)

register(
    id="DroneSupplier-v0",
    entry_point="bt_as_reward.envs.drone_supplier:DroneSupplierEnv",
    kwargs={"obstacle_path": "data/drone_obstacles.npy", "size": 24},
    max_episode_steps=500,
)
