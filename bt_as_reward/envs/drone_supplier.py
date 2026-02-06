from pathlib import Path

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Wall, Door, Key, Box
from minigrid.core.constants import IDX_TO_COLOR, COLOR_NAMES


class BoxNonPickup(Box):
    def can_pickup(self):
        return False


class DroneSupplierEnv(MiniGridEnv):
    def __init__(
        self, obstacle_path: Path, size=50, max_steps: int | None = None, **kwargs
    ):
        if max_steps is None:
            max_steps = 4 * size**2
        mission_space = MissionSpace(
            mission_func=self._gen_mission, ordered_placeholders=[COLOR_NAMES] * 3
        )
        self.obstacles = np.load(obstacle_path)
        self.episode_num = 0
        super().__init__(
            mission_space=mission_space, width=size, height=size, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission(region_color: str, key_color: str, door_color: str) -> str:
        return f"open the {region_color} box, pick up the {key_color} key, then open the {door_color} door"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        rows, cols = 3, 2
        cell_width = (width - 2) / cols
        cell_height = (height - 2) / rows

        for obstacle in self.obstacles:
            x, y = obstacle
            self.grid.set(x, y, Wall())
        region = self._rand_elem(range(0, rows * cols))
        row = region // cols
        col = region % cols

        self.box = None
        colors = set(COLOR_NAMES)
        for i, region_tuple in enumerate(
            [
                (cell_width - 1, cell_height - 1),
                (0, cell_height - 1),
                (cell_width - 1, cell_height // 2 - 1),
                (0, cell_height // 2 - 1),
                (cell_width - 1, 0),
                (0, 0),
            ]
        ):
            region_row = i // cols
            region_col = i % cols
            box = BoxNonPickup(color=IDX_TO_COLOR[i % len(IDX_TO_COLOR)])
            if i == region:
                self.box = box
            self.place_obj(
                box,
                top=(
                    int(region_col * cell_width + 1 + region_tuple[0]),
                    int(region_row * cell_height + 1 + region_tuple[1]),
                ),
                size=(1, 1),
            )

        door_color = self._rand_elem(sorted(colors))
        colors.remove(door_color)
        self.door = Door(door_color, is_locked=True, is_open=False)
        self.place_obj(
            self.door,
            top=(int(col * cell_width + 2), int(row * cell_height + 2)),
            size=(cell_width - 2, cell_height - 2),
        )

        regions = set(range(0, rows * cols))
        regions.remove(region)
        # Place more doors
        for _ in range(rows * cols - 1):
            other_region = self._rand_elem(sorted(regions))
            regions.remove(other_region)
            other_row = other_region // cols
            other_col = other_region % cols
            other_door_color = self._rand_elem(sorted(colors))
            colors.remove(other_door_color)
            other_door = Door(other_door_color, is_locked=False, is_open=False)
            self.place_obj(
                other_door,
                top=(int(other_col * cell_width + 2), int(other_row * cell_height + 2)),
                size=(cell_width - 2, cell_height - 2),
            )

        self.place_agent(top=(width // 2, height // 2), size=(1, 1), rand_dir=False)

        self.agent_dir = 0
        self.box.contains = Key(door_color)

        self.mission = f"open the {IDX_TO_COLOR[region]} box, pick up the {door_color} key, then open the {door_color} door"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.toggle:
            if self.door.is_open:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info


class DroneSupplierSmallEnv(MiniGridEnv):
    
    def __init__(
        self, size=8, max_steps: int | None = None, **kwargs
    ):
        if max_steps is None:
            max_steps = 4 * size**2
        mission_space = MissionSpace(
            mission_func=self._gen_mission, ordered_placeholders=[COLOR_NAMES] * 3
        )
        self.episode_num = 0
        super().__init__(
            mission_space=mission_space, width=size, height=size, max_steps=max_steps, **kwargs
        )
    
    @staticmethod
    def _gen_mission(region_color: str, key_color: str, door_color: str) -> str:
        return f"open the {region_color} box, pick up the {key_color} key, then open the {door_color} door"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        self.np_random = self._np_random
        
        door_positions = [
            (1, 1), (6, 1),
            (1, 3), (6, 3),
            (1, 6), (6, 6)
        ]
        
        box_positions = [
            (3, 1), (4, 1),
            (3, 3), (4, 3),
            (3, 6), (4, 6)
        ]
        
        region = self._rand_elem(door_positions)

        self.box = None
        colors = set(COLOR_NAMES)
        for i in range(len(box_positions)):
            box = BoxNonPickup(color=IDX_TO_COLOR[i % len(IDX_TO_COLOR)])
            if box_positions[i][1] == region[1] and abs(box_positions[i][0] - region[0]) == 2:
                self.box = box
            self.place_obj(
                box,
                top=(
                    box_positions[i][0],
                    box_positions[i][1],
                ),
                size=(1, 1),
            )
        door_color = self._rand_elem(sorted(colors))
        colors.remove(door_color)
        self.door = Door(door_color, is_locked=True, is_open=False)
        self.place_obj(
            self.door,
            top=region,
            size=(1, 1),
        )

        regions = set(door_positions)
        regions.remove(region)
        # Place more doors
        for _ in range(len(door_positions) - 1):
            other_region = self._rand_elem(sorted(regions))
            regions.remove(other_region)
            other_door_color = self._rand_elem(sorted(colors))
            colors.remove(other_door_color)
            other_door = Door(other_door_color, is_locked=False, is_open=False)
            self.place_obj(
                other_door,
                top=other_region,
                size=(1, 1),
            )

        self.place_agent(top=(width // 2, height // 2), size=(1, 1), rand_dir=False)

        self.agent_dir = 0
        self.box.contains = Key(door_color)

        self.mission = f"open the {IDX_TO_COLOR[door_positions.index(region)]} box, pick up the {door_color} key, then open the {door_color} door"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.toggle:
            if self.door.is_open:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info
