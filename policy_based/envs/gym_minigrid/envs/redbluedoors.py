from policy_based.envs.gym_minigrid.minigrid import *
from policy_based.envs.gym_minigrid.register import register

class RedBlueDoorEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8):
        self.size = size

        super().__init__(
            width=2*size,
            height=size,
            max_steps=10*size*size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the grid walls
        self.grid.wall_rect(0, 0, 2*self.size, self.size)
        self.grid.wall_rect(self.size//2, 0, self.size, self.size)

        # Place the agent in the top-left corner
        self.place_agent(top=(self.size//2, 0), size=(self.size, self.size))

        # Add a red door at a random position in the left wall
        pos = self._rand_int(1, self.size - 1)
        self.red_door = Door("red")
        self.grid.set(self.size//2, pos, self.red_door)

        # Add a blue door at a random position in the right wall
        pos = self._rand_int(1, self.size - 1)
        self.blue_door = Door("blue")
        self.grid.set(self.size//2 + self.size - 1, pos, self.blue_door)

        # Generate the mission string
        self.mission = "open the red door then the blue door"

    def step(self, action):
        red_door_opened_before = self.red_door.is_open
        blue_door_opened_before = self.blue_door.is_open

        obs, reward, done, info = MiniGridEnv.step(self, action)

        red_door_opened_after = self.red_door.is_open
        blue_door_opened_after = self.blue_door.is_open

        if blue_door_opened_after:
            if red_door_opened_before:
                reward = self._reward()
                done = True
            else:
                reward = 0
                done = True

        elif red_door_opened_after:
            if blue_door_opened_before:
                reward = 0
                done = True

        return obs, reward, done, info

class RedBlueDoorEnv6x6(RedBlueDoorEnv):
    def __init__(self):
        super().__init__(size=6)

register(
    id='MiniGrid-RedBlueDoors-6x6-v0',
    entry_point='envs.gym_minigrid.envs:RedBlueDoorEnv6x6'
)

register(
    id='MiniGrid-RedBlueDoors-8x8-v0',
    entry_point='envs.gym_minigrid.envs:RedBlueDoorEnv'
)
