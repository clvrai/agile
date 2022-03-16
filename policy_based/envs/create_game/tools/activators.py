from .poly import FixedPoly
import pymunk
import pygame as pg
from pygame import gfxdraw
# from ..constants import goal_color
from .gravity_obj import MOVING_OBJ_COLLISION_TYPE
from .img_tool import ImageTool

GOAL_RADIUS = 4.0
GOAL_OBJ_COLLISION_TYPE = 15
ACTIVATOR_FIXED_SIZE = 5.0

FIRE_COLLISION_TYPE = 20
WATER_COLLISION_TYPE = 21
ELECTRIC_COLLISION_TYPE = 22
MAGNET_COLLISION_TYPE = 23
SPRING_COLLISION_TYPE = 24

class ActivatorObj(FixedPoly):
    def __init__(self, pos, type="Fire"):
        super().__init__(pos, n_sides=4, angle=0.0, size=ACTIVATOR_FIXED_SIZE,
                color='black')

        if type == 'Fire':
            self.img = ImageTool('fire.png',
                    angle=0.0,
                    pos=pos[:],
                    use_shape=self.shape,
                    debug_render=False,
                    opacity=0.7)
            self.collision_type = FIRE_COLLISION_TYPE
        elif type == 'Water':
            self.img = ImageTool('water.png',
                    angle=0.0,
                    pos=pos[:],
                    use_shape=self.shape,
                    debug_render=False,
                    opacity=0.7)
            self.collision_type = WATER_COLLISION_TYPE
        elif type == 'Electric':
            self.img = ImageTool('electric.png',
                    angle=0.0,
                    pos=pos[:],
                    use_shape=self.shape,
                    debug_render=False,
                    opacity=0.7)
            self.collision_type = ELECTRIC_COLLISION_TYPE
        elif type == 'Magnet':
            self.img = ImageTool('magnet.png',
                    angle=0.0,
                    pos=pos[:],
                    use_shape=self.shape,
                    debug_render=False,
                    opacity=0.7)
            self.collision_type = MAGNET_COLLISION_TYPE
        elif type == 'Spring':
            self.img = ImageTool('spring.png',
                    angle=0.0,
                    pos=pos[:],
                    use_shape=self.shape,
                    debug_render=False,
                    opacity=0.7)
            self.collision_type = SPRING_COLLISION_TYPE

        self.activator_type = type
        self.sensor = True
        self.shape.sensor = True
        self.shape.collision_type = self.collision_type
        self.shape.is_goal = True

    def add_to_space(self, space):
        super().add_to_space(space)

    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1

        self.img.render(screen, scale, self.flipy)
