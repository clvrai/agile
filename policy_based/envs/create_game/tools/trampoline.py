from .segment import LINE_THICKNESS
from .ramp import Ramp
import pygame as pg
import numpy as np
from .basic_obj import BasicObj
from .img_tool import ImageTool

TRAMPOLINE_ELASTICITY = 1.2


class Trampoline(Ramp):
    def __init__(self, pos, length, angle, elasticity=TRAMPOLINE_ELASTICITY, friction=1.0):
        if angle > np.pi/2:
            self.actual_angle = angle + np.pi
        else:
            self.actual_angle = angle

        self.angle = angle
        self.length = length
        super().__init__(pos, length, self.actual_angle, friction, elasticity)
        self.required_activator = 'Spring'


    def add_to_space(self, space):
        super().add_to_space(space, use_friction=False)
        self.img = ImageTool('trampoline.png', self.actual_angle,
                self.pos[:],
                debug_render=False,
                use_shape=self.shape)

    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1

        self.img.render(screen, scale, self.flipy)

    def set_settings(self, settings):
        self.settings = settings
        self.is_active = not settings.activator_tools
        self.shape.is_active = self.is_active
        self.shape.sensor = not self.is_active

    def activate(self):
        self.is_active = True
        self.shape.is_active = True
        self.shape.sensor = False


class ActiveTrampoline(Trampoline):
    def set_settings(self, settings):
        self.settings = settings
        self.is_active = True
        self.shape.is_active = True

# The sizes refer to the power of the bounce.
class MediumTrampoline(ActiveTrampoline):
    def __init__(self, pos, elasticity=TRAMPOLINE_ELASTICITY):
        super().__init__(pos, length=16.0, angle=0.0, elasticity=elasticity)
        self.color = 'purple'


class Trampoline45(ActiveTrampoline):
    def __init__(self, pos):
        super().__init__(pos, length=16.0, angle=np.pi/4)
        self.color = 'purple'


class ReverseTrampoline45(ActiveTrampoline):
    def __init__(self, pos):
        super().__init__(pos, length=16.0, angle=-np.pi/4)
        self.color = 'purple'

class VerticalTrampoline(ActiveTrampoline):
    def __init__(self, pos):
        super().__init__(pos, length=16.0, angle=np.pi/2)
        self.color = 'purple'
