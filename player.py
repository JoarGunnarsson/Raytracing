from constants import *
import numpy as np
import scene_object


class Player(scene_object.Object):
    def __init__(self, x=0, y=0, z=0, viewing_direction=None):
        if viewing_direction is None:
            viewing_direction = np.array([1, 0, 0])
        super().__init__(x, y, z)
        self.viewing_direction = viewing_direction
