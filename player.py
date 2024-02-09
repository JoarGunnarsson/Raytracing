from constants import *
import numpy as np
import scene_object


class Player(scene_object.Object):
    def __init__(self, x=0, y=0, z=0, viewing_direction=None):
        if viewing_direction is None:
            viewing_direction = np.array([1/2**0.5, 0, -1/2**0.5])
        super().__init__(x, y, z)
        self.viewing_direction = viewing_direction
        self.y_vector = np.array([0.1, 0, 0.97])
        if np.dot(viewing_direction, self.y_vector) != 0:
            orthogonal_vector = np.cross(viewing_direction, self.y_vector)
            self.y_vector = np.cross(orthogonal_vector, viewing_direction)
            self.y_vector = self.y_vector / np.linalg.norm(self.y_vector)
        screen_x, screen_y, screen_z = self.position + viewing_direction
        self.screen = Screen(screen_x, screen_y, screen_z, normal_vector=-viewing_direction, y_vector=self.y_vector)


class Screen(scene_object.Object):
    def __init__(self, x=1, y=0, z=0, normal_vector=np.array([1, 0, 0]),
                 y_vector=np.array([0, 0, 1]), width=1, height=1):
        super().__init__(x, y, z)
        self.width = width
        self.height = height
        self.pixels_x = WIDTH
        self.pixels_y = HEIGHT
        self.image = np.zeros((WIDTH, HEIGHT, 3))
        self.normal_vector = normal_vector
        self.y_vector = y_vector
        self.x_vector = np.cross(self.normal_vector, self.y_vector)

    def index_to_position(self, i, j):
        x = self.x_vector * (-self.width / 2 + i * self.width / self.pixels_x)
        y = self.y_vector * (-self.height / 2 + (self.pixels_y - j) * self.height / self.pixels_y)
        return x + y + self.position
