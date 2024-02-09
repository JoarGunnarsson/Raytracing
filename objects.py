import numpy as np

from constants import *
import scene_object


EPSILON = 0.001


class Sphere(scene_object.Object):
    def __init__(self, x=4, y=0, z=0, radius=1, color=YELLOW):
        super().__init__(x, y, z)
        self.radius = radius
        self.color = color

    def small_normal_offset(self, position):
        normal_vector = position - self.position
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        return normal_vector * EPSILON

    def intersection(self, starting_position, direction_vector):
        a = np.dot(direction_vector, direction_vector)
        b = - 2 * (direction_vector[0] * self.position[0] + direction_vector[1] * self.position[1] + direction_vector[
            2] * self.position[2]) + 2 * (starting_position[0] * direction_vector[0] + starting_position[1] * direction_vector[1] + starting_position[2] * direction_vector[2])
        c = np.dot(self.position, self.position) + np.dot(starting_position, starting_position) - self.radius ** 2 - 2 * (starting_position[0] * self.position[0] + starting_position[1] * self.position[1] + starting_position[2] * self.position[2])
        solution = solve_quadratic(a, b, c)
        if solution is not None:
            t1, t2 = solution
            t = min(t1, t2)
            if t < 0:
                t = max(t1, t2)
                if t < 0:
                    return False
            return t
        return False


class PointSource(scene_object.Object):
    def __init__(self, x=0, y=0, z=10):
        super().__init__(x, y, z)
        self.radius = 10


def solve_quadratic(a, b, c):
    if a == 0 and b == 0:
        return None

    if (b ** 2 - 4 * a * c) < 0:
        return None

    if a == 0:
        return - c / b

    x1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
    x2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)

    return x1, x2
