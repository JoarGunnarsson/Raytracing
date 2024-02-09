import numpy as np

from constants import *
import scene_object
import materials

EPSILON = 0.001


class Sphere(scene_object.Object):
    def __init__(self, x=4, y=0, z=0, radius=1, material=materials.Material(YELLOW)):
        super().__init__(x, y, z)
        self.radius = radius
        self.material = material

    def normal_vector(self, intersection_point):
        normal_vector = intersection_point - self.position
        return normal_vector / np.linalg.norm(normal_vector)

    def small_normal_offset(self, position):
        return self.normal_vector(position) * EPSILON

    def compute_surface_color(self, intersection_point, direction_vector):
        #return self.material.color
        return self.material.compute_color(self.normal_vector(intersection_point), direction_vector)

    def intersection(self, starting_position, direction_vector):
        a = np.dot(direction_vector, direction_vector)
        b = - 2 * np.dot(direction_vector, self.position) + 2 * np.dot(starting_position, direction_vector)
        c = np.dot(self.position, self.position) + np.dot(starting_position, starting_position) - self.radius ** 2 - 2 * np.dot(starting_position, self.position)
        solution = solve_quadratic(a, b, c)
        if solution is not None:
            t1, t2 = solution
            t = min(t1, t2)
            if t < 0:
                t = max(t1, t2)
                if t < 0:
                    return False, None
            return True, t
        return False, None


class PointSource(scene_object.Object):
    def __init__(self, x=4, y=0, z=1000):
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
