import numpy as np
import random
from constants import *
import materials
import math
EPSILON = 0.0001


class Object:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.position = np.array([x, y, z], dtype=float)


class Camera(Object):
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


class Screen(Object):
    def __init__(self, x=1, y=0, z=0, normal_vector=np.array([1.0, 0.0, 0.0]),
                 y_vector=np.array([0.0, 0.0, 1.0]), width=1, height=1):
        super().__init__(x, y, z)
        self.width = width
        self.height = height
        self.pixels_x = WIDTH
        self.pixels_y = HEIGHT
        self.image = np.zeros((HEIGHT, WIDTH, 3))
        self.normal_vector = normal_vector
        self.y_vector = y_vector
        self.x_vector = np.cross(self.normal_vector, self.y_vector)

    def index_to_position(self, i, j):
        x = self.x_vector * (-self.width / 2 + i * self.width / self.pixels_x)
        y = self.y_vector * (-self.height / 2 + (self.pixels_y - j) * self.height / self.pixels_y)
        return x + y + self.position


class Sphere(Object):
    def __init__(self, x=4, y=0, z=0, radius=1, material=materials.Material(YELLOW)):
        super().__init__(x, y, z)
        self.radius = radius
        self.material = material

    def normal_vector(self, intersection_point):
        normal_vector = intersection_point - self.position
        return normal_vector / np.linalg.norm(normal_vector)

    def small_normal_offset(self, position):
        return self.normal_vector(position) * EPSILON

    def compute_surface_color(self, intersection_point, direction_vector, light_direction_vector):
        return self.material.compute_color(self.normal_vector(intersection_point), direction_vector,
                                           light_direction_vector)

    def intersection(self, starting_position, direction_vector):
        b = 2 * (np.dot(starting_position, direction_vector) - np.dot(direction_vector, self.position))
        c = np.linalg.norm(self.position - starting_position) ** 2 - self.radius ** 2
        solution = solve_quadratic(b, c)

        if solution is None:
            return None

        t1, t2 = solution
        if t1 < 0 and t2 < 0:
            return None
        elif t1 < 0:
            return t2
        elif t2 < 0:
            return t1
        return min(t1, t2)


class LightSource(Object):
    def __init__(self, x=4, y=0, z=1000):
        super().__init__(x, y, z)
        self.intensity = 1

    def compute_light_intensity(self, *args, **kwargs):
        """Virtual method to be implemented in child classes."""
        pass


class PointSource(LightSource):
    def __init__(self, x=4, y=0, z=20):
        super().__init__(x, y, z)
        self.intensity = 15

    def compute_light_intensity(self, intersection_point, scene_objects):
        light_vector = self.position - intersection_point
        light_vector = light_vector / np.linalg.norm(light_vector)
        obscuring_object, _ = find_closes_intersected_object(intersection_point, light_vector, scene_objects)
        if obscuring_object is not None:
            return 0, [light_vector]
        distance = np.linalg.norm(intersection_point - self.position)
        return self.intensity / distance**2, [light_vector]


class DiskSource(LightSource):
    def __init__(self, x=4, y=0, z=20):
        super().__init__(x, y, z)
        self.radius = 3
        self.intensity = 15
        self.n_points = 30
        self.normal_vector = np.array([0.0, 0.0, -1.0])

    def compute_light_intensity(self, intersection_point, scene_objects):
        total_intensity = 0

        if self.normal_vector[0] != 0 and self.normal_vector[1] == 0 and self.normal_vector[2] == 0:
            perpendicular_vector = np.array([0.0, 1.0, 0.0])
        else:
            perpendicular_vector = np.array([1.0, 0.0, 0.0])

        x_hat = np.cross(self.normal_vector, perpendicular_vector)

        y_hat = np.cross(self.normal_vector, x_hat)
        light_vectors = []
        for i in range(self.n_points):
            theta = random.random() * 2 * math.pi
            d = random.random() * self.radius
            random_light_point = self.position + d ** 0.5 * (math.cos(theta) * x_hat + math.sin(theta) * y_hat)
            light_vector = random_light_point - intersection_point
            light_vector = light_vector / np.linalg.norm(light_vector)
            obscuring_object, _ = find_closes_intersected_object(intersection_point, light_vector, scene_objects)
            if obscuring_object is not None:
                intensity = 0
            else:
                intensity = self.intensity / self.n_points
            distance = np.linalg.norm(intersection_point - self.position)
            total_intensity += intensity / distance**2
            light_vectors.append(light_vector)

        return total_intensity, light_vectors


def solve_quadratic(b, c):
    """Solves a special case quadratic equation with a = 1."""
    discriminant = b ** 2 - 4 * c
    if discriminant <= 0:
        return None

    root_discriminant = discriminant ** 0.5
    x1 = -b / 2 + root_discriminant / 2
    x2 = -b / 2 - root_discriminant / 2

    return x1, x2


def find_closes_intersected_object(starting_position, direction_vector, object_list):
    min_t = np.inf
    closest_object = None
    for obj in object_list:
        t = obj.intersection(starting_position, direction_vector)
        if t is None:
            continue
        if t <= min_t:
            min_t = t
            closest_object = obj

    return closest_object, min_t
