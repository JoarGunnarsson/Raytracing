import numpy as np
from constants import *
import materials
import math


class Object:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.position = np.array([x, y, z], dtype=float)


class Camera(Object):
    def __init__(self, x=0, y=0, z=0, viewing_direction=None):
        if viewing_direction is None:
            viewing_direction = np.array([1 / 2 ** 0.5, 0, -1 / 2 ** 0.5])
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
    def __init__(self, x=1, y=0, z=0, normal_vector=np.array([1, 0, 0]),
                 y_vector=np.array([0, 0, 1]), width=3, height=1):
        super().__init__(x, y, z)
        self.width = width
        self.height = width * HEIGHT / WIDTH
        self.pixels_x = WIDTH
        self.pixels_y = HEIGHT
        self.image = np.zeros((HEIGHT, WIDTH, 3))
        self.normal_vector = normal_vector
        self.y_vector = y_vector
        self.x_vector = np.cross(self.normal_vector, self.y_vector)

    def index_to_position(self, X, Y):
        X = X * self.width / self.pixels_x - self.width / 2
        X = X * self.x_vector

        Y = (self.pixels_y - Y) * self.height / self.pixels_y - self.height / 2
        Y = Y * self.y_vector
        return X + Y + self.position


class Sphere(Object):
    def __init__(self, x=4, y=0, z=0, radius=1, material=materials.Material(YELLOW)):
        super().__init__(x, y, z)
        self.radius = radius
        self.material = material

    def intersection(self, starting_positions, direction_vectors):
        # TODO: Input vectors can be inf and nan, if we are checking a reflection... Perhaps this does not need to be
        # computed etc.
        dot_product = np.sum(direction_vectors * starting_positions, axis=-1)
        B = 2 * (dot_product - np.dot(direction_vectors, self.position))
        difference_in_positions = self.position - starting_positions
        C = np.sum(difference_in_positions * difference_in_positions, axis=-1) - self.radius ** 2
        solutions = solve_quadratic(B, C)
        return solutions


class LightSource(Object):
    def __init__(self, x=4, y=0, z=1000, intensity=15):
        super().__init__(x, y, z)
        self.intensity = intensity
        self.normal_vector = np.array([0.0, 0.0, -1.0])

    def compute_light_intensity(self, *args, **kwargs):
        """Virtual method to be implemented in child classes."""
        pass


class PointSource(LightSource):
    def __init__(self, x=4, y=0, z=20, intensity=15, angle=90):
        super().__init__(x, y, z, intensity=intensity)

    def compute_light_intensity(self, intersection_points, scene_objects):
        size, _ = intersection_points.shape
        intensities = np.zeros(size)

        light_vectors = self.position - intersection_points
        norms = np.linalg.norm(light_vectors, axis=-1, keepdims=True)
        light_vectors = light_vectors / norms
        obscuring_objects, _ = find_closest_intersected_object(intersection_points, light_vectors, scene_objects)

        obscured_indices = obscuring_objects != -1
        intensities[obscured_indices] = 0

        non_obscured_indices = obscuring_objects == -1
        distances = norms.reshape(size)

        intensities[non_obscured_indices] = self.intensity / distances[non_obscured_indices] ** 2
        return intensities, [light_vectors]


class DiskSource(LightSource):
    def __init__(self, x=4, y=0, z=20, radius=3, intensity=15, angle=90):
        super().__init__(x, y, z, intensity=intensity)
        self.radius = radius
        self.n_points = 30
        self.angle = angle * np.pi / 180
        self.fall_off_angle = 1 * np.pi / 180

    def compute_light_intensity(self, intersection_points, scene_objects):
        # TODO: Add fall_off_angle
        size, _ = intersection_points.shape
        total_intensities = np.zeros(size)

        if self.normal_vector[0] != 0 and self.normal_vector[1] == 0 and self.normal_vector[2] == 0:
            perpendicular_vector = np.array([0.0, 1.0, 0.0])
        else:
            perpendicular_vector = np.array([1.0, 0.0, 0.0])

        x_hat = np.cross(self.normal_vector, perpendicular_vector)
        y_hat = np.cross(self.normal_vector, x_hat)

        x = np.sum(x_hat * (intersection_points - self.position), axis=-1)
        y = np.sum(y_hat * (intersection_points - self.position), axis=-1)
        z = np.sum(self.normal_vector * (intersection_points - self.position), axis=-1)
        distance_from_normal_axis = (x ** 2 + y ** 2)**0.5
        allowed_distance = self.radius + np.tan(self.angle) * z
        light_vectors_matrix = []
        for i in range(self.n_points):
            theta = np.random.random(size) * 2 * math.pi
            d = np.random.random(size) ** 0.5 * self.radius

            random_point_local = d[:, None] * (np.cos(theta)[:, None] * x_hat + np.sin(theta)[:, None] * y_hat)
            random_light_point = self.position + random_point_local
            light_vectors = random_light_point - intersection_points
            norms = np.linalg.norm(light_vectors, axis=-1, keepdims=True)
            light_vectors = light_vectors / norms
            obscuring_objects, _ = find_closest_intersected_object(intersection_points, light_vectors, scene_objects)

            non_obscured_indices = obscuring_objects == -1

            distances = norms.reshape(size)
            modifier = np.ones(intersection_points[non_obscured_indices].shape[0])

            if self.angle != np.deg2rad(90):
                not_ok_indices = distance_from_normal_axis[non_obscured_indices] > allowed_distance[non_obscured_indices]
                modifier[not_ok_indices] = 0

            intensities = modifier * self.intensity / self.n_points / distances[non_obscured_indices] ** 2

            total_intensities[non_obscured_indices] += intensities

            light_vectors_matrix.append(light_vectors)

        return total_intensities, light_vectors_matrix


def solve_quadratic(B, C):
    """Solves a special case quadratic equation with a = 1."""
    solutions = -np.ones(B.shape)

    discriminants = B ** 2 - 4 * C
    real_solution_indices = 0 <= discriminants

    root_discriminant = discriminants[real_solution_indices] ** 0.5
    B = B[real_solution_indices]
    x1 = -B / 2 + root_discriminant / 2
    x2 = -B / 2 - root_discriminant / 2

    minimum_solutions = np.minimum(x1, x2)
    maximum_solutions = np.maximum(x1, x2)

    valid_solutions = solutions[real_solution_indices]
    max_ok_indices = 0 < maximum_solutions
    valid_solutions[max_ok_indices] = maximum_solutions[max_ok_indices]

    min_ok_indices = 0 < minimum_solutions
    valid_solutions[min_ok_indices] = minimum_solutions[min_ok_indices]

    solutions[real_solution_indices] = valid_solutions
    return solutions


def find_closest_intersected_object(starting_positions, direction_vectors, object_list):
    size, _ = direction_vectors.shape
    min_t = np.full(size, np.inf)
    closest_objects = np.full(size, -1)
    for i, obj in enumerate(object_list):
        t = obj.intersection(starting_positions, direction_vectors)
        positive_indices = t > 0
        min_t[positive_indices] = np.minimum(min_t[positive_indices], t[positive_indices])
        new_closest_object_found_indices = min_t == t
        closest_objects[new_closest_object_found_indices] = i

    return closest_objects, min_t[:, None]
