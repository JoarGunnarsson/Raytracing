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
        self.position = np.array([x, y, z])


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
    def __init__(self, x=1, y=0, z=0, normal_vector=np.array([1, 0, 0]),
                 y_vector=np.array([0, 0, 1]), width=1, height=1):
        super().__init__(x, y, z)
        self.width = width
        self.height = height
        self.pixels_x = WIDTH
        self.pixels_y = HEIGHT
        self.image = np.zeros((HEIGHT, WIDTH, 3))
        self.normal_vector = normal_vector
        self.y_vector = y_vector
        self.x_vector = np.cross(self.normal_vector, self.y_vector)

    def index_to_position(self, X, Y):
        X = X * self.width / self.pixels_x - self.width / 2
        X = multiply_matrix_by_vector_elementwise(X, self.x_vector)

        Y = (self.pixels_y - Y) * self.height / self.pixels_y - self.height / 2
        Y = multiply_matrix_by_vector_elementwise(Y, self.y_vector)
        return X + Y + self.position


def multiply_matrix_by_vector_elementwise(A, v):
    A_height, A_width = A.shape
    A = A.reshape(-1, 1, 1)
    A = A * v
    return A.reshape(A_height, A_width, 3)


class Sphere(Object):
    def __init__(self, x=4, y=0, z=0, radius=1, material=materials.Material(YELLOW)):
        super().__init__(x, y, z)
        self.radius = radius
        self.material = material

    def normal_vector(self, intersection_points):
        normal_vector = intersection_points - self.position
        norms = np.linalg.norm(normal_vector)
        return normal_vector / norms

    def small_normal_offset(self, position):
        return self.normal_vector(position) * EPSILON

    def compute_surface_color(self, intersection_points, direction_vectors, light_vector_matrix):
        return self.material.compute_color(self.normal_vector(intersection_points), direction_vectors,
                                           light_vector_matrix)

    def intersection(self, starting_positions, direction_vectors):
        dot_product = np.sum(direction_vectors * starting_positions, axis=2)
        B = 2 * (dot_product - np.dot(direction_vectors, self.position))
        # TODO: Issue with starting_positions.
        difference_in_positions = self.position - starting_positions
        c = np.sum(difference_in_positions * difference_in_positions, axis=2) - self.radius ** 2
        C = np.full(B.shape, c)
        solutions = solve_quadratic(B, C)
        return solutions


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

    def compute_light_intensity(self, intersection_points, scene_objects):
        width, height, _ = intersection_points.shape
        intensities = np.zeros((height, width))
        light_vectors = self.position - intersection_points
        norms = np.linalg.norm(light_vectors, axis=-1, keepdims=True)
        light_vectors = light_vectors / norms
        obscuring_objects, _ = find_closes_intersected_object(intersection_points, light_vectors, scene_objects)

        obscured_indices = obscuring_objects != None
        intensities[obscured_indices] = 0

        non_obscured_indices = obscuring_objects == None
        distances = np.sum((intersection_points - self.position) * (intersection_points - self.position), axis=2)
        intensities[non_obscured_indices] = self.intensity / distances[non_obscured_indices]
        #print(obscuring_objects)
        #print()
        #print(np.array([[round(value, 3) for value in x] for x in intensities]))
        return intensities, [light_vectors]


class DiskSource(LightSource):
    def __init__(self, x=4, y=0, z=20):
        super().__init__(x, y, z)
        self.radius = 3
        self.intensity = 15
        self.n_points = 30
        self.normal_vector = np.array([0, 0, -1])

    def compute_light_intensity(self, intersection_point, scene_objects):
        total_intensity = 0

        if self.normal_vector[0] != 0 and self.normal_vector[1] == 0 and self.normal_vector[2] == 0:
            perpendicular_vector = np.array([0, 1, 0])
        else:
            perpendicular_vector = np.array([1, 0, 0])

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


def solve_quadratic(B, C):
    """Solves a special case quadratic equation with a = 1."""
    solutions = np.full(B.shape, None, dtype=object)

    discriminants = B ** 2 - 4 * C
    imaginary_solutions_indices = discriminants <= 0

    discriminants[imaginary_solutions_indices] = 0

    root_discriminant = discriminants ** 0.5
    x1 = -B / 2 + root_discriminant / 2
    x2 = -B / 2 - root_discriminant / 2

    minimum_solutions = np.minimum(x1, x2)
    maximum_solutions = np.maximum(x1, x2)
    min_ok_indices = minimum_solutions > 0
    solutions[min_ok_indices] = minimum_solutions[min_ok_indices]

    max_ok_indices = minimum_solutions.any() <= 0 < maximum_solutions.any()
    solutions[max_ok_indices] = maximum_solutions[max_ok_indices]

    negative_solutions_indices = minimum_solutions.any() <= 0 and maximum_solutions.any() <= 0
    solutions[negative_solutions_indices] = None
    solutions[imaginary_solutions_indices] = None
    return solutions


def find_closes_intersected_object(starting_positions, direction_vectors, object_list):
    height, width, _ = direction_vectors.shape
    min_t = np.zeros((height, width)) + np.inf
    closest_objects = np.full((height, width), None, dtype=Object)

    for obj in object_list:
        t = obj.intersection(starting_positions, direction_vectors)
        print(t)
        print()
        none_indices = t == None
        t[none_indices] = -1
        positive_indices = t > 0
        min_t[positive_indices] = np.minimum(min_t[positive_indices], t[positive_indices])

        closest_objects[positive_indices] = obj

    return closest_objects, min_t
