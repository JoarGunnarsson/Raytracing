from constants import *
import materials as materials
import math
import numpy as np


class Object:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.position = np.array([x, y, z], dtype=float)


class Camera(Object):
    def __init__(self, x=0, y=0, z=0, viewing_direction=None):
        super().__init__(x, y, z)
        if viewing_direction is None:
            viewing_direction = np.array([1 / 2 ** 0.5, 0, -1 / 2 ** 0.5])
        viewing_direction /= np.linalg.norm(viewing_direction)
        self.viewing_direction = viewing_direction
        self.y_vector = np.array([0.1, 0, 0.97])
        self.y_vector /= np.linalg.norm(self.y_vector)
        if np.dot(viewing_direction, self.y_vector) != 0:
            orthogonal_vector = np.cross(viewing_direction, self.y_vector)
            self.y_vector = np.cross(orthogonal_vector, viewing_direction)
            self.y_vector = self.y_vector / np.linalg.norm(self.y_vector)
        screen_x, screen_y, screen_z = self.position + viewing_direction
        self.screen = Screen(screen_x, screen_y, screen_z, normal_vector=-viewing_direction, y_vector=self.y_vector)


class Screen(Object):
    def __init__(self, x=1, y=0, z=0, normal_vector=np.array([1, 0, 0]),
                 y_vector=np.array([0, 0, 1]), width=1):
        super().__init__(x, y, z)
        self.width = width
        self.height = width * HEIGHT / WIDTH
        self.pixels_x = float(WIDTH)
        self.pixels_y = float(HEIGHT)
        self.image = np.zeros((HEIGHT, WIDTH, 3), dtype=float)
        self.normal_vector = normal_vector
        self.y_vector = y_vector
        self.x_vector = np.cross(self.normal_vector, self.y_vector)

    def index_to_position(self, X, Y):
        X = X * float(self.width) / self.pixels_x - float(self.width) / 2
        X = X * self.x_vector

        Y = (self.pixels_y - Y) * float(self.height) / self.pixels_y - float(self.height) / 2
        Y = Y * self.y_vector
        return X + Y + self.position


class Sphere(Object):
    def __init__(self, x=4, y=0, z=0, radius=1.0, material=materials.Material(colors.YELLOW)):
        super().__init__(x, y, z)
        self.radius = radius
        self.material = material

    def intersection(self, starting_positions, direction_vectors, mode="first"):
        dot_product = np.sum(direction_vectors * starting_positions, axis=-1)
        B = 2 * (dot_product - np.dot(direction_vectors, self.position))
        difference_in_positions = self.position - starting_positions
        C = np.sum(difference_in_positions * difference_in_positions, axis=-1) - self.radius ** 2
        solutions = solve_quadratic(B, C, mode)
        return solutions

    def get_normal_vectors(self, intersection_points):
        normal_vectors = intersection_points - self.position
        return normal_vectors / np.linalg.norm(normal_vectors, axis=-1, keepdims=True)


class Plane(Object):
    def __init__(self, x=4, y=0, z=0, v1=np.array([1.0, 0.0, 0.0]), v2=np.array([0.0, 1.0, 0.0]),
                 material=materials.Material(colors.WHITE)):
        super().__init__(x, y, z)
        self.v1 = v1
        self.v2 = v2
        self.normal_vector = np.cross(v1, v2)
        self.normal_vector = self.normal_vector / np.linalg.norm(self.normal_vector)
        self.material = material

    def intersection(self, starting_positions, direction_vectors, mode="first"):
        distances = -np.ones(starting_positions.shape[0])
        shifted_points = starting_positions - self.position
        distances_to_start = np.sum(shifted_points * self.normal_vector, axis=-1)
        direction_dot_normal = np.sum(direction_vectors * -self.normal_vector, axis=-1)
        non_perpendicular_indices = direction_dot_normal != 0
        distances[non_perpendicular_indices] = distances_to_start[non_perpendicular_indices] / direction_dot_normal[non_perpendicular_indices]
        return distances

    def get_normal_vectors(self, intersection_points):
        normal_vectors = np.full(intersection_points.shape, self.normal_vector)
        return normal_vectors


class LightSource(Object):
    def __init__(self, x=4.0, y=0.0, z=1000.0, intensity=1.05):
        super().__init__(x, y, z)
        self.intensity = intensity
        self.ambient_intensity = self.intensity / 50
        self.normal_vector = np.array([0.0, 0.0, -1.0])
        self.ambient_color = colors.WHITE
        self.diffuse_color = colors.WHITE
        self.specular_color = colors.WHITE

    def compute_light_intensity(self, *args, **kwargs):
        """Virtual method to be implemented in child classes."""
        pass


class AmbientLight:
    def __init__(self, intensity=0, color=colors.WHITE):
        self.intensity = intensity
        self.color = color


class PointSource(LightSource):
    def __init__(self, x=4.0, y=0.0, z=20.0, intensity=15.0):
        super().__init__(x, y, z, intensity=intensity)

    def compute_light_intensity(self, intersection_points, scene_objects):
        light_vectors = self.position - intersection_points

        return self.intensities_from_vectors(intersection_points, light_vectors, scene_objects)

    def intensities_from_vectors(self, intersection_points, light_vectors, scene_objects):
        size = intersection_points.shape[0]

        norms = np.linalg.norm(light_vectors, axis=-1, keepdims=True)
        light_vectors = light_vectors / norms
        multiplier = shadow_objects_multipliers(intersection_points, light_vectors, scene_objects)

        distances = norms.reshape(size)

        diffuse_intensities = self.diffuse_color * self.intensity / distances[:, None] ** 2 * multiplier
        specular_intensities = self.specular_color * self.intensity / distances[:, None] ** 2 * multiplier
        return np.clip(diffuse_intensities, 0, 1), np.clip(specular_intensities, 0, 1), light_vectors[None, :, :]


class DiskSource(LightSource):
    def __init__(self, x=4.0, y=0.0, z=20.0, radius=3.0, intensity=15.0, N=30):
        super().__init__(x, y, z, intensity=intensity)
        self.radius = radius
        self.n_points = N

    def compute_light_intensity(self, intersection_points, scene_objects):
        size = intersection_points.shape[0]

        extended_intersection_points = np.tile(intersection_points, (self.n_points, 1))

        if self.normal_vector[0] != 0 and self.normal_vector[1] == 0 and self.normal_vector[2] == 0:
            perpendicular_vector = np.array([0.0, 1.0, 0.0])
        else:
            perpendicular_vector = np.array([1.0, 0.0, 0.0])

        x_hat = np.cross(self.normal_vector, perpendicular_vector)
        y_hat = np.cross(self.normal_vector, x_hat)

        theta = np.random.random((self.n_points * size)) * 2 * math.pi
        d = np.random.random((self.n_points * size)) ** 0.5 * self.radius

        random_point_local = d[:, None] * (np.cos(theta)[:, None] * x_hat + np.sin(theta)[:, None] * y_hat)
        random_light_point = self.position + random_point_local
        light_vectors = random_light_point - extended_intersection_points
        return self.intensities_from_vectors(extended_intersection_points, light_vectors, scene_objects)

    def intensities_from_vectors(self, intersection_points, light_vectors, scene_objects):
        size = int(intersection_points.shape[0] / self.n_points)
        point_source = PointSource(intensity=self.intensity / self.n_points)
        diffuse_intensities, specular_intensities, light_vectors_matrix = point_source.intensities_from_vectors(
            intersection_points, light_vectors, scene_objects)

        diffuse_intensities = diffuse_intensities.reshape((self.n_points, size, 3))
        diffuse_intensities = np.sum(diffuse_intensities, axis=0) / self.n_points

        specular_intensities = specular_intensities.reshape((self.n_points, size, 3))
        specular_intensities = np.sum(specular_intensities, axis=0) / self.n_points

        light_vectors_matrix = light_vectors_matrix.reshape((self.n_points, size, 3))

        return np.clip(diffuse_intensities, 0, 1), np.clip(specular_intensities, 0, 1), light_vectors_matrix


class EasingModes:
    NONE = "none"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    EXPONENTIAL = "exponential"


class DirectionalDiskSource(DiskSource):
    def __init__(self, x=4, y=0, z=20, radius=3, intensity=15, angle=30, easing_mode=EasingModes.QUADRATIC):
        super().__init__(x, y, z, radius=radius, intensity=intensity)
        self.angle = angle * np.pi / 180
        self.fall_off_angle = 20 * np.pi / 180
        self.easing_mode = easing_mode
        self.n_points = 30
        if angle == 90:
            print("Using a directional disk source with an angle of 90 degrees is not recommended. "
                  "Use DiskSource instead.")

    def ease_fall_off_beam(self, x, a, d):
        eased_matrix = np.ones(x.shape, dtype=float)
        valid_indices = d != 0
        if self.easing_mode == EasingModes.LINEAR:
            eased_matrix[valid_indices] = linear_easing(x[valid_indices], a[valid_indices], d[valid_indices])

        elif self.easing_mode == EasingModes.QUADRATIC:
            eased_matrix[valid_indices] = quadratic_easing(x[valid_indices], a[valid_indices], d[valid_indices])

        elif self.easing_mode == EasingModes.CUBIC:
            eased_matrix[valid_indices] = cubic_easing(x[valid_indices], a[valid_indices], d[valid_indices])

        elif self.easing_mode == EasingModes.EXPONENTIAL:
            eased_matrix[valid_indices] = exponential_easing(x[valid_indices], a[valid_indices], d[valid_indices])

        return eased_matrix

    def compute_light_intensity(self, intersection_points, scene_objects):
        size = intersection_points.shape[0]

        if self.normal_vector[0] != 0 and self.normal_vector[1] == 0 and self.normal_vector[2] == 0:
            perpendicular_vector = np.array([0.0, 1.0, 0.0])
        else:
            perpendicular_vector = np.array([1.0, 0.0, 0.0])

        x_hat = np.cross(self.normal_vector, perpendicular_vector)
        y_hat = np.cross(self.normal_vector, x_hat)

        x = np.sum(x_hat * (intersection_points - self.position), axis=-1)
        y = np.sum(y_hat * (intersection_points - self.position), axis=-1)
        z = np.sum(self.normal_vector * (intersection_points - self.position), axis=-1)

        distance_from_normal_axis = (x ** 2 + y ** 2) ** 0.5
        allowed_distance = self.radius + np.tan(self.angle) * np.abs(z)
        distance_to_edge_of_fall_off_region = self.radius + np.tan(self.angle + self.fall_off_angle) * np.abs(z)
        allowed_fall_off_distance = distance_to_edge_of_fall_off_region - allowed_distance

        inside_light_beam_collapsed_indices = distance_from_normal_axis <= distance_to_edge_of_fall_off_region
        inside_light_beam_indices = np.tile(inside_light_beam_collapsed_indices, self.n_points)

        diffuse_intensities, specular_intensities, light_vectors_matrix = super().compute_light_intensity(
            intersection_points[inside_light_beam_collapsed_indices], scene_objects)

        x = distance_from_normal_axis[inside_light_beam_collapsed_indices]
        d = allowed_fall_off_distance[inside_light_beam_collapsed_indices]
        a = allowed_distance[inside_light_beam_collapsed_indices]
        fall_off_factors = self.ease_fall_off_beam(x, a, d)[:, None]

        total_diffuse_intensities = np.zeros((size, 3))
        total_diffuse_intensities[inside_light_beam_collapsed_indices] = diffuse_intensities * fall_off_factors

        total_specular_intensities = np.zeros((size, 3))
        total_specular_intensities[inside_light_beam_collapsed_indices] = specular_intensities * fall_off_factors

        light_vectors = light_vectors_matrix.reshape(-1, 3)
        total_light_vectors = np.zeros((size * self.n_points, 3))
        total_light_vectors[inside_light_beam_indices] = light_vectors
        total_light_vectors = total_light_vectors.reshape((self.n_points, size, 3))

        return np.clip(total_diffuse_intensities, 0, 1), np.clip(total_specular_intensities, 0, 1), total_light_vectors


def solve_quadratic(B, C, mode):
    """Solves a special case quadratic equation with a = 1. Returns one solution, determined by 'mode'."""
    solutions = -np.ones(B.shape, dtype=float)

    discriminants = B ** 2 - 4 * C
    real_solution_indices = discriminants > 0

    root_discriminant = discriminants[real_solution_indices] ** 0.5
    B = B[real_solution_indices]
    x1 = -B / 2 + root_discriminant / 2
    x2 = -B / 2 - root_discriminant / 2

    minimum_solutions = np.minimum(x1, x2)
    maximum_solutions = np.maximum(x1, x2)
    valid_solutions = solutions[real_solution_indices]
    if mode == "first":
        max_ok_indices = 0 < maximum_solutions
        valid_solutions[max_ok_indices] = maximum_solutions[max_ok_indices]

        min_ok_indices = 0 < minimum_solutions
        valid_solutions[min_ok_indices] = minimum_solutions[min_ok_indices]

    elif mode == "second":
        min_ok_indices = 0 < minimum_solutions
        valid_solutions[min_ok_indices] = minimum_solutions[min_ok_indices]

        max_ok_indices = 0 < maximum_solutions
        valid_solutions[max_ok_indices] = maximum_solutions[max_ok_indices]

    else:
        raise ValueError("Not a valid mode.")

    solutions[real_solution_indices] = valid_solutions
    return solutions


def shadow_objects_multipliers(starting_positions, direction_vectors, object_list):
    size, _ = direction_vectors.shape
    multiplier = np.ones((size, 3))
    for i, obj in enumerate(object_list):
        first_t = obj.intersection(starting_positions, direction_vectors, mode="first")
        second_t = obj.intersection(starting_positions, direction_vectors, mode="second")
        ok_indices = second_t > 0
        distance_travelled_through_object = second_t[ok_indices] - first_t[ok_indices]
        attenuation_factor = obj.material.transparency_coefficient * np.exp(-obj.material.attenuation_coefficient * obj.material.absorption_color * distance_travelled_through_object[:, None])
        multiplier[ok_indices] *= attenuation_factor

    return multiplier


def find_closest_intersected_object(starting_positions, direction_vectors, object_list):
    size, _ = direction_vectors.shape
    min_t = np.full(size, np.inf)
    closest_objects = np.full(size, -1)
    for i, obj in enumerate(object_list):
        t = obj.intersection(starting_positions, direction_vectors, "first")
        positive_indices = t > 0
        min_t[positive_indices] = np.minimum(min_t[positive_indices], t[positive_indices])
        new_closest_object_found_indices = min_t == t
        closest_objects[new_closest_object_found_indices] = i

    return closest_objects, min_t[:, None]


def linear_easing(x, a, d):
    return np.minimum(np.maximum((a - x) / d + 1, 0), 1)


def quadratic_easing(x, a, d):
    before_easing_starts_indices = x < a
    after_easing_stops_indices = x > a + d
    easing_indices = np.logical_not(np.logical_or(before_easing_starts_indices, after_easing_stops_indices))

    result = np.zeros(x.shape, dtype=float)

    x = x[easing_indices]
    a = a[easing_indices]
    d = d[easing_indices]
    result[easing_indices] = (x - a + d) * (1 / d - (x - a) / d ** 2)

    result[before_easing_starts_indices] = 1
    return result


def cubic_easing(x, a, d):
    before_easing_starts_indices = x < a
    after_easing_stops_indices = x > a + d
    easing_indices = np.logical_not(np.logical_or(before_easing_starts_indices, after_easing_stops_indices))
    result = np.zeros(x.shape, dtype=float)
    x = x[easing_indices]
    a = a[easing_indices]
    d = d[easing_indices]
    q = (4 * a + 3 * d) / d ** 3
    p = (1 - q * d ** 2) / (d ** 2 * (d + 2 * a))
    result[easing_indices] = 1 - (x - a) * (p * (x ** 2 - (a + d) ** 2) + q * (x - (a + d)) + 1 / d)
    result[before_easing_starts_indices] = 1
    return result


def exponential_easing(x, a, d):
    return 1 - 1 / (1 + np.exp(-10 / d * (x - a - d / 2)))
