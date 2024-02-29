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
                 y_vector=np.array([0, 0, 1]), width=1):
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
    def __init__(self, x=4, y=0, z=0, radius=1.0, material=materials.Material(YELLOW)):
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

    def exit_point(self, starting_positions, direction_vectors):
        # TODO: Input vectors can be inf and nan, if we are checking a reflection... Perhaps this does not need to be
        # computed etc.
        dot_product = np.sum(direction_vectors * starting_positions, axis=-1)
        B = 2 * (dot_product - np.dot(direction_vectors, self.position))
        difference_in_positions = self.position - starting_positions
        C = np.sum(difference_in_positions * difference_in_positions, axis=-1) - self.radius ** 2
        solutions = solve_quadratic_2(B, C)
        return solutions


class LightSource(Object):
    def __init__(self, x=4, y=0, z=1000, intensity=15):
        super().__init__(x, y, z)
        self.intensity = intensity
        self.normal_vector = np.array([0.0, 0.0, -1.0])
        self.diffuse_color = WHITE.copy()
        self.specular_color = WHITE.copy()

    def compute_light_intensity(self, *args, **kwargs):
        """Virtual method to be implemented in child classes."""
        pass


class PointSource(LightSource):
    def __init__(self, x=4, y=0, z=20, intensity=15):
        super().__init__(x, y, z, intensity=intensity)
        self.orb = Sphere(x, y, z, 0.1, material=materials.Material(diffuse_color=WHITE))

    def compute_light_intensity(self, intersection_points, scene_objects):
        size = intersection_points.shape[0]
        diffuse_intensities = np.zeros((size, 3))
        specular_intensities = np.zeros((size, 3))

        light_vectors = self.position - intersection_points
        norms = np.linalg.norm(light_vectors, axis=-1, keepdims=True)
        light_vectors = light_vectors / norms
        obscuring_objects, distances_to_intersections = find_closest_intersected_object(intersection_points, light_vectors, scene_objects)

        obscured_indices = np.logical_and(obscuring_objects != -1, distances_to_intersections.flatten() < norms.flatten())
        non_obscured_indices = np.logical_not(obscured_indices)

        distances = norms.reshape(size)

        non_obscured_distances = distances[non_obscured_indices][:, None]

        diffuse_intensities[non_obscured_indices] = self.diffuse_color * self.intensity / non_obscured_distances ** 2
        specular_intensities[non_obscured_indices] = self.specular_color * self.intensity / non_obscured_distances ** 2

        exit_points, refracted_vectors = compute_refraction_for_shadows(scene_objects,
                                                                        obscuring_objects[obscured_indices],
                                                                        intersection_points[obscured_indices],
                                                                        light_vectors[obscured_indices])

        intersection_solutions = self.orb.intersection(exit_points, refracted_vectors)
        secondary_obscured_indices = intersection_solutions <= 0

        obscured_intensity_factor = np.zeros((size, 3))
        for i, obj in enumerate(scene_objects):
            relevant_indices = obscuring_objects == i
            obscured_intensity_factor[relevant_indices] = obj.material.transparency_coefficient * obj.material.diffuse_color

        # TODO: Do this part recursively or something similar.
        temp1 = obscured_intensity_factor[obscured_indices]
        temp1[secondary_obscured_indices] = 0
        obscured_intensity_factor[obscured_indices] = temp1
        obscured_distances = distances[obscured_indices][:, None]
        diffuse_intensities[obscured_indices] = obscured_intensity_factor[obscured_indices] * (
                    self.diffuse_color * self.intensity / obscured_distances ** 2)
        specular_intensities[obscured_indices] = obscured_intensity_factor[obscured_indices] * (
                    self.specular_color * self.intensity / obscured_distances ** 2)

        return np.clip(diffuse_intensities, 0, 1), np.clip(specular_intensities, 0, 1), [light_vectors]


class DiskSource(LightSource):
    def __init__(self, x=4, y=0, z=20, radius=3, intensity=15):
        super().__init__(x, y, z, intensity=intensity)
        self.radius = radius
        self.n_points = 30

    def compute_light_intensity(self, intersection_points, scene_objects):
        size = intersection_points.shape[0]

        diffuse_intensities = np.zeros((size, 3))
        specular_intensities = np.zeros((size, 3))

        if self.normal_vector[0] != 0 and self.normal_vector[1] == 0 and self.normal_vector[2] == 0:
            perpendicular_vector = np.array([0.0, 1.0, 0.0])
        else:
            perpendicular_vector = np.array([1.0, 0.0, 0.0])

        x_hat = np.cross(self.normal_vector, perpendicular_vector)
        y_hat = np.cross(self.normal_vector, x_hat)

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

            reshaped_distances = distances[non_obscured_indices][:, None]

            diffuse_intensities[non_obscured_indices] += np.clip(
                self.diffuse_color * self.intensity / self.n_points / reshaped_distances ** 2, 0, 1)
            specular_intensities[non_obscured_indices] += np.clip(
                self.specular_color * self.intensity / self.n_points / reshaped_distances ** 2, 0, 1)

            light_vectors_matrix.append(light_vectors)

        return np.clip(diffuse_intensities, 0, 1), np.clip(specular_intensities, 0, 1), light_vectors_matrix


class EasingModes:
    NONE = "none"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    EXPONENTIAL = "exponential"


class DirectionalDiskSource(DiskSource):
    def __init__(self, x=4, y=0, z=20, radius=3, intensity=15, angle=30, easing_mode=EasingModes.EXPONENTIAL):
        super().__init__(x, y, z, radius=radius, intensity=intensity)
        self.angle = angle * np.pi / 180
        self.fall_off_angle = 20 * np.pi / 180
        self.easing_mode = easing_mode
        if angle == 90:
            print("Using a directional disk source with an angle of 90 degrees is not recommended. "
                  "Use DiskSource instead.")

    def ease_fall_off_beam(self, x, a, d):
        eased_matrix = np.ones(x.shape)
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
        distance_from_normal_axis = (x ** 2 + y ** 2) ** 0.5

        allowed_distance = self.radius + np.tan(self.angle) * np.abs(z)
        distance_to_edge_off_fall_off_region = self.radius + np.tan(self.angle + self.fall_off_angle) * np.abs(z)
        allowed_fall_off_distance = distance_to_edge_off_fall_off_region - allowed_distance

        inside_light_beam_indices = distance_from_normal_axis <= distance_to_edge_off_fall_off_region
        relevant_total_intensities = total_intensities[inside_light_beam_indices]

        visible_intersection_points = intersection_points[inside_light_beam_indices]

        light_vectors_matrix = []
        for i in range(self.n_points):
            theta = np.random.random(size) * 2 * math.pi
            d = np.random.random(size) ** 0.5 * self.radius

            random_point_local = d[:, None] * (np.cos(theta)[:, None] * x_hat + np.sin(theta)[:, None] * y_hat)
            random_light_point = self.position + random_point_local
            light_vectors = random_light_point - intersection_points
            norms = np.linalg.norm(light_vectors, axis=-1, keepdims=True)
            light_vectors = light_vectors / norms
            obscuring_objects, _ = find_closest_intersected_object(visible_intersection_points,
                                                                   light_vectors[inside_light_beam_indices],
                                                                   scene_objects)

            non_obscured_indices = obscuring_objects == -1

            distances = norms[inside_light_beam_indices].reshape(visible_intersection_points.shape[0])

            intensities = self.intensity / self.n_points / distances[non_obscured_indices] ** 2

            x = distance_from_normal_axis[inside_light_beam_indices]
            x = x[non_obscured_indices]

            d = allowed_fall_off_distance[inside_light_beam_indices]
            d = d[non_obscured_indices]

            a = allowed_distance[inside_light_beam_indices]
            a = a[non_obscured_indices]

            fall_off_factors = self.ease_fall_off_beam(x, a, d)
            relevant_total_intensities[non_obscured_indices] += intensities * fall_off_factors

            light_vectors_matrix.append(light_vectors)

        total_intensities[inside_light_beam_indices] = relevant_total_intensities

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


def solve_quadratic_2(B, C):
    """Solves a special case quadratic equation with a = 1. Returns the maximum solution"""
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

    min_ok_indices = 0 < minimum_solutions
    valid_solutions[min_ok_indices] = minimum_solutions[min_ok_indices]

    max_ok_indices = 0 < maximum_solutions
    valid_solutions[max_ok_indices] = maximum_solutions[max_ok_indices]

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


def linear_easing(x, a, d):
    return np.minimum(np.maximum((a - x) / d + 1, 0), 1)


def quadratic_easing(x, a, d):
    before_easing_starts_indices = x < a
    after_easing_stops_indices = x > a + d
    easing_indices = np.logical_not(np.logical_or(before_easing_starts_indices, after_easing_stops_indices))

    result = np.zeros(x.shape)

    x = x[easing_indices]
    a = a[easing_indices]
    d = d[easing_indices]
    result[easing_indices] = 1 - (x - a) * (1 / d - (x - (a + d)) / d ** 2)

    result[before_easing_starts_indices] = 1
    return result


def cubic_easing(x, a, d):
    before_easing_starts_indices = x < a
    after_easing_stops_indices = x > a + d
    easing_indices = np.logical_not(np.logical_or(before_easing_starts_indices, after_easing_stops_indices))
    result = np.zeros(x.shape)
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


def compute_refraction_for_shadows(scene_objects, seen_objects, intersection_points, direction_vectors):
    """Computes the color for the refracted rays. Function currently assumes the same index of refraction."""
    # TODO: This refraction depth should be necessary, but refraction should also stop when hitting the sky.
    exit_points = np.zeros(intersection_points.shape)
    transmitted_vectors_out_of_surface = np.zeros(intersection_points.shape)
    for i, obj in enumerate(scene_objects):
        relevant_indices = seen_objects == i
        transmitted_vectors_into_surface = refract_vectors(intersection_points[relevant_indices],
                                                           direction_vectors[relevant_indices], 1,
                                                           obj.material.refractive_index, obj)
        t = obj.exit_point(intersection_points[relevant_indices], transmitted_vectors_into_surface)
        exit_points[relevant_indices] = intersection_points[relevant_indices] + transmitted_vectors_into_surface * (
                    t[:, None] + EPSILON)
        transmitted_vectors_out_of_surface[relevant_indices] = refract_vectors(exit_points[relevant_indices],
                                                                               transmitted_vectors_into_surface,
                                                                               obj.material.refractive_index, 1, obj, 1)
    return exit_points, transmitted_vectors_out_of_surface


def refract_vectors(starting_points, incident_vectors, n1, n2, obj, dir=-1):
    # TODO: Add handling of total internal reflection. Do this by checking when the square root becomes invalid. These index should be separated, be reflected,
    # returned as a tuple
    # Also, a refracted ray should be able to be reflected on the exit region of the object.
    # Perhaps merge reflections and refractions into a single function.
    mu = n1 / n2
    normal_vectors = dir * (starting_points - obj.position)
    normal_vectors = normal_vectors / np.linalg.norm(normal_vectors, axis=-1, keepdims=True)
    length_in_normal_direction = ((1 - mu ** 2 * (1 - np.sum(normal_vectors * incident_vectors, axis=-1) ** 2)) ** 0.5)[
                                 :, None]
    temp = np.sum(normal_vectors * incident_vectors, axis=-1)[:, None]
    transmitted_vectors = length_in_normal_direction * normal_vectors + mu * (incident_vectors - temp * normal_vectors)
    return transmitted_vectors
