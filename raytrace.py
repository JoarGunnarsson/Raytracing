from constants import *
import objects
import numpy as np


def get_pixel_color(X, Y, scene):
    pixel_vector = scene.screen.index_to_position(X, Y)
    direction_vectors = pixel_vector - scene.camera.position
    norms = np.linalg.norm(direction_vectors, axis=-1, keepdims=True)
    direction_vectors = direction_vectors / norms

    starting_positions = np.full(direction_vectors.shape, scene.camera.position)

    color = recursive_function(scene, starting_positions, direction_vectors, MAX_REFLECTION_DEPTH, MAX_REFRACTION_DEPTH)
    color = np.clip(color, 0, 1)
    return color


def recursive_function(scene, starting_points, direction_vectors, reflection_depth, refraction_depth):
    # Input: Ray starting point, ray direction vectors.
    # Returns: The color corresponding to the vectors.
    initial_size = starting_points.shape[0]

    # 0: Define color array, and return early if the number of vectors is small.
    combined_colors = np.zeros(starting_points.shape, dtype=float)
    if initial_size < SMALL_NUM:
        return combined_colors + BACKGROUND_COLOR

    # 1: Check closes object the ray intersects.
    seen_objects, distances_to_intersection = objects.find_closest_intersected_object(starting_points,
                                                                                      direction_vectors, scene.objects)

    no_object_intersection_indices = seen_objects == -1
    object_intersection_indices = seen_objects != -1

    seen_objects = seen_objects[object_intersection_indices]
    starting_points = starting_points[object_intersection_indices]
    direction_vectors = direction_vectors[object_intersection_indices]
    distances_to_intersection = distances_to_intersection[object_intersection_indices]

    if starting_points.shape[0] == 0:
        return np.ones((initial_size, 3), dtype=float) * BACKGROUND_COLOR

    object_intersection_size = seen_objects.shape[0]
    surface_colors = np.zeros((object_intersection_size, 3), dtype=float)
    refraction_colors = np.zeros((object_intersection_size, 3), dtype=float)
    reflection_colors = np.zeros((object_intersection_size, 3), dtype=float)

    # 2: If the ray does not intersect an object, set the color to the background color.
    combined_colors[no_object_intersection_indices] = BACKGROUND_COLOR.copy()

    # 3: Compute normal vectors
    intersection_points = starting_points + distances_to_intersection * direction_vectors

    positions = np.zeros((object_intersection_size, 3), dtype=float)
    for i, obj in enumerate(scene.objects):
        intersects_this_object_indices = seen_objects == i
        positions[intersects_this_object_indices] = obj.position

    normal_vectors = intersection_points - positions
    distances_to_object_center = np.linalg.norm(normal_vectors, axis=-1, keepdims=True)
    normal_vectors = normal_vectors / distances_to_object_center

    # 4: Compute the fresnel multipliers.
    n1 = np.zeros(object_intersection_size, dtype=float)
    n2 = np.zeros(object_intersection_size, dtype=float)

    inside_an_object_indices = np.sum(direction_vectors * normal_vectors, axis=-1) > 0

    for i, obj in enumerate(scene.objects):
        intersects_this_object_indices = seen_objects == i

        inside_this_object_indices = np.logical_and(inside_an_object_indices, intersects_this_object_indices)
        outside_this_object_indices = np.logical_and(np.logical_not(inside_an_object_indices),
                                                     intersects_this_object_indices)

        n1[inside_this_object_indices] = obj.material.refractive_index
        n2[inside_this_object_indices] = AIR_REFRACTIVE_INDEX

        n1[outside_this_object_indices] = AIR_REFRACTIVE_INDEX
        n2[outside_this_object_indices] = obj.material.refractive_index

        # TODO: This code assumes the object is surrounded by air. This is not necessarily the case, if the object
        # is inside another object.

    refraction_normal_vectors = -normal_vectors.copy()
    refraction_normal_vectors[inside_an_object_indices] *= -1.0

    intersection_points -= EPSILON * refraction_normal_vectors

    # 5: Compute the refraction vectors.
    transmitted_vectors, transmitted_indices = refract_vectors(refraction_normal_vectors, direction_vectors, n1, n2)
    transmitted_vectors_for_fresnel = np.zeros((object_intersection_size, 3), dtype=float)
    transmitted_vectors_for_fresnel[transmitted_indices] = transmitted_vectors
    R = fresnel_multiplier(direction_vectors, transmitted_vectors_for_fresnel, refraction_normal_vectors, n1, n2)

    # 6: Split the refraction vectors into transmitted vectors and reflected vectors.
    total_reflection_indices = np.logical_not(transmitted_indices)
    R[total_reflection_indices] = 1
    direction_vectors_to_be_reflected = direction_vectors[total_reflection_indices]

    # 7: Recursively compute the colors for these vectors.
    if refraction_depth != 0:
        adjusted_intersections = intersection_points[transmitted_indices] + 2 * EPSILON * refraction_normal_vectors[
            transmitted_indices]
        refraction_colors[transmitted_indices] = recursive_function(scene, adjusted_intersections, transmitted_vectors,
                                                                    reflection_depth, refraction_depth - 1)
        for i, obj in enumerate(scene.objects):
            intersects_this_object_indices = seen_objects == i
            outside_this_object_indices = np.logical_and(np.logical_not(inside_an_object_indices),
                                                         intersects_this_object_indices)
            transmitted_into_this_object_indices = np.logical_and(outside_this_object_indices, transmitted_indices)
            intersections_into_this_object = intersection_points[transmitted_into_this_object_indices]
            rel_ind = seen_objects[transmitted_indices] == i
            ind = np.logical_and(rel_ind, np.logical_not(inside_an_object_indices[transmitted_indices]))
            transmitted_vectors_into_this_object = transmitted_vectors[ind]
            distance_travelled_through_object = obj.intersection(intersections_into_this_object,
                                                                 transmitted_vectors_into_this_object, "furthest")
            attenuation_factor = np.exp(-obj.material.attenuation_coefficient * obj.material.absorption_color * distance_travelled_through_object[:, None])
            refraction_colors[transmitted_into_this_object_indices] *= attenuation_factor

    if refraction_depth != 0 and reflection_depth != 0:
        if intersection_points[total_reflection_indices].shape[0] > 0:
            total_reflection_vectors = reflect_vectors(direction_vectors_to_be_reflected,
                                                       -refraction_normal_vectors[total_reflection_indices])
            refraction_colors[total_reflection_indices] = recursive_function(scene, intersection_points[
                total_reflection_indices], total_reflection_vectors, reflection_depth - 1, refraction_depth - 1)

    # 8: Compute the reflected vectors, and recursively compute the color for them.
    if reflection_depth != 0:
        reflection_vectors = reflect_vectors(direction_vectors, -refraction_normal_vectors)
        reflection_colors = recursive_function(scene, intersection_points, reflection_vectors, reflection_depth - 1,
                                               refraction_depth)

    # 9: Compute the surface color of the object.
    ambient_intensities = np.zeros((object_intersection_size, 3))
    if scene.ambient_light is not None:
        ambient_intensities += scene.ambient_light.intensity * scene.ambient_light.color

    for light_source in scene.light_sources:
        diffuse_intensities, specular_intensities, light_vector_matrix = light_source.compute_light_intensity(
            intersection_points, scene.objects)
        surface_colors += compute_surface_color(scene, seen_objects, direction_vectors,
                                                normal_vectors, ambient_intensities,
                                                diffuse_intensities,
                                                specular_intensities,
                                                light_vector_matrix)

    for i, obj in enumerate(scene.objects):
        relevant_indices = seen_objects == i
        surface_colors[relevant_indices] += np.clip(obj.material.ambient_color * ambient_intensities[relevant_indices], 0, 1)
    surface_colors = np.clip(surface_colors, 0, 1)

    # 10: Combine all of these colors together.
    internal_reflection_multiplier = np.ones(object_intersection_size)
    internal_reflection_multiplier[inside_an_object_indices] = 0
    for i, obj in enumerate(scene.objects):
        intersects_this_object_indices = seen_objects == i
        max_reflection = obj.material.smoothness
        minimum_reflection = obj.material.reflection_coefficient * internal_reflection_multiplier[intersects_this_object_indices]
        amount_of_reflected_light = (minimum_reflection + (max_reflection - minimum_reflection) * R[intersects_this_object_indices])
        amount_of_reflected_light = amount_of_reflected_light[:, None]

        surface_colors[intersects_this_object_indices] *= (1 - obj.material.transparency_coefficient) * (1 - amount_of_reflected_light)
        refraction_colors[intersects_this_object_indices] *= obj.material.transparency_coefficient * (1 - amount_of_reflected_light)
        reflection_colors[intersects_this_object_indices] *= amount_of_reflected_light

    combined_colors[object_intersection_indices] = surface_colors + refraction_colors + reflection_colors

    return np.clip(combined_colors, 0, 1)


def fresnel_multiplier(incident_vectors, transmitted_vectors, normal_vectors, n1, n2):
    R_0 = ((n1 - n2) / (n1 + n2)) ** 2
    n2_larger_indices = n2 >= n1
    n1_larger_indices = np.logical_not(n2_larger_indices)
    R = np.zeros(incident_vectors.shape[0])

    cos_theta_incident = np.sum(incident_vectors[n2_larger_indices] * normal_vectors[n2_larger_indices], axis=-1)
    R[n2_larger_indices] = schlick_approximation(R_0[n2_larger_indices], cos_theta_incident)

    cos_theta_transmitted = np.sum(transmitted_vectors[n1_larger_indices] * normal_vectors[n1_larger_indices], axis=-1)
    R[n1_larger_indices] = schlick_approximation(R_0[n1_larger_indices], cos_theta_transmitted)
    return R


def schlick_approximation(R_0, cos_theta):
    R = R_0 + (1 - R_0) * (1 - cos_theta) ** 5
    return R


def reflect_vectors(direction_vectors, normal_vectors):
    normal_dot_direction_vectors = np.sum(normal_vectors * direction_vectors, axis=-1, keepdims=True)
    reflection_vectors = direction_vectors - 2 * normal_vectors * normal_dot_direction_vectors
    return reflection_vectors


def refract_vectors(normal_vectors, incident_vectors, n1, n2):
    mu = n1 / n2
    length_in_normal_direction_squared = (1 - mu ** 2 * (1 - np.sum(normal_vectors * incident_vectors, axis=-1) ** 2))
    transmitted_indices = length_in_normal_direction_squared >= 0

    mu = mu[transmitted_indices]
    normal_vectors = normal_vectors[transmitted_indices]
    incident_vectors = incident_vectors[transmitted_indices]

    length_in_normal_direction = (length_in_normal_direction_squared[transmitted_indices] ** 0.5)[:, None]
    cos_incident_angle = np.sum(normal_vectors * incident_vectors, axis=-1, keepdims=True)
    perpendicular_vectors = incident_vectors - cos_incident_angle * normal_vectors

    transmitted_vectors = length_in_normal_direction * normal_vectors + mu[:, None] * perpendicular_vectors
    return transmitted_vectors, transmitted_indices


def compute_surface_color(scene, seen_objects, direction_vectors, normal_vectors, ambient_intensities,
                          diffuse_intensities,
                          specular_intensities, light_vectors_matrix):
    surface_color = np.zeros(direction_vectors.shape, dtype=float)
    for k, light_vec in enumerate(light_vectors_matrix):
        normal_dot_light_vectors = np.sum(normal_vectors * light_vec, axis=-1, keepdims=True)
        reflection_vectors = - 2 * normal_vectors * normal_dot_light_vectors + light_vec
        reflection_dot_direction_vectors = np.abs(
            np.sum(reflection_vectors * direction_vectors, axis=-1, keepdims=True))
        for i, obj in enumerate(scene.objects):
            relevant_indices = seen_objects == i
            diffusive_color = get_diffusive_color(obj)
            specular_color = get_specular_color(obj)
            shininess = get_shininess(obj)
            I_diffuse = np.clip(diffusive_color * normal_dot_light_vectors[relevant_indices], 0, 1)
            I_specular = np.clip(specular_color * reflection_dot_direction_vectors[relevant_indices] ** shininess, 0, 1)
            surface_color[relevant_indices] += np.clip(I_diffuse * diffuse_intensities[relevant_indices], 0, 1)
            surface_color[relevant_indices] += np.clip(I_specular * specular_intensities[relevant_indices], 0, 1)

    return np.clip(surface_color, 0, 1)


def get_position(obj):
    if obj is None:
        return np.array([0.0, 0.0, 0.0])
    return obj.position


def get_diffusive_color(obj):
    if obj is None:
        return BLACK
    return obj.material.diffuse_color * obj.material.diffuse_coefficient


def get_specular_color(obj):
    if obj is None:
        return BLACK
    return obj.material.specular_color * obj.material.specular_coefficient


def get_alpha(obj):
    if obj is None:
        return 0
    return obj.material.reflection_coefficient


def get_transparency(obj):
    if obj is None:
        return 0
    return obj.material.transparency_coefficient


def get_shininess(obj):
    if obj is None:
        return 0
    return obj.material.shininess


def raytrace(scene):
    Y, X = np.indices((HEIGHT, WIDTH), dtype=float)
    X = X.reshape(SIZE, 1)
    Y = Y.reshape(SIZE, 1)
    image_array = get_pixel_color(X, Y, scene)
    image_array = image_array.reshape((HEIGHT, WIDTH, 3))
    return image_array
