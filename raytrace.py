from constants import *
import objects as objects
import materials as materials
import numpy as np


def get_pixel_color(X, Y, scene):
    pixel_vector = scene.screen.index_to_position(X, Y)
    direction_vectors = pixel_vector - scene.camera.position
    norms = np.linalg.norm(direction_vectors, axis=-1, keepdims=True)
    direction_vectors = direction_vectors / norms

    starting_positions = np.full(direction_vectors.shape, scene.camera.position)
    color = get_intersection_color(starting_positions, direction_vectors, scene, depth=reflection_depth)
    color = np.clip(color, 0, 1)
    return color


def get_intersection_color(starting_positions, direction_vectors, scene, depth=1, refraction_depth=3):
    combined_colors = np.full(starting_positions.shape, BLACK.copy())
    seen_objects, T = objects.find_closest_intersected_object(starting_positions, direction_vectors, scene.objects)
    invalid_indices = seen_objects == -1
    valid_indices = seen_objects != -1

    seen_objects = seen_objects[valid_indices]
    if len(seen_objects) == 0:
        combined_colors[invalid_indices] = BACKGROUND_COLOR
        return combined_colors

    starting_positions = starting_positions[valid_indices]
    direction_vectors = direction_vectors[valid_indices]
    T = T[valid_indices]

    intersection_points = starting_positions + direction_vectors * T
    positions = np.zeros(starting_positions.shape)
    for i, obj in enumerate(scene.objects):
        relevant_indices = seen_objects == i
        positions[relevant_indices] = obj.position

    normals = intersection_points - positions
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normal_vectors = normals / norms

    intersection_points += normal_vectors * EPSILON

    if depth != 0:
        normal_dot_direction_vectors = np.sum(normal_vectors * direction_vectors, axis=-1)[:, None]
        reflection_vectors = - 2 * normal_vectors * normal_dot_direction_vectors + direction_vectors
        reflection_colors = get_intersection_color(intersection_points, reflection_vectors, scene,
                                                   depth - 1)

        alpha = np.zeros(normal_dot_direction_vectors.shape)
        for i, obj in enumerate(scene.objects):
            relevant_indices = seen_objects == i
            alpha[relevant_indices] = get_alpha(obj)

    # TODO: Compute refraction vectors here (start with no change in angles).

    if refraction_depth != 0:
        refraction_colors = compute_refraction(scene, seen_objects, intersection_points, direction_vectors, depth, refraction_depth)
    transparency = np.zeros(normal_vectors.shape)

    for i, obj in enumerate(scene.objects):
        relevant_indices = seen_objects == i
        transparency[relevant_indices] = get_transparency(obj)

    if refraction_depth != 0:
        combined_colors[valid_indices] += transparency * refraction_colors

    combined_colors[invalid_indices] = BACKGROUND_COLOR
    for light in scene.light_sources:
        diffuse_intensities, specular_intensities, light_vectors_matrix = light.compute_light_intensity(intersection_points, scene.objects)

        surface_color = compute_surface_color(scene, seen_objects, direction_vectors, normal_vectors, diffuse_intensities, specular_intensities,
                                              light_vectors_matrix)

        if depth == 0:
            combined_colors[valid_indices] += (1 - transparency) * surface_color
            continue

        surface_colors_weighted = surface_color * (1 - alpha)
        reflection_colors_weighted = alpha * reflection_colors

        combined_colors[valid_indices] += (1 - transparency) * (surface_colors_weighted + reflection_colors_weighted)

    return np.clip(combined_colors, 0, 1)


def compute_refraction(scene, seen_objects, intersection_points, direction_vectors, depth, refraction_depth):
    """Computes the color for the refracted rays. Function currently assumes the same index of refraction."""
    # TODO: This refraction depth should be necessary, but refraction should also stop when hitting the sky.
    exit_points = np.zeros(intersection_points.shape)
    transmitted_vectors_out_of_surface = np.zeros(intersection_points.shape)
    for i, obj in enumerate(scene.objects):
        relevant_indices = seen_objects == i
        transmitted_vectors_into_surface = refract_vectors(intersection_points[relevant_indices], direction_vectors[relevant_indices], 1, obj.material.refractive_index, obj)
        t = obj.exit_point(intersection_points[relevant_indices], transmitted_vectors_into_surface)
        exit_points[relevant_indices] = intersection_points[relevant_indices] + transmitted_vectors_into_surface * (t[:, None] + EPSILON)
        transmitted_vectors_out_of_surface[relevant_indices] = refract_vectors(exit_points[relevant_indices], transmitted_vectors_into_surface, obj.material.refractive_index, 1, obj, 1)
    return get_intersection_color(exit_points, transmitted_vectors_out_of_surface, scene, depth, refraction_depth-1)


def refract_vectors(starting_points, incident_vectors, n1, n2, obj, dir=-1):
    # TODO: Add handling of total internal reflection. Do this by checking when the square root becomes invalid. These index should be separated, be reflected,
    # returned as a tuple
    # Also, a refracted ray should be able to be reflected on the exit region of the object.
    # Perhaps merge reflections and refractions into a single function.
    mu = n1 / n2
    normal_vectors = dir * (starting_points - obj.position)
    normal_vectors = normal_vectors / np.linalg.norm(normal_vectors, axis=-1, keepdims=True)
    length_in_normal_direction = ((1 - mu ** 2 * (1 - np.sum(normal_vectors * incident_vectors, axis=-1) ** 2)) ** 0.5)[:, None]
    temp = np.sum(normal_vectors * incident_vectors, axis=-1)[:, None]
    transmitted_vectors = length_in_normal_direction * normal_vectors + mu * (incident_vectors - temp * normal_vectors)
    return transmitted_vectors


def compute_surface_color(scene, seen_objects, direction_vectors, normal_vectors, diffuse_intensities, specular_intensities, light_vectors_matrix):
    # TODO: Only do this for array elements that have non_zero intensities.
    surface_color = np.full(direction_vectors.shape, BLACK.copy())
    for k, light_vec in enumerate(light_vectors_matrix):
        normal_dot_light_vectors = np.sum(normal_vectors * light_vec, axis=-1)
        reflection_vectors = - 2 * normal_vectors * normal_dot_light_vectors[:, None] + light_vec
        reflection_dot_direction_vectors = np.sum(reflection_vectors * direction_vectors, axis=-1)
        for i, obj in enumerate(scene.objects):
            relevant_indices = seen_objects == i
            diffusive_color = get_diffusive_color(obj)
            specular_color = get_specular_color(obj)
            shininess = get_shininess(obj)

            I_diffuse = np.clip(diffusive_color * normal_dot_light_vectors[relevant_indices][:, None], 0, 1)
            I_specular = np.clip(specular_color * reflection_dot_direction_vectors[relevant_indices][:, None] ** shininess, 0, 1)
            color = (I_diffuse * diffuse_intensities[relevant_indices] + I_specular * specular_intensities[relevant_indices]) / len(light_vectors_matrix)
            surface_color[relevant_indices] += np.clip(color, 0, 1)

    return np.clip(surface_color, 0, 1)


def get_position(obj):
    if obj is None:
        return np.array([0.0, 0.0, 0.0])
    return obj.position


def get_diffusive_color(obj):
    if obj is None:
        return BLACK
    return obj.material.diffuse_color * obj.material.diffusion_coefficient


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
    Y, X = np.indices((HEIGHT, WIDTH))
    X = X.reshape(SIZE, 1)
    Y = Y.reshape(SIZE, 1)
    image_array = get_pixel_color(X, Y, scene)
    image_array = image_array.reshape((HEIGHT, WIDTH, 3))
    return image_array
