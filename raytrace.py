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
    color = recursive_function(scene, starting_positions, direction_vectors, MAX_REFLECTION_DEPTH, MAX_REFRACTION_DEPTH)
    color = np.clip(color, 0, 1)
    return color


def get_intersection_color(starting_positions, direction_vectors, scene, reflection_depth=1, refraction_depth=10):
    return recursive_function(scene, starting_positions, direction_vectors, depth)
    combined_colors = np.full(starting_positions.shape, BLACK.copy())
    seen_objects, T = objects.find_closest_intersected_object(starting_positions, direction_vectors, scene.objects)

    invalid_indices = seen_objects == -1
    valid_indices = seen_objects != -1

    seen_objects = seen_objects[valid_indices]
    if len(seen_objects) == 0:
        combined_colors[invalid_indices] = BACKGROUND_COLOR.copy()
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
        combined_colors[valid_indices] = np.clip(combined_colors[valid_indices], 0, 1)

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

        combined_colors[valid_indices] += (1 - transparency) * np.clip(surface_colors_weighted + reflection_colors_weighted, 0, 1)

    return np.clip(combined_colors, 0, 1)


def recursive_function(scene, starting_points, direction_vectors, reflection_depth, refraction_depth):
    # Input: Ray starting point, ray direction vectors.
    # Returns: The color corresponding to the vectors.
    initial_size = starting_points.shape[0]
    # 0: Define color arrays.
    combined_colors = np.full(starting_points.shape, BLACK.copy())

    # 1: Check closes object for intersection.
    seen_objects, distances_to_intersection = objects.find_closest_intersected_object(starting_points, direction_vectors, scene.objects)
    no_object_intersection_indices = seen_objects == -1
    object_intersection_indices = seen_objects != -1

    seen_objects = seen_objects[object_intersection_indices]
    starting_points = starting_points[object_intersection_indices]
    direction_vectors = direction_vectors[object_intersection_indices]
    distances_to_intersection = distances_to_intersection[object_intersection_indices]
    if starting_points.shape[0] == 0:
        return np.zeros((initial_size, 3))

    object_intersection_size = seen_objects.shape[0]
    surface_colors = np.zeros((object_intersection_size, 3))
    refraction_colors = np.zeros((object_intersection_size, 3))
    reflection_colors = np.zeros((object_intersection_size, 3))

    # 2: If no intersecting object, set the color to background color.
    combined_colors[no_object_intersection_indices] = BACKGROUND_COLOR.copy()

    # 3: Otherwise, compute the surface color of the object. This is summed over all light sources
    intersection_points = starting_points + distances_to_intersection * direction_vectors

    positions = np.zeros((object_intersection_size, 3))
    for i, obj in enumerate(scene.objects):
        intersects_this_object_indices = seen_objects == i
        positions[intersects_this_object_indices] = obj.position

    normal_vectors = intersection_points - positions
    distances_to_object_center = np.linalg.norm(normal_vectors, axis=-1, keepdims=True)
    normal_vectors = normal_vectors / distances_to_object_center
    intersection_points += EPSILON * normal_vectors
    for light_source in scene.light_sources:
        diffuse_intensities, specular_intensities, light_vector_matrix = light_source.compute_light_intensity(intersection_points, scene.objects)
        surface_colors += compute_surface_color(scene, seen_objects, direction_vectors, normal_vectors, diffuse_intensities, specular_intensities, light_vector_matrix)

    # 5: If a ray intersects an object, compute the refraction vectors.
    n1 = np.ones(object_intersection_size)
    n2 = np.ones(object_intersection_size)
    refraction_normal_vectors = -normal_vectors.copy()
    inside_an_object_indices = np.sum(direction_vectors * normal_vectors, axis=-1) > 0
    refraction_normal_vectors[inside_an_object_indices] *= -1
    for i, obj in enumerate(scene.objects):
        intersects_this_object_indices = seen_objects == i

        inside_this_object_indices = np.logical_and(inside_an_object_indices, intersects_this_object_indices)
        outside_this_object_indices = np.logical_not(inside_this_object_indices)
        n1[inside_this_object_indices] = obj.material.refractive_index
        n2[inside_this_object_indices] = AIR_REFRACTIVE_INDEX

        n1[outside_this_object_indices] = AIR_REFRACTIVE_INDEX
        n2[outside_this_object_indices] = obj.material.refractive_index

    transmitted_vectors, transmitted_indices = refract_vectors(refraction_normal_vectors, direction_vectors, n1, n2)

    # 6: Split the refraction vectors into transmitted vectors and reflected vectors.
    total_reflection_indices = np.logical_not(transmitted_indices)
    direction_vectors_to_be_reflected = direction_vectors[total_reflection_indices]
    normal_vectors_for_reflection = normal_vectors[total_reflection_indices]

    total_reflection_vectors = reflect_vectors(direction_vectors_to_be_reflected, normal_vectors_for_reflection)

    # 7: Recursively compute the colors for these vectors.

    if refraction_depth != 0:
        x = intersection_points[transmitted_indices] + transmitted_vectors * EPSILON*2  # TODO: Skipping the interior fixes the issue, why?
        refraction_colors[transmitted_indices] = recursive_function(scene, x, transmitted_vectors, reflection_depth, refraction_depth-1)
        refraction_colors[total_reflection_indices] = recursive_function(scene, intersection_points[total_reflection_indices], total_reflection_vectors, reflection_depth-1, refraction_depth-1)

    # 8: If a ray intersects an object, compute the reflected vectors.
    reflection_vectors = reflect_vectors(direction_vectors, normal_vectors)

    # 9: Recursively compute the color for these vectors.
    if reflection_depth != 0:
        reflection_colors = recursive_function(scene, intersection_points, reflection_vectors, reflection_depth-1, refraction_depth)
        print(reflection_colors.shape, intersection_points.shape, reflection_vectors.shape)

    # 10: Combine all of these colors together depending on the coefficients of the intersected object.
    for i, obj in enumerate(scene.objects):
        intersects_this_object_indices = seen_objects == i
        surface_colors[intersects_this_object_indices] *= (1 - obj.material.transparency_coefficient) * (1 - obj.material.reflection_coefficient)
        refraction_colors[intersects_this_object_indices] *= obj.material.transparency_coefficient
        reflection_colors[intersects_this_object_indices] *= (1 - obj.material.transparency_coefficient) * obj.material.reflection_coefficient

    combined_colors[object_intersection_indices] = surface_colors + refraction_colors + reflection_colors

    # 11: Return the combined color.
    return np.clip(combined_colors, 0, 1)


def compute_refraction(scene, seen_objects, intersection_points, direction_vectors, depth, refraction_depth):
    """Computes the color for the refracted rays. Function currently assumes the same index of refraction."""
    # TODO: This refraction depth should be necessary, but refraction should also stop when hitting the sky.
    exit_points = np.zeros(intersection_points.shape)
    transmitted_vectors_out_of_surface = np.zeros(intersection_points.shape)
    for i, obj in enumerate(scene.objects):
        relevant_indices = seen_objects == i
        transmitted_vectors_into_surface = refract_vectors_old(intersection_points[relevant_indices], direction_vectors[relevant_indices], 1, obj.material.refractive_index, obj)
        t = obj.exit_point(intersection_points[relevant_indices], transmitted_vectors_into_surface)
        exit_points[relevant_indices] = intersection_points[relevant_indices] + transmitted_vectors_into_surface * (t[:, None] + EPSILON)
        transmitted_vectors_out_of_surface[relevant_indices] = refract_vectors_old(exit_points[relevant_indices], transmitted_vectors_into_surface, obj.material.refractive_index, 1, obj, 1)
    return get_intersection_color(exit_points, transmitted_vectors_out_of_surface, scene, depth, refraction_depth-1)


def refract_vectors_old(starting_points, incident_vectors, n1, n2, obj, dir=-1):
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


def reflect_vectors(direction_vectors, normal_vectors):
    normal_dot_direction_vectors = np.sum(normal_vectors * direction_vectors, axis=-1)[:, None]
    reflection_vectors = - 2 * normal_vectors * normal_dot_direction_vectors + direction_vectors
    return reflection_vectors


def refract_vectors(normal_vectors, incident_vectors, n1, n2):
    mu = n1 / n2
    length_in_normal_direction_squared = (1 - mu ** 2 * (1 - np.sum(normal_vectors * incident_vectors, axis=-1) ** 2))
    transmitted_indices = length_in_normal_direction_squared >= 0

    mu = mu[transmitted_indices]
    normal_vectors = normal_vectors[transmitted_indices]
    incident_vectors = incident_vectors[transmitted_indices]

    length_in_normal_direction = (length_in_normal_direction_squared[transmitted_indices] ** 0.5)[:, None]
    cos_incident_angle = np.sum(normal_vectors * incident_vectors, axis=-1)[:, None]
    perpendicular_vectors = incident_vectors - cos_incident_angle * normal_vectors

    transmitted_vectors = length_in_normal_direction * normal_vectors + mu[:, None] * perpendicular_vectors
    return transmitted_vectors, transmitted_indices


def compute_surface_color(scene, seen_objects, direction_vectors, normal_vectors, diffuse_intensities, specular_intensities, light_vectors_matrix):
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
