from constants import *
import non_vectorized.objects as objects
import non_vectorized.materials as materials
import numpy as np


def get_pixel_color(i, j, screen, camera, scene_objects, light_sources):
    pixel_vector = screen.index_to_position(i, j)
    direction_vector = pixel_vector - camera.position
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    color = get_intersection_color(camera.position, direction_vector, scene_objects, light_sources, depth=0)
    color = np.clip(color, 0, 1)
    return color


def get_intersection_color(start_position, direction_vector, scene_objects, light_sources, depth=1):
    seen_object, t = objects.find_closes_intersected_object(start_position, direction_vector, scene_objects)
    if seen_object is None:
        return BACKGROUND_COLOR.copy()

    intersection_point = start_position + direction_vector * t
    intersection_point += seen_object.small_normal_offset(intersection_point)
    combined_color = BLACK.copy()
    for light in light_sources:
        light_intensity, light_vectors = light.compute_light_intensity(intersection_point, scene_objects)
        if light_intensity == 0:
            if depth == 0:
                continue

            normal_vector = seen_object.normal_vector(intersection_point)
            alpha = seen_object.material.reflection_coefficient
            reflection_vector = - 2 * np.dot(normal_vector, direction_vector) * normal_vector + direction_vector
            color = get_intersection_color(intersection_point, reflection_vector, scene_objects, light_sources, depth-1)
            combined_color += alpha * color
            continue
        surface_color = BLACK.copy()
        for light_vector in light_vectors:
            surface_color += seen_object.compute_surface_color(intersection_point, direction_vector, light_vector) * light_intensity / len(light_vectors)
        surface_color = np.clip(surface_color, 0, 1)
        if depth == 0:
            combined_color += surface_color
            continue

        normal_vector = seen_object.normal_vector(intersection_point)
        reflection_vector = - 2 * np.dot(normal_vector, direction_vector) * normal_vector + direction_vector
        alpha = seen_object.material.reflection_coefficient
        color = get_intersection_color(intersection_point, reflection_vector, scene_objects, light_sources, depth - 1)

        combined_color += surface_color * (1 - alpha) + alpha * color
        continue
    return np.clip(combined_color, 0, 1)


def raytrace():
    scene_objects = [objects.Sphere(z=-1000, radius=1000, material=materials.Material(diffuse_color=WHITE, specular_coefficient=0.3, reflection_coefficient=0.24)),
                     objects.Sphere(z=1, radius=1, material=materials.Material(diffuse_color=BLUE, reflection_coefficient=0.1)),
                     objects.Sphere(y=2, z=1.25, radius=0.5)]
    light_sources = [objects.DirectionalDiskSource(x=4, y=0, z=5)]
    camera = objects.Camera(x=0, y=1, z=4)
    screen = camera.screen
    for j, column in enumerate(screen.image):
        for i, row in enumerate(column):
            screen.image[j][i] = get_pixel_color(i, j, screen, camera, scene_objects, light_sources)

    return screen.image
