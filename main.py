import numpy as np

from constants import *
import matplotlib.pyplot as plt
import objects
import player
import math
import materials


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


def get_pixel_color(i, j, screen, camera, scene_objects, light_sources):
    pixel_vector = screen.index_to_position(j, i)
    direction_vector = pixel_vector - camera.position
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    color = get_intersection_color(camera.position, direction_vector, scene_objects, light_sources, depth=1)
    color = [materials.clamp(value, 0, 1) for value in color]
    return color


def get_intersection_color(start_position, direction_vector, scene_objects, light_sources, depth=1):
    seen_object, t = find_closes_intersected_object(start_position, direction_vector, scene_objects)
    if seen_object is None:
        return SKY_BLUE

    intersection_point = start_position + direction_vector * t
    intersection_point += seen_object.small_normal_offset(intersection_point)
    for light in light_sources:
        direction_vector_towards_light = light.position - intersection_point
        direction_vector_towards_light = direction_vector_towards_light / np.linalg.norm(
            direction_vector_towards_light)
        obscuring_object, _ = find_closes_intersected_object(intersection_point, direction_vector_towards_light,
                                                             scene_objects)
        if obscuring_object is not None:
            if depth == 0:
                return BLACK
            normal_vector = seen_object.normal_vector(intersection_point)
            reflection_vector = - 2 * np.dot(normal_vector, direction_vector) * normal_vector + direction_vector
            return BLACK + seen_object.material.reflection_coefficient * np.array(get_intersection_color(intersection_point, reflection_vector, scene_objects, light_sources, depth-1))

        surface_color = seen_object.compute_surface_color(intersection_point, direction_vector, direction_vector_towards_light)
        if depth == 0:
            return surface_color
        normal_vector = seen_object.normal_vector(intersection_point)
        reflection_vector = - 2 * np.dot(normal_vector, direction_vector) * normal_vector + direction_vector

        return np.array(surface_color) + seen_object.material.reflection_coefficient * np.array(get_intersection_color(intersection_point, reflection_vector,
                                                                             scene_objects, light_sources, depth - 1))


def raytrace():
    scene_objects = [objects.Sphere(z=-1000, radius=1000, material=materials.Material(diffuse_color=GREY)),
                     objects.Sphere(z=1, radius=1, material=materials.Material(diffuse_color=BLUE, reflection_coefficient=0.1)),
                     objects.Sphere(y=2, z=1.25, radius=0.5)]
    light_sources = [objects.PointSource()]
    camera = player.Player(x=0, y=1, z=4)
    screen = camera.screen
    for j, column in enumerate(screen.image):
        for i, row in enumerate(column):
            screen.image[i][j] = get_pixel_color(i, j, screen, camera, scene_objects, light_sources)

    return screen.image


def main():
    image = raytrace()
    plt.imsave(image_directory + "test.png", image)


if __name__ == '__main__':
    main()
