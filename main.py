import numpy as np

from constants import *
import matplotlib.pyplot as plt
import objects
import player
import math


def find_closes_intersected_object(starting_position, direction_vector, object_list):
    min_t = np.inf
    closest_object = None
    for obj in object_list:
        intersects, t = obj.intersection(starting_position, direction_vector)
        if not t:
            continue
        if t <= min_t:
            min_t = t
            closest_object = obj

    return closest_object, min_t


def get_pixel_color(i, j, screen, camera, scene_objects, light_sources):
    pixel_vector = screen.index_to_position(j, i)
    direction_vector = pixel_vector - camera.position
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    seen_object, t = find_closes_intersected_object(camera.position, direction_vector, scene_objects)

    if seen_object is None:
        return SKY_BLUE

    intersection_point = camera.position + direction_vector * t
    intersection_point += seen_object.small_normal_offset(intersection_point)
    for light in light_sources:
        direction_vector_towards_light = light.position - intersection_point
        direction_vector_towards_light = direction_vector_towards_light / np.linalg.norm(
            direction_vector_towards_light)
        obscuring_object, _ = find_closes_intersected_object(intersection_point, direction_vector_towards_light, scene_objects)
        if obscuring_object is not None:
            return BLACK

        return seen_object.compute_surface_color(intersection_point, direction_vector, direction_vector_towards_light)


def raytrace():
    scene_objects = [objects.Sphere(z=0, radius=1), objects.Sphere(z=1, radius=0.5)]
    light_sources = [objects.PointSource()]
    camera = player.Player(x=1, z=3)
    screen = camera.screen
    for j, column in enumerate(screen.image):
        for i, row in enumerate(column):
            screen.image[i][j] = get_pixel_color(i, j, screen, camera, scene_objects, light_sources)

    return screen.image


def main():
    image = raytrace()
    plt.imsave(image_directory + "test.png", image)


if __name__ == '__main__':
    temp = 0
    main()
