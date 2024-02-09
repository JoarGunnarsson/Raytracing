import numpy as np

from constants import *
import matplotlib.pyplot as plt
import objects
import player
import scene_object
import math


class Screen(scene_object.Object):
    def __init__(self, x=-8, y=0, z=0, width=1, height=1):
        super().__init__(x, y, z)
        self.width = width
        self.height = height
        self.pixels_x = WIDTH
        self.pixels_y = HEIGHT
        self.image = np.zeros((WIDTH, HEIGHT, 3))
        self.normal_vector = np.array([-1, 0, 0])
        self.y_vector = np.array([0, 0, 1])
        self.x_vector = np.cross(self.normal_vector, self.y_vector)

    def index_to_position(self, i, j):
        x = self.x_vector * (-self.width/2 + i * self.width / self.pixels_x)
        y = self.y_vector * (-self.height / 2 + (self.pixels_y - j) * self.height / self.pixels_y)
        return x + y + self.position


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

        return seen_object.compute_surface_color(intersection_point, direction_vector_towards_light)


def raytrace():
    scene_objects = [objects.Sphere()]
    light_sources = [objects.PointSource()]
    camera = player.Player(x=-10)
    screen = Screen()

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
