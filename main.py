import numpy as np

from constants import *
import matplotlib.pyplot as plt
import objects
import player
import scene_object


class Screen(scene_object.Object):
    def __init__(self, x=1, y=0, z=0, width=1, height=1):
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
        y = self.y_vector * (-self.height / 2 + j * self.height / self.pixels_y)
        return x + y + self.position


def raytrace():
    # TODO: Take into account the camera position.
    # TODO: Check for object intersection.

    scene_objects = [objects.Sphere()]
    lights = [objects.PointSource()]
    camera = player.Player()
    screen = Screen()

    for i, row in enumerate(screen.image):
        for j, column in enumerate(row):
            pixel_vector = screen.index_to_position(i, j)
            direction_vector = pixel_vector - camera.position
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
            min_t = 10**10
            pixel_color = BLACK
            light = lights[0]
            for obj in scene_objects:
                t = obj.intersection(camera.position, direction_vector)
                if not t:
                    continue
                if t <= min_t:
                    direction_vector_towards_light = light.position - (camera.position + direction_vector * t)
                    direction_vector_towards_light = direction_vector_towards_light / np.linalg.norm(direction_vector_towards_light)
                    t_light = obj.intersection((camera.position + direction_vector * t) + obj.small_normal_offset(camera.position + direction_vector * t), direction_vector_towards_light)
                    if not t_light:
                        continue

                    min_t = t
                    pixel_color = obj.color
            screen.image[i][j] = pixel_color

    return screen.image


def main():
    image = raytrace()
    plt.imsave(image_directory + "test.png", image)


if __name__ == '__main__':
    main()
