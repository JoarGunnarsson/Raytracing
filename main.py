import numpy as np
from constants import *
import matplotlib.pyplot as plt
import objects
import materials
import time


def get_pixel_color(X, Y, screen, camera, scene_objects, light_sources):
    pixel_vector = screen.index_to_position(X, Y)
    direction_vectors = pixel_vector - camera.position
    norms = np.linalg.norm(direction_vectors, axis=-1, keepdims=True)
    direction_vectors = direction_vectors / norms
    color = get_intersection_color(camera.position, direction_vectors, scene_objects, light_sources, depth=0)
    color = np.clip(color, 0, 1)
    return color


def get_intersection_color(start_position, direction_vectors, scene_objects, light_sources, depth=1):
    # TODO: Start_position -> Start_positions
    starting_positions = np.full(direction_vectors.shape, start_position)
    colors = np.full(direction_vectors.shape, BLACK)
    seen_objects, T = objects.find_closes_intersected_object(starting_positions, direction_vectors, scene_objects)
    no_seen_object_indices = seen_objects == None
    colors[no_seen_object_indices] = SKY_BLUE
    # TODO: Invalid T-elements: None. Only look at good indices.
    intersection_points = start_position + direction_vectors * T[:, :, None]

    # intersection_points += seen_object.small_normal_offset(intersection_point)
    # TODO: Enable the above line again, somehow.

    combined_colors = np.full(direction_vectors.shape, BLACK)
    for light in light_sources:
        light_intensities, light_vectors_matrix = light.compute_light_intensity(intersection_points, scene_objects)
        for i, x in enumerate(seen_objects):
            for j, y in enumerate(x):
                if y is None:
                    continue

                clr = (y.compute_surface_color(intersection_points[i][j], direction_vectors[i][j],
                                              light_vectors_matrix[0][i][j]))
                colors[i][j] = np.array(clr) * light_intensities[i][j]

        return colors
        zero_light_intensity_indices = light_intensities == 0
        non_zero_light_intensity_indices = light_intensities != 0
        if depth == 0:
            combined_colors[zero_light_intensity_indices] += BLACK  # TODO: This might not be necessary, perhaps remove it.
            ###

            combined_colors[non_zero_light_intensity_indices] += surface_color
            continue

        if light_intensity == 0:
            normal_vector = seen_object.normal_vector(intersection_point)
            alpha = seen_object.material.reflection_coefficient
            reflection_vector = - 2 * np.dot(normal_vector, direction_vector) * normal_vector + direction_vector
            color = get_intersection_color(intersection_point, reflection_vector, scene_objects, light_sources, depth-1)

            combined_color += np.array(BLACK) * (1 - alpha) + alpha * np.array(color)
            continue
        surface_color = np.array(BLACK)
        for light_vector in light_vectors:
            surface_color += np.array(seen_object.compute_surface_color(intersection_point, direction_vector, light_vector)) * light_intensity / len(light_vectors)


        normal_vector = seen_object.normal_vector(intersection_point)
        reflection_vector = - 2 * np.dot(normal_vector, direction_vector) * normal_vector + direction_vector
        alpha = seen_object.material.reflection_coefficient
        color = get_intersection_color(intersection_point, reflection_vector, scene_objects, light_sources, depth - 1)

        combined_color += surface_color * (1 - alpha) + alpha * np.array(color)
        continue
    return [materials.clamp(value, 0, 1) for value in combined_color]


def raytrace():
    scene_objects = [objects.Sphere(z=-1000, radius=1000, material=materials.Material(diffuse_color=GREY, specular_coefficient=0.3, reflection_coefficient=0.24)),
                     objects.Sphere(z=1, radius=1, material=materials.Material(diffuse_color=BLUE, reflection_coefficient=0.1)),
                     objects.Sphere(y=2, z=1.25, radius=0.5)]
    light_sources = [objects.PointSource(x=4, y=0, z=5)]
    camera = objects.Camera(x=0, y=1, z=4)
    screen = camera.screen
    Y, X = np.indices((HEIGHT, WIDTH))
    image_array = get_pixel_color(X, Y, screen, camera, scene_objects, light_sources)
    screen.image = image_array
    return screen.image


def main():
    start = time.time()
    image = raytrace()
    plt.imsave(image_directory + "test.png", image)
    print(f"The program took {time.time() - start} seconds to run.")


if __name__ == '__main__':
    main()
