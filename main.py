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
    color = get_intersection_color(camera.position, direction_vectors, scene_objects, light_sources, depth=1)
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

    for i, x in enumerate(seen_objects):
        for j, obj in enumerate(x):
            if obj is None:
                continue
            intersection_points[i][j] += obj.small_normal_offset(intersection_points[i][j])

    combined_colors = np.full(direction_vectors.shape, BLACK)
    for light in light_sources:
        light_intensities, light_vectors_matrix = light.compute_light_intensity(intersection_points, scene_objects)
        if depth == 0:
            for k, light_vec in enumerate(light_vectors_matrix):
                for i, x in enumerate(seen_objects):
                    for j, obj in enumerate(x):
                        if obj is None:
                            combined_colors[i][j] += SKY_BLUE
                            continue
                        if light_intensities[i][j] == 0:
                            continue
                        combined_colors[i][j] += light_intensities[i][j] * obj.compute_surface_color(intersection_points[i][j], direction_vectors[i][j], light_vec[i][j])
            continue

        zero_light_intensity_indices = light_intensities == 0
        non_zero_light_intensity_indices = light_intensities != 0
        reflection_vectors = np.zeros((HEIGHT, WIDTH, 3))
        for i, x in enumerate(seen_objects):
            for j, obj in enumerate(x):
                if obj is None:
                    continue
                normal_vector = obj.normal_vector(intersection_points[i][j])
                reflection_vectors[i][j] = - 2 * np.dot(normal_vector, direction_vectors[i][j]) * normal_vector + direction_vectors[i][j]

        colors = get_intersection_color(intersection_points, reflection_vectors, scene_objects, light_sources, depth-1)
        for k, light_vec in enumerate(light_vectors_matrix):
            for i, x in enumerate(seen_objects):
                for j, obj in enumerate(x):
                    if obj is None:
                        surface_color = SKY_BLUE
                        alpha = 0
                    else:
                        alpha = obj.material.reflection_coefficient
                        surface_color = light_intensities[i][j] * obj.compute_surface_color(intersection_points[i][j], direction_vectors[i][j], light_vec[i][j])
                    combined_colors[i][j] += surface_color * (1 - alpha) + alpha * colors[i][j]

    return np.clip(combined_colors, 0, 1)


def raytrace():
    scene_objects = [objects.Sphere(z=-1000, radius=1000, material=materials.Material(diffuse_color=WHITE, specular_coefficient=0.3, reflection_coefficient=0.24)),
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
    # Vectorized function is about 5 times faster... Not very good :/ But not completely vectorized...

if __name__ == '__main__':
    main()
