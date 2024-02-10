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
    starting_positions = np.full(direction_vectors.shape, camera.position)
    color = get_intersection_color(starting_positions, direction_vectors, scene_objects, light_sources, depth=0)
    color = np.clip(color, 0, 1)
    return color


def get_intersection_color(starting_positions, direction_vectors, scene_objects, light_sources, depth=1):
    # TODO: Start_position -> Start_positions

    colors = np.full(direction_vectors.shape, BLACK)
    seen_objects, T = objects.find_closes_intersected_object(starting_positions, direction_vectors, scene_objects)
    no_seen_object_indices = seen_objects == None
    colors[no_seen_object_indices] = SKY_BLUE
    # TODO: Invalid T-elements: None. Only look at good indices.
    intersection_points = starting_positions + direction_vectors * T[:, :, None]

    get_position_vectorized = np.vectorize(get_position, otypes=[object])
    positions = get_position_vectorized(seen_objects)
    positions = np.apply_along_axis(lambda x: np.stack(x, axis=0), axis=-1, arr=positions)
    normals = intersection_points - positions
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normal_vectors = normals / norms
    EPSILON = 0.001

    intersection_points += normal_vectors * EPSILON

    combined_colors = np.full(direction_vectors.shape, BLACK)
    for light in light_sources:
        light_intensities, light_vectors_matrix = light.compute_light_intensity(intersection_points, scene_objects)
        if depth == 0:
            get_diffusive_color_vectorized = np.vectorize(get_diffusive_color, otypes=[object])
            get_specular_color_vectorized = np.vectorize(get_specular_color, otypes=[object])

            for k, light_vec in enumerate(light_vectors_matrix):
                normal_dot_light_vectors = np.sum(normal_vectors * light_vec, axis=-1)
                reflection_vectors = - 2 * normal_vectors * normal_dot_light_vectors[:, :, None] + light_vec
                reflection_dot_direction_vectors = np.sum(reflection_vectors * direction_vectors, axis=-1)

                diffusive_colors = get_diffusive_color_vectorized(seen_objects)
                diffusive_colors = np.apply_along_axis(lambda x: np.stack(x, axis=0), axis=-1, arr=diffusive_colors)

                specular_colors = get_specular_color_vectorized(seen_objects)
                specular_colors = np.apply_along_axis(lambda x: np.stack(x, axis=0), axis=-1, arr=specular_colors)

                I_diffuse = diffusive_colors * normal_dot_light_vectors[:, :, None]

                shininess = 30 # TODO: This needs to be taken from the material of the object!
                I_specular = specular_colors * reflection_dot_direction_vectors[:, :, None] ** shininess
                surface_color = (I_diffuse + I_specular) * light_intensities[:, :, None]
                combined_colors += np.clip(surface_color, 0, 1)
            continue

        normal_dot_direction_vectors = np.sum(normal_vectors * direction_vectors, axis=-1)
        reflection_vectors = - 2 * normal_vectors * normal_dot_direction_vectors[:, :, None] + direction_vectors

        colors = get_intersection_color(intersection_points, reflection_vectors, scene_objects, light_sources,
                                        depth - 1)
        for k, light_vec in enumerate(light_vectors_matrix):
            for i, x in enumerate(seen_objects):
                for j, obj in enumerate(x):
                    if obj is None:
                        surface_color = SKY_BLUE
                        alpha = 0
                    else:
                        alpha = obj.material.reflection_coefficient
                        surface_color = light_intensities[i][j] * obj.compute_surface_color(intersection_points[i][j],
                                                                                            direction_vectors[i][j],
                                                                                            light_vec[i][j])
                    combined_colors[i][j] += surface_color * (1 - alpha) + alpha * colors[i][j]

    return np.clip(combined_colors, 0, 1)


def get_position(obj):
    if obj is None:
        return np.array([0, 0, 0])
    return obj.position


def get_diffusive_color(obj):
    if obj is None:
        return BLACK
    return obj.material.get_diffuse_color()


def get_specular_color(obj):
    if obj is None:
        return BLACK
    return obj.material.get_specular_color()


def raytrace():
    scene_objects = [objects.Sphere(z=-1000, radius=1000,
                                    material=materials.Material(diffuse_color=WHITE, specular_coefficient=0.3,
                                                                reflection_coefficient=0.24)),
                     objects.Sphere(z=1, radius=1,
                                    material=materials.Material(diffuse_color=BLUE, reflection_coefficient=0.1)),
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
    # 6.738481760025024


if __name__ == '__main__':
    main()
