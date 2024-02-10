import numpy as np

from constants import *


class Material:
    def __init__(self, diffuse_color=GREEN, specular_color=WHITE, diffusion_coefficient=0.7, specular_coefficient=0.5,
                 shininess=30,
                 reflection_coefficient=0.5):
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.diffusion_coefficient = diffusion_coefficient
        self.specular_coefficient = specular_coefficient
        self.reflection_coefficient = reflection_coefficient
        self.shininess = shininess

    def compute_color(self, normal_vectors, direction_vectors, light_vectors):
        # TODO: Issue here, with getting HEIGHTxWIDTHx3 list of None here, if failiure
        normal_dot_light_vectors = np.sum(normal_vectors * light_vectors, axis=-1)
        R = - 2 * normal_vectors * normal_dot_light_vectors[:, :, None] + light_vectors
        reflection_dot_direction_vectors = np.sum(R * direction_vectors, axis=-1)
        I_diffuse = self.diffusion_coefficient * multiply_matrix_by_vector_elementwise(normal_dot_light_vectors,
                                                                                       self.diffuse_color)

        I_specular = self.specular_coefficient * multiply_matrix_by_vector_elementwise(reflection_dot_direction_vectors ** self.shininess, self.specular_color)
        new_color = np.clip(I_diffuse + I_specular, 0, 1)
        return new_color

    def get_specular_color(self):
        return self.specular_coefficient * self.specular_color

    def get_diffuse_color(self):
        return self.diffusion_coefficient * self.diffuse_color


def multiply_matrix_by_vector_elementwise(A, v):
    A_height, A_width = A.shape
    A = A.reshape(-1, 1, 1)
    A = A * v
    return A.reshape(A_height, A_width, 3)
