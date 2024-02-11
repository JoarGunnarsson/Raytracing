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

    def compute_color(self, normal_vector, direction_vector, light_vector):
        I_diffuse = self.diffusion_coefficient * np.dot(normal_vector, light_vector)

        R = - 2 * np.dot(normal_vector, light_vector) * normal_vector + light_vector
        I_specular = self.specular_coefficient * np.dot(R, direction_vector)**self.shininess

        new_color = [clamp(I_diffuse * diffuse_value + I_specular * specular_value, 0, 1) for diffuse_value, specular_value in zip(self.diffuse_color, self.specular_color)]
        return np.array(new_color)

    def get_specular_color(self):
        return self.specular_coefficient * self.specular_color

    def get_diffuse_color(self):
        return self.diffusion_coefficient * self.diffuse_color


def multiply_matrix_by_vector_elementwise(A, v):
    A_height, A_width = A.shape
    A = A.reshape(-1, 1, 1)
    A = A * v
    return A.reshape(A_height, A_width, 3)


def clamp(x, low, high):
    return min(max(x, low), high)
