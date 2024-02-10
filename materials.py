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
        I_diffuse = self.diffusion_coefficient * np.dot(normal_vector, light_vector) * self.diffuse_color

        R = - 2 * np.dot(normal_vector, light_vector) * normal_vector + light_vector
        I_specular = self.specular_coefficient * np.dot(R, direction_vector)**self.shininess * self.specular_color

        new_color = np.clip(I_diffuse + I_specular, 0, 1)
        return new_color


def clamp(x, minimum, maximum):
    return min(max(x, minimum), maximum)