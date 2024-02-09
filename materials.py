import numpy as np
from constants import *

class Material:
    def __init__(self, diffuse_color=GREEN, specular_color=WHITE, diffusion_coefficient=1, specular_coefficient=1):
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.diffusion_coefficient = diffusion_coefficient
        self.specular_coefficient = specular_coefficient

    def compute_color(self, normal_vector, direction_vector, light_vector):
        I_diffuse = self.diffusion_coefficient * np.dot(normal_vector, light_vector)

        n = 30
        R = - 2 * np.dot(normal_vector, light_vector) * normal_vector + light_vector
        I_specular = self.specular_coefficient * np.dot(R, direction_vector)**n

        new_color = [clamp(I_diffuse * diffuse_value + I_specular * specular_value, 0, 1) for diffuse_value, specular_value in zip(self.diffuse_color, self.specular_color)]
        return new_color


def clamp(x, minimum, maximum):
    return min(max(x, minimum), maximum)