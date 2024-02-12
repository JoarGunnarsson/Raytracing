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

        new_color = I_diffuse * self.diffuse_color + I_specular * self.specular_color
        return new_color
