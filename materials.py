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