from constants import *


class Material:
    def __init__(self, diffuse_color=YELLOW, specular_color=WHITE, diffuse_coefficient=0.8, specular_coefficient=0.3,
                 shininess=30, reflection_coefficient=0, transparency_coefficient=0, refractive_index=1,
                 smoothness=0):
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.diffuse_coefficient = diffuse_coefficient
        self.specular_coefficient = specular_coefficient
        self.reflection_coefficient = reflection_coefficient
        self.shininess = shininess
        self.transparency_coefficient = transparency_coefficient
        self.refractive_index = refractive_index
        self.smoothness = smoothness
