from constants import *


class Material:
    def __init__(self, diffuse_color=YELLOW, specular_color=WHITE,
                 shininess=30, reflection_coefficient=0, transparency_coefficient=0, refractive_index=1,
                 smoothness=0.8):
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.reflection_coefficient = reflection_coefficient
        self.shininess = shininess
        self.transparency_coefficient = transparency_coefficient
        self.refractive_index = refractive_index
        self.smoothness = smoothness
