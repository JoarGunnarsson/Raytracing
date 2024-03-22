from constants import *


class Material:
    def __init__(self, ambient_color=None, diffuse_color=YELLOW, specular_color=WHITE, diffuse_coefficient=0.8, specular_coefficient=0.3,
                 shininess=100, reflection_coefficient=0, transparency_coefficient=0, refractive_index=1,
                 smoothness=0, attenuation_coefficient=0.1, absorption_color=None):
        self.diffuse_color = diffuse_color
        if ambient_color is None:
            ambient_color = diffuse_color
        self.ambient_color = ambient_color
        self.specular_color = specular_color
        self.diffuse_coefficient = diffuse_coefficient
        self.specular_coefficient = specular_coefficient
        self.reflection_coefficient = reflection_coefficient
        self.shininess = shininess
        self.transparency_coefficient = transparency_coefficient
        self.refractive_index = refractive_index
        self.smoothness = smoothness
        self.attenuation_coefficient = attenuation_coefficient
        if absorption_color is None:
            absorption_color = 1 - self.diffuse_color
            color_norm = np.max(absorption_color)
            if color_norm == 0:
                absorption_color = WHITE.copy()
            else:
                absorption_color = absorption_color / color_norm

        self.absorption_color = absorption_color
