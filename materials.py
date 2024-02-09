import numpy as np


class Material:
    def __init__(self, color, diffusion_coefficient=1):
        self.color = color
        self.diffusion_coefficient = diffusion_coefficient

    def compute_color(self, normal_vector, direction_vector):
        color_multiplication = self.diffusion_coefficient * np.dot(normal_vector, direction_vector)
        color_multiplication = clamp(color_multiplication, 0, 1)
        new_color = [clamp(color_multiplication * value, 0, 1) for value in self.color]
        return new_color


def clamp(x, minimum, maximum):
    return min(max(x, minimum), maximum)