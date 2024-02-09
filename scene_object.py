import numpy as np


class Object:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.position = np.array([x, y, z])
