import numpy as np
WIDTH = 1000
HEIGHT = 1000
SIZE = HEIGHT * WIDTH
image_directory = "Images/"
WHITE = np.array([1.0, 1.0, 1.0])
BLACK = np.array([0.0, 0.0, 0.0])
RED = np.array([1.0, 0.0, 0.0])
GREEN = np.array([0.0, 1.0, 0.0])
BLUE = np.array([0.0, 0.0, 1.0])
YELLOW = np.array([1.0, 1.0, 0.0])
SKY_BLUE = np.array([0.251, 0.624, 0.769])
GREY = np.array([0.5, 0.5, 0.5])
BACKGROUND_COLOR = SKY_BLUE
MAX_REFLECTION_DEPTH = 1
MAX_REFRACTION_DEPTH = 100
EPSILON = 0.00001
AIR_REFRACTIVE_INDEX = 1
