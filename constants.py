import colors
WIDTH = 100
HEIGHT = 100
SIZE = HEIGHT * WIDTH
image_directory = "Images/"
BACKGROUND_COLOR = colors.SKY_BLUE
MAX_REFLECTION_DEPTH = 3
MAX_REFRACTION_DEPTH = 4
EPSILON = 0.00001
AIR_REFRACTIVE_INDEX = 1
SMALL_NUM = 3  # Reduces the number of recursive calls, ignoring when only a small number of light rays are reflected.
