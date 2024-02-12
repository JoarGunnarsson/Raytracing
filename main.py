import time
import matplotlib.pyplot as plt
import vectorized.raytrace as vectorized
import non_vectorized.raytrace as non_vectorized
from constants import *


def main():
    start = time.time()
    image = vectorized.raytrace()
    plt.imsave(image_directory + "test.png", image)
    print(f"The program took {time.time() - start} seconds to run.")


if __name__ == '__main__':
    main()
