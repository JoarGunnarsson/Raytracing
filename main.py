import time
import matplotlib.pyplot as plt
from constants import *
from raytrace import raytrace


def main():
    start = time.time()
    image = raytrace()
    plt.imsave(image_directory + "result.png", image)
    print(f"The program took {time.time() - start} seconds to run.")


if __name__ == '__main__':
    main()
