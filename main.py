import time
import matplotlib.pyplot as plt

import Scenes.scenes
from constants import *
from raytrace import raytrace


def example_1():
    scene = Scenes.scenes.scenes["example_1"]
    return raytrace(scene)


def example_2():
    scene = Scenes.scenes.scenes["example_2"]
    return raytrace(scene)


def main():
    start = time.time()
    image = example_2()
    plt.imsave(image_directory + "result.png", image)
    print(f"The program took {time.time() - start} seconds to run.")


if __name__ == '__main__':
    main()
