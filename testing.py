import numpy as np

# Define the Sphere class with a position attribute
class Sphere:
    def __init__(self, position):
        self.position = position


# Example array of different sphere objects with positions
array_of_spheres = np.array([[Sphere(position=np.array([1, 2, 3])),
                             Sphere(position=np.array([4, 5, 6])),
                             Sphere(position=np.array([7, 8, 9]))],
                             [Sphere(position=np.array([1, 2, 3])),
                              Sphere(position=np.array([4, 5, 6])),
                              Sphere(position=np.array([7, 8, 9]))]]
                            )


# Define a function to extract the position from a Sphere object
def get_position(sphere):
    return np.array(sphere.position)


# Vectorize the get_position function
get_position_vectorized = np.vectorize(get_position, otypes=[object])

# Get the positions using the vectorized function
positions = np.array(get_position_vectorized(array_of_spheres))

print(positions)

print(positions.shape)