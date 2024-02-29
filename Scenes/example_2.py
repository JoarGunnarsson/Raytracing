import objects
import materials
from constants import *
import Scenes.Scene as Scene
scene_objects = [objects.Sphere(z=-1000000, radius=1000000,
                                material=materials.Material(diffuse_color=WHITE, specular_coefficient=0.3,
                                                            reflection_coefficient=0, transparency_coefficient=0, refractive_index=1)),
                 objects.Sphere(z=1, radius=1,
                                material=materials.Material(diffuse_color=WHITE, reflection_coefficient=0.1, transparency_coefficient=0.9, refractive_index=1.05)),
                 objects.Sphere(y=2, z=1.25, radius=0.5),
                 objects.Sphere(x=2.5, y=1.5, z=2, radius=0.5,
                                material=materials.Material(diffuse_color=RED, transparency_coefficient=0.5, refractive_index=1.001))
                 ]
light_sources = [objects.PointSource(x=4, y=0, z=10, intensity=10**2)]
camera = objects.Camera(x=-2, y=1, z=1, viewing_direction=np.array([1.0, 0.0, 0.0]))
screen = camera.screen

scene = Scene.Scene(scene_objects, light_sources, camera, screen)
