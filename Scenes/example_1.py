import objects
import materials
from constants import *
import Scenes.Scene as Scene
scene_objects = [objects.Sphere(z=-1000000, radius=1000000,
                                material=materials.Material(diffuse_color=WHITE, specular_coefficient=0.3,
                                                            reflection_coefficient=0.24)),
                 objects.Sphere(z=1, radius=1,
                                material=materials.Material(diffuse_color=BLUE, reflection_coefficient=0.1)),
                 objects.Sphere(y=2, z=1.25, radius=0.5, material=materials.Material(reflection_coefficient=0.5))]
light_sources = [objects.PointSource(x=4, y=0, z=5)]
camera = objects.Camera(x=0, y=1, z=4)
screen = camera.screen

scene = Scene.Scene(scene_objects, light_sources, camera, screen)
