import objects
import materials
from constants import *
import Scenes.Scene as Scene
scene_objects = [objects.Sphere(z=-1000000, radius=1000000,
                                material=materials.Material(diffuse_color=WHITE,
                                                            reflection_coefficient=0)),
                 objects.Sphere(z=1, radius=1,
                                material=materials.Material(diffuse_color=WHITE, reflection_coefficient=0, transparency_coefficient=0.8, refractive_index=1.05)),
                objects.Sphere(x=30, z=1, radius=7,
                                material=materials.Material(diffuse_color=RED, reflection_coefficient=0, transparency_coefficient=0))
                 ]
light_sources = [objects.PointSource(x=4, y=0, z=40, intensity=40**2)]
camera = objects.Camera(x=0, y=1, z=1, viewing_direction=np.array([1.0, 0.0, -0.1]))
screen = camera.screen

scene = Scene.Scene(scene_objects, light_sources, camera, screen)
