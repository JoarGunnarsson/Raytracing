import objects
import materials
from constants import *
import Scenes.Scene as Scene
scene_objects = [objects.Sphere(z=-1000000, radius=1000000,
                                material=materials.Material(diffuse_color=WHITE, specular_coefficient=0.3,
                                                            reflection_coefficient=0.05, transparency_coefficient=0, refractive_index=1)),
                objects.Sphere(z=1, radius=1,
                                material=materials.Material(diffuse_color=WHITE, reflection_coefficient=0, transparency_coefficient=0.8, refractive_index=1.3)),
                objects.Sphere(x=1000 + 35, y=1, z=1, radius=1000,
                                material=materials.Material(diffuse_color=GREEN, reflection_coefficient=0, transparency_coefficient=0, refractive_index=1))
                 ]
light_sources = [objects.PointSource(x=4, y=-4, z=10, intensity=10**2)]
camera = objects.Camera(x=-80, y=1, z=40, viewing_direction=np.array([1, 0.0, -0.3]))
screen = camera.screen

scene = Scene.Scene(scene_objects, light_sources, camera, screen)
