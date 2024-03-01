import objects
import materials
from constants import *
import Scenes.Scene as Scene
scene_objects = [objects.Sphere(z=-1000000, radius=1000000,
                                material=materials.Material(diffuse_color=WHITE, specular_coefficient=0.3,
                                                            reflection_coefficient=0.05, transparency_coefficient=0, refractive_index=1)),
                objects.Sphere(z=2, radius=1,
                                material=materials.Material(diffuse_color=WHITE, reflection_coefficient=0, transparency_coefficient=0.9, refractive_index=1))
                 ]
light_sources = [objects.PointSource(x=4, y=-4, z=100, intensity=100**2)]
camera = objects.Camera(x=-4, y=1, z=2, viewing_direction=np.array([1, 0.0, 0.0]))
screen = camera.screen

scene = Scene.Scene(scene_objects, light_sources, camera, screen)
