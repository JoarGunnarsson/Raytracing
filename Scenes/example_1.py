import objects
import materials
from constants import *
import Scenes.Scene as Scene
scene_objects = [objects.Sphere(x=0, y=0, z=-1000000, radius=1000000,
                                material=materials.Material(diffuse_color=WHITE,
                                                            reflection_coefficient=0, smoothness=1)),
                 objects.Sphere(x=4, y=0, z=1, radius=1,
                                material=materials.Material(diffuse_color=BLUE,
                                                            reflection_coefficient=0.1, shininess=10)),
                 objects.Sphere(y=2, z=1.25, radius=0.5,
                                material=materials.Material(diffuse_color=YELLOW,
                                                            reflection_coefficient=0.5))]
light_sources = [objects.PointSource(x=4, y=0, z=5)]
ambient_light = objects.AmbientLight(intensity=1/10, color=WHITE)
camera = objects.Camera(x=0, y=1, z=4)
screen = camera.screen

scene = Scene.Scene(scene_objects, light_sources, camera, screen, ambient_light)
