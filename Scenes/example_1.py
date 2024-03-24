import objects
import materials
import colors
import Scenes.Scene as Scene
import numpy as np

scene_objects = [objects.Plane(x=0, y=0, v1=np.array([1.0, 0.0, 0.0]), v2=np.array([0.0, 1.0, 0.0]),
                                material=materials.Material(diffuse_color=colors.WHITE,
                                                            reflection_coefficient=0, smoothness=1)),
                 objects.Sphere(x=4, y=0, z=1, radius=1,
                                material=materials.Material(diffuse_color=colors.BLUE,
                                                            reflection_coefficient=0.1, shininess=10)),
                 objects.Sphere(y=2, z=1.25, radius=0.5,
                                material=materials.Material(diffuse_color=colors.YELLOW,
                                                            reflection_coefficient=0.5))]
light_sources = [objects.PointSource(x=4, y=0, z=5)]
ambient_light = objects.AmbientLight(intensity=1/10, color=colors.WHITE)
camera = objects.Camera(x=0, y=1, z=4)
screen = camera.screen

scene = Scene.Scene(scene_objects, light_sources, camera, screen, ambient_light)
