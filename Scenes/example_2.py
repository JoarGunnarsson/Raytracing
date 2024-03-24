import objects
import materials
import colors
import Scenes.Scene as Scene
import numpy as np

scene_objects = [objects.Sphere(x=0, y=0, z=-1000000, radius=1000000,
                                material=materials.Material(diffuse_color=colors.WHITE, smoothness=0.4)),
                 objects.Sphere(x=4, y=0, z=1, radius=1,
                                material=materials.Material(diffuse_color=colors.WHITE, reflection_coefficient=0.1,
                                                            transparency_coefficient=0.9, refractive_index=1.05)),
                 objects.Sphere(x=4, y=2, z=1.25, radius=0.5),
                 objects.Sphere(x=2.5, y=1.5, z=2, radius=0.5,
                                material=materials.Material(diffuse_color=colors.RED, transparency_coefficient=0.5))
                 ]
light_sources = [objects.PointSource(x=4, y=0, z=10, intensity=10**2)]
camera = objects.Camera(x=-2, y=1, z=1, viewing_direction=np.array([1.0, 0.0, 0.0]))
screen = camera.screen
ambient_light = objects.AmbientLight(intensity=1/10, color=colors.WHITE)

scene = Scene.Scene(scene_objects, light_sources, camera, screen, ambient_light)
