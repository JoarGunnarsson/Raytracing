import objects
import materials
import colors
import Scenes.Scene as Scene
import numpy as np

scene_objects = [objects.Plane(z=0,  v1=np.array([1.0, 0.0, 0.0]), v2=np.array([0.0, 1.0, 0.0]),
                                material=materials.Material(diffuse_color=colors.WHITE,
                                                            reflection_coefficient=0, smoothness=1)),
                 objects.Sphere(x=0, y=0, z=1, radius=1,
                                material=materials.Material(diffuse_color=colors.GREEN, reflection_coefficient=0, specular_coefficient=1,
                                                            transparency_coefficient=0.9, refractive_index=1.05,
                                                            smoothness=1, attenuation_coefficient=0.5)),
                 objects.Sphere(x=15, y=0, z=2, radius=4,
                                material=materials.Material(diffuse_color=colors.RED,
                                                            smoothness=0.6))
                 ]
light_sources = [objects.DiskSource(x=4, y=0, z=10, intensity=10 ** 2)]
camera = objects.Camera(x=-4, y=1, z=1, viewing_direction=np.array([1.0, 0.0, -0.1]))
screen = camera.screen
ambient_light = objects.AmbientLight(intensity=1/10, color=colors.WHITE)

scene = Scene.Scene(scene_objects, light_sources, camera, screen, ambient_light)
