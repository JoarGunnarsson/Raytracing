import objects
import materials
import colors
import Scenes.Scene as Scene
import numpy as np

scene_objects = [objects.Plane(z=0, v1=np.array([1.0, 0.0, 0.0]), v2=np.array([0.0, 1.0, 0.0]),
                                material=materials.Material(diffuse_color=colors.WHITE,
                                                            reflection_coefficient=0, smoothness=0.5)),
                 objects.Sphere(x=4, y=0, z=3, radius=3,
                                material=materials.Material(diffuse_color=colors.WHITE, reflection_coefficient=0,
                                                            transparency_coefficient=1, refractive_index=1.03,
                                                            smoothness=1)),
                 objects.Sphere(x=30, y=0, z=5, radius=5,
                                material=materials.Material(diffuse_color=colors.WHITE, reflection_coefficient=0,
                                                            transparency_coefficient=1, refractive_index=1.5,
                                                            smoothness=1)),

                 ]
light_sources = [objects.PointSource(x=4, y=20, z=10, intensity=50 ** 2)]
camera = objects.Camera(x=-4, y=1, z=1, viewing_direction=np.array([1.0, 0.0, -0.1]))
screen = camera.screen
ambient_light = objects.AmbientLight(intensity=1/10, color=colors.WHITE)
scene = Scene.Scene(scene_objects, light_sources, camera, screen, ambient_light)
