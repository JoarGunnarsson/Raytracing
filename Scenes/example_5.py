import objects
import materials
import colors
import Scenes.Scene as Scene
import numpy as np


my_cube = objects.Cuboid(x=0, y=0, z=0.7, v1=np.array([1.0, 0.0, 0.0]), v2=np.array([0.0, 1.0, 0.0]),
                         v3=np.array([0.0, 0.0, 1.0]), width=1.2, depth=1.2, height=1.2,
                         material=materials.Material(diffuse_color=colors.RED, smoothness=1, transparency_coefficient=0.5, refractive_index=1.5))

plane = objects.Rectangle(x=-0.3, y=0.3, z=0.4, v1=np.array([1.0, 0.0, 0.0]), v2=np.array([0.0, 1.0, 0.0]))
plane2 = objects.Rectangle(x=-0.3, y=0.3, z=0.1, v1=np.array([1.0, 0.0, 0.0]), v2=np.array([0.0, 1.0, 0.0]))
union = objects.ObjectUnion(objects=[plane], material=materials.Material(diffuse_color=colors.RED,
                                                            reflection_coefficient=0, smoothness=0, transparency_coefficient=0))

scene_objects = [objects.Plane(x=0, y=0, z=0,  v1=np.array([1.0, 0.0, 0.0]), v2=np.array([0.0, 1.0, 0.0]),
                                material=materials.Material(diffuse_color=colors.GREEN,
                                                            reflection_coefficient=0, smoothness=0, transparency_coefficient=0)),

                 my_cube,
                 ]
light_sources = [objects.PointSource(x=4, y=4, z=10, intensity=10 ** 2)]
camera = objects.Camera(x=-2.3, y=2.3, z=1, viewing_direction=np.array([1.0, -1.0, -0.1]))
screen = camera.screen
ambient_light = objects.AmbientLight(intensity=1/10, color=colors.WHITE)

scene = Scene.Scene(scene_objects, light_sources, camera, screen, ambient_light)
