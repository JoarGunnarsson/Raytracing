import objects
import materials
import colors
import Scenes.Scene as Scene
import numpy as np


floor = objects.Plane(x=0, y=0, z=-0.001, v1=np.array([1.0, 0.0, 0.0]), v2=np.array([0.0, 1.0, 0.0]),
                      material=materials.Material(diffuse_color=colors.RED))

cube = objects.Cuboid(x=-2.0, y=0.0, z=-0, v1=np.array([1.0, 0.0, 0.0]), v2=np.array([0.0, 1.0, 0.0]),
                         v3=np.array([0.0, 0.0, 1.0]), width=2, depth=2, height=1,
                         material=materials.Material(diffuse_color=colors.BLUE, smoothness=0,
                                                     transparency_coefficient=0.75, refractive_index=1,
                                                     attenuation_coefficient=0.05))

suzanne = objects.load_object_from_file('suzanneTri.obj', size=2, material=materials.Material(diffuse_color=colors.WHITE,
                                                                                             transparency_coefficient=0.75, refractive_index=1))
scene_objects = [floor, suzanne]

light_sources = [objects.PointSource(x=2, y=2, z=1, intensity=30 ** 2)]
camera = objects.Camera(x=0, y=0, z=3, viewing_direction=np.array([0.0, 0.0, -1.0]), y_vector=np.array([0.0, 1.0, 0.0]))
screen = camera.screen
ambient_light = objects.AmbientLight(intensity=1/3, color=colors.WHITE)

scene = Scene.Scene(scene_objects, light_sources, camera, screen, ambient_light)
