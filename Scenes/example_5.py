import objects
import materials
import colors
import Scenes.Scene as Scene
import numpy as np


floor = objects.Plane(x=0, y=0, z=-0.0001, v1=np.array([1.0, 0.0, 0.0]), v2=np.array([0.0, 1.0, 0.0]))

cube = objects.Cuboid(x=0.0, y=-0.5, z=0.5, v1=np.array([1.0, 0.0, 0.0]), v2=np.array([0.0, 1.0, 0.0]),
                         v3=np.array([0.0, 0.0, 1.0]), width=1, depth=1, height=1,
                         material=materials.Material(diffuse_color=colors.RED, smoothness=1,
                                                     transparency_coefficient=0.75, refractive_index=1.5,
                                                     attenuation_coefficient=0.05))

bottom = objects.Rectangle(x=0.5, y=0.5, z=0.0, v1=np.array([0.0, 1.0, 0.0]), v2=np.array([1.0, 0.0, 0.0]), L1=1, L2=1)
side1 = objects.Triangle(p1=np.array([0.0, 1.0, 0.0]), p2=np.array([0.0, 0.0, 0.0]), p3=np.array([0.5, 0.5, 1]))
side2 = objects.Triangle(p1=np.array([1.0, 1.0, 0.0]), p2=np.array([1.0, 0.0, 0.0]), p3=np.array([0.5, 0.5, 1]))
side3 = objects.Triangle(p1=np.array([1.0, 1.0, 0.0]), p2=np.array([0.0, 1.0, 0.0]), p3=np.array([0.5, 0.5, 1]))
side4 = objects.Triangle(p1=np.array([0.0, 0.0, 0.0]), p2=np.array([1.0, 0.0, 0.0]), p3=np.array([0.5, 0.5, 1]))
pyramid = objects.ObjectUnion([bottom, side1, side2, side3, side4],
                              material=materials.Material(diffuse_color=colors.RED, transparency_coefficient=0.75,
                                                          refractive_index=1.5))
scene_objects = [floor,
                 cube,
                 pyramid]

light_sources = [objects.PointSource(x=4, y=4, z=10, intensity=10 ** 2)]
camera = objects.Camera(x=-2, y=2, z=1.5, viewing_direction=np.array([1.0, -1.0, -0.3]))
screen = camera.screen
ambient_light = objects.AmbientLight(intensity=1/3, color=colors.WHITE)

scene = Scene.Scene(scene_objects, light_sources, camera, screen, ambient_light)