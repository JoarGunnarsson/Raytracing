class Scene:
    def __init__(self, objects, light_sources, camera, screen, ambient_light=None):
        self.objects = objects
        self.light_sources = light_sources
        self.ambient_light = ambient_light
        self.camera = camera
        self.screen = screen
        pass
