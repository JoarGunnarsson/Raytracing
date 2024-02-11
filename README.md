# Python Ray Tracing Project

This is a Python project for ray tracing, a technique used in computer graphics to generate an image by tracing the path of light takes, interaction with different objects and materials. This project provides a basic implementation of ray tracing using Python and NumPy. Diffusive and specular effects are available, as well as reflections. Currently, only spheres are implemented. Additionally, two different types of light sources have been implemented: a point light source and a disk light source


### Example scene
Below is an example scene, using two different kinds light sources.

| ![Point light](Images/point_reflections.png) Point light source | ![Disk light](Images/disk_reflections.png) Disk light source |
|:---------------------------------------------------------------:|:------------------------------------------------------------:|


## Overview

The project consists of several Python files, including:

- `main.py`: The main file that initiates the ray tracing process.
- `objects.py`: Defines classes for objects like the camera, screen, geometric shapes, and light sources.
  - Additional shapes and light sources can be added here
- `materials.py`: Defines material properties used for shading objects.
  - Additional material properties can be added for increased realism.

## Dependencies
- `numpy`
- `matplotlib`

## Usage

To run the ray tracing simulation and generate an image, simply execute the `main.py` file:

```
python main.py
```

You can define your own objects, light sources and materials in the `main.py`, `objects.py`, and `materials.py` to suit your needs.
