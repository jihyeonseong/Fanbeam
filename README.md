# 2D CT Reconstruction with Fanbeam Geometry
* Contributors: @fxnnxc and @jihyeonseong

## Overview of Fanbeam Geometry
![image](https://github.com/user-attachments/assets/ae9eea29-a08c-43fa-b367-15acd61c437c)

Fanbeam geometry is a common acquisition setup used in computed tomography (CT) reconstruction. In this geometry, X-rays are emitted from a single focal point (source) and spread out in a fan-shaped configuration, passing through the object being imaged before being detected by a linear array of detectors.

The main components of fanbeam geometry are:

* X-ray source: A point source emits a divergent beam of X-rays that form a fan-shaped pattern.
* Detector array: A 1D array of detectors is placed opposite the source to measure the attenuated X-ray intensities after they pass through the object.
* Fan angle ($\theta$): The angle of the X-ray beam divergence, which determines the spread of the rays within the fan.
* Source-to-object distance ($R_s$): The distance from the X-ray source to the center of the object being imaged.
* Source-to-detector distance ($R_d$): The distance from the X-ray source to the detector array.

In fanbeam geometry, the X-ray paths are parameterized by the source angle $\beta$ (relative to a fixed reference axis) and the detector position. The measurement $p(\beta, s)$ at angle $\beta$ and detector position $s$ corresponds to the line integral of the attenuation coefficient $\mu(x, y)$ along the ray:

$$p(\beta, s) = \int_{L(\beta, s)} \mu(x, y) \, d\ell,$$

where:

* $L(\beta, s)$ is the ray path defined by the source angle $\beta$ and detector position $s$.
* $\mu(x, y)$ is the spatially varying attenuation coefficient of the object.


## Running the codes
```
python test.py
```
You can set your hyperparameters as you want!
