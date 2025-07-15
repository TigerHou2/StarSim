[![Unit Tests](https://github.com/TigerHou2/star-tracker-image-sim/actions/workflows/unit_test.yml/badge.svg?branch=main)](https://github.com/TigerHou2/star-tracker-image-sim/actions/workflows/unit_test.yml)
[![codecov](https://codecov.io/github/TigerHou2/star-tracker-image-sim/branch/main/graph/badge.svg?token=S7B8ZHHLOR)](https://codecov.io/github/TigerHou2/star-tracker-image-sim)

# Star Tracker Image Simulator

This package simulates star tracker images of distant stars (and hopefully unresolved asteroids/planets in the future) given lens + sensor specifications and camera orientation. 

The effects considered include lens distortion, the point spread function, transmission/filter/quantum efficiencies, shot noise, dark current, read noise, saturation, and bloom. 

[__Check out an example here__](https://github.com/TigerHou2/star-tracker-image-sim/blob/main/examples/gaia.ipynb).


## Available Models

### Lens Distortion
- Brown-Conrady

### PSF
- Pillbox
- Gaussian
- Airy
- Defocus (Wyant and Creath)

### Filter/Quantum Efficiency
- Integrate over stellar spectra given transmission curve
- Average over designated bandpass if only magnitude + zero point available

### Dark Current
- Exponential temperature

### Bloom
- Unidirectional/Symmetric bloom along 0, 1, or 2 axes

### Shutter
- Global shutter

