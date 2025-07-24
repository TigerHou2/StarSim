import numpy as np
from astropy import units
from scipy.special import j0, j1
from scipy.integrate import quad_vec
from scipy.stats import multivariate_normal

import warnings

from typing import Union
Number = Union[int, float]


def airyPSFModel(wavelength, aperture, focal_length):
    wavelength = wavelength.to(units.micron).value
    aperture_radius = aperture.to(units.micron).value/2  # the Airy disk calcs use aperture **radius**, not diameter
    focal_length = focal_length.to(units.micron).value
    def airyPSF(x: Number, y: Number) -> float:
        r = np.sqrt(x**2 + y**2)
        v = 2 * np.pi * aperture_radius * r / wavelength / np.sqrt(r**2 + focal_length**2)
        v += (v==0) * 1e-16             # FIXME: find a better smooth way to handle discontinuity at x,y = 0?
        psf = (2.0 * j1(v) / v)**2

        # normally this would be the end of the Airy PSF calculations. 
        # However, since we expect the PSF to directly multiply the flux (power)
        # whereas the Airy PSF definition uses the maximum intensity (power * area / lambda^2 / f^2), 
        # we need to add the rest of the terms here.
        return psf * np.pi * aperture_radius**2 / wavelength**2 / focal_length**2
    return airyPSF


def gaussianPSFModel(cov):
    cov = cov.to(units.micron**2).value
    distr = multivariate_normal(mean=[0,0], cov=cov)
    def gaussianPSF(x: Number, y: Number) -> float:
        shape = x.shape
        return distr.pdf(np.array([np.ravel(x),np.ravel(y)]).T).reshape(shape)
    return gaussianPSF


def pillboxPSFModel(radius, sharpness=50):
    radius = radius.to(units.micron).value
    def pillboxPSF(x: Number, y: Number) -> float:
        r2 = x**2 + y**2
        return 1/(1+np.exp(sharpness*(r2-radius**2)/radius**2))/(np.pi*radius**2)
    return pillboxPSF


# https://www.strollswithmydog.com/a-simple-model-for-sharpness-in-digital-cameras-defocus/
# Eq. 40 of https://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/03-BasicAberrations_and_Optical_Testing.pdf
def defocusPSFModel(wavelength, aperture, focal_length, defocus, legendre_deg=50, nowarn=False):
    if not nowarn:
        warnings.warn("The Wyant and Creath defocused PSF model is very expensive to compute - you have been warned!")  # pragma: no cover
    wavelength = wavelength.to(units.micron).value
    aperture = aperture.to(units.micron).value
    focal_length = focal_length.to(units.micron).value
    defocus = defocus.to(units.micron).value

    rho, w = np.polynomial.legendre.leggauss(legendre_deg)
    rho = (rho + 1) / 2
    w /= 2

    N = focal_length / aperture

    def defocusPSF(x: Number, y: Number) -> float:
        r = np.sqrt(x**2 + y**2)[..., np.newaxis]
        integrand = np.exp(1j*2*np.pi/wavelength*defocus*rho**2) * j0(np.pi*r*rho/wavelength/N) * rho
        integral = np.sum(integrand * w, axis=-1)
        return np.pi / (wavelength * N)**2 *  (np.real(integral)**2 + np.imag(integral)**2)
    return defocusPSF
