import sys
sys.path.append('.')
from STLib.functions.psf import airyPSFModel, defocusPSFModel, pillboxPSFModel, gaussianPSFModel

from astropy import units
from scipy.integrate import dblquad
import numpy as np

wavelength = 550  # nm
aperture = 50  # mm
focal_length = 40  # mm

PSFs = [ \
    pillboxPSFModel(radius=4*units.micron),
    gaussianPSFModel(sigma=2*units.micron),
    airyPSFModel(wavelength=550*units.nm, aperture=aperture*units.mm, focal_length=focal_length*units.mm), 
    defocusPSFModel(wavelength=wavelength*units.nm, aperture=aperture*units.mm, focal_length=focal_length*units.mm, defocus=0.5*wavelength*units.nm, nowarn=True),
]

TOL = 1.0e-2

for psf in PSFs:
    _x = np.linspace(-10,10,100)
    _y = np.linspace(-10,10,100)
    integral = np.trapezoid(np.trapezoid(psf(*np.meshgrid(_x,_y)),_y,axis=0),_x,axis=0)
    integral_error = abs(integral-1)
    assert integral_error < TOL
    print(f"{psf.__name__} passed with error {integral_error}")