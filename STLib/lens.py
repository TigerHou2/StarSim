from astropy import units
import numpy as np

from numpy.typing import ArrayLike
from typing import Union, Callable

from .utils import type_checker, Number

class Lens:

    @type_checker
    def __init__(self, *, 
                 aperture       : units.Quantity,
                 focal_length   : units.Quantity,
                 transmission_efficiency: Number,
                 psf: Callable[[Number, Number], float],
                 psf_bounds: Union[Number, tuple[Number, Number]],
                 k1 : Number = 0,
                 k2 : Number = 0,
                 k3 : Number = 0,
                 k4 : Number = 0,
                 k5 : Number = 0,
                 k6 : Number = 0,
                 p1 : Number = 0,
                 p2 : Number = 0):
    
        self.D = aperture.to(units.mm)
        self.f = focal_length.to(units.mm)
        self.transmission_eff = float(transmission_efficiency)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.k3 = float(k3)
        self.k4 = float(k4)
        self.k5 = float(k5)
        self.k6 = float(k6)
        self.p1 = float(p1)
        self.p2 = float(p2)
        self.psf = psf
        self.area = np.pi * (self.D/2)**2
    
        if isinstance(psf_bounds, tuple):
            self.psf_bounds_x = float(psf_bounds[0])
            self.psf_bounds_y = float(psf_bounds[1])
        else:
            self.psf_bounds_x = float(psf_bounds)
            self.psf_bounds_y = float(psf_bounds)

    def applyDistortion(self, 
                         x: ArrayLike,   # normalized coordinates (0-1, relative to top left corner)
                         y: ArrayLike):
        x2 = (x-0.5)**2
        y2 = (y-0.5)**2
        xy = (x-0.5) * (y-0.5)
        r2 = x2 + y2
        r4 = r2**2
        r6 = r2*r4
        num = (1 + self.k1*r2 + self.k2*r4 + self.k3*r6)
        den = (1 + self.k4*r2 + self.k5*r4 + self.k6*r6)
        xd = (x-0.5) * num/den + 2*self.p1*xy + self.p2*(r2+2*x2) + 0.5
        yd = (y-0.5) * num/den + 2*self.p2*xy + self.p1*(r2+2*y2) + 0.5
        return xd, yd
        