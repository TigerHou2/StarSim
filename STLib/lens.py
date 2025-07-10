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
                 auto_tune_integration_params: bool = True,
                 psf_bounds: Union[None, units.Quantity, tuple[units.Quantity, units.Quantity]] = None,
                 psf_resolution: Union[None,units.Quantity] = None,
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
        self.psf_bounds_x = None
        self.psf_bounds_y = None

        if auto_tune_integration_params:
            psf_bounds, psf_resolution = self.tuneIntegrationParams()
        elif psf_bounds is None or psf_resolution is None:
            raise ValueError("If auto_tune_integration_params == False, you must specify the PSF bounds of"
                             "integration and grid resolution for integration. ")
    
        self.psd_bounds = psf_bounds
        self.psf_resolution = psf_resolution

    @property
    def psf_bounds(self):
        return (self.psf_bounds_x, self.psf_bounds_y)
    @psf_bounds.setter
    def psf_bounds(self, psf_bounds):
        if isinstance(psf_bounds, tuple):
            self.psf_bounds_x = psf_bounds[0].to(units.micron)
            self.psf_bounds_y = psf_bounds[1].to(units.micron)
        else:
            self.psf_bounds_x = psf_bounds.to(units.micron)
            self.psf_bounds_y = psf_bounds.to(units.micron)


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
        

    def tuneIntegrationParams(self,atol=1e-2,rtol=1e-3,btol=1e-1,gtol=1e-1):
        '''
        Calculate recommended psf_bounds and grid_size based on desired absolute and relative tolerance. 
        - atol: Limits absolute error of integrated PSF (integral from -∞ to ∞ should equal 1, 
                although this is not possible with either the Airy disk or the defocused PSF model).
        - rtol: Limits integrated PSF error difference between the recommended grid size and maximum grid size.
        - btol: minimum change for tuning psf_bounds.
        - gtol: minimum change for tuning grid_size.
        '''
        MIN_GRID_SIZE = 0.1  # microns
        bounds = 5  # microns, initial guess
        grid = MIN_GRID_SIZE

        bounds_lb = None
        bounds_ub = None
        grid_lb = None
        grid_ub = None

        def calcAbsError(b,g): 
            n = int(np.ceil(b/g))
            _x = np.linspace(-b,b,n)
            _y = np.linspace(-b,b,n)
            integral = np.trapezoid(np.trapezoid(self.psf(*np.meshgrid(_x,_y)),_y,axis=0),_x,axis=0)
            return abs(integral-1)

        abs_converged = False
        while True:
            abs_error_truth = calcAbsError(bounds, MIN_GRID_SIZE)

            if abs_converged:
                abs_error = calcAbsError(bounds, grid)
                rel_error = abs(abs_error - abs_error_truth)

                if rel_error > rtol:
                    grid_ub = grid
                    if grid_lb is not None:
                        grid = (grid_lb + grid_ub) / 2
                    else:
                        grid /= 2
                    continue
                else:
                    grid_lb = grid
                    if grid_ub is not None:
                        if (grid_ub - grid_lb) <= gtol:
                            return bounds * units.micron, grid * units.micron
                        grid = (grid_lb + grid_ub) / 2
                    else:
                        grid *= 2
                    continue

            else:
                if abs_error_truth > atol:
                    bounds_lb = bounds
                    if bounds_ub is not None:
                        bounds = (bounds_lb + bounds_ub) / 2
                    else:
                        bounds *= 2
                    continue
                else:
                    bounds_ub = bounds
                    if bounds_lb is not None:
                        if (bounds_ub - bounds_lb) < btol:
                            abs_converged = True
                            continue
                        bounds = (bounds_lb + bounds_ub) / 2
                    else:
                        bounds /= 2
                    continue

            


