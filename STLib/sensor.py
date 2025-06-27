import numpy as np
from .lens import Lens

from scipy.integrate import dblquad
from scipy.signal import convolve2d

from astropy import units
import astropy.constants as const

from numpy.typing import ArrayLike
from typing import Union, Tuple, Callable
from .utils import timer

Number = Union[int, float]

# create a new equivalency where 1 pA = 6.28e6 e-/s
electron_current_density = [(units.pA/units.m**2, units.electron/units.s/units.m**2, lambda x: x * 6.28e6, lambda x: x / 6.28e6)]

class Sensor:

    def __init__(self, *, 
                 width_px               : int, 
                 height_px              : int, 
                 px_len                 : Union[Number, Tuple[Number, Number]], 
                 px_pitch               : Union[Number, Tuple[Number, Number]], 
                 quantum_efficiency     : Number, 
                 filter_efficiency      : Number,
                 band                   : str,
                 dark_current           : Callable[[Number], float], 
                 read_noise             : Number,
                 gain                   : Number, 
                 full_well_capacity     : Number, 
                 bloom                  : set, 
                 readout_time           : Number):
        
        self.width_px  = int(width_px)
        self.height_px = int(height_px)

        if isinstance(px_len, tuple):
            self.px_len_x = float(px_len[0]) * units.micron
            self.px_len_y = float(px_len[1]) * units.micron
            self.px_area = self.px_len_x * self.px_len_y
        else:
            self.px_len_x = float(px_len) * units.micron
            self.px_len_y = float(px_len) * units.micron
            self.px_area = self.px_len_x * self.px_len_y

        if isinstance(px_pitch, tuple):
            self.px_pitch_x = float(px_pitch[0]) * units.micron
            self.px_pitch_y = float(px_pitch[1]) * units.micron
        else:
            self.px_pitch_x = float(px_pitch) * units.micron
            self.px_pitch_y = float(px_pitch) * units.micron

        self.width  = (self.width_px +1) * self.px_pitch_x - self.px_len_x
        self.height = (self.height_px+1) * self.px_pitch_y - self.px_len_y

        self.quantum_eff    = float(quantum_efficiency)
        self.filter_eff = float(filter_efficiency)
    
        if band not in ['U','B','V','R','I','J','H','K','u','g','r','i','z']:
            raise ValueError(f"Argument `band` must be in [U,B,V,R,I,J,H,K,u,g,r,i,z], got `{band}`.")
        self.band = band
    
        self.dark_current   = dark_current  # pA / cm**2
        self.read_noise     = float(read_noise) * units.electron
        self.gain           = float(gain) * units.electron / units.adu
        self.full_well      = float(full_well_capacity) * units.electron
        self.bloom          = bloom
        self.readout_time   = float(readout_time) * units.s
        self.pixels         = np.zeros((self.height_px, self.width_px)) * units.electron

        if not self.bloom.issubset({'+x','-x','+y','-y'}):
            raise ValueError("Argument `bloom` must be a subset of {'+x','-x','+y','-y'}.")
        
    
    def clear(self):
        self.pixels = np.zeros((self.height_px, self.width_px)) * units.electron


    def accumulate(self, 
                   lens: Lens,
                   sky_mag: float,
                   exposure_time : float,
                   temperature   : float,
                   xcoords       : ArrayLike,   # distance coordinates (relative to top left corner)
                   ycoords       : ArrayLike, 
                   magnitudes    : ArrayLike):
        
        exposure_time *= units.s
        
        # source flux

        intensities = units.Quantity([self._getSourceFlux(magnitude=mag, lens=lens) * exposure_time for mag in magnitudes])
        self._applyPSF(lens, xcoords, ycoords, intensities)

        # sky background flux

        bg_count = self._getBackgroundFluxPerPixel(sky_mag=sky_mag, lens=lens) * exposure_time * units.pixel
        self.pixels += np.random.poisson(bg_count.to(units.electron).value, (self.height_px, self.width_px)) * units.electron

        # dark current

        dark_current = self.dark_current(temperature) * units.pA / units.cm**2
        dark_count = dark_current.to(units.electron/units.s/units.micron**2, equivalencies=electron_current_density) * exposure_time * self.px_area
        self.pixels += np.random.poisson(dark_count.to(units.electron).value, (self.height_px, self.width_px)) * units.electron
        
        # saturation and bloom

        self._applyBloom()


    def readout(self):

        # global shutter

        if self.readout_time == 0:
            
            # read noise
            self.pixels += np.random.poisson(self.read_noise.to(units.electron).value, (self.height_px, self.width_px)) * units.electron

            # analog to digital conversion
            return np.floor(self.pixels / self.gain)

        # TODO: implement rolling shutter?

        else:

            return self.pixels


    @timer
    def _applyPSF(self, 
                  lens          : Lens, 
                  xcoords       : ArrayLike, 
                  ycoords       : ArrayLike, 
                  intensities   : ArrayLike):
        
        for x, y, i in zip(xcoords, ycoords, intensities):

            # find the center pixel

            xi = int(np.round(x / (self.width  - 2*self.px_pitch_x + self.px_len_x) * self.width_px ).to(units.dimensionless_unscaled).value)
            yi = int(np.round(y / (self.height - 2*self.px_pitch_y + self.px_len_y) * self.height_px).to(units.dimensionless_unscaled).value)

            # integrate PSF over each pixel within lens.psf_bounds_x/y

            xmin = max(int(np.floor(xi-lens.psf_bounds_x)), 0)
            ymin = max(int(np.floor(yi-lens.psf_bounds_y)), 0)
            xmax = min(int(np.ceil( xi+lens.psf_bounds_x)), self.width_px)
            ymax = min(int(np.ceil( yi+lens.psf_bounds_y)), self.height_px)

            for idx in range(xmin,xmax):
                for idy in range(ymin,ymax):
                    x_lb = ((self.px_pitch_x * idx - self.px_len_x) - x).to(units.micron).value  # convert to distance units relative to psf center
                    x_ub = ((self.px_pitch_x * idx)                 - x).to(units.micron).value
                    y_lb = ((self.px_pitch_y * idy - self.px_len_y) - y).to(units.micron).value
                    y_ub = ((self.px_pitch_y * idy)                 - y).to(units.micron).value
                    # note that y-bounds go first because dblquad expects the function to be f(y,x) not f(x,y)
                    # but the psf we defined is psf(x,y)

                    _x = np.linspace(x_lb,x_ub,100)
                    _y = np.linspace(y_lb,y_ub,100)
                    count = i * np.trapezoid(np.trapezoid(lens.psf(*np.meshgrid(_x,_y)),_y,axis=0),_x,axis=0)

                    # if more precision is needed, can use dblquad (much slower!)
                    # note that y-bounds go first because dblquad expects the function to be f(y,x) not f(x,y)
                    # but the psf we defined is psf(x,y)
                    # count = i * dblquad(lens.psf, y_lb, y_ub, x_lb, x_ub, epsabs=1e-1, epsrel=1e-1)[0]  # TODO: assess integration error?
                    
                    self.pixels[idy,idx] += np.random.poisson(count.to(units.electron).value) * units.electron
                    
        return
        

    def _applyBloom(self):

        if not self.bloom:
            self.pixels = np.minimum(self.pixels, self.full_well)
            return
        
        conv_filter = np.zeros((3,3))
        frac = 1.0 / len(self.bloom)
        conv_filter[1,0] = frac if '-x' in self.bloom else 0
        conv_filter[1,2] = frac if '+x' in self.bloom else 0
        conv_filter[0,1] = frac if '-y' in self.bloom else 0
        conv_filter[2,1] = frac if '+y' in self.bloom else 0
        while True:
            excess = np.maximum(0, self.pixels - self.full_well)
            if not any(excess.flatten().value > 0):
                break
            self.pixels -= excess
            self.pixels += np.floor(convolve2d(excess.to(units.electron).value, conv_filter, mode='same', boundary='fill', fillvalue=0)) * units.electron
        
        return


    def _getSourceFlux(self, *, 
                       magnitude: Union[int, float], 
                       lens: Lens):
        # flux zeropoint reference:
        # https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
        match self.band:
            case 'U':
                f_ref        = 1.79e-20
                eff_lambda   = 0.36
                delta_lambda = 0.06
            case 'B':
                f_ref        = 4.063e-20
                eff_lambda   = 0.438
                delta_lambda = 0.09
            case 'V':
                f_ref        = 3.636e-20
                eff_lambda   = 0.545
                delta_lambda = 0.085
            case 'R':
                f_ref        = 3.064e-20
                eff_lambda   = 0.641
                delta_lambda = 0.15
            case 'I':
                f_ref        = 2.416e-20
                eff_lambda   = 0.798
                delta_lambda = 0.15
            case 'J':
                f_ref        = 1.589e-20
                eff_lambda   = 1.22
                delta_lambda = 0.26
            case 'H':
                f_ref        = 1.021e-20
                eff_lambda   = 1.63
                delta_lambda = 0.29
            case 'K':
                f_ref        = 0.64e-20
                eff_lambda   = 2.19
                delta_lambda = 0.41
            case 'u':
                f_ref        = 3.631e-20
                eff_lambda   = 0.356
                delta_lambda = 0.0463
            case 'g':
                f_ref        = 3.631e-20
                eff_lambda   = 0.483
                delta_lambda = 0.0988
            case 'r':
                f_ref        = 3.631e-20
                eff_lambda   = 0.626
                delta_lambda = 0.0955
            case 'i':
                f_ref        = 3.631e-20
                eff_lambda   = 0.767
                delta_lambda = 0.1064
            case 'z':
                f_ref        = 3.631e-20
                eff_lambda   = 0.910
                delta_lambda = 0.1248
            case _:
                raise ValueError(f"Sensor parameter `band` must be in [U,B,V,R,I,J,H,K,u,g,r,i,z], got `{self.band}`.")
            
        # calculate the average electron energy in the filter
        f_ref *= units.erg / units.cm**2 / units.s / units.Hz
        eff_lambda *= units.micron
        delta_lambda *= units.micron

        delta_wavelength = (eff_lambda - delta_lambda/2).to(units.Hz, equivalencies=units.spectral()) \
                         - (eff_lambda + delta_lambda/2).to(units.Hz, equivalencies=units.spectral())
        E_e = const.h * const.c / eff_lambda / units.electron
        
        # convert magnitude to flux density
        f_source = f_ref * 10**((0. - magnitude) / 2.5)

        # convert flux density to photon flux density
        f_photon = f_source / E_e

        # multiply by area and bandwidth to get photons per second
        photon_rate = f_photon * np.pi * (lens.D/2)**2 * delta_wavelength

        # total efficiency
        photon_rate *= self.quantum_eff * self.filter_eff * lens.transmission_eff

        return photon_rate.to(units.electron/units.s)


    def _getBackgroundFluxPerPixel( self, 
                                    sky_mag: Union[int, float], 
                                    lens: Lens):
        
        plate_scale = 206265 * units.arcsec / lens.f.to(units.mm)
        pixel_scale = (plate_scale**2 * self.px_area / units.pixel).to(units.arcsec**2/units.pixel)

        # the sky background is everywhere, so the magnitude is given in mag/arcsec^2
        bg_rate = self._getSourceFlux(magnitude=sky_mag, lens=lens) / (units.arcsec**2)

        # multiply by pixel size to get rate per pixel
        bg_rate = bg_rate * pixel_scale

        return bg_rate.to(units.electron/units.s/units.pixel)