import numpy as np
from astropy import units
import astropy.constants as const
from astropy.time import Time
from astropy.modeling.rotations import EulerAngleRotation

import spiceypy as spice

from scipy.interpolate import CubicSpline

from .lens import Lens
from .filter import Filter
from .sensor import Sensor
from .sources import AstronomicalSources, SpectralSources, MagnitudeSources, SolarSystemSources
from .utils import type_checker, Number, timer
from . import LEAPSECONDS_KERNEL, DE_KERNEL, JOHNSON_V_FILTER_PATH

from typing import Union
from numpy.typing import ArrayLike

class Camera:

    @type_checker
    def __init__(self, *, 
                 lens   : Lens,
                 filter_: Filter,
                 sensor : Sensor,
                 sources: Union[AstronomicalSources, list],
                 sky_mag: Number,
                 kernel : str, 
                 ra     : units.Quantity = 0 * units.degree,
                 dec    : units.Quantity = 0 * units.degree,
                 roll   : units.Quantity = 0 * units.degree,
                 time   : Time = Time("2025-01-01", format='iso', scale='utc')
                 ):
        
        self._lens        = None
        self._filter      = None
        self._sensor      = None
        self._sources     = None
        self._source_ra   = None
        self._source_dec  = None
        self._photon_flux_densities = None
        self._plate_scale = None
        self._px_scale_x  = None
        self._px_scale_y  = None
        self._fov_x       = None
        self._fov_y       = None
        self._sky_mag     = None
        self._bg_photon_flux = None
        self._ra          = None
        self._dec         = None
        self._roll        = None
        self._kernel      = '_empty.bsp'
        self._spkid       = None
        self._pos         = None
        self._vel         = None
        self._time        = None
        zero_point_V = (363.1e-110 * units.erg / units.cm**2 / units.s / units.angstrom).to(units.W/units.nm/units.m**2)
        self._filter_V    = Filter(zp_flux=zero_point_V, file=JOHNSON_V_FILTER_PATH)

        self.kernel       = kernel
        self.sources      = sources
        self.filter_      = filter_
        self.lens         = lens
        self.sensor       = sensor
        self.sky_mag      = sky_mag
        self.orientation  = [ra, dec, roll]
        self.time         = time


    @property
    def sources(self):
        return self._sources
    @sources.setter
    def sources(self, sources: Union[AstronomicalSources, SolarSystemSources, list]):
        if not isinstance(sources, list):
            self._sources = [sources]
        else:
            self._sources = sources

        if self._time is not None:
            spice.furnsh(self._kernel)
            for source in self._sources:
                if isinstance(source, SolarSystemSources):
                    source.update(self._spkid, self._time)
            spice.unload(self._kernel)
                
        self.updateFlux()

    @property
    def filter_(self):
        return self._filter
    @filter_.setter
    def filter_(self, filter_: Filter):
        self._filter = filter_
        self.updateFlux()
    
    @property
    def lens(self):
        return self._lens
    @lens.setter
    def lens(self, lens: Lens):
        self._lens = lens
        self._plate_scale = 206265. * units.arcsec / lens.f.to(units.mm)
        if self._sensor is not None:
            self._fov_x = 2 * np.arctan(self._sensor.width.to( units.mm).value / 2 / self._lens.f.to(units.mm).value) * units.radian
            self._fov_y = 2 * np.arctan(self._sensor.height.to(units.mm).value / 2 / self._lens.f.to(units.mm).value) * units.radian
        self.updateFlux()

    @property
    def sensor(self):
        return self._sensor
    @sensor.setter
    def sensor(self, sensor: Sensor):
        self._sensor = sensor
        self._px_scale_x = (self._plate_scale * self._sensor.px_len_x).to(units.arcsec)
        self._px_scale_y = (self._plate_scale * self._sensor.px_len_y).to(units.arcsec)
        if self._lens is not None:
            self._fov_x = 2 * np.arctan(self._sensor.width.to( units.mm).value / 2 / self._lens.f.to(units.mm).value) * units.radian
            self._fov_y = 2 * np.arctan(self._sensor.height.to(units.mm).value / 2 / self._lens.f.to(units.mm).value) * units.radian
        self.updateFlux()

    @property
    def plate_scale(self):
        return self._plate_scale
    
    @property
    def px_scale(self):
        return [self._px_scale_x, self._px_scale_y]
    
    @property
    def fov(self):
        return [self._fov_x.to(units.degree), self._fov_y.to(units.degree)]
    
    @property
    def sky_mag(self):
        return self._sky_mag
    @sky_mag.setter
    def sky_mag(self, sky_mag: Number):
        self._sky_mag = sky_mag
        self.updateFlux()
    
    @property
    def ra(self):
        return self._ra
    @ra.setter
    def ra(self, ra: units.Quantity):
        self._ra = ra.to(units.degree)
        self._calcCoordTransforms()
    
    @property
    def dec(self):
        return self._dec
    @dec.setter
    def dec(self, dec: units.Quantity):
        self._dec = dec.to(units.degree)
        self._calcCoordTransforms()
    
    @property
    def roll(self):
        return self._roll
    @roll.setter
    def roll(self, roll: units.Quantity):
        self._roll = roll.to(units.degree)
        self._calcCoordTransforms()
    
    @property
    def orientation(self):
        return (self._ra, self._dec, self._roll)
    @orientation.setter
    def orientation(self, rotations: ArrayLike):
        self._ra   = rotations[0].to(units.deg)
        self._dec  = rotations[1].to(units.deg)
        self._roll = rotations[2].to(units.deg)
        self._calcCoordTransforms()

    @property
    def kernel(self):
        return self._kernel
    @kernel.setter
    def kernel(self, kernel: str):
        self._kernel = kernel
        spice.furnsh(self._kernel)
        self._spkid = str(spice.spkobj(self._kernel)[0])
        spice.unload(self._kernel)
        # TODO: recompute solar system objects, stellar aberration, and redshift

    @property
    def spkid(self):
        return self._spkid

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def time(self):
        return self._time
    @time.setter
    def time(self, time):
        self._time = time
        
        spice.furnsh([LEAPSECONDS_KERNEL, DE_KERNEL, self._kernel])

        et = spice.datetime2et(time.to_datetime())
        state = spice.spkezr(self._spkid, et, "J2000", "NONE", "SSB")[0]
        self._pos = state[:3] * units.km
        self._vel = state[3:] * units.km / units.s
        spice.unload([LEAPSECONDS_KERNEL, DE_KERNEL])

        # TODO: recompute solar system objects, stellar aberration, and redshift
        for source in self._sources:
            if isinstance(source, SolarSystemSources):
                source.update(self._spkid, self._time)
        spice.unload(self._kernel)

        self.updateFlux()


    
    def _calcCoordTransforms(self):
        '''
        The rotation from the camera frame to the world frame is given by the intrinsic rotation sequence x-y-z:
            Rz(ra) * Ry(-dec) * Rx(roll)  (note that the first rotation is the rightmost matrix)
        The image frame +x and +y axes align with the camera frame +y and -z axes (or ra and dec), respectively.
        Note that astropy's EulerAngleRotation applies intrinsic rotations. 
        '''
        # as of astropy 7.1.0, the implementation of EulerAngleRotation follows the **left** hand rule
        # contrary to the documentation (see https://github.com/astropy/astropy/issues/13134)
        # therefore, we need to invert the angles to use the right hand rule
        # additionally, since declination actually follows the left hand rule, it needs no inversion

        self._coordsCameraToWorld = EulerAngleRotation(-self._roll, self._dec, -self._ra, 'xyz')

        # the inverse transformation

        self._coordsWorldToCamera = EulerAngleRotation(self._ra, -self._dec, self._roll, 'zyx')


    def updateFlux(self):

        if (self._filter is None) or \
           (self._lens is None) or (self._sensor is None) or \
           (self._sky_mag is None):
            return
        
        photon_flux_densities = []
        ra = []
        dec = []

        for sources in self._sources:
        
            if len(sources.ra) == 0:
                photon_flux_densities.append(np.array([]))
            elif isinstance(sources, SpectralSources):
                photon_flux_densities.append(self.calcPhotonFluxDensityFromSpectral(sources))
            elif isinstance(sources, MagnitudeSources):
                photon_flux_densities.append(self.calcPhotonFluxDensityFromMagnitude(sources))
            elif isinstance(sources, SolarSystemSources):
                photon_flux_densities.append(self.calcPhotonFluxDensitySmallBody(sources))
            else:
                raise ValueError("Unknown light source format. ")
            
            ra.append(sources.ra)
            dec.append(sources.dec)
        
        self._photon_flux_densities = np.concatenate(photon_flux_densities)
        self._source_ra  = np.concatenate(ra)
        self._source_dec = np.concatenate(dec)
        
        self._bg_photon_flux = self.calcBackgroundFluxPerPixel()


    def snap(self, *, 
             exposure_time : float,
             temperature   : float,
             close_shutter : bool = False):
        
        # get the image coordinates of the stars in frame
        xcoords, ycoords, mask = self._getImageCoords(self._source_ra, self._source_dec)

        # if the shutter is closed, we see no stars or background
        if close_shutter:
            mask = []

        # TODO: implement rolling shutter (readout time)
        self._sensor.accumulate(lens=self._lens, exposure_time=exposure_time, temperature=temperature,
                                xcoords=xcoords, ycoords=ycoords, 
                                photon_flux_density=self._photon_flux_densities[mask],
                                background_flux=self._bg_photon_flux * (0.0 if close_shutter else 1.0))

        return self._sensor.readout().value


    def _getImageCoords(self, ra, dec):

        # get the undistorted light source coordinates on the image plane relative to the top left corner of the image
        # with right = +x, down = +y

        top_left_ra = -self._sensor.width/2 * self._plate_scale
        top_left_dec = self._sensor.height/2 * self._plate_scale
        
        rai_centered, deci_centered = self._coordsWorldToCamera(ra, dec)
        rai = rai_centered - top_left_ra
        deci = -(deci_centered - top_left_dec)

        # generate normalized coordinates and apply distortion

        ran  = (rai / (self._sensor.width * self._plate_scale)  ).to(units.dimensionless_unscaled).value
        decn = (deci / (self._sensor.height * self._plate_scale)).to(units.dimensionless_unscaled).value

        ran_distorted, decn_distorted = self._lens.applyDistortion(ran, decn)

        mask = (ran_distorted >=0) & (ran_distorted <= 1) \
             & (decn_distorted >=0) & (decn_distorted <= 1)

        x_distorted = ran_distorted[mask] * self._sensor.width
        y_distorted = decn_distorted[mask] * self._sensor.height

        return x_distorted, y_distorted, mask
    

    def calcPhotonFluxDensitySmallBody(self, sources: SolarSystemSources):

        # SolarSystemSources have phase curves defined in the V band, so we will need to do some extra work to be able to convert their spectra.
        # 1. using the normalized spectra, we can compute the normalized flux in V (P_V')
        # 2. using the phase curve, we can compute the magnitude in V -> therefore compute flux in V using the Vega zero point. (P_V)
        # 3. using the normalized spectra, we can compute the normalized flux in the camera's filter (P_F')
        # 4. finally, the actual flux in the camera's filter (P_F) = P_F' * (P_V / P_V')
        F_Vega = (363.1e-110 * units.erg / units.cm**2 / units.s / units.angstrom).to(units.W/units.nm/units.m**2)
        V_bandwidth = 85 * units.nm
        P_Vega = 995.5 * units.electron / units.cm**2 / units.s / units.angstrom * V_bandwidth

        tx_wavelengths = self._filter_V.wavelengths.to(units.nm)

        nsfd_spl = [CubicSpline(w.to(units.nm), s * F_Vega)(tx_wavelengths)
                    for w, s in zip(sources.sample_wavelengths, sources.normalized_spectra)]
        nsfd_spl = np.asarray(nsfd_spl) * units.W/units.nm/units.m**2 

        P_V_prime = self._calcPhotonFluxDensityFromSpectral(self._filter_V, nsfd_spl, tx_wavelengths)
        P_V = P_Vega * 10**(-sources.magnitudes/2.5)

        P_F_prime = self._calcPhotonFluxDensityFromSpectral(self._filter, nsfd_spl, tx_wavelengths)
        P_F = P_F_prime * P_V / P_V_prime
        return P_F.to(units.electron/units.s/units.m**2)
    

    def calcPhotonFluxDensityFromSpectral(self, sources: SpectralSources, sample_wavelengths=None):

        return self._calcPhotonFluxDensityFromSpectral(self._filter, sources.spectral_flux_density, sources.sfd_wavelength, sample_wavelengths)
    

    def _calcPhotonFluxDensityFromSpectral(self, filter_, spectral_flux_density, sfd_wavelengths, sample_wavelengths=None):

        spectral_flux_density = spectral_flux_density.to(units.W/units.nm/units.m**2)
        sfd_wavelengths = sfd_wavelengths.to(units.nm)
        transmission = filter_.transmission
        tx_wavelengths = filter_.wavelengths.to(units.nm)

        # use provided sampling wavelengths or choose the overlapping wavelengths between flux and transmission, preferring highest-density sampling
        if not sample_wavelengths:
            common_interval = [max(min(sfd_wavelengths),min(tx_wavelengths)), min(max(sfd_wavelengths),max(tx_wavelengths))]
            step = min(np.min(np.diff(sfd_wavelengths)), np.min(np.diff(tx_wavelengths)))
            sample_wavelengths = np.arange(start=common_interval[0].to(units.nm).value, 
                                           stop=common_interval[1].to(units.nm).value, 
                                           step=step.to(units.nm).value) * units.nm

        # cubic spline interp of spectral flux density and transmission
        # CubicSpline removes units to we need to add them back. Units already standardized at the beginning of the function.
        sfd_spl = CubicSpline(sfd_wavelengths, spectral_flux_density, axis=1)(sample_wavelengths, extrapolate=False) * units.W/units.nm/units.m**2
        tx_spl  = CubicSpline(tx_wavelengths, transmission)(sample_wavelengths, extrapolate=False)
        electron_energy = const.h * const.c / sample_wavelengths / units.electron

        # integrating just spectral flux density * transmission would give us irradiance [energy per time per area]
        # however, instead of energy, we care about the equivalent number of photons, i.e., photon flux density.
        photon_flux_density = np.trapezoid(sfd_spl * tx_spl / electron_energy, sample_wavelengths, axis=1) * self._lens.transmission_eff
        return photon_flux_density.to(units.electron/units.s/units.m**2)
    

    def calcPhotonFluxDensityFromMagnitude(self, sources: MagnitudeSources):

        return self._calcPhotonFluxDensityFromMagnitude(self._filter, sources.magnitude)
    

    def _calcPhotonFluxDensityFromMagnitude(self, filter_, magnitudes):

        flux = filter_.zp_flux * 10**((0. - magnitudes) / 2.5) * filter_.fwhm
        electron_energy = const.h * const.c / filter_.eff_wavelength / units.electron
        photon_flux_density = flux / electron_energy * self._lens.transmission_eff
        return photon_flux_density.to(units.electron/units.s/units.micron**2)


    def calcBackgroundFluxPerPixel(self):
        
        pixel_scale = (self._px_scale_x * self._px_scale_y / units.pixel).to(units.arcsec**2/units.pixel)

        # the sky background is everywhere, so the magnitude is given in mag/arcsec^2
        bg_photon_flux_density = self._calcPhotonFluxDensityFromMagnitude(self._filter, self._sky_mag) / (units.arcsec**2)

        # multiply by pixel scale to get rate per pixel
        bg_photon_flux = bg_photon_flux_density * pixel_scale

        return bg_photon_flux.to(units.electron/units.s/units.m**2/units.pixel)