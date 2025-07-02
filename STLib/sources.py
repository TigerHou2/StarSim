import numpy as np

from typing import Union
from numpy.typing import ArrayLike
from astropy import units
from .utils import type_checker

class Sources:

    @type_checker
    def __init__(self, *,
                 ra : Union[ArrayLike, None],
                 dec: Union[ArrayLike, None],
                 source_id: Union[ArrayLike, None]):
        
        '''
        Base class for astronomical sources. 
        '''
        if ra is None or dec is None or source_id is None:
            self.ra  = np.array([]) * units.degree
            self.dec = np.array([]) * units.degree
            self.source_id = []
            return

        self.ra = ra.to(units.degree)
        self.dec = dec.to(units.degree)
        self.source_id = source_id


class SpectralSources(Sources):

    @type_checker
    def __init__(self, *,
                 ra : ArrayLike,
                 dec: ArrayLike,
                 source_id: ArrayLike,
                 spectral_flux_density  : ArrayLike,
                 sfd_wavelength         : ArrayLike):
        
        '''
        Define a collection of sources (stars, planetary bodies, etc.) by their right ascension, declination, id,
            and a list of spectral flux densities and wavelength arrays defining the flux profile of each source:
                - spectral flux density (energy ⋅ time⁻¹ ⋅ area⁻¹ ⋅ wavelength⁻¹)
                - spectral flux density sample points (wavelength)
        '''
        super().__init__(ra=ra, dec=dec, source_id=source_id)
        self.spectral_flux_density = spectral_flux_density.to(units.W / units.nm / units.m**2)
        self.sfd_wavelength = sfd_wavelength.to(units.nm)


class MagnitudeSources(Sources):

    @type_checker
    def __init__(self, *,
                 ra : ArrayLike,
                 dec: ArrayLike,
                 source_id: ArrayLike,
                 magnitude: ArrayLike):
        
        '''
        Define a collection of sources (stars, planetary bodies, etc.) by their right ascension, declination, id,
            and magnitude.
        '''
        super().__init__(ra=ra, dec=dec, source_id=source_id)
        self.magnitude = magnitude