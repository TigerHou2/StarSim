import numpy as np
import os.path
import csv

from typing import Union
from numpy.typing import ArrayLike
from astropy import units
from astropy.time import Time
from datetime import datetime
from .utils import type_checker, Number, horizonsQuery, naifSchemaToExtended, naifSchemaToOriginal
from . import \
    DEFAULT_SMALL_BODIES_PATH, DEFAULT_CACHE_EXPIRATION_DAYS, \
    SMALL_BODIES_SPECTRA_CACHE_DIR, SMALL_BODIES_MEAN_SPECTRA_PATH, \
    LEAPSECONDS_KERNEL, DE_KERNEL

import spiceypy as spice


class AstronomicalSources:

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


class SpectralSources(AstronomicalSources):

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


class MagnitudeSources(AstronomicalSources):

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


class SolarSystemSources:

    @type_checker
    def __init__(self, *,
                 start_date: datetime,
                 end_date: datetime, 
                 small_bodies_query_results: str = DEFAULT_SMALL_BODIES_PATH,
                 max_cache_age: Number = DEFAULT_CACHE_EXPIRATION_DAYS):
        
        '''
        Define a collection of small bodies in the Solar System by:
        - a path to the SBDB query results obtained from `STLib.utils.sbdbQuery`, and
        - a list of SPICE kernels obtained from `STLib.utils.horizonsQuery`
        '''

        # use the results of an existing SBDB query to download the appropriate small body SPICE kernels
        with open(small_bodies_query_results, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            spkids = []
            names  = []
            spec_B = []
            spec_T = []
            H = []
            G = []
            for row in reader:
                spkids.append(naifSchemaToExtended(row[header.index("spkid")]))
                names.append(row[header.index("name")])
                spec_B.append(row[header.index("spec_B")])
                spec_T.append(row[header.index("spec_T")])
                H.append(float(row[header.index("H")]))
                _G_val = row[header.index("G")]
                if _G_val == '':
                    G.append(0.15)  # default slope parameter: https://adsabs.harvard.edu/full/2007JBAA..117..342D
                else:
                    G.append(float(_G_val))

        small_bodies_kernels = horizonsQuery(start_date, end_date, small_bodies_query_results, max_cache_age)

        self.names = names
        self.spkids = spkids
        self.kernels = small_bodies_kernels
        self.H = np.array(H)
        self.G = np.array(G)
        self.spec = [b if b != '' else t for b, t in zip(spec_B, spec_T)]
        self.pos = np.empty((len(self.kernels),3)) * units.km
        self.vel = np.empty((len(self.kernels),3)) * units.km / units.s
        self.ra  = np.empty(len(self.kernels)) * units.radian
        self.dec = np.empty(len(self.kernels)) * units.radian
        self.phase = np.empty(len(self.kernels)) * units.radian
        self.dBS = np.empty(len(self.kernels)) * units.km
        self.dBO = np.empty(len(self.kernels)) * units.km
        self.magnitudes = None
        self.time = None

        # get normalized spectra

        self.normalized_spectra = []
        self.sample_wavelengths = []

        for idx in range(len(self.names)):

            spkid = self.spkids[idx]
            spec = self.spec[idx]
            fname_smass2 = os.path.join(SMALL_BODIES_SPECTRA_CACHE_DIR, "smass2", naifSchemaToOriginal(spkid)[1:]+".[2]")
            fname_smass1 = os.path.join(SMALL_BODIES_SPECTRA_CACHE_DIR, "smass1", naifSchemaToOriginal(spkid)[1:]+".[1]")

            # search in smass 2 table
            if os.path.isfile(fname_smass2):
                wavelengths, normalized_flux = _readSmassSpectra(fname_smass2)

            # search in smass 1 table
            elif os.path.isfile(fname_smass1):
                wavelengths, normalized_flux = _readSmassSpectra(fname_smass1)

            # default to mean spectra
            else:
                wavelengths, normalized_flux = _readMeanSpectra(spec)
                
            self.sample_wavelengths.append(wavelengths)
            self.normalized_spectra.append(normalized_flux)



    def update(self, observer_id, time: Time):
        
        spice.furnsh([LEAPSECONDS_KERNEL, DE_KERNEL])

        self.time = time
        et = spice.datetime2et(time.to_datetime())

        for idx, targ in enumerate(self.spkids):
            spice.furnsh(self.kernels[idx])
            state, lt = spice.spkezr(targ, et, "J2000", "LT+S", observer_id)
            self.pos[idx,:] = state[:3] * units.km
            self.vel[idx,:] = state[3:] * units.km / units.s
            _, ra, dec = spice.recrad(state[:3])
            self.ra[idx]  = ra  * units.radian
            self.dec[idx] = dec * units.radian
            self.phase[idx] = spice.phaseq(et, targ, "Sun", observer_id, "LT+S") * units.radian
            self.dBO[idx] = _fnorm3(self.pos[idx,:])
            self.dBS[idx] = _fnorm3(spice.spkezp(int(targ), et-lt, "J2000", "NONE", 10)[0]) * units.km
            spice.unload(self.kernels[idx])

        self.magnitudes = None
        # https://en.wikipedia.org/wiki/Absolute_magnitude#Asteroids
        self.magnitudes = \
            self.H + 5 * np.log10(self.dBS * self.dBO / ((1*units.au).to(units.km))**2) - 2.5 * np.log10(_q(self.phase, self.G))
        
        spice.unload([LEAPSECONDS_KERNEL, DE_KERNEL])
        


def _readSmassSpectra(fname):
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        wavelengths = []
        normalized_flux = []
        for row in reader:
            wavelengths.append(float(row[0]) * 1000)
            normalized_flux.append(float(row[1]))
        wavelengths = np.asarray(wavelengths) * units.nm
        normalized_flux = np.asarray(normalized_flux)
    return wavelengths, normalized_flux


def _readMeanSpectra(spec):
    with open(SMALL_BODIES_MEAN_SPECTRA_PATH, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        wavelengths = []
        normalized_flux = []
        # the columns corresponding to each spectral class are defined in:
        # https://sbnarchive.psi.edu/pds4/non_mission/ast.bus-demeo.taxonomy/document/classdesc.asc
        # note that these are 1-indexed, so we will need to subtract 1
        column_lut = {
            "A":    2,
            "B":    4,
            "C":    6,
            "Cb":   8,
            "Cg":   10,
            "Cgh":  12,
            "Ch":   14,
            "D":    16,
            "K":    18,
            "L":    20,
            "O":    22,
            "Q":    24,
            "R":    24,
            "S":    28,
            "Sa":   30,
            "Sq":   32,
            "Sr":   34,
            "Sv":   36,
            "T":    38,
            "V":    40,  # originally 37 -- typo?
            "X":    42,
            "Xc":   44,
            "Xe":   46,
            "Xk":   48,
            # map Tholen taxonomy to SMASSII taxonomy (NOTE: not accurate!)
            # https://en.wikipedia.org/wiki/Asteroid_spectral_types#Overview_of_Tholen_and_SMASS
            "F":    4,
            "E":    42,
            "M":    42,
            "P":    42,
            "G":    10,
            "" :    6  # map unknown types to C for now
        }
        try:
            col = column_lut[spec] - 1
        except KeyError:
            base_spec = spec[0]
            col = column_lut[base_spec] - 1
        for row in reader:
            wavelengths.append(float(row[0]) * 1000)
            normalized_flux.append(float(row[col]))
        wavelengths = np.asarray(wavelengths) * units.nm
        normalized_flux = np.asarray(normalized_flux)
    return wavelengths, normalized_flux


def _fnorm3(a):
    return np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)


def _q(alpha, G):
    # https://en.wikipedia.org/wiki/Absolute_magnitude#Asteroids
    A1 = 3.332
    A2 = 1.862
    B1 = 0.631
    B2 = 1.218
    mask = alpha > 120.0 * units.degree
    ret = (1.0-G) * np.exp(-A1 * np.tan(alpha/2.0)**B1) \
        +      G  * np.exp(-A2 * np.tan(alpha/2.0)**B2)
    ret[mask] = np.nan
    return ret
