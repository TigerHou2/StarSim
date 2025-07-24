import numpy as np
import os.path
import csv

from typing import Union
from astropy import units
from astropy.time import Time
from astropy.table import Column
from .utils import type_checker, Number, horizonsQuery, naifSchemaToExtended, naifSchemaToOriginal
from . import \
    DEFAULT_SMALL_BODIES_PATH, DEFAULT_CACHE_EXPIRATION_DAYS, \
    SMALL_BODIES_SPECTRA_CACHE_DIR, SMALL_BODIES_MEAN_SPECTRA_PATH, \
    LEAPSECONDS_KERNEL, DE_KERNEL

import spiceypy as spice


class AstronomicalSources:

    """Astronomical sources base class.

    This is the base class for :class:`~STLib.sources.SpectralSources` and :class:`~STLib.sources.MagnitudeSources`.

    Parameters
    ----------
    ra: (N, ) `astropy.units.Quantity` ['angle'], optional
        Right ascension of the sources(s).
    dec: (N, ) :class:`~astropy.units.Quantity` ['angle'], optional
        Declination of the sources(s).
    source_id: (N, ) :class:`list` | :class:`~astropy.table.Column`, optional
        Corresponding source IDs (e.g. from input catalogs).
    """

    @type_checker
    def __init__(self, *,
                 ra : units.Quantity = np.array([]) * units.degree,
                 dec: units.Quantity = np.array([]) * units.degree,
                 source_id: Union[list, Column] = []):
        
        """Constructor for the AstronomicalSources class.
        """

        self.ra = np.atleast_1d(ra).to(units.degree)
        self.dec = np.atleast_1d(dec).to(units.degree)
        self.source_id = source_id


class SpectralSources(AstronomicalSources):

    """Astronomical sources defined by their spectral flux density profiles.

    Spectral flux density is the energy per area per unit wavelength. Integrating the product of spectral flux density
    with the transmission curve of a given filter allows for accurate computation of the measured flux.

    Parameters
    ----------
    ra: (N, ) :class:`~astropy.units.Quantity` ['angle']
        Right ascension of the sources(s).
    dec: (N, ) :class:`~astropy.units.Quantity` ['angle']
        Declination of the sources(s).
    source_id: (N, ) Union[list, :class:`~astropy.table.Column`]
        Corresponding source IDs (e.g. from input catalogs).
    sfd_wavelength: (M, ) :class:`~astropy.units.Quantity` ['length']
        Spectral flux density sample points.
    spectral_flux_density: (N, M) :class:`~astropy.units.Quantity` ['spectral flux density']
        Spectral flux density values corresponding to each sample point. Can be either:

        - irradiance / wavelength (e.g., :class:`~astropy.units.Jy`), or
        - irradiance / frequency (e.g., :class:`~astropy.units.W`/:class:`~astropy.units.m**2`/:class:`~astropy.units.nm`)
    """

    @type_checker
    def __init__(self, *,
                 ra : units.Quantity,
                 dec: units.Quantity,
                 source_id: Union[list, Column],
                 spectral_flux_density  : units.Quantity,
                 sfd_wavelength         : units.Quantity):
        
        """Constructor for the SpectralSources class.
        """

        super().__init__(ra=ra, dec=dec, source_id=source_id)
        self.spectral_flux_density = np.atleast_2d(spectral_flux_density).to(units.W / units.nm / units.m**2)
        self.sfd_wavelength = sfd_wavelength.to(units.nm)


class MagnitudeSources(AstronomicalSources):

    """ Astronomical sources defined by their magnitude.

    Magnitude is defined with respect to a given filter and zero point flux. It is not possible to determine the
    measured flux of a source with a different filter than the one used to define its magnitude. This class should
    only be used when a source's spectral flux density is not available, and only with a matching filter + zero 
    point flux (see :class:`~STLib.filter.Filter`).

    Parameters
    ----------
    ra: (N, ) `~astropy.units.Quantity` ['angle']
        Right ascension of the sources(s).
    dec: (N, ) `~astropy.units.Quantity` ['angle']
        Declination of the sources(s).
    source_id: (N, ) array_like
        Corresponding source IDs (e.g. from input catalogs).
    magnitude: (N, ) :class:`~numpy.ndarray`
        Magnitudes of the source(s).
    """

    @type_checker
    def __init__(self, *,
                 ra : units.Quantity,
                 dec: units.Quantity,
                 source_id: Union[list, Column],
                 magnitude: np.ndarray):
        
        """Constructor for the MagnitudeSources class.
        """

        super().__init__(ra=ra, dec=dec, source_id=source_id)
        self.magnitude = magnitude


class SolarSystemSources:

    """Solar System sources, e.g., asteroids. 
    
    Defined by a file containing JPL SBDB query results and desired ephemeris start/end dates. Upon instantiation, 
    queries the JPL Small Body Database for SPICE kernels corresponding to the small bodies listed in the query for
    the coverage defined by the start and end dates. Cached kernels that do not exceed the max cache age and with
    sufficient coverage are reused.

    Parameters
    ----------
    start_date: :class:`~astropy.time.Time`
        Start date of the ephemeris.
    end_date: :class:`~astropy.time.Time`
        End date of the ephemeris.
    small_bodies_query_results: :class:`str`
        Path to the SBDB query results.
    max_cache_age: :class:`float`
        Maximum age of the downloaded kernels in days.
    """

    @type_checker
    def __init__(self, *,
                 start_date: Time,
                 end_date: Time, 
                 small_bodies_query_results: str = DEFAULT_SMALL_BODIES_PATH,
                 max_cache_age: Number = DEFAULT_CACHE_EXPIRATION_DAYS):
        
        """Constructor for the SolarSystemSources class.
        """

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

        """
        Update the state of each source as seen by an observer at a given time. Includes relative position, velocity,
        right ascension, declination, and magnitude.

        Parameters
        ----------
        observer_id: str
            NAIF ID of the observer.
        time: :class:`~astropy.time.Time`
            Time of observation.
        """
        
        spice.furnsh([LEAPSECONDS_KERNEL, DE_KERNEL])

        self.time = time
        et = spice.str2et(time.tdb.strftime("%Y-%m-%d %H:%M:%S TDB"))

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
            self.dBS[idx] = _fnorm3(spice.spkezp(int(targ), et-lt, "J2000", "LT+S", 10)[0]) * units.km
            spice.unload(self.kernels[idx])

        # https://en.wikipedia.org/wiki/Absolute_magnitude#Asteroids
        self.magnitudes = \
            self.H + 5 * np.log10(self.dBS * self.dBO / ((1*units.au).to(units.km))**2) - 2.5 * np.log10(_q(self.phase, self.G))
        
        spice.unload([LEAPSECONDS_KERNEL, DE_KERNEL])
        


def _readSmassSpectra(fname):
    """
    Reads a SMASS spectrum file and returns the wavelengths and normalized fluxes.

    Parameters
    ----------
    fname : str
        path to the SMASS spectrum file

    Returns
    -------
    wavelengths : :class:`~astropy.units.Quantity`
        an array of wavelengths in nanometers
    normalized_flux : :class:`~numpy.ndarray`
        an array of normalized fluxes
    """
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
    """
    Reads a mean spectrum file for a given spectral class and returns the wavelengths and normalized fluxes.

    Parameters
    ----------
    spec : str
        A string representing the spectral class to be used for fetching the respective data column.

    Returns
    -------
    wavelengths : :class:`~astropy.units.Quantity`
        An array of wavelengths in nanometers.
    normalized_flux : :class:`~numpy.ndarray`
        An array of normalized fluxes for the specified spectral class.
    """

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
    """
    Calculate the Euclidean norm of a 3-element vector.

    Parameters
    ----------
    a : array_like
        3-element vector

    Returns
    -------
    norm : float
        Euclidean norm of the vector
    """
    return np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)


def _q(alpha, G):
    # https://en.wikipedia.org/wiki/Absolute_magnitude#Asteroids
    """
    Calculate the phase integral q(alpha, G) used in the HG12 system.

    Parameters
    ----------
    alpha : float or array_like
        Phase angle in degrees
    G : float
        Slope parameter

    Returns
    -------
    q : float or array_like
        Phase integral

    Notes
    -----
    The phase integral q(alpha, G) is defined as:
        q(alpha, G) = (1-G) * exp(-A1 * tan(alpha/2)**B1) \
                    + G * exp(-A2 * tan(alpha/2)**B2)
    where A1, A2, B1, and B2 are constants.  The phase integral is
    undefined for alpha > 120.0 degrees.
    """
    A1 = 3.332
    A2 = 1.862
    B1 = 0.631
    B2 = 1.218
    mask = alpha > 120.0 * units.degree
    ret = (1.0-G) * np.exp(-A1 * np.tan(alpha/2.0)**B1) \
        +      G  * np.exp(-A2 * np.tan(alpha/2.0)**B2)
    ret[mask] = np.nan
    return ret
