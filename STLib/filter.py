from astropy.io import votable, ascii
from astropy import units
import numpy as np

from typing import Union
from .utils import type_checker, Number

VOTABLE_COLS = {"Wavelength", "Transmission"}
ASCII_COLS_DEFAULT = {"col1", "col2"}
ASCII_COLS = {"Wavelength", "Transmission"}

class Filter:

    @type_checker
    def __init__(self, *, 
                 zp_flux        : units.Quantity,
                 file           : Union[str, None] = None,
                 eff_wavelength : Union[units.Quantity, None] = None,
                 fwhm           : Union[units.Quantity, None] = None):
        
        '''
        The filter can be defined either by:
        A transmission profile from an ASCII table/VOTable consisting of two columns:
            - wavelength (assumed to be in Angstrom if no units are provided)
            - transmission [0-1]
        OR
        The effective wavelength and FWHM. 
            - effective wavelength
            - FWHM
        In both cases, the zero point of the filter must be specified in units of 
            - W / nm / m^2 or equivalent.
        '''

        self.zp_flux = zp_flux.to(units.W/units.nm/units.m**2)

        if file is not None:

            if file.lower().endswith(".xml"):

                tab = votable.parse_single_table(file).to_table()

                self._validateColumns(tab, VOTABLE_COLS)
                self.wavelengths = tab['Wavelength'].to(units.nm)
                self.transmission = tab['Transmission'].to(units.dimensionless_unscaled).value
                
            elif file.lower().endswith((".dat", ".csv", ".txt")):

                tab = ascii.read(file)

                try:
                    self._validateColumns(tab, ASCII_COLS)
                    self.wavelengths = (tab['Wavelength'] * units.Angstrom).to(units.nm)
                    self.transmission = tab['Transmission']
                except IndexError:
                    self._validateColumns(tab, ASCII_COLS_DEFAULT)
                    self.wavelengths = (tab['col1'] * units.Angstrom).to(units.nm)
                    self.transmission = tab['col2']

            else:
                
                raise ValueError("Only .xml, .dat, .csv, .txt extensions are supported.")
            
            max_transmission = np.max(self.transmission)
            fwhm_indices = np.where(self.transmission > max_transmission/2)[0]
            self.fwhm = self.wavelengths[fwhm_indices[-1]] - self.wavelengths[fwhm_indices[0]]
            self.eff_wavelength = np.trapezoid(self.wavelengths * self.transmission, self.wavelengths) \
                                / np.trapezoid(self.transmission, self.wavelengths)

        elif eff_wavelength is not None and fwhm is not None:

            self.eff_wavelength = eff_wavelength.to_value(units.nm)
            self.fwhm = fwhm.to_value(units.nm)
            self.wavelengths = np.array([self.eff_wavelength - self.fwhm/2, self.eff_wavelength + self.fwhm/2]) * units.nm
            self.transmission = np.array([1.0, 1.0])

        else:

            raise ValueError("Please specify a transmission profile either by "
                             "the file path to a VOTable/two-column ASCII table OR "
                             "an effective wavelength and FWHM.")



    @staticmethod
    def _validateColumns(table, cols):
        if cols.issubset(set(table.columns)):
            return True
        raise IndexError(f"The provided table does not contain the required columns {cols}.")
    