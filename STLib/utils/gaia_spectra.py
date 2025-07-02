import numpy as np
from astropy import units
import pyvo
tap_service = pyvo.dal.TAPService("https://gaia.aip.de/tap")

def spectraConeSearch(ra, dec, radius, mag_cutoff):

    _ra     = ra.to(units.degree).value
    _dec    = dec.to(units.degree).value
    _radius = radius.to(units.degree).value

    query = f"""
    SELECT 
        source.source_id, source.ref_epoch, 
        source.ra, source.dec, 
        source.pmra, source.pmdec, 
        source.phot_g_mean_mag,
        source.phot_bp_mean_mag,
        source.phot_rp_mean_mag,
        spec.flux
    FROM
        gaiadr3.gaia_source AS source
    LEFT OUTER JOIN
        gaiadr3.xp_sampled_mean_spectrum AS spec
    ON
        source.source_id = spec.source_id
    WHERE
        source.phot_g_mean_mag < {mag_cutoff}
    AND 
        1 = CONTAINS(
            POINT('ICRS', source.ra, source.dec),
            CIRCLE('ICRS', {_ra}, {_dec}, {_radius})
            )
   """

    results = tap_service.search(query).to_table()

    spec_indices = np.flatnonzero([len(x)>0 for x in results['flux']])
    mag_indices = np.delete(np.arange(len(results)), spec_indices)
    print(f"Missing spectra for {len(mag_indices)}/{len(results)} ({(len(mag_indices)/len(results))*100:.1f}%) "
          f"of stars; their magnitudes will be supplied instead.")

    sfd_unit = units.W / units.nm / units.m**2  # parsing will fail, but we know the units from Gaia documentation
    ra_unit = results.columns['ra'].unit
    dec_unit = results.columns['dec'].unit

    spectral_flux_densities = np.asarray([np.asarray(flux) for flux in results['flux'][spec_indices]]) * sfd_unit
    spectral_flux_wavelengths = np.arange(336, 1022, 2) * units.nm  # fixed sample grid from Gaia documentation
    spectral_source_id = results['source_id'][spec_indices]
    spectral_ra  = results['ra' ][spec_indices].filled().data * ra_unit
    spectral_dec = results['dec'][spec_indices].filled().data * dec_unit

    magnitudes_g  = np.asarray(results['phot_g_mean_mag'][mag_indices])
    magnitudes_rp = np.asarray(results['phot_rp_mean_mag'][mag_indices])
    magnitudes_bp = np.asarray(results['phot_bp_mean_mag'][mag_indices])
    magnitudes_source_id = results['source_id'][mag_indices]
    magnitudes_ra  = results['ra' ][mag_indices].filled().data * ra_unit
    magnitudes_dec = results['dec'][mag_indices].filled().data * dec_unit

    return {"spectral_flux_densities": spectral_flux_densities, 
            "wavelengths": spectral_flux_wavelengths,
            "source_id": spectral_source_id, 
            "ra": spectral_ra,
            "dec": spectral_dec}, \
           {"magnitudes_g": magnitudes_g, 
            "magnitudes_rp": magnitudes_rp, 
            "magnitudes_bp": magnitudes_bp, 
            "source_id": magnitudes_source_id,
            "ra": magnitudes_ra,
            "dec": magnitudes_dec}
