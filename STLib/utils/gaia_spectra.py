import numpy as np
from astropy import units
from astropy.table import Table
import pyvo

import time
import hashlib
import os.path
from .. import GAIA_CACHE_DIR, DEFAULT_CACHE_EXPIRATION_DAYS


def spectraConeSearch(ra, dec, radius, mag_cutoff, cache=False, max_cache_age=DEFAULT_CACHE_EXPIRATION_DAYS):

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
    
    # use the hash of the query parameters as the file name for query results
    # can't use the query itself because salt is added for strings: 
    # https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED

    # note that the hash may not be consistent across different interpreters (https://stackoverflow.com/a/64356731),
    # so upgrading Python *may* invalidate previously cached files (although this was not the case during testing).
    m = hashlib.sha1(usedforsecurity=False)
    m.update(query.encode("utf-8"))
    hash_val = m.hexdigest()
    fname = os.path.join(GAIA_CACHE_DIR, str(hash_val)+".xml")
    
    # do ADQL query unless file exists and hasn't expired
    if os.path.isfile(fname) and (time.time() - os.path.getmtime(fname)) < max_cache_age * 86400:
        query_flag = False
    else:
        query_flag = True

    if not query_flag:
        print(f"Using cached results from {fname} .")
        results = Table.read(fname)
    else:
        print("Performing TAP query to https://gaia.aip.de/tap ...")
        tap_service = pyvo.dal.TAPService("https://gaia.aip.de/tap")

        job = tap_service.submit_job(query, queue="30s")
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=30.0)
        print('JOB %s: %s' % (job.job.runid, job.phase))
        job.raise_if_error()

        # there seems to be a disconnect between what pyvo expects and what the gaia aip TAP provides for identifying VOTables
        # we need to manually change the id_ of the appropriate job._job.results member for fetch_result() to work...
        votable_id = [r.id_ for r in job._job.results].index('votable')
        job._job.results[votable_id].id_ = 'result'
        results = job.fetch_result().to_table()

        # the query results table denotes exponents with ** (e.g., nm**-2), which does not conform to
        # any standard recognized by astropy. We can remove the exponents and match the default cds standard. 
        for ii in range(len(results.columns)):
            if results.columns[ii].unit is not None:
                results.columns[ii].unit = units.Unit(results.columns[ii].unit._names[0].replace('**',''))
        if cache:
            results.write(fname, format="votable", overwrite=True)
            print(f"Search results cached to {fname} .")

    # split the results into two groups:
    # stars with spectra
    # stars without spectra (magnitude always available)
    spec_indices = np.flatnonzero([len(x)>0 for x in results['flux']])
    mag_indices = np.delete(np.arange(len(results)), spec_indices)
    print(f"Missing spectra for {len(mag_indices)}/{len(results)} ({(len(mag_indices)/len(results))*100:.1f}%) "
          f"of stars; their magnitudes will be supplied instead.")

    sfd_unit = results.columns['flux'].unit
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
    magnitudes_ra  = results['ra' ][mag_indices].filled().data * ra_unit  # `.filled()` gets rid of the masked array;
    magnitudes_dec = results['dec'][mag_indices].filled().data * dec_unit # there are no masked elements.

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
