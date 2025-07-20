# although astroquery.jplsbdb provides the basic means for querying a list of objects using wildcards
# (see https://astroquery.readthedocs.io/en/latest/jplsbdb/jplsbdb.html#name-search),
# this is insufficient and not recommended by JPL for bulk requests. We will write our own simplified version.

import requests
import json
import urllib
import csv

import time
import hashlib
import os.path
from .. import SMALL_BODIES_SBDB_CACHE_DIR, DEFAULT_CACHE_EXPIRATION_DAYS

def sbdbQuery(orbit_class, 
              file_name=None, 
              fields=[], 
              Hmax=None, 
              constraints=None, 
              max_cache_age=DEFAULT_CACHE_EXPIRATION_DAYS,
              suppress=False):

    # construct query

    if constraints is not None and Hmax is not None:
        print("Warning: `Hmax` is ignored when `constraints` is defined.")
    if constraints is None and Hmax is None:
        raise ValueError("Please specify the search constraints.")
    else:
        constraints = f'{{"AND":["H|LT|{Hmax}"]}}'

    MANDATORY_FIELDS = ["spkid","name","pdes","H","G","BV","UB","albedo","spec_B","spec_T"]
    fields = list(set(MANDATORY_FIELDS).union(set(fields)))
    fields.sort()  # set() operations don't guarantee order -- sort() to maintain consistent hashing

    base = "https://ssd-api.jpl.nasa.gov/sbdb_query.api?"
    constraints = "&sb-cdata=" + urllib.parse.quote(constraints)
    url = base  + "fields=" + ",".join(fields) \
                + "&sb-class=" + ",".join(orbit_class) \
                + constraints
    

    # check cache (see gaia_spectra.py for caching caveats)

    if file_name is None:
        m = hashlib.sha1(usedforsecurity=False)
        m.update(url.encode("utf-8"))
        file_name = m.hexdigest()
    fname = os.path.join(SMALL_BODIES_SBDB_CACHE_DIR, str(file_name)+".csv")
    
    # do SBDB query unless file exists and hasn't expired
    if os.path.isfile(fname) and (time.time() - os.path.getmtime(fname)) < max_cache_age * 86400:

        if not suppress:
            print("Using cached results for SBDB query.")
    
    else:

        # Submit the API request and decode the JSON-response:
        if not suppress:
            print("Performing SBDB query...")

        response = requests.get(url)
        try:
            data = json.loads(response.text)
        except ValueError:
            print("Unable to decode JSON results")

        with open(fname, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(fields)
            writer.writerows(data['data'])

    return fname