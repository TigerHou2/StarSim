import json
import base64
import requests
import csv

import os.path
import time
import spiceypy as spice

from .. import SMALL_BODIES_KERNEL_CACHE_DIR, DEFAULT_CACHE_EXPIRATION_DAYS, LEAPSECONDS_KERNEL, DE_KERNEL


def horizonsQuery(start_time, stop_time, sbdb_path, max_cache_age=DEFAULT_CACHE_EXPIRATION_DAYS):

    # download small body SPICE kernels and merge the resulting .bsp files together.
    # individual kernels are first placed into an artifact cache.

    kernels = []

    with open(sbdb_path, 'r') as f:

        spice.furnsh([LEAPSECONDS_KERNEL, DE_KERNEL])

        # use the results of an existing SBDB query to download the appropriate small body SPICE kernels
        reader = csv.reader(f, delimiter=',')

        # for SBDB queries, the first row is the header; we are looking for the `spkid` column
        id_col = next(reader).index("spkid")
        spkids = [row[id_col] for row in reader]

        counter = 0
        for spkid in spkids:

            counter += 1

            # skip if this kernel exists
            kernel = os.path.join(SMALL_BODIES_KERNEL_CACHE_DIR, spkid + ".bsp")
            if os.path.isfile(kernel) and (time.time() - os.path.getmtime(kernel)) < max_cache_age * 86400:

                # check if kernel covers requested time range
                spice.furnsh(kernel)
                spkid = spice.spkobj(kernel)[0]
                coverage = spice.spkcov(kernel, spkid)
                cov_start = spice.et2datetime(coverage[0])
                cov_end   = spice.et2datetime(coverage[-1])
                spice.unload(kernel)

                if cov_start <= start_time and stop_time <= cov_end:
                    kernels.append(kernel)
                    continue

            print(f"\rDownloading kernel {counter}/{len(spkids)}...", end='')

            # Define API URL and SPK filename:
            url = 'https://ssd.jpl.nasa.gov/api/horizons.api'

            # Build the appropriate URL for this API request:
            # IMPORTANT: You must encode the "=" as "%3D" and the ";" as "%3B" in the
            #            Horizons COMMAND parameter specification.
            url += "?format=json&EPHEM_TYPE=SPK&OBJ_DATA=NO"
            url += "&COMMAND='DES%3D{}%3B'&START_TIME='{}'&STOP_TIME='{}'".format( \
                spkid, start_time.date().strftime("%Y-%m-%d"), stop_time.date().strftime("%Y-%m-%d"))

            # Submit the API request and decode the JSON-response:
            response = requests.get(url)
            try:
                data = json.loads(response.text)
            except ValueError:
                print("Unable to decode JSON results")

            # If the request was valid...
            if (response.status_code == 200):
                #
                # If the SPK file was generated, decode it and write it to the output file:
                if "spk" in data:
                    try:
                        f = open(kernel, "wb")
                    except OSError as err:
                        print("Unable to open SPK file '{0}': {1}".format(kernel, err))
                    #
                    # Decode and write the binary SPK file content:
                    f.write(base64.b64decode(data["spk"]))
                    f.close()
                    kernels.append(kernel)
                    continue
                #
                # Otherwise, the SPK file was not generated so output an error:
                print("ERROR: SPK file not generated")
                if "result" in data:
                    print(data["result"])
                else:
                    print(response.text)
                continue

            # If the request was invalid, extract error content and display it:
            if (response.status_code == 400):
                data = json.loads(response.text)
                if "message" in data:
                    print("MESSAGE: {}".format(data["message"]))
                else:
                    print(json.dumps(data, indent=2))

            # Otherwise, some other error occurred:
            print("response code: {0}".format(response.status_code))
            continue

    return kernels