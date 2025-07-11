from astropy.config import get_cache_dir
import requests
import tarfile
import io
import os
from pathlib import Path
from datetime import datetime, UTC

# TODO: put these into a dedicated `settings` module?

DEFAULT_CACHE_EXPIRATION_DAYS = 30
CACHE_DIR = get_cache_dir(rootname="starsim")

GAIA_CACHE_DIR = os.path.join(CACHE_DIR, "gaia")

FILTER_CACHE_DIR = os.path.join(CACHE_DIR, "filters")
JOHNSON_V_FILTER_PATH = os.path.join(FILTER_CACHE_DIR, "Generic.Johnson.V.xml")

SMALL_BODIES_CACHE_DIR = os.path.join(CACHE_DIR, "asteroids")
SMALL_BODIES_SPECTRA_CACHE_DIR = os.path.join(SMALL_BODIES_CACHE_DIR, "spectra")
SMALL_BODIES_MEAN_SPECTRA_PATH = os.path.join(SMALL_BODIES_SPECTRA_CACHE_DIR, "meanspectra.tab")

SMALL_BODIES_SBDB_CACHE_DIR = os.path.join(SMALL_BODIES_CACHE_DIR, "sbdb")
DEFAULT_SMALL_BODIES_PATH = os.path.join(SMALL_BODIES_SBDB_CACHE_DIR, "default.csv")
DEFAULT_SMALL_BODIES_GROUPS = \
    ["IEO", "ATE", "APO", "AMO", "MCA", "IMB", "MBA", "OMB", "TJN", "AST"]  # asteroids within the orbit of Jupiter
DEFAULT_SMALL_BODIES_HMAX = 10

SMALL_BODIES_KERNEL_CACHE_DIR = os.path.join(SMALL_BODIES_CACHE_DIR, "kernels")
DEFAULT_SMALL_BODIES_KERNELS = []  # populated later in this script

DE_KERNEL = os.path.join(SMALL_BODIES_KERNEL_CACHE_DIR, "de440s.bsp")
GM_KERNEL = os.path.join(SMALL_BODIES_KERNEL_CACHE_DIR, "gm_de440.tpc")

if os.name == 'nt':
    LEAPSECONDS_KERNEL = os.path.join(SMALL_BODIES_KERNEL_CACHE_DIR, "naif0012.tls.pc")
    lsk_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls.pc"
else:
    LEAPSECONDS_KERNEL = os.path.join(SMALL_BODIES_KERNEL_CACHE_DIR, "naif0012.tls")
    lsk_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"
                                      

Path(GAIA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(FILTER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(SMALL_BODIES_SPECTRA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(SMALL_BODIES_SBDB_CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(SMALL_BODIES_KERNEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)

from .sensor import Sensor
from .filter import Filter
from .lens import Lens
from .camera import Camera


def downloadBasicData():

    # core SPICE kernels
    print("Downloading SPICE kernels...")
    _download("https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp", DE_KERNEL)
    _download("https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc", GM_KERNEL)
    _download(lsk_url, LEAPSECONDS_KERNEL)

    # asteroid-related
    downloadAsteroidSpectra()
    downloadAstroidKernels()

    # filters
    _download("https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Generic/Johnson.V", JOHNSON_V_FILTER_PATH)


def downloadAsteroidSpectra():

    # asteroid spectra
    print("Downloading asteroid spectra...")
    url_smass1 = "http://smass.mit.edu/data/smass/smass1data_new.tar.gz"
    url_smass2 = "http://smass.mit.edu/data/smass/smass2data.tar.gz"
    _download_and_extract(url_smass1, SMALL_BODIES_SPECTRA_CACHE_DIR)
    _download_and_extract(url_smass2, SMALL_BODIES_SPECTRA_CACHE_DIR)

    # rename provisional designations to permanent designations for easier matching with SPICE kernels
    smass1_prov_id = ["au1991XB", "au1992NA", "au1992UB"]
    smass1_perm_id = ["", "65706", "415690"]
    smass2_prov_id = ["au1995BM2", "au1995WQ5", "au1996PW", "au1996UK", "au1996VC", "au1997CZ5", "au1997RD1", "au1998WS"]
    smass2_perm_id = ["129493", "376713", "", "100480", "22449", "99913", "26209", "47035"]
    for prov_id, perm_id in zip(smass1_prov_id, smass1_perm_id):
        if perm_id == "":
            continue
        os.rename(os.path.join(SMALL_BODIES_SPECTRA_CACHE_DIR, "smass1", prov_id+".[1]"),
                  os.path.join(SMALL_BODIES_SPECTRA_CACHE_DIR, "smass1", "a"+f"{perm_id:>06}"+".[1]"))
    for prov_id, perm_id in zip(smass2_prov_id, smass2_perm_id):
        if perm_id == "":
            continue
        os.rename(os.path.join(SMALL_BODIES_SPECTRA_CACHE_DIR, "smass2", prov_id+".[2]"),
                  os.path.join(SMALL_BODIES_SPECTRA_CACHE_DIR, "smass2", "a"+f"{perm_id:>06}"+".[2]"))
        os.rename(os.path.join(SMALL_BODIES_SPECTRA_CACHE_DIR, "smass2", prov_id+".spfit.[2]"),
                  os.path.join(SMALL_BODIES_SPECTRA_CACHE_DIR, "smass2", "a"+f"{perm_id:>06}"+".spfit.[2]"))

    # asteroid mean spectra templates
    print("Downloading asteroid mean spectra templates...")
    url_mean_spectra = "https://sbnarchive.psi.edu/pds4/non_mission/ast.bus-demeo.taxonomy/data/meanspectra.tab"
    _download(url_mean_spectra, SMALL_BODIES_MEAN_SPECTRA_PATH)


def downloadAstroidKernels():

    # default population of asteroids
    from .utils.sbdb_query import sbdbQuery as _sbdbQuery
    default_small_body_groups = DEFAULT_SMALL_BODIES_GROUPS
    _sbdbQuery(file_name='default', 
               orbit_class=default_small_body_groups, 
               Hmax=DEFAULT_SMALL_BODIES_HMAX, 
               max_cache_age=DEFAULT_CACHE_EXPIRATION_DAYS)

    # kernels of default asteroids
    from .utils.horizons_query import horizonsQuery as _horizonsQuery
    global DEFAULT_SMALL_BODIES_KERNELS
    start_time = datetime(2010,1,1, tzinfo=UTC)
    stop_time  = datetime(2049,1,1, tzinfo=UTC)
    DEFAULT_SMALL_BODIES_KERNELS = _horizonsQuery(start_time, stop_time, DEFAULT_SMALL_BODIES_PATH)



def _download_and_extract(url, destination):
    response = requests.get(url)
    response.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        tar.extractall(destination, filter="data")


def _download(url, destination):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def getAllCachedKernels():
    all_kernels = []
    for file in os.listdir(SMALL_BODIES_KERNEL_CACHE_DIR):
        if file.endswith(".bsp"):
            all_kernels.append(os.path.join(SMALL_BODIES_KERNEL_CACHE_DIR, file))
    return all_kernels


DEFAULT_SMALL_BODIES_KERNELS = getAllCachedKernels()