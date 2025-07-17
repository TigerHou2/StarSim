import pytest
import sys
sys.path.append('.')

import STLib
from STLib.utils import type_checker, Number, timer

import astropy.units as u
from typing import Union
from collections.abc import Callable




def test_timer_enable(mocker):
    mock_print = mocker.patch("builtins.print")
    @timer
    def func(*args, **kwargs):
        a = 2
        return a**2
    STLib.utils.debug.ENABLE_TIMER = True
    func()
    mock_print.assert_called_once()

def test_timer_disable(mocker):
    mock_print = mocker.patch("builtins.print")
    @timer
    def func(*args, **kwargs):
        a = 2
        return a**2
    STLib.utils.debug.ENABLE_TIMER = False
    func()
    mock_print.assert_not_called()





def test_type_checker_union():
    @type_checker
    def typed_func(arg: Number):
        return  # pragma: no cover
    with pytest.raises(TypeError):
        typed_func("bad arg")

def test_type_checker_tuple():
    @type_checker
    def typed_func(arg: tuple[str, Union[int, float]]):
        return  # pragma: no cover
    with pytest.raises(TypeError):
        typed_func(lambda x: None)
    with pytest.raises(TypeError):
        typed_func(("test", 1, "extra arg"))
    typed_func(("test", 1))
    
def test_type_checker_astropy_unit():
    @type_checker
    def typed_func(arg: u.Quantity):
        return  # pragma: no cover
    with pytest.raises(TypeError):
        typed_func("bad arg")
    
def test_type_checker_nested_union():
    @type_checker
    def typed_func(arg: Union[Number, u.Quantity]):
        return  # pragma: no cover
    with pytest.raises(TypeError):
        typed_func("bad arg")
    
def test_type_checker_func_as_arg():
    @type_checker
    def typed_func(arg: Callable[[int, int], str]):
        return  # pragma: no cover
    
    with pytest.raises(TypeError):  # incorrect arg type
        def f(a: int, b: float) -> str:
            return  # pragma: no cover
        typed_func(f)
    
    with pytest.raises(TypeError):  # incorrect return type
        def f(a: int, b: int) -> None:
            return  # pragma: no cover
        typed_func(f)
    
    with pytest.raises(TypeError):  # too few args
        def f(a: int) -> str:
            return  # pragma: no cover
        typed_func(f)
    
    with pytest.raises(TypeError):  # too many args
        def f(a: int, b: int, c) -> str:
            return  # pragma: no cover
        typed_func(f)
    
    with pytest.raises(TypeError):  # not a function
        typed_func(0)

    with pytest.raises(TypeError):  # non-introspectable function
        typed_func(len)

    def f(a: int, b: int) -> str:
        return " "  # pragma: no cover
    typed_func(f)  # correctly typed
    typed_func(lambda a, b: None)  # untyped

    @type_checker
    def func_arb_call_signature(arg: Callable[..., str]):
        return " "  # pragma: no cover
    def f() -> str:
        return " "  # pragma: no cover
    func_arb_call_signature(f)





from STLib.utils import naifSchemaToExtended, naifSchemaToOriginal

def test_naif_schema_upgrade():
    with pytest.raises(ValueError, match="Expected integer string"):
        naifSchemaToExtended("123abcd")
    with pytest.raises(ValueError, match="Expected NAIF ID"):
        naifSchemaToExtended("4000001")

    with pytest.warns(UserWarning, match="already in"):
        assert(naifSchemaToExtended("20002956") == "20002956")
    with pytest.warns(UserWarning, match="already in"):
        assert(naifSchemaToExtended("920065803") == "920065803")

    assert(naifSchemaToExtended("2002956") == "20002956")
    assert(naifSchemaToExtended("3000001") == "50000001")

def test_naif_schema_downpgrade():
    with pytest.raises(ValueError, match="Expected integer string"):
        naifSchemaToOriginal("123abcde")
    with pytest.raises(ValueError, match="Expected NAIF ID"):
        naifSchemaToOriginal("10000001")
    with pytest.raises(ValueError, match="cannot be accommodated"):
        naifSchemaToOriginal("21000001")
    with pytest.raises(ValueError, match="multi-body"):
        naifSchemaToOriginal("920065803")

    with pytest.warns(UserWarning, match="already in"):
        assert(naifSchemaToOriginal("2002956") == "2002956")

    assert(naifSchemaToOriginal("20002956") == "2002956")
    assert(naifSchemaToOriginal("50000001") == "3000001")





from STLib.functions.psf import airyPSFModel, defocusPSFModel, pillboxPSFModel, gaussianPSFModel

import numpy as np

def test_psf_pillbox():
    psf = pillboxPSFModel(radius=5*u.micron)

    bounds = 6
    grid = 100
    _x = np.linspace(-bounds,bounds,grid)
    _y = np.linspace(-bounds,bounds,grid)
    integral = np.trapezoid(np.trapezoid(psf(*np.meshgrid(_x,_y)),_y,axis=0),_x,axis=0)
    integral_error = abs(integral-1)
    assert integral_error < 1e-5, f"Expected 1 (± 1e-5), got error of {integral_error}"

def test_psf_gaussian():
    psf = gaussianPSFModel(sigma=3*u.micron)

    bounds = 10
    grid = 100
    _x = np.linspace(-bounds,bounds,grid)
    _y = np.linspace(-bounds,bounds,grid)
    integral = np.trapezoid(np.trapezoid(psf(*np.meshgrid(_x,_y)),_y,axis=0),_x,axis=0)
    integral_error = abs(integral-1)
    assert integral_error < 1e-5, f"Expected 1 (± 1e-5), got error of {integral_error}"

def test_psf_airy():
    wavelength = 550*u.nm
    d = 1*u.cm
    f = 10*u.cm
    A = (d/2 / np.sqrt(f**2 + (d/2)**2)).to_value(u.dimensionless_unscaled)  # numerical aperture
    psf = airyPSFModel(wavelength=wavelength, aperture=d, focal_length=f)

    bounds = (1.22 * wavelength / 2 / A).to_value(u.micron)
    power = 0.838
    grid = 100
    _x = np.linspace(-bounds,bounds,grid)
    _y = np.linspace(-bounds,bounds,grid)
    integral = np.trapezoid(np.trapezoid(psf(*np.meshgrid(_x,_y)),_y,axis=0),_x,axis=0)
    integral_error = abs(integral-power)
    assert integral_error < 1e-2, f"Expected {power} (± 1e-2), got error of {integral_error}"

def test_psf_defocused():
    wavelength = 550*u.nm
    a = 1*u.cm
    f = 1*u.cm
    psf = defocusPSFModel(wavelength=wavelength, aperture=a, focal_length=f, defocus=2*wavelength, nowarn=True)

    bounds = 20
    grid = 100
    _x = np.linspace(-bounds,bounds,grid)
    _y = np.linspace(-bounds,bounds,grid)
    integral = np.trapezoid(np.trapezoid(psf(*np.meshgrid(_x,_y)),_y,axis=0),_x,axis=0)
    integral_error = abs(integral-1)
    assert integral_error < 1e-2, f"Expected 1 (± 1e-2), got error of {integral_error}"





from STLib import Lens

@pytest.fixture
def psf():
    return gaussianPSFModel(sigma=3*u.micron)

def test_lens_autotune():
    psf = gaussianPSFModel(sigma=10*u.micron)
    lens = Lens(aperture=1*u.cm, focal_length=1*u.cm, transmission_efficiency=1.0, psf=psf, 
                auto_tune_integration_params=True)
    lens.focal_length=3*u.cm
    lens.tuneIntegrationParams(atol=1e-4, rtol=1e-4)

def test_lens_rw_params(psf):
    lens = Lens(aperture=1*u.cm, focal_length=1*u.cm, transmission_efficiency=1.0, psf=psf, 
                auto_tune_integration_params=False, psf_bounds=10*u.micron, psf_resolution=1*u.micron)
    lens.psf_bounds = 20 * u.micron
    lens.psf_bounds = np.array([10,10]) * u.micron
    psf_bounds_x, psf_bounds_y = lens.psf_bounds
    assert isinstance(psf_bounds_x, u.Quantity) and isinstance(psf_bounds_y, u.Quantity)
    
def test_lens_autotune_off_error(psf):
    with pytest.raises(ValueError):
        lens = Lens(aperture=1*u.cm, focal_length=1*u.cm, transmission_efficiency=1.0, psf=psf, 
                    auto_tune_integration_params=False)
    with pytest.raises(ValueError):
        lens = Lens(aperture=1*u.cm, focal_length=1*u.cm, transmission_efficiency=1.0, psf=psf, 
                    auto_tune_integration_params=False, psf_bounds=10*u.micron)
    with pytest.raises(ValueError):
        lens = Lens(aperture=1*u.cm, focal_length=1*u.cm, transmission_efficiency=1.0, psf=psf, 
                    auto_tune_integration_params=False, psf_resolution=1*u.micron)

def test_lens_center(psf):
    lens = Lens(aperture=1*u.cm, focal_length=1*u.cm, transmission_efficiency=1.0, psf=psf,
                k1=0.1, k2=0.1, k3=0.1, k4=0.01, k5=0.01, k6=0.01)
    x = np.array([0.5])
    y = np.array([0.5])
    xd, yd = lens.applyDistortion(x,y)
    assert xd[0] == 0.5 and yd[0] == 0.5, "Lens center should not change under distortion."

def test_lens_flat(psf):
    lens = Lens(aperture=1*u.cm, focal_length=1*u.cm, transmission_efficiency=1.0, psf=psf)
    x, y = np.meshgrid(np.linspace(0,1,3), np.linspace(0,1,3))
    xd, yd = lens.applyDistortion(x,y)
    np.testing.assert_array_almost_equal(x, xd)
    np.testing.assert_array_almost_equal(y, yd)

def test_lens_barrel_distortion(psf):
    lens = Lens(aperture=1*u.cm, focal_length=1*u.cm, transmission_efficiency=1.0, psf=psf, k1=-0.1)
    x = np.linspace(0, 1, 5)
    y = np.ones(5) * 0.75
    xd, yd = lens.applyDistortion(x,y)
    np.testing.assert_array_equal(np.diff(yd) > 0, [True, True, False, False])
    np.testing.assert_array_equal(np.diff(np.diff(yd)) > 0, [False, False, False])

def test_lens_pincushion_distortion(psf):
    lens = Lens(aperture=1*u.cm, focal_length=1*u.cm, transmission_efficiency=1.0, psf=psf, k1=0.1)
    x = np.linspace(0, 1, 5)
    y = np.ones(5) * 0.75
    xd, yd = lens.applyDistortion(x,y)
    np.testing.assert_array_equal(np.diff(yd) > 0, [False, False, True, True])
    np.testing.assert_array_equal(np.diff(np.diff(yd)) > 0, [True, True, True])

def test_lens_plotting(mocker, psf):
    mock_ax = mocker.Mock()
    mock_show = mocker.patch("matplotlib.pyplot.show")
    mocker.patch("matplotlib.pyplot.subplots", return_value=(None, mock_ax))
    lens = Lens(aperture=1*u.cm, focal_length=1*u.cm, transmission_efficiency=1.0, psf=psf,
                k1=0.1, k2=-0.1, k3=-0.1, k4=0.01, k5=-0.01, k6=0.01, p1=0.02, p2=-0.02)
    lens.showDistortion()
    assert mock_ax.plot.call_count == 44
    mock_show.assert_called_once()





from STLib import Filter, Sensor, Camera
from STLib.sources import AstronomicalSources, MagnitudeSources, SpectralSources, SolarSystemSources

