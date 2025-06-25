import numpy as np
from astropy import units
from astropy.modeling.rotations import EulerAngleRotation

from .lens import Lens
from .sensor import Sensor

from numpy.typing import ArrayLike
from typing import Union

Number = Union[int, float]

class Camera:

    # @enforce_types
    def __init__(self, *, 
                 lens   : Lens,
                 sensor : Sensor,
                 ra     : Number = 0,
                 dec    : Number = 0,
                 roll   : Number = 0):
        self.lens        = lens
        self.sensor      = sensor
        self.plate_scale = 206265. * units.arcsec / lens.f.to(units.mm)
        self.px_scale_x  = (self.plate_scale * self.sensor.px_len_x).to(units.arcsec)
        self.px_scale_y  = (self.plate_scale * self.sensor.px_len_y).to(units.arcsec)
        self.fov_x       = 2 * np.arctan(self.sensor.width.to( units.mm).value / 2 / lens.f.to(units.mm).value)
        self.fov_y       = 2 * np.arctan(self.sensor.height.to(units.mm).value / 2 / lens.f.to(units.mm).value)
        self.orient(ra=ra, dec=dec, roll=roll)

    
    '''
    The rotation from the camera frame to the world frame is given by the intrinsic rotation sequence x-y-z:
        Rz(ra) * Ry(-dec) * Rx(roll)  (note that the first rotation is the rightmost matrix)
    The image frame +x and +y axes align with the camera frame +y and -z axes (or ra and dec), respectively.
    '''
    def orient(self, *,
               ra: Number,
               dec: Number,
               roll: Number):
        
        self.ra   = float(ra)   * units.rad  # orientation of the camera in the world frame
        self.dec  = float(dec)  * units.rad
        self.roll = float(roll) * units.rad

        '''
        Express the ra, dec of an array of targets in the camera's frame, given their ra, dec in the world frame.
        Note that astropy's EulerAngleRotation applies intrinsic rotations. 
        '''
        # as of astropy 7.1.0, the implementation of EulerAngleRotation follows the **left** hand rule
        # contrary to the documentation (see https://github.com/astropy/astropy/issues/13134)
        # therefore, we need to invert the angles to use the right hand rule
        # additionally, since declination actually follows the left hand rule, it needs no inversion

        self.coordsCameraToWorld = EulerAngleRotation(-roll, dec, -ra, 'xyz')

        # the inverse transformation

        self.coordsWorldToCamera = EulerAngleRotation(ra, -dec, roll, 'zyx')


    def snap(self, *, 
             sky_mag       : float,
             exposure_time : float,
             temperature   : float,
             ra            : ArrayLike,
             dec           : ArrayLike, 
             magnitudes    : ArrayLike):
        
        # get the image coordinates of the stars in frame
        xcoords, ycoords, mask = self._getImageCoords(ra, dec)

        if len(mask) == 0:
            mags = []
        else:
            mags = magnitudes[mask]
        # TODO: implement rolling shutter (readout time)
        self.sensor.accumulate(lens=self.lens, sky_mag=sky_mag, exposure_time=exposure_time, temperature=temperature, xcoords=xcoords, ycoords=ycoords, magnitudes=mags)

        return self.sensor.readout().value.T


    def _getImageCoords(self, ra, dec):

        # get the undistorted light source coordinates on the image plane relative to the top left corner of the image
        # with right = +x, down = +y

        top_left_ra = -self.sensor.width/2 * self.plate_scale
        top_left_dec = self.sensor.height/2 * self.plate_scale
        
        rai_centered, deci_centered = self.coordsWorldToCamera(ra, dec)
        rai = rai_centered - top_left_ra
        deci = -(deci_centered - top_left_dec)

        # generate normalized coordinates and apply distortion

        ran = rai / (self.sensor.width * self.plate_scale)
        decn = deci / (self.sensor.height * self.plate_scale)

        ran_distorted, decn_distorted = self.lens.applyDistortion(ran, decn)

        mask = (ran_distorted >=0) & (ran_distorted <= 1) \
             & (decn_distorted >=0) & (decn_distorted <= 1)

        x_distorted = ran_distorted[mask] * self.sensor.width
        y_distorted = decn_distorted[mask] * self.sensor.height

        return x_distorted, y_distorted, mask