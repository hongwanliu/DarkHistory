"""Functions and classes for processing lists of transfer functions."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

class IonRSInterp:

    """Interpolation function over list of objects

    Parameters
    ----------
    val_arr : list of objects
        List of objects
    xe_arr : ndarray
        List of xe values corresponding to val_arr. 
    rs_arr : ndarray
        List of redshift values corresponding to val_arr. 
    in_eng : ndarray
        Injection energy abscissa of entries of val_arr.
    eng : ndarray
        Energy abscissa of entries of val_arr.

    Attributes
    ----------
    interp_func : function
        A 2D interpolation function over xe and rs. 
    _grid_vals : ndarray
        a nD array of input data
    
    """

    def __init__(self, xe_arr, rs_arr, val_arr, in_eng=None, eng=None, logInterp=False):

        if str(type(val_arr)) != "<class 'numpy.ndarray'>":
            raise TypeError('val_arr must be an ndarray')
        
        if len(xe_arr) != np.size(val_arr, 0):
            raise TypeError('0th dimension of val_arr must be the xe dimension')

        if len(rs_arr) != np.size(val_arr, 1):
            raise TypeError('1st dimension of val_arr (val_arr[0,:,0,0,...]) must be the rs dimension')

        self.rs     = rs_arr
        self.xe     = xe_arr
        self.in_eng = in_eng
        self.eng    = eng
        self._grid_vals = val_arr
        self.logInterp = logInterp

        if self.rs[0] - self.rs[1] > 0:
            # data points have been stored in decreasing rs.
            self.rs = np.flipud(self.rs)
            self._grid_vals = np.flip(self._grid_vals, 1)

        # Now, data is stored in *increasing* rs.

        if not logInterp:
            self.interp_func = RegularGridInterpolator((np.log(self.xe), np.log(self.rs)), self._grid_vals)
        else:
            self.interp_func = RegularGridInterpolator((np.log(self.xe), np.log(self.rs)), np.log(self._grid_vals))


    def get_val(self, xe, rs):

        # xe must lie between these values.
        if xe > self.xe[-1]:
            xe = self.xe[-1]
        if xe < self.xe[0]:
            xe = self.xe[0]

        if rs > self.rs[-1]:
            rs = self.rs[-1]
        if rs < self.rs[0]:
            rs = self.rs[0]

        if not self.logInterp:
            return np.squeeze(self.interp_func([np.log(xe), np.log(rs)]))
        else:
            return np.exp(np.squeeze(self.interp_func([np.log(xe), np.log(rs)])))
        
