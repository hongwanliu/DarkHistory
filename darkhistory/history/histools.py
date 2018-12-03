"""Functions and classes for processing lists of transfer functions."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

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

        if xe_arr is not None:
            if len(xe_arr) != np.size(val_arr, 0):
                raise TypeError('0th dimension of val_arr must be the xe dimension')

        if len(rs_arr) != np.size(val_arr, 1):
            raise TypeError('1st dimension of val_arr (val_arr[0,:,0,0,...]) must be the rs dimension')

        self.rs         = rs_arr
        self.xes        = xe_arr
        self.in_eng     = in_eng
        self.eng        = eng
        self._grid_vals = val_arr
        self._logInterp  = logInterp

        if self.rs[0] - self.rs[1] > 0:
            # data points have been stored in decreasing rs.
            self.rs = np.flipud(self.rs)
            self._grid_vals = np.flip(self._grid_vals, 1)

        # Now, data is stored in *increasing* rs.

        if self._logInterp:
            self._grid_vals[self._grid_vals<=0] = 1e-200
            func = np.log
        else:
            print('noninterp')
            def func(obj):
                return obj

        if xe_arr is not None:
            self.interp_func = RegularGridInterpolator(
                (func(self.xes), func(self.rs)), func(self._grid_vals)
            )
        else:
            self.interp_func = interp1d(
                func(self.rs), func(self._grid_vals[0]), axis=0
            )


    def get_val(self, xe, rs):

        if self._logInterp:
            func = np.log
            invFunc = np.exp
        else:
            def func(obj):
                return obj
            invFunc = func

        if rs > self.rs[-1]:
            rs = self.rs[-1]
        if rs < self.rs[0]:
            rs = self.rs[0]

        if self.xes is not None:
            # xe must lie between these values.
            if xe > self.xes[-1]:
                xe = self.xes[-1]
            if xe < self.xes[0]:
                xe = self.xes[0]

            return invFunc(np.squeeze(self.interp_func([func(xe), func(rs)])))
        else:
            return invFunc(self.interp_func(func(rs)))


class IonRSInterps:
    """Interpolation function over multiple list of objects

    Parameters
    ----------
    ...

    Attributes
    ----------

    """

    def __init__(self, ionRSinterps, xe_arr, inverted=False):

        length = len(ionRSinterps)

        self.rs = np.array([None for i in np.arange(length)])
        self.rs_nodes = np.zeros(length-1)
        self.xe_arr = xe_arr

        for i, ionRSinterp in enumerate(ionRSinterps):

            if np.any(np.diff(ionRSinterp.rs)<0):
                raise TypeError('redshifts in ionRSinterp[%d] should be increasing' % i+1)

            self.rs[i] = ionRSinterp.rs

            if i != length-1:
                if ionRSinterps[i].rs[0] > ionRSinterps[i+1].rs[0]:
                    raise TypeError(
                        'IonRSInterp object number %d should have redshifts smaller than object number %d (we demand ascending order of redshifts between objects)' % (i,i+1)
                    )
                if ionRSinterps[i].rs[-1] < ionRSinterps[i+1].rs[0]:
                    raise TypeError(
                        'The largest redshift in ionRSinterps[%d] is smaller '
                        +'than the largest redshift in ionRSinterps[%d] (i.e. there\'s a missing interpolation window)' % (i,i+1)
                    )
                if not inverted:
                    self.rs_nodes[i] = ionRSinterps[i].rs[-1]
                else:
                    self.rs_nodes[i] = ionRSinterps[i+1].rs[0]

        self.ionRSinterps = ionRSinterps


    def get_val(self, xe, rs):
        interpInd = np.searchsorted(self.rs_nodes, rs)
        return self.ionRSinterps[interpInd].get_val(xe,rs)
