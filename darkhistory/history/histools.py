"""Functions and classes for processing lists of transfer functions."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

class IonRSArray:
    """Array of objects indexed by ionization and redshift. 

    Parameters
    ----------
    val_arr : ndarray
        Array of objects, indexed by (rs, ...), (xH, rs, ...) or (xH, xHe, rs, ...)
    x_arr : ndarray
        List of x values corresponding to val_arr, either None, 1D array or 3D array indexed by (xH, xHe), with each entry being [xH, xHe]. 
    rs_arr : ndarray
        List of redshift values corresponding to val_arr. 
    in_eng : ndarray
        Injection energy abscissa of entries of val_arr. 
    eng : ndarray
        Energy abscissa of entries of val_arr. 

    Attributes
    ----------
    interp_func : function
        An interpolation function over ionization and redshift.
    """
    def __init__(
        self, val_arr, x_arr, rs_arr, in_eng=None, eng=None
    ):

        if str(type(val_arr)) != "<class 'numpy.ndarray'>":
            raise TypeError('val_arr must be an ndarray.')

        self.rs          = rs_arr
        self.x           = x_arr
        self.in_eng      = in_eng
        self.eng         = eng
        self._grid_vals  = val_arr

        if self.rs[0] - self.rs[1] > 0:
            # Data points have been stored in decreasing rs. 
            self.rs = np.flipud(self.rs)
            if self.x is not None:
                if self.x.ndim == 1:
                    # dim 0: xH. 
                    self._grid_vals = np.flip(self._grid_vals, 1)
                elif self.x.ndim == 3:
                    # dim 0: xH, dim 1: xHe. 
                    self._grid_vals = np.flip(self._grid_vals, 2)
                else:
                    # dim 0: rs.
                    raise TypeError('invalid dimensions for x_arr.')
            else:
                self._grid_vals = np.flip(self._grid_vals, 0)
            # Now, data stored in *increasing* rs. 
            
        def __iter__(self):
            return iter(self.tflist_arr)

        def __getitem__(self, key):
            return self.tflist_arr[key]

        def __setitem__(self, key, value):
            self.tflist_arr[key] = value

class IonRSInterp:
    """Interpolation function over list of IonRSArray objects. 

    Parameters
    -----------
    ionrsarrays : list of IonRSArray
        IonRSArray objects to interpolate over. 
    rs_nodes : ndarray
        List of redshifts to transition between redshift regimes. 
    log_interp : bool, optional
        If true, performs an interpolation over log of the grid values. 
    """

    def __init__(self, ionrsarrays, rs_nodes=None, log_interp=False):

        # rs_nodes must have 1 less entry than tflistarrs.
        if (
            (rs_nodes is not None and len(rs_nodes) != len(ionrsarrays)-1)
            or (rs_nodes is None and len(ionrsarrays) > 1)
        ):
            raise TypeError('rs_nodes incompatible with given ionrsarrays.')

        # rs_nodes must be in *increasing* redshift
        if rs_nodes is not None and len(rs_nodes) > 1:
            if not np.all(np.diff(rs_nodes) > 0):
                raise TypeError('rs_nodes must be in increasing order.')

        self.rs       = [ionrsarr.rs for ionrsarr in ionrsarrays]
        self.in_eng   = [ionrsarr.in_eng for ionrsarr in ionrsarrays]
        self.eng      = [ionrsarr.eng for ionrsarr in ionrsarrays]
        self.rs_nodes = rs_nodes

        self.grid_vals = [ionrsarr._grid_vals for ionrsarr in ionrsarrays]
        self.x         = []

        for ionrsarr in ionrsarrays:
            try:
                self.x.append(ionrsarr.x)
            except:
                self.x.append(None)

        self._log_interp = log_interp

        if self._log_interp:
            for grid in self.grid_vals:
                grid[grid <= 0] = 1e-200
            func = np.log
        else:
            print('noninterp')
            def func(obj):
                return obj

        self.interp_func = []
        for x_vals,z,grid in zip(self.x, self.rs, self.grid_vals):
            if x_vals is None:
                print(z)
                print(grid.shape)
                self.interp_func.append(
                    interp1d(func(z), func(np.squeeze(grid)), axis=0)
                )
            elif x_vals.ndim == 1:
                # xH dependence.
                self.interp_func.append(
                    RegularGridInterpolator(
                        (func(x_vals), func(z)), func(grid)
                    )
                )
            elif x_vals.ndim == 3:
                # xH, xHe dependence.
                xH_arr = x_vals[:,0,0]
                xHe_arr = x_vals[0,:,1]
                self.interp_func.append(
                    RegularGridInterpolator(
                        (func(xH_arr), func(xHe_arr), func(z)), func(grid)
                    )
                )
            else:
                raise TypeError('grid has anomalous dimensions (and not in a good QFT way).')


# class IonRSInterp:

#     """Interpolation function over list of objects

#     Parameters
#     ----------
#     val_arr : list of objects
#         List of objects
#     xe_arr : ndarray
#         List of xe values corresponding to val_arr.
#     rs_arr : ndarray
#         List of redshift values corresponding to val_arr.
#     in_eng : ndarray
#         Injection energy abscissa of entries of val_arr.
#     eng : ndarray
#         Energy abscissa of entries of val_arr.

#     Attributes
#     ----------
#     interp_func : function
#         A 2D interpolation function over xe and rs.
#     _grid_vals : ndarray
#         a nD array of input data

#     """

#     def __init__(
#         self, xe_arr, rs_arr, val_arr, in_eng=None, eng=None, logInterp=False
#     ):

#         if str(type(val_arr)) != "<class 'numpy.ndarray'>":
#             raise TypeError('val_arr must be an ndarray')

#         if xe_arr is not None:
#             if len(xe_arr) != np.size(val_arr, 0):
#                 raise TypeError('0th dimension of val_arr must be the xe dimension')

#         if len(rs_arr) != np.size(val_arr, 1):
#             raise TypeError('1st dimension of val_arr (val_arr[0,:,0,0,...]) must be the rs dimension')

#         self.rs         = rs_arr
#         self.xes        = xe_arr
#         self.in_eng     = in_eng
#         self.eng        = eng
#         self._grid_vals = val_arr
#         self._logInterp  = logInterp

#         if self.rs[0] - self.rs[1] > 0:
#             # data points have been stored in decreasing rs.
#             self.rs = np.flipud(self.rs)
#             self._grid_vals = np.flip(self._grid_vals, 1)

#         # Now, data is stored in *increasing* rs.

#         if self._logInterp:
#             self._grid_vals[self._grid_vals<=0] = 1e-200
#             func = np.log
#         else:
#             print('noninterp')
#             def func(obj):
#                 return obj

#         if xe_arr is not None:
#             self.interp_func = RegularGridInterpolator(
#                 (func(self.xes), func(self.rs)), func(self._grid_vals)
#             )
#         else:
#             self.interp_func = interp1d(
#                 func(self.rs), func(self._grid_vals[0]), axis=0
#             )


#     def get_val(self, xe, rs):

#         if self._logInterp:
#             func = np.log
#             invFunc = np.exp
#         else:
#             def func(obj):
#                 return obj
#             invFunc = func

#         if rs > self.rs[-1]:
#             rs = self.rs[-1]
#         if rs < self.rs[0]:
#             rs = self.rs[0]

#         if self.xes is not None:
#             # xe must lie between these values.
#             if xe > self.xes[-1]:
#                 xe = self.xes[-1]
#             if xe < self.xes[0]:
#                 xe = self.xes[0]

#             return invFunc(np.squeeze(self.interp_func([func(xe), func(rs)])))
#         else:
#             return invFunc(self.interp_func(func(rs)))


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
