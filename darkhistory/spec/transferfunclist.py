"""Functions and classes for processing lists of transfer functions."""

import numpy as np
from numpy.linalg import matrix_power
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

from darkhistory.utilities import arrays_close
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectra import Spectra
import darkhistory.spec.transferfunction as tf
from darkhistory.utilities import dict_from_inhom_list, inhom_list_from_dict
from darkhistory.utilities import dict_from_inhom_list, inhom_list_from_dict

class TransferFuncList:
    """List of transfer functions.

    Parameters
    ----------
    tflist : list of TransFuncAtRedshift or TransFuncAtEnergy

    Attributes
    ----------
    tflist : list of TransFuncAtRedshift or TransFuncAtEnergy
        List of transfer functions that is part of this class.
    tftype : {'rs', 'in_eng'}
        Type of transfer functions in the list: 'rs' for TransFuncAtRedshift, 'in_eng' for TransFuncAtEnergy
    rs : ndarray
        Redshift abscissa of the transfer functions.
    in_eng : ndarray
        Injection energy abscissa of the transfer functions.
    spec_type : {'N', 'dNdE'}
        The type of spectra stored in the transfer functions.
    dlnz : float
        The d ln(1+z) step for the transfer functions.
    """

    def __init__(self, tflist):

        self._tflist = tflist
        self.spec_type = tflist[0].spec_type

        if (not np.all([isinstance(tfunc, tf.TransFuncAtRedshift)
                for tfunc in tflist]) and
            not np.all([isinstance(tfunc, tf.TransFuncAtEnergy)
                for tfunc in tflist])
        ):

            raise TypeError('transfer functions must be of the same type.')

        if not arrays_close(
            [tfunc.eng for tfunc in self._tflist]
        ):
            raise TypeError('all transfer functions must have the same \
                energy abscissa.')

        if len(set([tfunc.dlnz for tfunc in self._tflist])) > 1:
            raise TypeError('all transfer functions must have the same \
                dlnz.')

        if isinstance(tflist[0], tf.TransFuncAtRedshift):
            self._tftype = 'rs'
            self._eng = tflist[0].eng
            self._rs = np.array([tfunc.rs[0] for tfunc in self.tflist])
            self._in_eng = tflist[0].in_eng
            self._dlnz = tflist[0].dlnz
            self._grid_vals = np.atleast_3d(
                np.stack([tf.grid_vals for tf in tflist])
            )
        elif isinstance(tflist[0], tf.TransFuncAtEnergy):
            self._tftype = 'in_eng'
            self._eng = tflist[0].eng
            self._rs = tflist[0].rs
            self._in_eng = np.array([tfunc.in_eng[0] for tfunc in self.tflist])
            self._dlnz = tflist[0].dlnz
            self._grid_vals = np.atleast_3d(
                np.stack([tf.grid_vals for tf in tflist])
            )
        else:
            raise TypeError('can only be list of valid transfer functions.')

    @property
    def eng(self):
        return self._eng

    @property
    def in_eng(self):
        return self._in_eng

    @property
    def rs(self):
        return self._rs

    @property
    def grid_vals(self):
        return self._grid_vals

    @property
    def tflist(self):
        return self._tflist

    @property
    def dlnz(self):
        return self._dlnz

    @property
    def tftype(self):
        return self._tftype

    def __iter__(self):
        return iter(self.tflist)

    def __getitem__(self, key):
        return self.tflist[key]

    def __setitem__(self, key, value):
        self.tflist[key] = value

    def at_val(self, axis, new_val, bounds_error=None, fill_value=np.nan):
        """Returns the transfer functions at the new abscissa.

        Parameters
        ----------
        axis : {'rs', 'in_eng', '2D_in_eng'}
            The axis along which to perform the interpolation. If the axis is 'rs', then the list will be transposed into tftype 'in_eng' and vice-versa.
        new_val : ndarray or tuple of ndarrays (in_eng, eng)
            The new redshift or injection energy abscissa.
        bounds_error : bool, optional
            See scipy.interpolate.interp1d
        fill_value : array-like or (array-like, array-like) or "extrapolate", optional
            See scipy.interpolate.interp1d
        """

        # i enables the use of tqdm.

        transposed = False

        if axis == 'in_eng':
            if self.tftype != 'rs':
                self.transpose()
                transposed = True

            new_tflist = [
            tf.at_in_eng(
                new_val, bounds_error=bounds_error, fill_value=fill_value
            ) for i,tf in zip(
                    np.arange(len(self.tflist)), self.tflist
                )
            ]
            self._tflist = new_tflist
            self._in_eng = new_val

        elif axis == 'rs':
            if self.tftype != 'in_eng':
                self.transpose()
                transposed = True

            new_tflist = [
                tf.at_rs(
                    new_val, bounds_error=bounds_error, fill_value=fill_value
                ) for i,tf in zip(
                        np.arange(len(self.tflist)), self.tflist
                    )
            ]

            self._tflist = new_tflist
            self._rs = new_val

        elif axis == '2D_in_eng':

            if self.tftype != 'rs':
                self.transpose()
                transposed = True

            new_tflist = [
                tf.at_val(
                    new_val[0], new_val[1],
                    bounds_error=bounds_error, fill_value=fill_value
                ) for i,tf in zip(
                        np.arange(len(self.tflist)), self.tflist
                    )
            ]

            self._tflist = new_tflist
            self._in_eng = new_val[0]
            self._eng    = new_val[1]

        else:
            raise TypeError('TransferFuncList.tftype is neither rs nor eng')

        if transposed:
            self.transpose()

    def transpose(self):
        """ Transposes the list of transfer functions.

        This takes a TransferFuncList made of TransFuncAtEnergy into a list a TransferFuncList made of TransFuncAtRedshift and vice-versa.
        """

        if self.tftype == 'in_eng':

            new_tflist = [tf.TransFuncAtRedshift(
                    [tfunc[i] for tfunc in self.tflist],
                    self.dlnz
                ) for i,rs in enumerate(self.rs)
            ]

            self._tflist = new_tflist
            self._grid_vals = np.transpose(self.grid_vals, (1,0,2))
            self._tftype = 'rs'

        elif self.tftype == 'rs':

            new_tflist = [tf.TransFuncAtEnergy(
                    [tfunc[i] for tfunc in self.tflist],
                    self.dlnz
                ) for i,in_eng in enumerate(self.in_eng)
            ]

            self._tflist = new_tflist
            self._grid_vals = np.transpose(self.grid_vals, (1,0,2))
            self._tftype = 'in_eng'

        else:

            raise TypeError('TransferFuncList.tftype is neither rs nor eng')

    def coarsen(
        self, dlnz_factor, delete_tfs=True, coarsen_type='prop',
        prop_transfunclist=None
    ):
        """Coarsens the new transfer function with larger dlnz.

        This is obtained by multiplying the transfer function by itself several times, and removing intermediate transfer functions.

        Parameters
        ----------
        dlnz_factor : int
            The factor to increase dlnz by.
        delete_tfs : bool
            If true, only retains transfer functions in tflist that have an index that is a multiple of dlnz_factor.
        coarsen_type : {'prop', 'dep'}
            The type of coarsening. Use 'prop' to coarsen by taking powers of the transfer function. Use 'dep' for deposition transfer functions, where coarsening is done by taking self * sum_i prop_tf**i.
        prop_tflist : TransferFuncList
            The transfer function for propagation, if the transfer function represents deposition.

        """

        if self.tftype != 'rs':
            self.transpose()
        if coarsen_type == 'dep' and prop_transfunclist.tftype != 'rs':
            prop_transfunclist.transpose()

        if delete_tfs:
            new_tflist = [
                self.tflist[i] for i in np.arange(
                    0, len(self.tflist), dlnz_factor
                )
            ]
        else:
            # list() needed to create a new copy, not just point.
            new_tflist = list(self.tflist)

        self._tflist = []

        if coarsen_type == 'dep':

            for i,(tfunc,prop_tfunc) in enumerate(
                zip(new_tflist, prop_transfunclist.tflist)
            ):
                in_eng_arr = tfunc.in_eng
                if prop_tfunc.in_eng.size != prop_tfunc.eng.size:
                    raise TypeError('propagation matrix is not square.')
                prop_part = np.zeros_like(prop_tfunc._grid_vals)
                for i in np.arange(dlnz_factor):
                    prop_part += matrix_power(prop_tfunc._grid_vals, i)
                # We need to take eng x in_eng times the propagating part.
                # Need to return new_grid_val to in_eng x eng in the end.
                # new_grid_val = np.transpose(
                #     np.dot(np.transpose(tfunc._grid_vals), prop_part)
                # )
                new_grid_val = np.matmul(prop_part, tfunc._grid_vals)
                new_spec_arr = [
                    Spectrum(
                        tfunc.eng, new_grid_val[i],
                        spec_type = tfunc.spec_type,
                        rs = tfunc.rs[0], in_eng = in_eng_arr[i]
                    )
                    for i in np.arange(in_eng_arr.size)
                ]

                self._tflist.append(
                    tf.TransFuncAtRedshift(
                        new_spec_arr, self.dlnz*dlnz_factor
                    )
                )

        elif coarsen_type == 'prop':

            for (i,tfunc) in enumerate(new_tflist):

                in_eng_arr = tfunc.in_eng
                new_grid_val = matrix_power(
                    tfunc._grid_vals,dlnz_factor
                )
                new_spec_arr = [
                    Spectrum(
                        tfunc.eng, new_grid_val[i],
                        spec_type = tfunc.spec_type,
                        rs = tfunc.rs[0], in_eng = in_eng_arr[i]
                    )
                    for i in np.arange(in_eng_arr.size)
                ]

                self._tflist.append(
                    tf.TransFuncAtRedshift(
                        new_spec_arr, self.dlnz*dlnz_factor
                    )
                )

        else:
            raise TypeError('invalid coarsen_type.')

        self._rs = np.array([tfunc.rs[0] for tfunc in new_tflist])
        self._dlnz *= dlnz_factor

class TransferFuncListArray:

    """Array of TransferFuncList for array of xH, xHe values.

    Parameters
    -----------
    tflist_arr : list of TransferFuncList
        TransferFuncList objects to add to the array. If 2D, should be indexed by (xH, xHe). 
    x_arr : ndarray
        Array of xH or (xH, xHe) values corresponding to tflist_arr.

    Attributes
    ----------
    rs : ndarray
        Redshift abscissa of the transfer functions. 
    in_eng : ndarray
        Injection energy abscissa of the transfer functions. 
    eng : ndarray
        Energy abscissa of the transfer functions. 
    dlnz : float
        The d ln(1+z) step for the transfer functions. 
    spec_type : {'N', 'dNdE'}
        The type of spectra stored in the transfer functions. 

    """

    def __init__(self, tflist_arr, x_arr):
        if x_arr is not None:
            ndim = x_arr.ndim
        else:
            ndim = 0

        if ndim == 0:
            self.rs     = tflist_arr[0].rs
            self.in_eng = tflist_arr[0].in_eng
            self.eng    = tflist_arr[0].eng
            self.dlnz   = tflist_arr[0].dlnz
            self.x      = x_arr
            self.spec_type  = tflist_arr[0].spec_type
            self.tflist_arr = tflist_arr 

            self._grid_vals = tflist_arr[0].grid_vals
            if tflist_arr[0].tftype == 'eng':
                # grid_vals should have indices corresponding to 
                # (rs, in_eng, eng). 
                grid_vals = np.transpose(grid_vals, (1, 0, 2))
                # grid_vals are now (rs, in_eng, eng).

            if self.rs[0] - self.rs[1] > 0:
                # data points have been stored in decreasing rs. 
                self.rs = np.flipud(self.rs)
                self._grid_vals = np.flip(self._grid_vals, 1)

        elif ndim == 1:
            if not arrays_close(
                [tflist.rs for tflist in tflist_arr]    
            ):
                raise TypeError('All redshift bins must be identical.')
            if not arrays_close(
                [tflist.in_eng for tflist in tflist_arr]
            ):
                raise TypeError('All in_eng bins must be identical.')
            if not arrays_close(
                [tflist.eng for tflist in tflist_arr]
            ):
                raise TypeError('All eng bins must be identical.')
            if len(set([tflist.dlnz for tflist in tflist_arr])) != 1:
                raise TypeError('All dlnz steps must be identical.')
            if len(set([tflist.spec_type for tflist in tflist_arr])) != 1:
                raise TypeError('All spec_type must be identical.')
            if len(set([tflist.tftype for tflist in tflist_arr])) != 1:
                raise TypeError('All tftype must be the same.')

            self.rs     = tflist_arr[0].rs
            self.in_eng = tflist_arr[0].in_eng
            self.eng    = tflist_arr[0].eng
            self.dlnz   = tflist_arr[0].dlnz
            self.x      = x_arr
            self.spec_type  = tflist_arr[0].spec_type
            self.tflist_arr = tflist_arr 


            grid_vals = np.array(
                np.stack(
                    [tflist.grid_vals for tflist in tflist_arr]
                ),
                ndmin = 4
            )
            if tflist_arr[0].tftype == 'eng':
                # grid_vals should have indices corresponding to 
                # (xH, rs, in_eng, eng). 
                grid_vals = np.transpose(grid_vals, (0, 2, 1, 3))

            # grid_vals are now (xH, rs, in_eng, eng).

            self._grid_vals = grid_vals

            if self.rs[0] - self.rs[1] > 0:
                # data points have been stored in decreasing rs. 
                self.rs = np.flipud(self.rs)
                self._grid_vals = np.flip(self._grid_vals, 1)

            # Now, data is stored in *increasing* rs. 

        elif ndim == 3:
            if not all(
                arrays_close([tflist.rs for tflist in tflist_xHe_arr]) 
                for tflist_xHe_arr in tflist_arr
            ):
                raise TypeError('All redshift bins must be identical.')
            if not all(
                arrays_close([tflist.in_eng for tflist in tflist_xHe_arr]) 
                for tflist_xHe_arr in tflist_arr
            ):
                raise TypeError('All in_eng bins must be identical.')
            if not all(
                arrays_close([tflist.eng for tflist in tflist_xHe_arr]) 
                for tflist_xHe_arr in tflist_arr
            ):
                raise TypeError('All eng bins must be identical.')
            if len(set(
                tflist.dlnz for tflist_xHe_arr in tflist_arr
                for tflist in tflist_xHe_arr
            )) != 1:
                raise TypeError('All dlnz must be identical.')
            if len(set(
                tflist.spec_type for tflist_xHe_arr in tflist_arr
                for tflist in tflist_xHe_arr
            )) != 1:
                raise TypeError('All spec_type must be identical.')
            if len(set(
                tflist.tftype for tflist_xHe_arr in tflist_arr 
                for tflist in tflist_xHe_arr
            )) != 1:
                raise TypeError('All tftype must be identical.')

            self.rs     = tflist_arr[0][0].rs
            self.in_eng = tflist_arr[0][0].in_eng
            self.eng    = tflist_arr[0][0].eng
            self.dlnz   = tflist_arr[0][0].dlnz
            self.x      = x_arr
            self.spec_type  = tflist_arr[0][0].spec_type
            self.tflist_arr = tflist_arr

            grid_vals = np.array(
                np.stack([
                    np.stack(
                        [tflist.grid_vals for tflist in tflist_xHe_arr]
                    ) for tflist_xHe_arr in tflist_arr
                ]), 
                ndmin = 5
            )

            if tflist_arr[0][0].tftype == 'eng':
                # grid_vals should have indices corresponding to
                # (xH, xHe, rs, in_eng, eng). 
                grid_vals = np.transpose(grid_vals, (0, 1, 3, 2, 4))

            # grid_vals are now (xH, xHe, rs, in_eng, eng)

            self._grid_vals = grid_vals

            if self.rs[0] - self.rs[1] > 0:
                # data points have been stored in decreasing rs. 
                self.rs = np.flipud(self.rs)
                self._grid_vals = np.flip(self._grid_vals, 2)

            # Now, data is stored in *increasing* rs. 

        else:
            raise TypeError('x_arr dimensions is anomalous (and not in the good QFT way).')

        def __iter__(self):
            return iter(self.tflist_arr)

        def __getitem__(self, key):
            return self.tflist_arr[key]

        def __setitem__(self, key, value):
            self.tflist_arr[key] = value



class TransferFuncInterp:
    """Interpolation function over list of TransferFuncList objects.

    Parameters
    ----------
    tflist_arr : list of TransferFuncList
         TransferFuncList objects to interpolate over. Should be indexed by xH, (redshift regime) or (redshift regime, xH, xHe) 
    x_arr : None or ndarray
        Array of xH or (xH, xHe) values corresponding to tflist_arr.
    rs_nodes : ndarray
        List of redshifts to transition between redshift regimes. 
    log_interp : bool, optional
        If True, performs an interpolation over log of the grid values.

    Attributes
    ----------
    rs : list of ndarray
        Redshift abscissa of the transfer functions.
    in_eng : list of ndarray
        Injection energy abscissa of the transfer functions.
    eng : list of ndarray
        Energy abscissa of the spectrum.
    dlnz : list of float
        The d ln(1+z) step for the transfer functions.
    spec_type : tuple of {'N', 'dNdE'}
        The type of spectra stored in the transfer functions.
    rs_nodes : ndarray
        List of redshifts to transition between redshift regimes.
    grid_vals : tuple of ndarray
        The grid values in each redshift regime.
    x : tuple of ndarray
        Array of xH or (xH, xHe) in each redshift regime.
    interp_func : function
        An interpolation function over xH (optionally xHe) and rs.
    """

    def __init__(self, tflistarrs, rs_nodes=None, log_interp=False):

        if isinstance(tflistarrs, dict): # initialize from dictionary.
            self.from_dict(tflistarrs)

        else: # original initialization.
            if not np.all(arrays_close([tfla.in_eng for tfla in tflistarrs])):
                raise TypeError('All in_eng bins must be identical.')

            if not np.all(arrays_close([tfla.eng for tfla in tflistarrs])):
                raise TypeError('All eng bins must be identical.')

            if ( # rs_nodes must have 1 less entry than tflistarrs.
                (rs_nodes is not None and len(rs_nodes) != len(tflistarrs)-1)
                or (rs_nodes is None and len(tflistarrs) > 1)
            ):
                raise TypeError('rs_nodes incompatible with given tflistarrs.')

            if rs_nodes is not None and len(rs_nodes) > 1: # rs_nodes must be in *increasing* redshift
                if not np.all(np.diff(rs_nodes) > 0):
                    raise TypeError('rs_nodes must be in increasing order.')

            if len(set([tfla.spec_type for tfla in tflistarrs])) != 1: # check all the same spec_type
                raise TypeError('all spec_type must be the same.')

            self.rs       = [tfla.rs for tfla in tflistarrs]
            self.in_eng   = tflistarrs[0].in_eng
            self.eng      = tflistarrs[0].eng
            self.dlnz     = [tfla.dlnz for tfla in tflistarrs]
            self.rs_nodes = rs_nodes
            self.spec_type = tflistarrs[0].spec_type

            self.grid_vals = [tfla._grid_vals for tfla in tflistarrs]
            self.x  = []
            for tfla in tflistarrs:
                try:
                    self.x.append(tfla.x)
                except:
                    self.x.append(None)

            self._log_interp = log_interp

        # common: build interpolation function
        if self._log_interp:
            for grid in self.grid_vals:
                grid[grid <= 0] = 1e-200
            func = np.log
        else:
            func = lambda x: x

        self.interp_func = []
        for x_vals,z,grid in zip(self.x, self.rs, self.grid_vals):
            if grid.ndim == 3: # No xe dependence.
                self.interp_func.append(interp1d(func(z), func(np.squeeze(grid)), axis=0))
            elif grid.ndim == 4: # xH dependence.
                self.interp_func.append(RegularGridInterpolator((func(x_vals), func(z)), func(grid)))
            elif grid.ndim == 5: # xH, xHe dependence.
                xH_arr = x_vals[:,0,0]
                xHe_arr = x_vals[0,:,1]
                self.interp_func.append(RegularGridInterpolator((func(xH_arr), func(xHe_arr), func(z)), func(grid)))
            else:
                raise ValueError('grid has anomalous dimensions (and not in a good QFT way).')
    
    
    def to_dict(self):
        """Return hdf5 compatible dictionary."""
        d = {
            'dlnz' : self.dlnz,
            'rs_nodes' : self.rs_nodes,
            'log_interp' : self._log_interp,
            'eng' : self.eng,
            'in_eng' : self.in_eng,
            'spec_type' : self.spec_type,
        }
        d.update(dict_from_inhom_list(self.rs, 'rs'))
        x_save = [(-1 if x is None else x) for x in self.x]
        d.update(dict_from_inhom_list(x_save, 'x'))
        d.update(dict_from_inhom_list(self.grid_vals, 'grid_vals'))
        return d
    

    def from_dict(self, d):
        """Initialize from hdf5 compatible dictionary."""
        self.dlnz = d['dlnz']
        self.rs_nodes = d['rs_nodes']
        self._log_interp = d['log_interp']
        self.eng = d['eng']
        self.in_eng = d['in_eng']
        self.spec_type = d['spec_type'].decode()
        self.rs = inhom_list_from_dict(d, 'rs')
        self.x = inhom_list_from_dict(d, 'x')
        for i, x in enumerate(self.x):
            if np.isscalar(x) and x == -1:
                self.x[i] = None
        self.grid_vals = inhom_list_from_dict(d, 'grid_vals')

    def get_tf(self, xH, xHe, rs):

        if self._log_interp:
            func = np.log
            inv_func = np.exp
        else:
            inv_func = func = lambda x: x

        rs_regime_ind = np.searchsorted(self.rs_nodes, rs)
        if rs > self.rs[rs_regime_ind][-1] or rs < self.rs[rs_regime_ind][0]:
            raise TypeError('redshift lies outside of range.')

        rs_regime_interp_func = self.interp_func[rs_regime_ind]

        # Make sure xH, xHe and rs are within bounds.
        if rs > self.rs[rs_regime_ind][-1]:
            rs = self.rs[-1]
        if rs < self.rs[rs_regime_ind][0]:
            rs = self.rs[0]

        if self.x[rs_regime_ind] is not None:
            if self.x[rs_regime_ind].ndim == 1:
                if xH > self.x[rs_regime_ind][-1]:
                    xH = self.x[rs_regime_ind][-1]
                if xH < self.x[rs_regime_ind][0]:
                    xH = self.x[rs_regime_ind][0]
            elif self.x[rs_regime_ind].ndim == 3:
                xH_arr = self.x[rs_regime_ind][:,0,0]
                xHe_arr = self.x[rs_regime_ind][0,:,1]
                if xH > xH_arr[-1]:
                    xH = xH_arr[-1]
                if xH < xH_arr[0]:
                    xH = xH_arr[0]
                if xHe > xHe_arr[-1]:
                    xHe = xHe_arr[-1]
                if xHe < xHe_arr[0]:
                    xHe = xHe_arr[0]

        if self.grid_vals[rs_regime_ind].ndim == 3:
            out_grid_vals = inv_func(
                np.squeeze(rs_regime_interp_func(func(rs)))
            )
        elif self.grid_vals[rs_regime_ind].ndim == 4:
            out_grid_vals = inv_func(
                np.squeeze(rs_regime_interp_func([func(xH), func(rs)]))
            )
        elif self.grid_vals[rs_regime_ind].ndim == 5:
            out_grid_vals = inv_func(
                np.squeeze(
                    rs_regime_interp_func([func(xH), func(xHe), func(rs)])
                )
            )

        return tf.TransFuncAtRedshift(
            out_grid_vals, eng=self.eng, in_eng=self.in_eng,
            rs = rs*np.ones_like(out_grid_vals[:,0]), dlnz=self.dlnz,
            spec_type = self.spec_type
        )
    

class TransferFuncInterps:

    def __init__(self, tfInterps, xe_arr):

        length = len(tfInterps)
        self.rs = np.array([None for i in np.arange(length)])
        self.rs_nodes = np.zeros(length-1)
        self.xe_arr = xe_arr
        self.eng = tfInterps[0].eng
        self.in_eng = tfInterps[0].in_eng
        self.dlnz = tfInterps[0].dlnz
        self._log_interp = [tf._log_interp for tf in tfInterps]

        for i, tfInterp in enumerate(tfInterps):
            if np.any(np.diff(tfInterp.rs)<0):
                raise TypeError('redshifts in tfInterp[%d] should be increasing' % i+1)
            self.rs[i] = tfInterp.rs


            if i != length-1:
                if np.all(self.eng != tfInterps[i+1].eng):
                    raise TypeError('All TransferFuncInterp objects must have same eng')
                if np.all(self.in_eng != tfInterps[i+1].in_eng):
                    raise TypeError('All TransferFuncInterp objects must have same in_eng')
                if self.dlnz != tfInterps[i+1].dlnz:
                    raise TypeError('All TransferFuncInterp objects must have same dlnz')

                if tfInterps[i].rs[0] > tfInterps[i+1].rs[0]:
                    raise TypeError(
                        'TransferFuncInterp object number %d should have redshifts smaller than object number %d (we demand ascending order of redshifts between objects)' % (i,i+1)
                    )
                if tfInterps[i].rs[-1] < tfInterps[i+1].rs[0]:
                    raise TypeError(
                        'The largest redshift in ionRSinterp_list[%d] is smaller '
                        +'than the largest redshift in ionRSinterp_list[%d] (i.e. there\'s a missing interpolation window)' % (i,i+1)
                    )
                self.rs_nodes[i] = tfInterps[i+1].rs[0]

        self.tfInterps = tfInterps

    def get_tf(self, xe, rs):
        interpInd = np.searchsorted(self.rs_nodes, rs)
        return self.tfInterps[interpInd].get_tf(xe,rs)
