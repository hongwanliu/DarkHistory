"""Functions and classes for processing lists of transfer functions."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from darkhistory.utilities import dict_from_inhom_list, inhom_list_from_dict

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
                    raise ValueError('invalid dimensions for x_arr.')
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
    ----------
    ionrsarrays : list of IonRSArray
        IonRSArray objects to interpolate over. 
    rs_nodes : ndarray
        List of redshifts to transition between redshift regimes. 
    log_interp : bool, optional
        If true, performs an interpolation over log of the grid values. 
    """

    def __init__(self, ionrsarrays, rs_nodes=None, log_interp=False):

        if isinstance(ionrsarrays, dict): # initialize from dictionary.
            self.from_dict(ionrsarrays)

        else: # original initialization.
            if ( # rs_nodes must have 1 less entry than tflistarrs.
                (rs_nodes is not None and len(rs_nodes) != len(ionrsarrays)-1)
                or (rs_nodes is None and len(ionrsarrays) > 1)
            ):
                raise ValueError('rs_nodes incompatible with given ionrsarrays.')

            if rs_nodes is not None and len(rs_nodes) > 1: # rs_nodes must be in *increasing* redshift
                if not np.all(np.diff(rs_nodes) > 0):
                    raise ValueError('rs_nodes must be in increasing order.')

            self.rs        = [ionrsarr.rs for ionrsarr in ionrsarrays]
            # self.in_eng    = [ionrsarr.in_eng for ionrsarr in ionrsarrays]
            # self.eng       = [ionrsarr.eng for ionrsarr in ionrsarrays]
            self.rs_nodes  = rs_nodes
            self.grid_vals = [ionrsarr._grid_vals for ionrsarr in ionrsarrays]

            self.x         = []
            for ionrsarr in ionrsarrays:
                try:
                    self.x.append(ionrsarr.x)
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
        for x_vals, z, grid in zip(self.x, self.rs, self.grid_vals):
            if x_vals is None: # no x dependence.
                self.interp_func.append(interp1d(func(z), func(np.squeeze(grid)), axis=0))
            elif x_vals.ndim == 1: # xH dependence.
                self.interp_func.append(RegularGridInterpolator((func(x_vals), func(z)), func(grid)))
            elif x_vals.ndim == 3: # xH, xHe dependence.
                xH_arr = x_vals[:,0,0]
                xHe_arr = x_vals[0,:,1]
                self.interp_func.append(RegularGridInterpolator((func(xH_arr), func(xHe_arr), func(z)), func(grid)))
            else:
                raise ValueError('x_vals has anomalous dimensions (and not in a good QFT way).')
            

    def to_dict(self):
        """Return hdf5 compatible dictionary. Does not save eng and in_eng information."""
        d = {
            'rs_nodes': self.rs_nodes,
            'log_interp': self._log_interp,
        }
        d.update(dict_from_inhom_list(self.rs, 'rs'))
        x_save = [(-1 if x is None else x) for x in self.x]
        d.update(dict_from_inhom_list(x_save, 'x'))
        d.update(dict_from_inhom_list(self.grid_vals, 'grid_vals'))
        return d
    

    def from_dict(self, d):
        """Initialize from hdf5 compatible dictionary."""
        self.rs_nodes = d['rs_nodes']
        self._log_interp = d['log_interp']
        self.rs = inhom_list_from_dict(d, 'rs')
        self.x = inhom_list_from_dict(d, 'x')
        for i, x in enumerate(self.x):
            if np.isscalar(x) and x == -1:
                self.x[i] = None
        self.grid_vals = inhom_list_from_dict(d, 'grid_vals')


    def get_val(self, xH, xHe, rs):

        if self._log_interp:
            func = np.log
            inv_func = np.exp
        else:
            inv_func = func = lambda x: x

        rs_regime_ind = np.searchsorted(self.rs_nodes, rs)
        if rs > self.rs[rs_regime_ind][-1] or rs < self.rs[rs_regime_ind][0]:
            raise ValueError('redshift lies outside of range.')

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

        if self.x[rs_regime_ind] is None:
            out_grid_vals = inv_func(
                np.squeeze(rs_regime_interp_func(func(rs)))
            )
        elif self.x[rs_regime_ind].ndim == 1:
            out_grid_vals = inv_func(
                np.squeeze(rs_regime_interp_func([func(xH), func(rs)]))
            )
        elif self.x[rs_regime_ind].ndim == 3:
            out_grid_vals = inv_func(
                np.squeeze(
                    rs_regime_interp_func([func(xH), func(xHe), func(rs)])
                )
            )
        else:
            raise ValueError('x has an anomalous dimension (and not in a good QFT way).')

        return out_grid_vals