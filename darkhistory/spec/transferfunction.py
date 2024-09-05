"""Functions and classes for processing transfer functions."""

import numpy as np
from scipy import interpolate
from tqdm import tqdm_notebook as tqdm

import darkhistory.physics as phys
from darkhistory.spec.spectools import rebin_N_arr
from darkhistory.spec.spectra import Spectra
# from darkhistory.spec.spectrum import Spectrum


class TransFuncAtEnergy(Spectra):
    """Transfer function at a given injection energy.

    Collection of Spectrum objects, each at different redshifts.

    Parameters
    ----------
    spec_arr : list of Spectrum
        List of Spectrum to be stored together.
    eng : ndarray, optional
        Energy abscissa.
    in_eng : ndarray, optional
        Injection energy abscissa.
    rs : ndarray, optional
        The redshift of the spectra.
    spec_type : {'N', 'dNdE'}, optional
        Type of data stored, 'dNdE' is the default.
    dlnz : float
        The d ln(1+z) step for the transfer function.
    rebin_eng : ndarray, optional
        New abscissa to rebin all of the Spectrum objects into.

    Attributes
    ----------
    in_eng : ndarray
        Array of injection energies corresponding to each spectrum.
    eng : ndarray
        Array of energy abscissa of each spectrum.
    rs : ndarray
        Array of redshifts corresponding to each spectrum.
    spec_type : {'N', 'dNdE'}
        The type of values stored.
    dlnz : float
        The d ln(1+z) step for the transfer function.

    """
    def __init__(
        self, spec_arr, eng=None, in_eng=None, rs=None,spec_type='dNdE', dlnz=-1, rebin_eng=None
    ):

        self.dlnz = dlnz
        super().__init__(
            spec_arr, eng=eng, in_eng=in_eng, rs=rs,
            spec_type=spec_type, rebin_eng=rebin_eng
        )

        # if spec_arr != []:
        #     self._grid_vals = np.atleast_2d(
        #             np.stack([spec._data for spec in spec_arr])
        #         )
        #     self._spec_type = spec_arr[0].spec_type
        #     self._eng = spec_arr[0].eng
        #     self._in_eng = np.array([spec.in_eng for spec in spec_arr])
        #     if len(set(self._in_eng)) > 1:
        #         raise TypeError('injection energies must be the same.')

        #     if np.any(self.rs <= 0):
        #         raise TypeError("injection energy of all spectra must be set.")
        #     self._rs = np.array([spec.rs for spec in spec_arr])
        #     self._N_underflow = np.array(
        #         [spec.underflow['N'] for spec in spec_arr]
        #     )
        #     self._eng_underflow = np.array(
        #         [spec.underflow['eng'] for spec in spec_arr]
        #     )

        # else:

        #     self._grid_vals = np.atleast_2d([])
        #     self._spec_type = spec_type
        #     self._eng = np.array([])
        #     self._in_eng = np.array([])
        #     self._rs = np.array([])
        #     self._N_underflow = np.array([])
        #     self._eng_underflow = np.array([])

    def __iter__(self):
        return iter(self.grid_vals)

    def at_rs(
        self, new_rs, interp_type='val', bounds_error=None, fill_value=np.nan
    ):
        """Interpolates the transfer function at a new redshift.

        Interpolation is logarithmic.

        Parameters
        ----------
        new_rs : ndarray
            The redshifts or redshift bin indices at which to interpolate.
        interp_type : {'val', 'bin'}
            The type of interpolation. 'bin' uses bin index, while 'val' uses the actual redshift.
        bounds_error : bool, optional
            See scipy.interpolate.interp1d.
        fill_value : array-like or (array-like, array-like) or "extrapolate", optional
            See scipy.interpolate.interp1d.
        """

        if (
            not np.all(np.diff(self.rs)) > 0
            and not np.all(np.diff(self.rs)) < 0
        ):
            raise TypeError('redshift abscissa must be strictly increasing or decreasing for interpolation.')

        non_zero_grid = self.grid_vals
        # set zero values to some small value for log interp.
        non_zero_grid[np.abs(non_zero_grid) < 1e-100] = 1e-200

        interp_func = interpolate.interp1d(
            np.log(self.rs), np.log(non_zero_grid), axis=0,
            bounds_error=bounds_error, fill_value=fill_value
        )

        if interp_type == 'val':

            new_tf = TransFuncAtEnergy([])

            new_tf._spec_type = self.spec_type
            interp_vals = np.exp(interp_func(np.log(new_rs)))
            interp_vals[interp_vals < 1e-100] = 0
            new_tf._grid_vals = interp_vals
            new_tf._eng = self.eng
            new_tf._in_eng = self.in_eng[0]*np.ones_like(new_rs)
            new_tf._rs = new_rs
            new_tf.dlnz = self.dlnz

            return new_tf

        elif interp_type == 'bin':

            log_new_rs = np.interp(
                np.log(new_rs),
                np.arange(self.rs.size),
                np.log(self.rs)
            )

            return self.at_rs(np.exp(log_new_rs))

        else:
            raise TypeError("invalid interp_type specified.")

    def sum_specs(self, weight=None):
        """Sums the spectrum in each energy bin, weighted by `weight`.

        Applies Spectra.sum_specs, but sets `in_eng` of the output `Spectrum` correctly.

        Parameters
        ----------
        weight : ndarray or Spectrum, optional
            The weight in each redshift bin, with weight of 1 for every bin if not specified.

        Returns
        -------
        ndarray or Spectrum
            An array or `Spectrum` of weight sums, one for each energy in `self.eng`, with length `self.length`.

        """
        out_spec = super().sum_specs(weight)
        # Remember that self.in_eng is an array, all
        # having the same value.
        out_spec.in_eng = self.in_eng[0]

        return out_spec

    def append(self, spec):
        """Appends a new Spectrum.

        Applies Spectra.append, but first checks that the appended `Spectrum` has the same injection energy, and is correctly ordered.

        Parameters
        ----------
        spec : Spectrum
            The new spectrum to append.
        """
        if self.rs[-1] < spec.rs:
            raise TypeError("new Spectrum has a larger redshift than the current last entry.")

        if spec.in_eng != self.in_eng[-1]:
            raise TypeError("cannot append new spectrum with different injection energy.")

        super().append(spec)


class TransFuncAtRedshift (Spectra):
    """Transfer function at a given redshift.

    Collection of Spectrum objects, each at different injection energies.

    Parameters
    ----------
    spec_arr : list of Spectrum or ndarray
        List of Spectrum to be stored together.
    eng : ndarray
        The energy abscissa of each Spectrum.
    in_eng : ndarray
        The injection energy abscissa.
    rs : ndarray
    dlnz : float
        d ln(1+z) associated with this transfer function.
    spec_type : {'N', 'dNdE'}, optional
        The type of spectrum saved.
    rs : float
        Redshift of this transfer function.
    rebin_eng : ndarray, optional
        New abscissa to rebin all of the Spectrum objects into.
    with_interp_func : bool
        If true, also returns an interpolation function of the grid.


    Attributes
    ----------
    spec_arr : list of Spectrum
        List of Spectrum to be stored together.
    dlnz : float
        d ln(1+z) associated with this transfer function.
    rs : float
        Redshift of this transfer function.
    interp_func : function
        The 2D interpolation function.

    """

    def __init__(
        self, spec_arr, eng=None, in_eng=None, rs=None,
        dlnz=-1, spec_type='dNdE', rebin_eng=None, with_interp_func=False
    ):

        if isinstance(spec_arr, dict): # initialize from dictionary.
            d = spec_arr
            super().__init__(
                d['grid_vals'], eng=d['eng'], in_eng=d['in_eng'], rs=d['rs'],
                spec_type=d['spec_type'].decode(), rebin_eng=rebin_eng
            )
            with_interp_func = d['with_interp_func'] # force from dictionary.
        else: # original initialization.
            super().__init__(
                spec_arr, eng=eng, in_eng=in_eng, rs=rs, spec_type=spec_type, rebin_eng=rebin_eng
            )

        # common: build interpolation function
        self.dlnz = dlnz
        self.with_interp_func = with_interp_func
        if self.with_interp_func:
            non_zero_grid = self.grid_vals
            non_zero_grid[np.abs(non_zero_grid) < 1e-100] = 1e-200 # set zero values to some small value for log interp.
            interp_grid = non_zero_grid
            self.interp_func = interpolate.interp2d(
                np.log(self.in_eng), np.log(self.eng),
                np.transpose(interp_grid),
                bounds_error = False,
                fill_value = 1e-200
            )

    def to_dict(self):
        """Return hdf5 compatible dictionary."""
        return {
            'eng' : self.eng,
            'in_eng' : self.in_eng,
            'rs' : self.rs,
            'dlnz' : self.dlnz,
            'spec_type' : self.spec_type,
            'grid_vals' : self.grid_vals,
            'with_interp_func' : self.with_interp_func,
        }

    def at_in_eng(self, new_eng, interp_type='val', log_interp=False, bounds_error=None, fill_value=np.nan):
        """Interpolates the transfer function at a new injection energy.

        Interpolation is logarithmic.

        Parameters
        ----------
        new_eng : ndarray
            The injection energies or injection energy bin indices at which to interpolate.
        interp_type : {'val', 'bin'}
            The type of interpolation. 'bin' uses bin index, while 'val' uses the actual injection energies.
        log_interp : bool, optional
            Whether to perform an interpolation over log of the grid values. Default is False.
        bounds_error : bool, optional
            See scipy.interpolate.interp1d.
        fill_value : array-like or (array-like, array-like) or "extrapolate", optional
            See scipy.interpolate.interp1d.

        Returns
        -------
        TransFuncAtRedshift
            New transfer function at the new injection energy.
        """

        if (
            not np.all(np.diff(self.in_eng)) > 0
            and not np.all(np.diff(self.in_eng)) < 0
        ):
            raise TypeError('injection energy must be strictly increasing or decreasing for interpolation.')

        non_zero_grid = self.grid_vals
        non_zero_N_und = self.N_underflow
        non_zero_eng_und = self.eng_underflow
        # set zero values to some small value for log interp.
        non_zero_grid[np.abs(non_zero_grid) < 1e-100] = 1e-200
        non_zero_N_und[np.abs(non_zero_N_und) < 1e-100] = 1e-200
        non_zero_eng_und[np.abs(non_zero_eng_und) < 1e-100] = 1e-200

        if interp_type == 'val':

            if log_interp:
                interp_grid  = np.log(non_zero_grid)
                N_und_grid   = np.log(non_zero_N_und)
                eng_und_grid = np.log(non_zero_eng_und)
            else:
                interp_grid  = non_zero_grid
                N_und_grid   = non_zero_N_und
                eng_und_grid = non_zero_eng_und

            interp_func = interpolate.interp1d(
                np.log(self.in_eng), interp_grid, axis=0,
                bounds_error=bounds_error, fill_value=fill_value
            )

            interp_func_N_und = interpolate.interp1d(
                np.log(self.in_eng), N_und_grid,
                bounds_error=bounds_error, fill_value=fill_value
            )

            interp_func_eng_und = interpolate.interp1d(
                np.log(self.in_eng), eng_und_grid,
                bounds_error=bounds_error, fill_value=fill_value
            )

            new_tf = TransFuncAtRedshift([])

            new_tf._spec_type = self.spec_type

            if log_interp:

                interp_vals = np.exp(interp_func(np.log(new_eng)))
                interp_vals_N_und = np.exp(interp_func_N_und(np.log(new_eng)))
                interp_vals_eng_und = np.exp(
                    interp_func_eng_und(np.log(new_eng))
                )

            else:
                interp_vals = interp_func(np.log(new_eng))
                interp_vals_N_und = interp_func_N_und(np.log(new_eng))
                interp_vals_eng_und = interp_func_eng_und(np.log(new_eng))

            interp_vals[interp_vals < 1e-100] = 0
            interp_vals_N_und[interp_vals_N_und < 1e-100] = 0
            interp_vals_eng_und[interp_vals_eng_und < 1e-100] = 0
            new_tf._grid_vals = interp_vals
            new_tf._eng = self.eng
            new_tf._in_eng = new_eng
            new_tf._rs = self.rs
            new_tf._N_underflow = interp_vals_N_und
            new_tf._eng_underflow = interp_vals_eng_und
            new_tf.dlnz = self.dlnz

            return new_tf

        elif interp_type == 'bin':

            log_new_eng = np.interp(
                np.log(new_eng),
                np.arange(self.in_eng.size),
                np.log(self.in_eng)
            )

            return self.at_in_eng(
                np.exp(log_new_eng), interp_type='val',
                log_interp = log_interp,
                bounds_error=bounds_error, fill_value=fill_value
            )

    def at_eng(
        self, new_eng, interp_type='val',
        bounds_error=None, fill_value= np.nan
    ):
        """Interpolates the transfer function at a new energy abscissa.

        Interpolation is logarithmic.

        Parameters
        ----------
        new_eng : ndarray
            The energy abscissa or energy abscissa bin indices at which to interpolate.
        interp_type : {'val', 'bin'}
            The type of interpolation. 'bin' uses bin index, while 'val' uses the actual injection energies.
        bounds_error : bool, optional
            See scipy.interpolate.interp1d.
        fill_value : array-like or (array-like, array-like) or "extrapolate", optional
            See scipy.interpolate.interp1d.

        Returns
        -------
        TransFuncAtRedshift
            New transfer function at the new energy abscissa.
        """
        non_zero_grid = self.grid_vals
        non_zero_N_und = self.N_underflow
        non_zero_eng_und = self.eng_underflow
        # set zero values to some small value for log interp.
        non_zero_grid[np.abs(non_zero_grid) < 1e-100] = 1e-200
        non_zero_N_und[np.abs(non_zero_N_und) < 1e-100] = 1e-200
        non_zero_eng_und[np.abs(non_zero_eng_und) < 1e-100] = 1e-200

        interp_func = interpolate.interp1d(
            np.log(self.eng), np.log(non_zero_grid), axis=1,
            bounds_error=bounds_error, fill_value=fill_value
        )

        interp_func_N_und = interpolate.interp1d(
            np.log(self.in_eng), np.log(non_zero_N_und),
            bounds_error=bounds_error, fill_value=fill_value
        )

        interp_func_eng_und = interpolate.interp1d(
            np.log(self.in_eng), np.log(non_zero_eng_und),
            bounds_error=bounds_error, fill_value=fill_value
        )

        if interp_type == 'val':

            new_tf = TransFuncAtRedshift([])

            new_tf._spec_type = self.spec_type
            interp_vals = np.exp(interp_func(np.log(new_eng)))
            interp_vals_N_und = np.exp(
                interp_func_N_und(np.log(new_eng))
            )
            interp_vals_eng_und = np.exp(
                interp_func_eng_und(np.log(new_eng))
            )
            interp_vals[interp_vals < 1e-100] = 0
            interp_vals_N_und[interp_vals_N_und < 1e-100] = 0
            interp_vals_eng_und[interp_vals_eng_und < 1e-100] = 0
            new_tf._grid_vals = interp_vals
            new_tf._eng = new_eng
            new_tf._in_eng = self.in_eng
            new_tf._rs = self.rs
            new_tf._N_underflow = interp_vals_N_und
            new_tf._eng_underflow = interp_vals_eng_und
            new_tf.dlnz = self.dlnz

            return new_tf

        elif interp_type == 'bin':

            log_new_eng = np.interp(
                np.log(new_eng),
                np.arange(self.eng.size),
                np.log(self.eng)
            )

            return self.at_eng(
                np.exp(log_new_eng), interp_type='val',
                bounds_error=bounds_error, fill_value=fill_value
            )

    def at_val(
        self, new_in_eng, new_eng, interp_type='val', log_interp=False,
        bounds_error=None, fill_value= np.nan
    ):
        """2D interpolation at specified abscissa.

        Interpolation is logarithmic. 2D interpolation should be preferred over 1D interpolation over each abscissa in the interest of accuracy.

        Parameters
        ----------
        new_in_eng : ndarray
            The injection energy abscissa or injection energy bin indices at which to interpolate.
        new_eng : ndarray
            The energy abscissa or energy abscissa bin indices at which to interpolate.
        interp_type : {'val', 'bin'}
            The type of interpolation. 'bin' uses bin index, while 'val' uses the actual injection energies.
        log_interp : bool, optional
            Whether to perform an interpolation over log of the grid values. Default is False.
        bounds_error : bool, optional
            See scipy.interpolate.interp1d.
        fill_value : array-like or (array-like, array-like) or "extrapolate", optional
            See scipy.interpolate.interp1d.

        Returns
        -------
        TransFuncAtRedshift
            New transfer function at the new abscissa.
        """

        # 2D interpolation, specified by vectors of length eng, in_eng,
        # and grid dimensions in_eng x eng.
        # interp_func takes (eng, in_eng) as argument.

        non_zero_grid = self.grid_vals
        non_zero_N_und = self.N_underflow
        non_zero_eng_und = self.eng_underflow
        # set zero values to some small value for log interp.
        non_zero_grid[np.abs(non_zero_grid) < 1e-100] = 1e-200
        non_zero_N_und[np.abs(non_zero_N_und) < 1e-100] = 1e-200
        non_zero_eng_und[np.abs(non_zero_eng_und) < 1e-100] = 1e-200

        if interp_type == 'val':

            if log_interp:
                interp_grid  = np.log(non_zero_grid)
                N_und_grid   = np.log(non_zero_N_und)
                eng_und_grid = np.log(non_zero_eng_und)
            else:
                interp_grid  = non_zero_grid
                N_und_grid   = non_zero_N_und
                eng_und_grid = non_zero_eng_und

            interp_func = interpolate.interp2d(
                np.log(self.eng), np.log(self.in_eng),
                interp_grid,
                bounds_error=bounds_error,
                fill_value=np.log(fill_value)
            )

            interp_func_N_und = interpolate.interp1d(
                np.log(self.in_eng), N_und_grid,
                bounds_error=False, fill_value=0
            )

            interp_func_eng_und = interpolate.interp1d(
                np.log(self.in_eng), eng_und_grid,
                bounds_error=False, fill_value=0
            )

            new_tf = TransFuncAtRedshift([])

            new_tf._spec_type = self.spec_type

            if log_interp:
                new_tf._grid_vals = np.atleast_2d(
                    np.exp(interp_func(np.log(new_eng), np.log(new_in_eng)))
                )
                interp_vals_N_und = np.exp(
                    interp_func_N_und(np.log(new_in_eng))
                )
                interp_vals_eng_und = np.exp(
                    interp_func_eng_und(np.log(new_in_eng))
                )
            else:
                new_tf._grid_vals   = np.atleast_2d(
                    interp_func(np.log(new_eng), np.log(new_in_eng))
                )
                interp_vals_N_und   = interp_func_N_und(np.log(new_in_eng))
                interp_vals_eng_und = interp_func_eng_und(np.log(new_in_eng))

            # Re-zero small values.
            new_tf._grid_vals[new_tf.grid_vals < 1e-100] = 0
            interp_vals_N_und[interp_vals_N_und < 1e-100] = 0
            interp_vals_eng_und[interp_vals_eng_und < 1e-100] = 0

            new_tf._eng = new_eng
            new_tf._in_eng = new_in_eng
            new_tf._rs = self.rs
            new_tf._N_underflow = interp_vals_N_und
            new_tf._eng_underflow = interp_vals_eng_und

            return new_tf

        elif interp_type == 'bin':

            if issubclass(new_eng.dtype.type, np.integer):
                return self.at_in_eng(
                    new_in_eng, interp_type='bin',
                    log_interp=log_interp
                ).at_eng(
                    new_eng, interp_type='bin'
                )

            log_new_in_eng = np.interp(
                np.log(new_in_eng),
                np.arange(self.in_eng.size),
                np.log(self.in_eng)
            )

            log_new_eng = np.interp(
                np.log(new_eng),
                np.arange(self.eng.size),
                np.log(self.eng)
            )

            return self.at_val(
                np.exp(log_new_in_eng), np.exp(log_new_eng),
                interp_type = 'val',
                log_interp = log_interp,
                bounds_error = bounds_error,
                fill_value = fill_value
            )


    def plot(
        self, ax, ind=None, step=1, indtype='ind', fac=1, **kwargs
    ):
        """Plots the contained `Spectrum` objects.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis handle of the figure to show the plot in.
        ind : int, float, tuple or ndarray, optional.
            Index or injected energy of Spectrum to plot, or a tuple of indices or injected energies providing a range of Spectrum to plot, or a list of indices or injected energies of Spectrum to plot.
        step : int, optional
            The number of steps to take before choosing one Spectrum to plot.
        indtype : {'ind', 'in_eng'}, optional
            Specifies whether ind is an index or an abscissa value.
        fac : ndarray, optional
            Factor to multiply the array by.
        **kwargs : optional
            All additional keyword arguments to pass to matplotlib.plt.plot.

        Returns
        -------
        matplotlib.figure
        """

        if ind is None:
            return self.plot(
                ax, ind=np.arange(self.in_eng.size), fac=fac, **kwargs
            )

        if indtype == 'ind':

            if np.issubdtype(type(ind), np.int64):
                return ax.plot(
                    self.eng, self.grid_vals[ind]*fac, **kwargs
                )

            elif isinstance(ind, tuple):
                spec_to_plot = np.stack(
                    [self.grid_vals[i]*fac
                        for i in np.arange(ind[0], ind[1], step)
                    ], axis = -1
                )
                return ax.plot(self.eng, spec_to_plot, **kwargs)


            elif isinstance(ind, np.ndarray) or isinstance(ind, list):
                spec_to_plot = np.stack(
                    [self.grid_vals[i]*fac for i in ind], axis=-1
                )
                return ax.plot(self.eng, spec_to_plot, **kwargs)

            else:
                raise TypeError("invalid ind.")

        elif indtype == 'in_eng':

            if (
                np.issubdtype(type(ind),np.int64)
                or np.issubdtype(type(ind), np.float64)
            ):
                return self.at_val(
                    np.array([ind]), self.eng, interp_type='val'
                ).plot(ax, ind=0, fac=fac, **kwargs)

            elif isinstance(ind, tuple):
                eng_to_plot = np.arange(ind[0], ind[1], step)
                return self.at_val(
                        eng_to_plot, self.eng, interp_type='val'
                    ).plot(ax, fac=fac, **kwargs)

            elif isinstance(ind, np.ndarray):
                return self.at_val(
                    ind, self.eng, interp_type='val'
                ).plot(ax, fac=fac, **kwargs)

        else:
            raise TypeError("indtype must be either ind or in_eng.")

    def sum_specs(self, weight=None):
        """Sums the spectrum in each energy bin, weighted by `weight`.

        Applies Spectra.sum_specs, but sets `rs` of the output `Spectrum` correctly.

        Parameters
        ----------
        weight : ndarray or Spectrum, optional
            The weight in each redshift bin, with weight of 1 for every bin if not specified.

        Returns
        -------
        ndarray or Spectrum
            An array or `Spectrum` of weight sums, one for each energy in `self.eng`, with length `self.length`.

        """
        out_spec = super().sum_specs(weight)
        # Remember that self.rs is an array, all with the
        # same value of rs.
        out_spec.rs = self.rs[0]
        if self.spec_type == 'dNdE':
            out_spec._spec_type = 'dNdE'
        return out_spec

    def append(self, spec):
        """Appends a new Spectrum.

        Applies Spectra.append, but first checks that the appended spectrum has the same redshift, and is correctly ordered.

        Parameters
        ----------
        spec : Spectrum
            The new spectrum to append.
        """
        if self.in_eng.size > 0:
            if self.in_eng[-1] > spec.in_eng:
                raise TypeError("new Spectrum has a smaller injection energy than the current last entry.")
            if self.rs[-1] != spec.rs:
                raise TypeError('redshift of the new Spectrum must be the same.')

        super().append(spec)


# def process_raw_tf(file):
#     """Processes raw data to return transfer functions.

#     Parameters
#     ----------
#     file : str
#         File to be processed.

#     Returns
#     -------
#     list of TransferFunction
#         List indexed by injection energy.


#     """

#     from darkhistory.spec.transferfunclist import TransferFuncList

#     def get_out_eng_absc(in_eng):
#         """ Returns the output energy abscissa for a given input energy.

#         Parameters
#         ----------
#         in_eng : float
#             Input energy (in eV).

#         Returns
#         -------
#         ndarray
#             Output energy abscissa.
#         """
#         log_bin_width = np.log((phys.me + in_eng)/1e-4)/500
#         bin_boundary = 1e-4 * np.exp(np.arange(501) * log_bin_width)
#         bin_boundary_low = bin_boundary[0:500]
#         bin_boundary_upp = bin_boundary[1:501]

#         return np.sqrt(bin_boundary_low * bin_boundary_upp)

#     #Redshift abscissa. In decreasing order.
#     rs_step = 50
#     rs_upp  = 31.
#     rs_low  = 4.

#     log_rs_absc = (np.log(rs_low) + (np.arange(rs_step) + 1)
#                  *(np.log(rs_upp) - np.log(rs_low))/rs_step)
#     log_rs_absc = np.flipud(log_rs_absc)

#     # Input energy abscissa.

#     in_eng_step = 500
#     low_in_eng_absc = 3e3 + 100.
#     upp_in_eng_absc = 5e3 * np.exp(39 * np.log(1e13/5e3) / 40)
#     in_eng_absc = low_in_eng_absc * np.exp((np.arange(in_eng_step)) *
#                   np.log(upp_in_eng_absc/low_in_eng_absc) / in_eng_step)

#     # Output energy abscissa
#     out_eng_absc_arr = np.array([get_out_eng_absc(in_eng)
#                                 for in_eng in in_eng_absc])

#     # Initial injected bin in output energy abscissa
#     init_inj_eng_arr = np.array([out_eng_absc[out_eng_absc < in_eng][-1]
#         for in_eng,out_eng_absc in zip(in_eng_absc, out_eng_absc_arr)
#     ])

#     # Import raw data.
#     # Raw data has shape in_eng, rs, xe, out_eng,
#     # type:{photonspectrum, lowengphot, lowengelec}

#     tf_raw = np.load(file)
#     tf_raw = np.swapaxes(tf_raw, 0, 1)
#     tf_raw = np.swapaxes(tf_raw, 1, 2)
#     tf_raw = np.swapaxes(tf_raw, 2, 3)
#     tf_raw = np.flip(tf_raw, axis=0)

#     # tf_raw has indices (redshift, xe, out_eng, in_eng), redshift in decreasing order.

#     # Prepare the output.

#     norm_fac = (in_eng_absc/init_inj_eng_arr)*2
#     # The transfer function is expressed as a dN/dE spectrum as a result of injecting approximately 2 particles in out_eng_absc[-1]. The exact number is computed and the transfer function appropriately normalized to 1 particle injection (at energy out_eng_absc[-1]).

#     tf_raw_list = [
#         [
#             Spectrum(
#                 out_eng_absc_arr[i], tf_raw[j,0,:,i]/norm_fac[i],
#                 rs=np.exp(log_rs_absc[j]), in_eng = init_inj_eng_arr[i]
#             ) for j in np.arange(tf_raw.shape[0])
#         ]
#         for i in np.arange(tf_raw.shape[-1])
#     ]

#     normfac2 = rebin_N_arr(np.ones(init_inj_eng_arr.size),
#         init_inj_eng_arr
#     ).dNdE
#     # This rescales the transfer function so that it is now normalized to
#     # dN/dE = 1.

#     # print(normfac2)


#     transfer_func_table = TransferFuncList([
#         TransFuncAtEnergy(
#             spec_arr/N, dlnz=0.002, rebin_eng = init_inj_eng_arr
#         ) for N, spec_arr in zip(normfac2, tqdm(tf_raw_list))
#     ])

#     # This further rescales the spectrum so that it is now the transfer
#     # function for dN/dE = 1 in in_eng_absc.


#     #Rebin to the desired abscissa, which is in_eng_absc.
#     # for spec_list,out_eng_absc in zip(tqdm(tf_raw_list),out_eng_absc_arr):
#     #     for spec in spec_list:
#     #         spec.rebin(in_eng_absc)
#     #     # Note that the injection energy is out_eng_absc[-1] due to our conventions in the high energy code.
#     #     transfer_func_table.append(TransferFunction(spec_list, out_eng_absc[-1]))

#     return transfer_func_table
