"""Contains the `Spectra` class."""

import numpy as np
from darkhistory import utilities as utils
from darkhistory.spec.spectools import get_bin_bound
from darkhistory.spec.spectools import get_log_bin_width
from darkhistory.spec.spectrum import Spectrum
import matplotlib.pyplot as plt
import warnings

from scipy import integrate
from scipy import interpolate


class Spectra:
    """Structure for a collection of `Spectrum` objects.

    Parameters
    ---------- 
    spec_arr : list of Spectrum
        List of Spectrum to be stored together.
    rebin_eng : ndarray, optional
        New abscissa to rebin all of the `Spectrum` objects into.

    Attributes
    ----------
    spec_arr : list of Spectrum
        List of Spectrum stored together.    

    """
    # __array_priority__ must be larger than 0, so that radd can work.
    # Otherwise, ndarray + Spectrum works by iterating over the elements of
    # ndarray first, which isn't what we want.
    __array_priority__ = 1           

    def __init__(self, spec_arr, rebin_eng=None):

        self.spec_arr = spec_arr

        if rebin_eng is not None:
            self.rebin(rebin_eng)

        if spec_arr != []:

            if not utils.arrays_equal([spec.eng for spec in spec_arr]):
                raise TypeError("all abscissae must be the same.")

    def __iter__(self):
        return iter(self.spec_arr)

    def __getitem__(self,key):
        if np.issubdtype(type(key), int) or isinstance(key, slice):
            return self.spec_arr[key]
        else:
            raise TypeError("index must be int.")

    def __setitem__(self,key,value):
        if np.issubdtype(type(key), int):
            if not isinstance(value, (list, tuple)):
                if np.issubclass_(type(value), Spectrum):
                    self.spec_arr[key] = value
                else:
                    raise TypeError("can only add Spectrum.")
            else:
                raise TypeError("can only add one spectrum per index.")
        elif isinstance(key, slice):
            if len(self.spec_arr[key]) == len(value):
                for i,spec in zip(key,value): 
                    if np.issubclass_(type(spec), Spectrum):
                        self.spec_arr[i] = spec
                    else: 
                        raise TypeError("can only add Spectrum.")
            else:
                raise TypeError("can only add one spectrum per index.")
        else:
            raise TypeError("index must be int.")

    def get_rs(self):
        return np.array([spec.rs for spec in self.spec_arr])

    def get_eng(self):
        return self.spec_arr[0].eng

    def get_in_eng(self):
        return np.array([spec.in_eng for spec in self.spec_arr])

    def get_grid_values(self):
        return np.stack([spec.dNdE for spec in self.spec_arr])


    def __add__(self, other): 
        """Adds two `Spectra` instances together.

        Parameters
        ----------
        other : Spectra

        Returns
        -------
        Spectra
            New `Spectra` instance which is an element-wise sum of the `Spectrum` objects in each Spectra.

        Notes
        -----
        This special function, together with `Spectra.__radd__`, allows the use of the symbol + to add `Spectra` objects together. 

        See Also
        --------
        spectra.Spectra.__radd__
        """
        if np.issubclass_(type(other), Spectra):

            if not np.array_equal(self.get_eng(), other.get_eng()):
                raise TypeError('abscissae are different for the two Spectra.')

            # Need to remove this in order to add transfer functions for TransferFuncList.at_val
            
            # if not np.array_equal(self.rs, other.rs):
            #     raise TypeError('redshifts are different for the two Spectra.')

            return Spectra([spec1 + spec2 for spec1,spec2 in zip(self.spec_arr, other.spec_arr)])

        else: raise TypeError('adding an object that is not of class Spectra.')


    def __radd__(self, other): 
        """Adds two `Spectra` instances together.

        Parameters
        ----------
        other : Spectra

        Returns
        -------
        Spectra
            New `Spectra` instance which is an element-wise sum of the `Spectrum` objects in each Spectra.

        Notes
        -----
        This special function, together with `Spectra.__add__`, allows the use of the symbol + to add `Spectra` objects together. 

        See Also
        --------
        spectra.Spectra.__add__
        """
        if np.issubclass_(type(other), Spectra):

            if not np.array_equal(self.get_eng(), other.get_eng()):
                raise TypeError('abscissae are different for the two Spectra.')
            
            # Need to remove this in order to add transfer functions for TransferFuncList.at_val

            # if not np.array_equal(self.rs, other.rs):
            #     raise TypeError('redshifts are different for the two Spectra.')

            return Spectra([spec1 + spec2 for spec1,spec2 in zip(self.spec_arr, other.spec_arr)])

        else: raise TypeError('adding an object that is not of class Spectra.')

    def __sub__(self, other):
        """Subtracts one `Spectra` instance from another. 

        Parameters
        ----------
        other : Spectra

        Returns
        -------
        Spectra
            New `Spectra` instance which has the subtracted list of `dNdE`. 

        Notes
        -----
        This special function, together with `Spectra.__rsub__`, allows the use of the symbol - to subtract or subtract from `Spectra` objects. 

        See Also
        --------
        spectrum.Spectra.__rsub__
        """
        return self + -1*other 

    def __rsub__(self, other):
        """Subtracts one `Spectra` instance from another. 

        Parameters
        ----------
        other : Spectra

        Returns
        -------
        Spectra
            New `Spectra` instance which has the subtracted list of `dNdE`. 

        Notes
        -----
        This special function, together with `Spectra.__rsub__`, allows the use of the symbol - to subtract or subtract from `Spectra` objects. 

        See Also
        --------
        spectrum.Spectra.__sub__
        """    
        return other + -1*self

    def __neg__(self):
        """Negates all of the `dNdE`. 

        Returns
        -------
        Spectra
            New `Spectra` instance with the `dNdE` negated.
        """
        return -1*self

    def __mul__(self, other):
        """Takes the product of two `Spectra` instances. 

        Parameters
        ----------
        other : Spectra, int, float, list or ndarray

        Returns
        -------
        Spectra
            New `Spectra` instance which is an element-wise product of the `Spectrum` objects in each Spectra. 

        Notes
        -----
        This special function, together with `Spectra.__rmul__`, allows the use of the symbol * to add `Spectra` objects together.

        See Also
        --------
        spectrum.Spectra.__rmul__
        """
        if np.issubdtype(type(other), float) or np.issubdtype(type(other), int):
            return Spectra([other*spec for spec in self])
        elif isinstance(other, list) or isinstance(other, np.ndarray):
            if len(other) != len(self.spec_arr):
                raise TypeError("list must be the same length as self.spec_arr.")
            return Spectra(
                [num*spec for num,spec in zip(other,self)]
            )
        elif np.issubclass_(type(other), Spectra):
            if (not np.array_equal(self.get_rs(), other.get_rs()) 
                or not np.array_equal(self.get_eng(), other.get_eng())):
                raise TypeError("the two spectra do not have the same redshift or abscissae.")
            return Spectra([spec1*spec2 for spec1,spec2 in zip(self, other)])
        else:
            raise TypeError("can only multiply Spectra or scalars.")

    def __rmul__(self, other):
        """Takes the product of two `Spectra` instances. 

        Parameters
        ----------
        other : Spectra, int, float, list or ndarray

        Returns
        -------
        Spectra
            New `Spectra` instance which is an element-wise product of the `Spectrum` objects in each Spectra. 

        Notes
        -----
        This special function, together with `Spectra.__mul__`, allows the use of the symbol * to add `Spectra` objects together.

        See Also
        --------
        spectrum.Spectra.__mul__
        """
        if np.issubdtype(type(other), float) or np.issubdtype(type(other), int):
            return Spectra([other*spec for spec in self])
        elif isinstance(other, list) or isinstance(other, np.ndarray):
            if len(other) != len(self.spec_arr):
                raise TypeError("list must be the same length as self.spec_arr.")
            return Spectra(
                [spec*num for num,spec in zip(other,self)]
            )
        elif np.issubclass_(type(other), Spectra):
            if (
                self.get_rs() != other.get_rs() 
                or self.get_eng() != other.get_eng()
            ):
                raise TypeError("the two spectra do not have the same redshift or abscissae.")
            return Spectra([spec2*spec1 for spec1,spec2 in zip(self, other)])
        else:
            raise TypeError("can only multiply Spectra or scalars.")

    def __truediv__(self,other):
        """Divides Spectra by a number or another Spectra. 

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectra

        Notes
        -----
        This special function, together with `Spectra.__rtruediv__`, allows the use of the symbol / to divide `Spectra` objects by a number or another Spectra. 

        See Also
        --------
        spectrum.Spectra.__rtruediv__
        """
        if np.issubclass_(type(other), Spectra):
            invSpec = Spectra([1./spec for spec in other])
            return self*invSpec
        else:
            return self*(1/other)

    def __rtruediv__(self,other):
        """Divides Spectra by a number or another Spectra. 

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectra

        Notes
        -----
        This special function, together with `Spectra.__truediv__`, allows the use of the symbol / to divide `Spectra` objects by a number or another Spectra. 

        See Also
        --------
        spectrum.Spectra.__truediv__
        """
        invSpec = Spectra([1./spec for spec in self])

        return other*invSpec 

    def totN(self, bound_type=None, bound_arr=None):
        """Returns the total number of particles in the spectra.

        The part of the `Spectrum` objects to find the total number of particles can be specified in two ways, and is specified by `bound_type`. Multiple totals can be obtained through `bound_arr`. 

        Parameters
        ----------
        bound_type : {'bin', 'eng', None}
            The type of bounds to use. Bound values do not have to be within the [0:eng.size] for `'bin'` or within the abscissa for `'eng'`. `None` should only be used when computing the total particle number in the spectrum. For `'bin'`, bounds are specified as the bin boundary, with 0 being the left most boundary, 1 the right-hand of the first bin and so on. This is equivalent to integrating over a histogram. For `'eng'`, bounds are specified by energy values.

        bound_arr : ndarray, optional
            An array of boundaries (bin or energy), between which the total number of particles will be computed. If bound_arr = None, but bound_type is specified, the total number of particles in each bin is computed. If both bound_type and bound_arr = None, then the total number of particles in the spectrum is computed.

        Returns
        -------
        ndarray
            Total number of particles in the spectrum. 

        """

        in_eng = self.get_in_eng()
        eng = self.get_eng()
        gridval = self.get_grid_values()

        # np.dot is sum(x[i,j,:] * y[:, k])
        # dNdlogE has size in_eng x eng, we make use of broadcasting.
        dNdlogE = gridval * eng 
        log_bin_width = get_log_bin_width(eng)

        if bound_type is not None:

            if bound_arr is None:

                bound_type = 'bin'
                bound_arr  = np.arange(eng.size + 1)

            if bound_type == 'bin':

                if not all(np.diff(bound_arr) >= 0):
                    raise TypeError(
                        "bound_arr must have increasing entries."
                    )

                # Size is number of totals requested x number of Spectrums
                N_in_bin = np.zeros((bound_arr.size - 1, in_eng.size))

                if bound_arr[0] > eng.size or bound_arr[-1] < 0:
                    return N_in_bin

                for i, (low,upp) in enumerate(
                    (bound_arr[:-1], bound_arr[1:])
                ):

                    # Set the lower and upper bounds, including case where
                    # low and upp are outside of the bins. 
                    if low > eng.size or upp < 0:
                        continue

                    low_ceil  = int(np.ceil(low))
                    low_floor = int(np.floor(low))
                    upp_ceil  = int(np.ceil(upp))
                    upp_floor = int(np.floor(upp))

                    # Sum the bins that are completely between the bounds. 
                    N_full_bins = np.dot(
                        dNdlogE[:,low_ceil:upp_floor],
                        log_bin_width[low_ceil:upp_floor]
                    )

                    N_part_bins = np.zeros_like(in_eng)

                    if low_floor == upp_floor or low_ceil == upp_ceil:
                        # Bin indices are within the same bin. The second requirement covers the case where upp_ceil is eng.size.
                        N_part_bins += (
                            dNdlogE[:,low_floor] * (upp - low)
                            * log_bin_width[low_floor]
                        )
                    else:
                        # Add up part of the bin for the low partial bin and
                        # the high partial bin. 
                        N_part_bins += (
                            dNdlogE[:,low_floor] * (low_ceil - low)
                            * log_bin_width[low_floor]
                        )
                        if upp_floor < eng.size:
                            # If upp_floor is eng.size, then there is no
                            # partial bin for the upper index. 
                            N_part_bins += (
                                dNdlogE[:,upp_floor] * (upp - upp_floor)
                                * log_bin_width[upp_floor]
                            )

                    N_in_bin[i] = N_full_bins + N_part_bins

                return N_in_bin

            if bound_type == 'eng':
                bin_boundary = get_bin_bound(eng)
                eng_bin_ind = np.interp(
                    np.log(bound_arr),
                    np.log(bin_boundary), np.arange(bin_boundary.size),
                    left = -1, right = eng.size + 1
                )

                return self.totN('bin', eng_bin_ind)

        else:
            underflow_vec = np.array([spec.underflow['N'] for spec in self])
            return np.dot(dNdlogE, log_bin_width) + underflow_vec






    def integrate_each_spec(self,weight=None):
        """Sums each `Spectrum`, each `eng` bin weighted by `weight`. 

        Equivalent to contracting `weight` with each `dNdE` in `Spectra`, `weight` should have length `self.length`. 

        Parameters
        ----------
        weight : ndarray, optional
            The weight in each energy bin, with weight of 1 for every bin if not specified. 

        Returns
        -------
        ndarray
            An array of weighted sums, one for each redshift in `self.rs`, with length `self.rs.size`. 
        """
        if weight is None:
            weight = np.ones(self.get_eng().size)

        if isinstance(weight,np.ndarray):
            return np.array([spec.contract(weight) for spec in self])

        else:
            raise TypeError("mat must be an ndarray.")

    def sum_specs(self,weight=None):
        """Sums the spectrum in each energy bin, weighted by `weight`. 

        Equivalent to contracting `weight` with `[spec.dNdE[i] for spec in spec_arr]` for all `i`. `weight` should have length `self.rs.size`. 

        Parameters
        ----------
        weight : ndarray or Spectrum, optional
            The weight in each redshift bin, with weight of 1 for every bin if not specified.

        Returns
        -------
        ndarray or Spectrum
            An array or `Spectrum` of weight sums, one for each energy in `self.eng`, with length `self.length`. 

        """
        if weight is None:
            weight = np.ones(self.get_rs().size)
    
        if isinstance(weight, np.ndarray):
            new_dNdE = np.dot(weight, self.get_grid_values())
            return Spectrum(self.get_eng(), new_dNdE)
        elif isinstance(weight, Spectrum):
            new_dNdE = np.dot(weight.dNdE, self.get_grid_values())
            return Spectrum(self.get_eng(), new_dNdE)
        else:
            raise TypeError("weight must be an ndarray or Spectrum.")

    def rebin(self, out_eng):
        """ Re-bins all `Spectrum` objects according to a new abscissa.

        Rebinning conserves total number and total energy.
        
        Parameters
        ----------
        out_eng : ndarray
            The new abscissa to bin into. If `self.eng` has values that are smaller than `out_eng[0]`, then the new underflow will be filled. If `self.eng` has values that exceed `out_eng[-1]`, then an error is returned.

        See Also
        --------
        spectrum.Spectrum.rebin
        """

        if not np.all(np.diff(out_eng) > 0):
            raise TypeError('new abscissa must be ordered in increasing energy.')

        # Get the bin indices that the current abscissa (self.eng) corresponds to in the new abscissa (new_eng). Can be any number between 0 and self.eng.size-1. Bin indices are wrt the bin centers.

        # Add an additional bin at the lower end of out_eng so that underflow can be treated easily.

        first_bin_eng = np.exp(np.log(out_eng[0]) - (np.log(out_eng[1]) - np.log(out_eng[0])))
        new_eng = np.insert(out_eng, 0, first_bin_eng)

        # Find the relative bin indices for self.eng wrt new_eng. The first bin in new_eng has bin index -1. 
        bin_ind = np.interp(self.eng, new_eng, 
            np.arange(new_eng.size)-1, left = -2, right = new_eng.size)

        # Locate where bin_ind is below 0, above self.length-1 and in between.
        ind_low = np.where(bin_ind < 0)
        ind_high = np.where(bin_ind == new_eng.size)
        ind_reg = np.where( (bin_ind >= 0) & (bin_ind <= new_eng.size - 1) )

        if ind_high[0].size > 0: 
            warnings.warn("The new abscissa lies below the old one: only bins that lie within the new abscissa will be rebinned, bins above the abscissa will be discarded.", RuntimeWarning)





        # for spec in self:
        #     spec.rebin(out_eng)


    def append(self, spec):
        """Appends a new Spectrum. 

        Parameters
        ----------
        spec : Spectrum
            The new spectrum to append.
        """
        
        # Checks if spec_arr is empty
        if self.spec_arr:
            if not np.array_equal(self.get_eng(), spec.eng):
                raise TypeError("new Spectrum does not have the same energy abscissa.")
        
        self.spec_arr.append(spec)

    def at_rs(self, new_rs, interp_type='val',bounds_err=True):
        """Interpolates the transfer function at a new redshift. 

        Interpolation is logarithmic. 

        Parameters
        ----------
        new_rs : ndarray
         The redshifts or redshift bin indices at which to interpolate. 
        interp_type : {'val', 'bin'}
         The type of interpolation. 'bin' uses bin index, while 'val' uses the actual redshift. 
        bounds_err : bool, optional
         Whether to return an error if outside of the bounds for the interpolation. 
        """
        if (
            not np.all(np.diff(self.get_rs()) > 0) 
            and not np.all(np.diff(self.get_rs()) < 0)
        ):
            raise TypeError('redshift abscissa must be strictly increasing or decreasing for interpolation to be correct.')
         
        interp_func = interpolate.interp1d(
            np.log(self.get_rs()), self.get_grid_values(), axis=0
        )
         
        if interp_type == 'val':
             
            new_spec_arr = [
                Spectrum(self.get_eng(), interp_func(np.log(rs)), rs=rs)
                    for rs in new_rs
            ]
            return Spectra(new_spec_arr)

        elif interp_type == 'bin':
             
            log_new_rs = np.interp(
                np.log(new_rs), 
                np.arange(self.get_rs().size), 
                np.log(self.get_rs())
            )

            return self.at_rs(np.exp(log_new_rs))

        else:
             raise TypeError("invalid interp_type specified.")

    def plot(self, ax, ind=None, step=1, indtype='ind', 
        abs_plot=False, fac=1, **kwargs):
        """Plots the contained `Spectrum` objects. 

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis handle of the figure to show the plot in.
        ind : int, float, tuple or ndarray, optional.
            Index or redshift of Spectrum to plot, or a tuple of indices or redshifts providing a range of Spectrum to plot, or a list of indices or redshifts of Spectrum to plot.
        step : int, optional
            The number of steps to take before choosing one Spectrum to plot.
        indtype : {'ind', 'rs'}, optional
            Specifies whether ind is an index or an abscissa value.
        abs_plot :  bool, optional
            Plots the absolute value if true.
        fac : ndarray, optional
            Factor to multiply the dN/dE array by. 
        **kwargs : optional
            All additional keyword arguments to pass to matplotlib.plt.plot. 

        Returns
        -------
        matplotlib.figure
        """
        
        if ind is None:
            return self.plot(
                ax, ind=np.arange(self.get_rs().size), 
                abs_plot=abs_plot, fac=fac, **kwargs
            )

        if indtype == 'ind':

            if np.issubdtype(type(ind), int):
                if abs_plot:
                    return ax.plot(
                        self.get_eng(), 
                        np.abs(self.spec_arr[ind].dNdE*fac), 
                        **kwargs
                    )
                else:
                    return ax.plot(
                        self.get_eng(), 
                        self.spec_arr[ind].dNdE*fac, 
                        **kwargs
                    )

            elif isinstance(ind, tuple):
                if abs_plot:
                    spec_to_plot = np.stack(
                        [np.abs(self.spec_arr[i].dNdE*fac) 
                            for i in 
                                np.arange(ind[0], ind[1], step)
                        ], 
                        axis=-1
                    )
                else:
                    spec_to_plot = np.stack(
                        [self.spec_arr[i].dNdE*fac
                            for i in 
                                np.arange(ind[0], ind[1], step)
                        ], 
                        axis=-1
                    )
                return ax.plot(self.get_eng(), spec_to_plot, **kwargs)
                
            
            elif isinstance(ind, np.ndarray):
                if abs_plot:
                    spec_to_plot = np.stack(
                        [np.abs(self.spec_arr[i].dNdE*fac)
                            for i in ind
                        ], axis=-1
                    ) 
                else:
                    spec_to_plot = np.stack(
                        [self.spec_arr[i].dNdE*fac
                            for i in ind
                        ], axis=-1
                    )
                return ax.plot(self.get_eng(), spec_to_plot, **kwargs)
                

            else:
                raise TypeError("ind should be either int, tuple of int or ndarray.")

        if indtype == 'rs':

            if (np.issubdtype(type(ind),int) or 
                    np.issubdtype(type(ind), float)):
                return self.at_rs(np.array([ind])).plot(
                    ax, ind=0, abs_plot=abs_plot, fac=fac, **kwargs
                )

            elif isinstance(ind, tuple):
                rs_to_plot = np.arange(ind[0], ind[1], step)
                return self.at_rs(rs_to_plot).plot(
                    ax, abs_plot=abs_plot, fac=fac, **kwargs
                )

            elif isinstance(ind, np.ndarray):
                return self.at_rs(ind).plot(
                    ax, abs_plot=abs_plot, fac=fac, **kwargs
                )

        else:
            raise TypeError("indtype must be either ind or rs.")

