"""Contains the `Spectrum` class."""

import numpy as np
from darkhistory import utilities as utils
from darkhistory.spec.spectools import get_bin_bound
from darkhistory.spec.spectools import get_log_bin_width
from darkhistory.spec.spectools import rebin_N_arr
import matplotlib.pyplot as plt
import warnings

from scipy import integrate

class Spectrum:
    """Structure for photon and electron spectra with log-binning in energy. 

    Parameters
    ----------
    eng : ndarray
        Abscissa for the spectrum. 
    data : ndarray
        Spectrum stored as N or dN/dE. 
    rs : float, optional
        The redshift (1+z) of the spectrum. Set to -1 if not specified.
    in_eng : float, optional
        The injection energy of the primary, if this is a secondary spectrum. Set to -1 if not specified.
    mode : {'N', 'dNdE'}, optional
        Whether the input is N or dN/dE in each bin. Default is 'dNdE'.

    Attributes
    ----------
    eng : ndarray
        Abscissa for the spectrum. 
    dNdE : ndarray
        dN/dE of the spectrum.
    N : ndarray
        N of the spectrum.
    rs : float, optional
        The redshift (1+z) of the spectrum. Set to -1 if not specified.
    length : int
        The length of the `eng` and `dNdE`.
    underflow : dict of str: float
        The underflow total number of particles and total energy.
    """

    # __array_priority__ must be larger than 0, so that radd can work.
    # Otherwise, ndarray + Spectrum works by iterating over the elements of
    # ndarray first, which isn't what we want.
    __array_priority__ = 1

    def __init__(self, eng, data, rs=-1., in_eng=-1., spec_type='dNdE'):

        if eng.size != data.size:
            raise TypeError("""abscissa and spectrum need to be of the
             same size.""")
        if eng.size == 1:
            raise TypeError("abscissa must be more than length 1.")
        if not all(np.diff(eng) > 0):
            raise TypeError("abscissa must be ordered in increasing energy.")
        if spec_type != 'N' and spec_type != 'dNdE':
            raise TypeError("invalid spec_type specified.")

        self.eng = eng
        self._data = data
        self.rs = rs
        self.in_eng = in_eng
        self._spec_type = spec_type 
        self.length = eng.size
        self.underflow = {'N': 0., 'eng': 0.}

    @property
    def dNdE(self):
        if self._spec_type == 'dNdE':
            return self._data
        elif self._spec_type == 'N':
            return self._data/(self.eng * get_log_bin_width(self.eng))
            
    @dNdE.setter
    def dNdE(self, value):
        if self._spec_type == 'dNdE':
            self._data = value
        elif self._spec_type == 'N':
            self._data = value/(self.eng * get_log_bin_width(self.eng))

    @property
    def N(self):
        if self._spec_type == 'dNdE':
            return self._data * self.eng * get_log_bin_width(self.eng)
        elif self._spec_type == 'N':
            return self._data

    @N.setter
    def N(self, value):
        if self._spec_type == 'dNdE':
            self._data = value * self.eng * get_log_bin_width(self.eng)
        elif self._spec_type == 'N':
            self._data = value

    @property
    def spec_type(self):
        return self._spec_type


    def __add__(self, other):
        """Adds two `Spectrum` instances together, or an array to the spectrum. The `Spectrum` object is on the left.
        
        Parameters
        ----------
        other : Spectrum, ndarray, float or int

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the summed spectrum. 

        Notes
        -----
        This special function, together with `Spectrum.__radd__`, allows the use of the symbol + to add `Spectrum` objects together.

        The returned `Spectrum` object `underflow` is reset to zero if `other` is not a `Spectrum` object.

        See Also
        --------
        spectrum.Spectrum.__radd__

        """

        # Removed ability to add int or float. Not likely to be useful I think?

        if np.issubclass_(type(other), Spectrum):
            # Some typical errors.
            if not np.array_equal(self.eng, other.eng):
                raise TypeError("abscissae are different for the two Spectrum objects.")
            if self._spec_type != other._spec_type:
                raise TypeError("cannot add N to dN/dE.")
            new_rs = -1
            new_in_eng = -1
            if np.array_equal(self.rs, other.rs):
                new_rs = self.rs 
            if np.array_equal(self.in_eng, other.in_eng):
                new_in_eng = self.in_eng 

            new_spectrum = Spectrum(
                self.eng, self._data+other._data, 
                rs = new_rs, in_eng = new_in_eng, 
                spec_type = self._spec_type
            )
            new_spectrum.underflow['N'] = (self.underflow['N'] 
                                          + other.underflow['N'])
            new_spectrum.underflow['eng'] = (self.underflow['eng']
                                            + other.underflow['eng'])

            return new_spectrum

        elif isinstance(other, np.ndarray):

            return Spectrum(
                self.eng, self._data + other, 
                rs = self.rs, in_eng = self.in_eng, 
                spec_type = self._spec_type
            )

        else:

            raise TypeError("cannot add object to Spectrum.")

    def __radd__(self, other):
        """Adds two `Spectrum` instances together, or an array to the spectrum. The `Spectrum` object is on the right.
        
        Parameters
        ----------
        other : Spectrum, ndarray

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the summed spectrum. 

        Notes
        -----
        This special function, together with `Spectrum.__add__`, allows the use of the symbol + to add `Spectrum` objects together.

        The returned `Spectrum` object `underflow` is reset to zero if `other` is not a `Spectrum` object.

        See Also
        --------
        spectrum.Spectrum.__add__

        """
        
        # Removed ability to add int or float. Not likely to be useful I think?

        if np.issubclass_(type(other), Spectrum):
            # Some typical errors.
            if not np.array_equal(self.eng, other.eng):
                raise TypeError("abscissae are different for the two `Spectrum` objects.")
            if self._spec_type != other._spec_type:
                raise TypeError("cannot add N to dN/dE.")
            new_rs = -1
            new_in_eng = -1
            if self.rs == other.rs:
                new_rs = self.rs 
            if self.in_eng == other.in_eng:
                new_in_eng = self.in_eng 

            new_spectrum = Spectrum(
                self.eng, self._data+other._data, 
                rs = new_rs, in_eng = new_in_eng,
                spec_type = self._spec_type
            )
            new_spectrum.underflow['N'] = (self.underflow['N'] 
                                          + other.underflow['N'])
            new_spectrum.underflow['eng'] = (self.underflow['eng']
                                            + other.underflow['eng'])

            return new_spectrum

        elif isinstance(other, np.ndarray):

            return Spectrum(
                self.eng, self._data + other, 
                rs = self.rs, in_eng = self.in_eng, 
                spec_type = self._spec_type 
                )

        else:

            raise TypeError("cannot add object to Spectrum.")

    def __sub__(self, other):
        """Subtracts one `Spectrum` instance from another, or subtracts an array from the spectrum. 
        
        Parameters
        ----------
        other : Spectrum or ndarray

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the subtracted `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__rsub__`, allows the use of the symbol - to subtract or subtract from `Spectrum` objects.

        The returned `Spectrum` object underflow is reset to zero if `other` is not a `Spectrum` object.

        See Also
        --------
        spectrum.Spectrum.__rsub__

        """
        return self + -1*other

    def __rsub__(self, other):
        """Subtracts one `Spectrum` instance from another, or subtracts the spectrum from an array.
        
        Parameters
        ----------
        other : Spectrum or ndarray

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the subtracted `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__sub__`, allows the use of the symbol - to subtract or subtract from `Spectrum` objects.

        See Also
        --------
        spectrum.Spectrum.__sub__

        """
        return other + -1*self

    def __neg__(self):
        """Negates the spectrum.

        Returns
        -------
        Spectrum
            New `Spectrum` instance with the spectrum negated. 
        """
        return -1*self

    def __mul__(self,other):
        """Takes the product of the spectrum with an array or number. `Spectrum` object is on the left.

        Parameters
        ----------
        other : Spectrum, ndarray, float or int

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the multiplied `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__rmul__`, allows the use of the symbol * to multiply `Spectrum` objects or an array and Spectrum.

        The returned `Spectrum` object `underflow` is reset to zero if `other` is not a `Spectrum` object.

        See Also
        --------
        spectrum.Spectrum.__rmul__

        """
        if (
            np.issubdtype(type(other),np.float64) 
            or np.issubdtype(type(other),np.int64)
        ):
            new_spectrum = Spectrum(
                self.eng, self._data*other, 
                rs = self.rs, in_eng = self.in_eng,
                spec_type = self._spec_type
            )
            new_spectrum.underflow['N'] = self.underflow['N']*other
            new_spectrum.underflow['eng'] = self.underflow['eng']*other
            return new_spectrum

        elif isinstance(other, np.ndarray):

            return Spectrum(
                self.eng, self._data*other, 
                rs = self.rs, in_eng = self.in_eng, 
                spec_type = self._spec_type
            )

        elif isinstance(other, Spectrum):

            fin_spec_type = self._spec_type
            if self._spec_type != other._spec_type:
                # If they are not the same, defaults to dNdE.
                fin_spec_type = 'dNdE'

            new_rs = -1
            new_in_eng = -1
            if self.rs == other.rs:
                new_rs = self.rs 
            if self.in_eng == other.in_eng:
                new_in_eng = self.in_eng 
            if not np.array_equal(self.eng, other.eng):
                raise TypeError("energy abscissae are not the same.")
            return Spectrum(
                self.eng, self._data*other._data, 
                rs = new_rs, in_eng = new_in_eng,
                spec_type = fin_spec_type
            )

        else:

            raise TypeError("cannot multiply object to Spectrum.")

    def __rmul__(self,other):
        """Takes the product of the spectrum with an array or number. `Spectrum` object is on the right.

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the multiplied `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__mul__`, allows the use of the symbol * to multiply `Spectrum` objects or an array and Spectrum.

        The returned `Spectrum` object `underflow` is reset to zero if `other` is not a `Spectrum` object.

        See Also
        --------
        spectrum.Spectrum.__mul__

        """
        if (np.issubdtype(type(other),np.float64) 
            or np.issubdtype(type(other),np.int64)
        ):
            new_spectrum = Spectrum(
                self.eng, self._data*other, 
                rs = self.rs, in_eng = self.in_eng,
                spec_type = self._spec_type 
            )
            new_spectrum.underflow['N'] = self.underflow['N']*other
            new_spectrum.underflow['eng'] = self.underflow['eng']*other
            return new_spectrum

        # Multiplication by Spectrum covered by __mul__

        elif isinstance(other, np.ndarray):

            return Spectrum(
                self.eng, self._data*other, 
                self.rs, self.in_eng,
                spec_type = self._spec_type
            )

        else:

            raise TypeError("cannot multiply object with Spectrum.")
    
    def __truediv__(self,other):
        """Divides the spectrum by an array or number.

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the divided `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__truediv__`, allows the use of the symbol / to multiply `Spectrum` objects or an array and Spectrum.

        The returned `Spectrum` object `underflow` is reset to zero.

        """
        return self*(1/other)

    def __rtruediv__(self,other):
        """Divides a number or array by the spectrum.

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the divided `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__truediv__`, allows the use of the symbol / to multiply `Spectrum` objects or an array and Spectrum.

        The returned `Spectrum` object `underflow` is reset to zero.

        """
        invSpec = Spectrum(self.eng, 1/self._data, self.rs, self.in_eng)
        return other*invSpec

    def switch_spec_type(self):
        """Switches between the type of values to be stored.

        Although both N and dN/dE can be accessed regardless of which values
        are stored, performing a switch before repeated computations can
        speed up the computation. 

        """
        log_bin_width = get_log_bin_width(self.eng)
        if self._spec_type == 'N':
            self._data = self._data/(self.eng*log_bin_width) 
            self._spec_type = 'dNdE'
        elif self._spec_type == 'dNdE':
            self._data = self._data*self.eng*log_bin_width 
            self._spec_type = 'N'

    def contract(self, mat):
        """Performs a dot product on the spectrum with `mat`.

        Parameters
        ----------
        mat : ndarray
            The array to dot into the spectrum with.

        Returns
        -------
        float
            The resulting dot product.

        """
        return np.dot(mat,self._data)

    def totN(self, bound_type=None, bound_arr=None):
        """Returns the total number of particles in part of the spectrum. 

        The part of the spectrum can be specified in two ways, and is specified by `bound_type`. Multiple totals can be obtained through `bound_arr`. 
        
        Parameters
        ----------
        bound_type : {'bin', 'eng', None}
            The type of bounds to use. Bound values do not have to be within the [0:length] for `'bin'` or within the abscissa for `'eng'`. `None` should only be used when computing the total particle number in the spectrum. For `'bin'`, bounds are specified as the bin boundary, with 0 being the left most boundary, 1 the right-hand of the first bin and so on. This is equivalent to integrating over a histogram. For `'eng'`, bounds are specified by energy values.

        bound_arr : ndarray, optional
            An array of boundaries (bin or energy), between which the total number of particles will be computed. If bound_arr = None, but bound_type is specified, the total number of particles in each bin is computed. If both bound_type and bound_arr = None, then the total number of particles in the spectrum is computed.

        Returns
        -------
        ndarray or float
            Total number of particles in the spectrum. 
        """

        length = self.length
        log_bin_width = get_log_bin_width(self.eng)
        if self._spec_type == 'dNdE':
            dNdlogE = self.eng*self.dNdE 
        elif self._spec_type == 'N':
            dNdlogE = self.N/log_bin_width 

        if bound_type is not None:

            if bound_arr is None:

                return dNdlogE * log_bin_width 

            if bound_type == 'bin':

                if not all(np.diff(bound_arr) >= 0):
                    raise TypeError("bound_arr must have increasing entries.")

                N_in_bin = np.zeros(bound_arr.size-1)

                if bound_arr[0] > length or bound_arr[-1] < 0:
                    return N_in_bin

                for low,upp,i in zip(bound_arr[:-1], bound_arr[1:], 
                    np.arange(N_in_bin.size)):
                    # Set the lower and upper bounds, including case where low and upp are outside of the bins.

                    if low > length or upp < 0:
                        N_in_bin[i] = 0
                        continue

                    low_ceil  = int(np.ceil(low))
                    low_floor = int(np.floor(low))
                    upp_ceil  = int(np.ceil(upp))
                    upp_floor = int(np.floor(upp))
                    # Sum the bins that are completely between the bounds.
                    N_full_bins = np.dot(
                        dNdlogE[low_ceil:upp_floor],
                        log_bin_width[low_ceil:upp_floor]
                    )

                    N_part_bins = 0

                    if low_floor == upp_floor or low_ceil == upp_ceil:
                        # Bin indices are within the same bin. The second requirement covers the case where upp_ceil is length. 
                        N_part_bins += (
                            dNdlogE[low_floor] * (upp - low)
                            * log_bin_width[low_floor]
                        )
                    else:
                        # Add up part of the bin for the low partial bin and the high partial bin. 
                        N_part_bins += (
                            dNdlogE[low_floor] * (low_ceil - low)
                            * log_bin_width[low_floor]
                        )
                        if upp_floor < length:
                        # If upp_floor is length, then there is no partial bin for the upper index. 
                            N_part_bins += (
                                dNdlogE[upp_floor]
                                * (upp-upp_floor) * log_bin_width[upp_floor]
                            )

                    N_in_bin[i] = N_full_bins + N_part_bins

                return N_in_bin

            if bound_type == 'eng':
                bin_boundary = get_bin_bound(self.eng)
                eng_bin_ind = np.interp( 
                    np.log(bound_arr), 
                    np.log(bin_boundary), np.arange(bin_boundary.size), 
                    left = -1, right = length + 1
                )

                return self.totN('bin', eng_bin_ind)

        else:
            return np.dot(dNdlogE,log_bin_width) + self.underflow['N']

    def toteng(self, bound_type=None, bound_arr=None):
        """Returns the total energy of particles in part of the spectrum. 

        The part of the spectrum can be specified in two ways, and is specified by `bound_type`. Multiple totals can be obtained through `bound_arr`. 

        Parameters
        ----------
        bound_type : {'bin', 'eng', None}
            The type of bounds to use. Bound values do not have to be within the [0:length] for `'bin'` or within the abscissa for `'eng'`. `None` should only be used to obtain the total energy. With `'bin'`, bounds are specified as the bin boundary, with 0 being the left most boundary, 1 the right-hand of the first bin and so on. This is equivalent to integrating over a histogram. With `'eng'`, bounds are specified by energy values. 

        bound_arr : ndarray, optional
            An array of boundaries (bin or energy), between which the total number of particles will be computed. If unspecified, the total number of particles in the whole spectrum is computed.


        Returns
        -------
        ndarray or float
            Total energy in the spectrum. 
        """
        eng = self.eng
        length = self.length
        log_bin_width = get_log_bin_width(self.eng)

        if self._spec_type == 'dNdE':
            dNdlogE = self.eng*self.dNdE
        elif self._spec_type == 'N':
            dNdlogE = self.N/log_bin_width 

        if bound_type is not None:

            if bound_arr is None:

                return dNdlogE * eng * log_bin_width 

            if bound_type == 'bin':

                if not all(np.diff(bound_arr) >= 0):
                    
                    raise TypeError("bound_arr must have increasing entries.")

                eng_in_bin = np.zeros(bound_arr.size-1)

                if bound_arr[0] > length or bound_arr[-1] < 0:
                    return eng_in_bin

                for low,upp,i in zip(bound_arr[:-1], bound_arr[1:], 
                    np.arange(eng_in_bin.size)):
                    
                    if low > length or upp < 0:
                        eng_in_bin[i] = 0
                        continue

                    low_ceil  = int(np.ceil(low))
                    low_floor = int(np.floor(low))
                    upp_ceil  = int(np.ceil(upp))
                    upp_floor = int(np.floor(upp))
                    # Sum the bins that are completely between the bounds.
                    eng_full_bins = np.dot(eng[low_ceil:upp_floor]
                        * dNdlogE[low_ceil:upp_floor],
                        log_bin_width[low_ceil:upp_floor])

                    eng_part_bins = 0

                    if low_floor == upp_floor or low_ceil == upp_ceil:
                        # Bin indices are within the same bin. The second requirement covers the case where upp_ceil is length. 
                        eng_part_bins += (eng[low_floor] * dNdlogE[low_floor]
                            * (upp - low) * log_bin_width[low_floor])
                    else:
                        # Add up part of the bin for the low partial bin and the high partial bin. 
                        eng_part_bins += (eng[low_floor] * dNdlogE[low_floor]
                            * (low_ceil - low) * log_bin_width[low_floor])
                        if upp_floor < length:
                        # If upp_floor is length, then there is no partial bin for the upper index. 
                            eng_part_bins += (eng[upp_floor]
                                * dNdlogE[upp_floor] * (upp-upp_floor) 
                                * log_bin_width[upp_floor])

                    eng_in_bin[i] = eng_full_bins + eng_part_bins

                return eng_in_bin

            if bound_type == 'eng':
                bin_boundary = get_bin_bound(self.eng)
                eng_bin_ind = np.interp( 
                    np.log(bound_arr), 
                    np.log(bin_boundary), np.arange(bin_boundary.size), 
                    left = -1, right = length + 1)

                return self.toteng('bin', eng_bin_ind)

        else:
            return (np.dot(dNdlogE, eng * log_bin_width) 
                + self.underflow['eng'])

    def shift_eng(self, new_eng):
        """ Shifts the abscissa while conserving number. 

        This function can be used to subtract or add some amount of energy from each bin in the spectrum. The dN/dE is adjusted to conserve number in each bin. 

        Parameters
        ----------
        new_eng : ndarray
            The new energy abscissa.
        """
        if new_eng.size != self.eng.size:
            return TypeError("The new abscissa must have the same length as the old one.")
        if not all(np.diff(new_eng) > 0):
            raise TypeError("abscissa must be ordered in increasing energy.")
        
        new_log_bin_width = get_log_bin_width(new_eng)
        
        if self._spec_type == 'dNdE':
            new_dNdE = self.totN('bin')/(new_eng * new_log_bin_width)
            self.eng = new_eng
            self._data = new_dNdE
        elif self._spec_type == 'N':
            self.eng = new_eng

    def rebin(self, out_eng):
        """ Re-bins the `Spectrum` object according to a new abscissa.

        Rebinning conserves total number and total energy.
        
        Parameters
        ----------
        out_eng : ndarray
            The new abscissa to bin into. If `self.eng` has values that are smaller than `out_eng[0]`, then the new underflow will be filled. If `self.eng` has values that exceed `out_eng[-1]`, then an error is returned.

        Raises
        ------
        OverflowError
            The maximum energy in `out_eng` cannot be smaller than any bin in `self.eng`. 

        
        Note
        ----
        The total number and total energy is conserved by assigning the number of particles N in a bin of energy eng to two adjacent bins in new_eng, with energies eng_low and eng_upp such that eng_low < eng < eng_upp. Then dN_low_dE_low = (eng_upp - eng)/(eng_upp - eng_low)*(N/(E * dlogE_low)), and dN_upp_dE_upp = (eng - eng_low)/(eng_upp - eng_low)*(N/(E*dlogE_upp)).

        If a bin in `self.eng` is below the lowest bin in `out_eng`, then the total number and energy not assigned to the lowest bin are assigned to the underflow. Particles will only be assigned to the lowest bin if there is some overlap between the bin index with respect to `out_eng` bin centers is larger than -1.0.

        If a bin in `self.eng` is above the highest bin in `out_eng`, then an `OverflowError` is thrown. 

        See Also
        --------
        spec.spectools.rebin_N_arr

        """
        
        if not np.all(np.diff(out_eng) > 0):
            raise TypeError("new abscissa must be ordered in increasing energy.")
        # if out_eng[-1] < self.eng[-1]:
        #     raise OverflowError("the new abscissa lies below the old one: this function cannot handle overflow (yet?).")
        # Get the bin indices that the current abscissa (self.eng) corresponds to in the new abscissa (new_eng). Can be any number between 0 and self.length-1. Bin indices are wrt the bin centers.

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
            # raise OverflowError("the new abscissa lies below the old one: this function cannot handle overflow (yet?).")

        # Get the total N and toteng in each bin of self._data
        if self._spec_type == 'dNdE':
            N_arr = self.totN('bin')
            toteng_arr = self.toteng('bin')
        elif self._spec_type == 'N':
            N_arr = self.N
            toteng_arr = self.N*self.eng 

        N_arr_low = N_arr[ind_low]
        N_arr_high = N_arr[ind_high]
        N_arr_reg = N_arr[ind_reg]

        toteng_arr_low = toteng_arr[ind_low]

        # Bin width of the new array. Use only the log bin width, so that dN/dE = N/(E d log E)
        if self._spec_type == 'dNdE':
            new_E_dlogE = new_eng * get_log_bin_width(new_eng)

        # Regular bins first, done in a completely vectorized fashion. 

        # reg_bin_low is the array of the lower bins to be allocated the particles in N_arr_reg, similarly reg_bin_upp. This should also take care of the fact that bin_ind is an integer.
        reg_bin_low = np.floor(bin_ind[ind_reg]).astype(int)
        reg_bin_upp = reg_bin_low + 1

        # Takes care of the case where eng[-1] = new_eng[-1]
        reg_bin_low[reg_bin_low == new_eng.size-2] = new_eng.size - 3
        reg_bin_upp[reg_bin_upp == new_eng.size-1] = new_eng.size - 2

        if self._spec_type == 'dNdE':
            reg_dNdE_low = (
                (reg_bin_upp - bin_ind[ind_reg]) * N_arr_reg
                /new_E_dlogE[reg_bin_low+1]
            )
            reg_dNdE_upp = (
                (bin_ind[ind_reg] - reg_bin_low) * N_arr_reg
                           /new_E_dlogE[reg_bin_upp+1]
            )
        elif self._spec_type == 'N':
            reg_N_low = (reg_bin_upp - bin_ind[ind_reg]) * N_arr_reg
            reg_N_upp = (bin_ind[ind_reg] - reg_bin_low) * N_arr_reg

        # Low bins. 
        low_bin_low = np.floor(bin_ind[ind_low]).astype(int) 
        N_above_underflow = np.sum((bin_ind[ind_low] - low_bin_low) 
            * N_arr_low)
        eng_above_underflow = N_above_underflow * new_eng[1]

        N_underflow = np.sum(N_arr_low) - N_above_underflow
        eng_underflow = np.sum(toteng_arr_low) - eng_above_underflow
        if self._spec_type == 'dNdE':
            low_dNdE = N_above_underflow/new_E_dlogE[1]

        # Add up, obtain the new data.
        new_data = np.zeros(new_eng.size)
        if self._spec_type == 'dNdE':
            new_data[1] += low_dNdE
            # reg_dNdE_low = -1 refers to new_eng[0]
            np.add.at(new_data, reg_bin_low+1, reg_dNdE_low)
            np.add.at(new_data, reg_bin_upp+1, reg_dNdE_upp)
            # print(new_data[reg_bin_low+1])
            # new_data[reg_bin_low+1] += reg_dNdE_low
            # new_data[reg_bin_upp+1] += reg_dNdE_upp
        elif self._spec_type == 'N':
            new_data[1] += N_above_underflow
            np.add.at(new_data, reg_bin_low+1, reg_N_low)
            np.add.at(new_data, reg_bin_upp+1, reg_N_upp)
            # new_data[reg_bin_low+1] += reg_N_low
            # new_data[reg_bin_upp+1] += reg_N_upp 
        
        # Implement changes.
        self.eng = new_eng[1:]
        self._data = new_data[1:]
        self.length = self.eng.size 
        self.underflow['N'] += N_underflow
        self.underflow['eng'] += eng_underflow

    def rebin_smooth(self, out_eng):
        """ Re-bins the `Spectrum` object according to a new abscissa.

        Rebinning conserves total number and total energy. This method also ensures smoothness of the rebinning.
        
        Parameters
        ----------
        out_eng : ndarray
            The new abscissa to bin into. If `self.eng` has values that are smaller than `out_eng[0]`, then the new underflow will be filled. If `self.eng` has values that exceed `out_eng[-1]`, then an error is returned.

        Raises
        ------
        OverflowError
            The maximum energy in `out_eng` cannot be smaller than any bin in `self.eng`. 

        
        Note
        ----
        Before passing the spectrum to the actual rebin function, we use spec.Spectrum.totN to find the total number of particles `N` lying in the new binning given by `out_eng`, and spec.Spectrum.toteng to find the total energy in these bins `E_tot`. Then we minimize a Lagrangian that penalizes deviations from local conservation of number and energy, while conserving total number and energy.  

        See Also
        --------
        spec.Spectrum.rebin

        """

        eng_bin_bound = get_bin_bound(out_eng)
        
        N_tot_arr   = self.totN(bound_type='eng', bound_arr=eng_bin_bound)
        eng_tot_arr = self.toteng(bound_type='eng', bound_arr=eng_bin_bound)

        out_eng_absc = np.divide(
            eng_tot_arr, N_tot_arr, 
            out=np.zeros_like(N_tot_arr), 
            where=N_tot_arr != 0
        )

        out_eng_absc[out_eng_absc == 0] = out_eng[out_eng_absc == 0]

        self.eng = out_eng_absc
        self.length = out_eng_absc.size
        self.underflow = {'N': 0., 'eng': 0.}
        
        switch = False
        if self.spec_type == 'dNdE':
            switch = True
        
        self._data = N_tot_arr
        self._spec_type = 'N'
        if switch:
            self.switch_spec_type()

        self.rebin(out_eng)
         
        


    def engloss_rebin(self, in_eng, out_eng):
        """ Converts an energy loss spectrum to a secondary spectrum.

        Parameters
        ----------
        in_eng : float
            The injection energy of the primary which gives rise to self.dNdE as the energy loss spectrum. 
        out_eng : ndarray
            The final energy abscissa to bin into. If not specified, it is assumed to be the same as the initial abscissa.

        Note
        ----
        This is primarily useful only when the spectrum represents an energy loss spectrum, i.e. when `self.eng` represents some *loss* in energy Delta. The loss spectrum can be directly converted into a secondary spectrum by using dN/(d Delta)* (d Delta) = dN/dE*dE, where the LHS is evaluated at Delta, and the RHS is evaluated at in_eng - Delta. 

        """ 

        # sec_spec_eng is the injected energy - delta, 
        sec_spec_eng = np.flipud(in_eng - self.eng)
        if self._spec_type == 'dNdE':
            N_arr = np.flipud(self.totN('bin'))
        elif self._spec_type == 'N':
            N_arr = self.N 

        # consider only positive energy
        pos_eng = sec_spec_eng > 0

        new_spec = rebin_N_arr(
            N_arr[pos_eng], sec_spec_eng[pos_eng], 
            out_eng, spec_type = self._spec_type
        )

        self.eng  = out_eng 
        self._data = new_spec._data
        self.length = out_eng.size
        self.underflow['N'] = new_spec.underflow['N']
        self.underflow['eng'] = new_spec.underflow['eng']

    def at_eng(self, new_eng, left=-200, right=-200):
        """Interpolates the spectrum at a new abscissa. 

        Interpolation is logarithmic. 

        Parameters
        ----------
        new_eng : ndarray
            The new energies to interpolate at. 
        left : float, optional
            Returns the value if beyond the first bin on the left. Default is to return -200, so that the exponential is small. 
        right : float, optional
            Returns the value if beyond the last bin on the right. Default is to return -200, so that the exponential is small.
        """

        self._data[self._data <= 1e-200] = 1e-200
        
        log_new_data = np.interp(
            np.log(new_eng), np.log(self.eng), np.log(self._data), 
            left=left, right=right
        )

        self.eng = new_eng
        self._data = np.exp(log_new_data)
        self._data[self._data <= 1e-200] = 0

    def redshift(self, new_rs):
        """Redshifts the `Spectrum` object as a photon spectrum. 

        Parameters
        ----------
        new_rs : float
            The new redshift (1+z) to redshift to.

        """

        fac = new_rs/self.rs
        
        eng_orig = self.eng

        self.eng = self.eng*fac
        if self._spec_type == 'dNdE':
            self.dNdE = self.dNdE/fac
        self.underflow['eng'] *= fac
        
        self.rebin(eng_orig)
        self.rs = new_rs
