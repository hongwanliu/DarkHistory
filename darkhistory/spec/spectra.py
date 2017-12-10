"""Contains the `Spectra` class."""

import numpy as np
from darkhistory import utilities as utils
from darkhistory.spec.spectools import get_log_bin_width
from darkhistory.spec.spectrum import Spectrum

import matplotlib.pyplot as plt
import warnings

from scipy import interpolate

class Spectra:
    """Structure for a collection of `Spectrum` objects. 

    Parameters
    ----------
    spec_arr : list of Spectrum
        List of `Spectrum` to be stored together. 
    rebin_eng : ndarray, optional
        New abscissa to rebin all of the spectra into. 
    rebin_type : {'1D', '2D'}, optional
        Whether to rebin each spectrum separately (`'1D'`), or the whole `Spectra` object at once (`'2D'`). 1D rebinning is only allowed when `spec_arr` is a list of `Spectrum`. Default is `'2D'`. 

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
    """

    # __array_priority__ must be larger than 0, so that radd can work.
    # Otherwise, ndarray + Spectrum works by iterating over the elements of
    # ndarray first, which isn't what we want.

    __array_priority__ = 1

    def __init__(self, spec_arr, rebin_eng=None, rebin_type='2D'):

        if spec_arr != []:

            if len(set([spec.spec_type for spec in spec_arr])) != 1:
                raise TypeError(
                    "all Spectrum must have spec_type 'N' or 'dNdE'."
                )

            _grid_vals = np.stack([spec._data for spec in spec_arr])
            _spec_type = spec_arr[0].spec_type

            if rebin_eng is not None:
                self.rebin(rebin_eng, rebin_type)

            if not utils.arrays_equal([spec.eng for spec in spec_arr]):
                raise TypeError("all abscissae must be the same.")

            _eng = spec_arr[0].eng
            _in_eng = np.array([spec.in_eng for spec in spec_arr])
            _rs = np.array([spec.rs for spec in spec_arr])
            _N_underflow = np.array(
                [spec.underflow['N'] for spec in spec_arr]
            )
            _eng_underflow = np.array(
                [spec.underflow['eng'] for spec in spec_arr]
            )

    @property
    def eng(self):
        return self._eng

    @eng.setter
    def eng(self, value):
        if value.size == _grid_vals.shape[1]:
            self._eng = value
        else:
            raise TypeError('not compatible with the grid.')

    @property
    def in_eng(self):
        return self._in_eng

    @in_eng.setter
    def in_eng(self, value):
        if value.size == _grid_vals.shape[0]:
            self._in_eng = value
        else:
            raise TypeError('not compatible with the grid.')

    @property
    def rs(self):
        return self._rs

    @rs.setter
    def rs(self, value):
        if value.size == _grid_vals.shape[0]:
            self.rs = value
        else:
            raise TypeError('not compatible with the grid.')

    @property
    def spec_type(self):
        return self._spec_type

    @property
    def N_underflow(self):
        return self._N_underflow

    @property
    def eng_underflow(self):
        return self._eng_underflow

    def __iter__(self):
        return iter(self.spec_arr)

    def __getitem__(self, key):
        if np.issubdtype(type(key), int):
            out_spec = Spectrum(
                self.eng, self.spec_arr[key], self.in_eng[key], self.rs[key]
            )
            out_spec.underflow['N']   = self.N_underflow[key]
            out_spec.underflow['eng'] = self.eng_underflow[key]
            return out_spec
        elif isinstance(key, slice):
            data_arr          = self._grid_vals[key]
            in_eng_arr        = self.in_eng[key]
            rs_arr            = self.rs[key]
            N_underflow_arr   = self.N_underflow[key]
            eng_underflow_arr = self.eng_underflow[key]
            out_spec_list = [
                Spectrum(self.eng, data, in_eng, rs) for (spec, in_eng, rs) 
                    in zip(data_arr, in_eng_arr, rs_arr)
            ]
            for (spec,N,eng) in zip(
                out_spec_list, N_underflow_arr, eng_underflow_arr
            ):
                spec.underflow['N'] = N
                spec.underflow['eng'] = eng
        else:
            raise TypeError("indexing is invalid.")

    def __setitem__(self, key, value):
        if np.issubdtype(type(key), int):
            if value.eng != self.eng:
                    raise TypeError("the energy abscissa of the new Spectrum does not agree with this Spectra.")
            self._in_eng[key] = value.in_eng
            self._rs[key] = value.rs
            if self.spec_type == 'N':
                self._grid_vals[key] = value.N
            elif self.spec_type == 'dNdE':
                self._grid_vals[key] = value.dNdE
            self._N_underflow[key] = value.underflow['N']
            self._eng_underflow[key] = value.underflow['eng']
        elif isinstance(key, slice):
            for i,spec in zip(key, value):
                if value.eng != self.eng:
                    raise TypeError("the energy abscissa of the new Spectrum does not agree with this Spectra.")
                self._in_eng[i] = spec.in_eng
                self._rs[i] = spec.rs
                if self.spec_type == 'N':
                    self._grid_vals[i] = spec.N
                elif self.spec_type == 'dNdE':
                    self._grid_vals[i] = spec.dNdE
                self._N_underflow[i] = spec.underflow['N']
                self._eng_underflow[i] = spec.underflow['eng']

    def __add__(self, other):
        """Adds two arrays of spectra together.

        Parameters
        ----------
        other : Spectra or ndarray

        Returns
        -------
        Spectra
            New `Spectra` instance which is an element-wise sum of the `Spectrum` objects in each Spectra.

        Notes
        -----
        This special function, together with `Spectra.__radd__`, allows the use of the symbol + to add arrays of spectra together. 

        See Also
        --------
        spectra.Spectra.__radd__
        """
        if np.issubclass_(type(other), Spectra):

            if not np.array_equal(self.eng, other.eng):
                raise TypeError('abscissae are different for the two spectra.')

            if self.spec_type != other.spec_type:
                raise TypeError('adding spectra of N to spectra of dN/dE.')

            out_spectra = Spectra([])
            out_spectra.spec_type = self.spec_type
            out_spectra._grid_vals = self._grid_vals + other._grid_vals
            out_spectra._eng = self.eng 
            if np.array_equal(self.in_eng, other.in_eng):
                out_spectra._in_eng = self.in_eng
            if np.array_equal(self.rs, other.rs):
                out_spectra.rs = self.rs

            return out_spectra

        elif isinstance(other, np.ndarray):

            self._grid_vals += other

        else:
            raise TypeError('adding an object that is not compatible.')

    def __radd__(self, other):
        """Adds two arrays of spectra together.

        Parameters
        ----------
        other : Spectra

        Returns
        -------
        Spectra
            New `Spectra` instance which is an element-wise sum of the `Spectrum` objects in each Spectra.

        Notes
        -----
        This special function, together with `Spectra.__add__`, allows the use of the symbol + to add two arrays of spectra together. 

        See Also
        --------
        spectra.Spectra.__add__
        """
        if npissubclass_(type(other), Spectra):

            if not np.array_equal(self.eng, other.eng):
                raise TypeError('abscissae are different from the two spectra.')

            if self.spec_type != other.spec_type:
                raise TypeError('adding spectra of N to spectra of dN/dE.')

            out_spectra = Spectra([])
            out_spectra.spec_type = self.spec_type
            out_spectra._grid_vals = self._grid_vals + other._grid_vals
            out_spectra.eng = self.eng 
            if np.array_equal(self.in_eng, other.in_eng):
                out_spectra.in_eng = self.in_eng
            if np.array_equal(self.rs, other.rs):
                out_spectra.rs = self.rs

            return out_spectra

        elif isinstance(other, np.ndarray):

            self._grid_vals += other

        else:
            raise TypeError('adding an object that is not compatible.')

    def __sub__(self, other):
        """Subtracts one array of spectra from another. 

        Parameters
        ----------
        other : Spectra or ndarray

        Returns
        -------
        Spectra

        Notes
        -----
        This special function, together with `Spectra.__rsub__`, allows the use of the symbol - to subtract or subtract from `Spectra` objects.

        See Also
        --------
        spectrum.Spectra.__rsub__
        """

        return self + -1*other

    def __rsub__(self, other):
        """Subtracts one array of spectra from another. 

        Parameters
        ----------
        other : Spectra or ndarray

        Returns
        -------
        Spectra

        Notes
        -----
        This special function, together with `Spectra.__rsub__`, allows the use of the symbol - to subtract or subtract from `Spectra` objects.

        See Also
        --------
        spectrum.Spectra.__sub__
        """

        return other + -1*self

    def __neg__(self):
        """Negates the spectra values. 

        Returns
        -------
        Spectra
        """

        return -1*self

    def__mul__(self, other):
        """Takes a product with this `Spectra`. 

        Parameters
        ----------
        other : Spectra, int, float, list or ndarray

        Returns
        -------
        Spectra

        Notes
        -----
        This special function, together with `Spectra.__rmul__`, allows the use of the symbol * to multiply objects with a `Spectra` object. 
        """

        if (
            np.issubdtype(type(other), float) 
            or np.issubdtype(type(other, int))
            or isinstance(other, list)
            or isinstance(other, ndarray)
        ):
            out_spectra = Spectra([])
            out_spectra.eng = self.eng
            out_spectra.in_eng = self.in_eng
            out_spectra.rs = self.rs
            out_spectra._grid_vals = self._grid_vals*other
            
            return out_spectra

        elif np.issubclass_(type(other), Spectra):

            if self.eng != other.eng:
                raise TypeError('the two spectra do not have the same abscissa.')

            out_spectra = Spectra([])
            out_spectra.eng = self.eng
            if np.array_equal(self.in_eng, other.in_eng):
                out_spectra.in_eng = self.in_eng
            if np.array_equal(self.rs, other.rs):
                out_spectra.rs = self.rs
            out_spectra._grid_vals = self._grid_vals * other._grid_vals

            return out_spectra

    def__rmul__(self, other):
        """Takes a product with this `Spectra`. 

        Parameters
        ----------
        other : Spectra, int, float, list or ndarray

        Returns
        -------
        Spectra

        Notes
        -----
        This special function, together with `Spectra.__mul__`, allows the use of the symbol * to multiply objects with a `Spectra` object. 
        """

        if (
            np.issubdtype(type(other), float) 
            or np.issubdtype(type(other, int))
            or isinstance(other, list)
            or isinstance(other, ndarray)
        ):
            out_spectra = Spectra([])
            out_spectra.eng = self.eng
            out_spectra.in_eng = self.in_eng
            out_spectra.rs = self.rs
            out_spectra._grid_vals = self._grid_vals*other
            
            return out_spectra

        elif np.issubclass_(type(other), Spectra):

            if self.eng != other.eng:
                raise TypeError('the two spectra do not have the same abscissa.')

            out_spectra = Spectra([])
            out_spectra.eng = self.eng
            if np.array_equal(self.in_eng, other.in_eng):
                out_spectra.in_eng = self.in_eng
            if np.array_equal(self.rs, other.rs):
                out_spectra.rs = self.rs
            out_spectra._grid_vals = self._grid_vals * other._grid_vals

            return out_spectra

    def __truediv__(self, other):
        """Divides Spectra by another object. 

        Parameters
        ----------
        other : ndarray, float, int, list or Spectra

        Returns
        -------
        Spectra

        Notes
        -----
        This special function, together with `Spectra.__rtruediv__`, allows the use fo the symbol / to divide `Spectra` objects. 

        See Also
        --------
        spectrum.Spectra.__rtruediv__
        """
        if np.issubclass_(type(other), Spectra):
            inv_spectra = Spectra([])
            inv_spectra.eng = other.eng
            inv_spectra.in_eng = other.in_eng
            inv_spectra._grid_vals = 1/other._grid_vals
            inv_spectra.rs = other.rs
            return self * inv_spectra
        else:
            return self * (1/other)

    def __rtruediv__(self, other):
        """Divides Spectra by another object. 

        Parameters
        ----------
        other : ndarray, float, int, list or Spectra

        Returns
        -------
        Spectra

        Notes
        -----
        This special function, together with `Spectra.__rtruediv__`, allows the use fo the symbol / to divide `Spectra` objects. 

        See Also
        --------
        spectrum.Spectra.__truediv__
        """
        inv_spectra = Spectra([])
        inv_spectra.eng = self.eng
        inv_spectra.in_eng = self.in_eng
        inv_spectra._grid_vals = 1/self._grid_vals
        inv_spectra.rs = self.rs

        return other * inv_spectra

    def totN(self, bound_type=None, bound_arr=None):
        """Returns the total number of particles in part of the spectra.

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
        log_bin_width = get_log_bin_width(self.eng)

        # Using the broadcasting rules here. 
        if self.spec_type == 'dNdE':
            dNdlogE = self._grid_vals * eng
        elif self.spec_type == 'N':
            dNdlogE = self._grid_vals/log_bin_width

        if bound_type is not None:

            if bound_arr is None:

                return dNdlogE * log_bin_width

            if bound_type == 'bin':

                if not all(np.diff(bound_arr) >= 0):
                    raise TypeError('bound_arr must have increasing entries.')

                # Size is number of totals requested x number of Spectrums.
                N_in_bin = np.zeros((bound_arr.size - 1, self.in_eng.size))

                if bound_arr[0] > self.eng.size or bound_arr[-1] < 0:
                    return N_in_bin

                for i, (low,upp) in enumerate(
                    (bound_arr[:-1], bound_arr[1:])
                ):
                    # Set the lower and upper bounds, including case where
                    # low and upp are outside fo the bins. 
                    if low > self.eng.size or upp < 0:
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

                    N_part_bins = np.zeros_like(self.in_eng)

                    if low_floor == upp_floor or low_ceil == upp_ceil:
                        # Bin indices are within the same bin. 
                        # The second requirement covers the case where
                        # upp_ceil is eng.size. 
                        N_part_bins += (
                            dNdlogE[:,low_floor] * (upp - low)
                            * log_bin_width[low_floor]
                        )
                    else:
                        # Add up part of the bin for the low partial bin
                        # and the high partial bin. 
                        N_part_bins += (
                            dNdlogE[:,low_floor] * (low_ceil - low)
                            * log_bin_width[low_floor]
                        )
                        if upp_floor < self.eng.size:
                            # If upp_floor is eng.size then there is
                            # no partial bin for the upper index. 
                            N_part_bins += (
                                dNdlogE[:,upp_floor] * (upp - upp_floor)
                                * log_bin_width[upp_floor]
                            )

                    N_in_bin[i] = N_full_bins + N_part_bins

                return N_in_bin

            if bound_type == 'eng':
                bin_boundary = get_bin_bound(self.eng)
                eng_bin_ind = np.interp(
                    np.log(bound_arr),
                    np.log(bin_boundary), np.arange(bin_boundary.size),
                    left = -1, right = self.eng.size + 1
                )

                return self.totN('bin', eng_bin_ind)

            else:
                return (
                    np.dot(dNdlogE, log_bin_width) + np.sum(self.N_underflow)
                )







































