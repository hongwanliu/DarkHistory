"""``spectra`` contains the ``Spectra`` class."""

import numpy as np
from darkhistory import utilities as utils
from darkhistory.spec.spectools import get_bin_bound
from darkhistory.spec.spectrum import Spectrum
import matplotlib.pyplot as plt
import warnings

from scipy import integrate

class Spectra:
    """Structure for a collection of ``Spectrum`` objects.

    Parameters
    ---------- 
    spec_arr : list of ``Spectrum``
        List of ``Spectrum`` to be stored together.
    rebin_eng : ndarray, optional
        New abscissa to rebin all of the ``Spectrum`` objects into.

    Attributes
    ----------
    eng : ndarray
        Energy abscissa for the ``Spectrum``.
    rs : ndarray
        The redshifts of the ``Spectrum`` objects.  
    grid_values : ndarray
        2D array with the spectra laid out in (rs, eng). 
    

    """
    # __array_priority__ must be larger than 0, so that radd can work.
    # Otherwise, ndarray + Spectrum works by iterating over the elements of
    # ndarray first, which isn't what we want.
    __array_priority__ = 1

    def __init__(self, spec_arr, rebin_eng=None):

        if len(set(spec.length for spec in spec_arr)) > 1:
            raise TypeError("all spectra must have the same length.")

        if not np.all(np.diff(spec_arr[0].eng) > 0):
            raise TypeError("abscissa must be ordered in increasing energy.")

        self.spec_arr = spec_arr
        
        if rebin_eng is not None:
            self.rebin(rebin_eng)

        if not utils.arrays_equal([spec.eng for spec in spec_arr]):
            raise TypeError("all abscissae must be the same.")

        self.eng = spec_arr[0].eng
        
        self.rs = np.array([spec.rs for spec in spec_arr])

        if self.rs.size > 1 and not np.all(np.diff(self.rs) <= 0):
            raise TypeError("redshift must be in increasing order.")

        # self.log_bin_width = np.diff(np.log(self.bin_boundary))

        self.grid_values = np.stack([spec.dNdE for spec in self.spec_arr])


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


    def __add__(self, other): 
        """Adds two ``Spectra`` instances together.

        Parameters
        ----------
        other : Spectra

        Returns
        -------
        Spectra
            New ``Spectra`` instance which is an element-wise sum of the ``Spectrum`` objects in each ``Spectra``.

        Notes
        -----
        This special function, together with `Spectra.__radd__`, allows the use of the symbol + to add ``Spectra`` objects together. 

        See Also
        --------
        spectrum.Spectra.__radd__
        """
        if np.issubclass_(type(other), Spectra):

            if not util.array_equal(self.eng, other.eng):
                raise TypeError('abscissae are different for the two Spectra.')
            if not util.array_equal(self.rs, other.rs):
                raise TypeError('redshifts are different for the two Spectra.')

            return Spectra([spec1 + spec2 for spec1,spec2 in zip(self.spec_arr, other.spec_arr)])

        else: raise TypeError('adding an object that is not of class Spectra.')


    def __radd__(self, other): 
        """Adds two ``Spectra`` instances together.

        Parameters
        ----------
        other : Spectra

        Returns
        -------
        Spectra
            New ``Spectra`` instance which is an element-wise sum of the ``Spectrum`` objects in each ``Spectra``.

        Notes
        -----
        This special function, together with `Spectra.__add__`, allows the use of the symbol + to add `Spectra` objects together. 

        See Also
        --------
        spectrum.Spectra.__add__
        """
        if np.issubclass_(type(other), Spectra):

            if not util.array_equal(self.eng, other.eng):
                raise TypeError('abscissae are different for the two Spectra.')
            if not util.array_equal(self.rs, other.rs):
                raise TypeError('redshifts are different for the two Spectra.')

            return Spectra([spec1 + spec2 for spec1,spec2 in zip(self.spec_arr, other.spec_arr)])

        else: raise TypeError('adding an object that is not of class Spectra.')

    def __sub__(self, other):
        """Subtracts one ``Spectra`` instance from another. 

        Parameters
        ----------
        other : Spectra

        Returns
        -------
        Spectra
            New ``Spectra`` instance which has the subtracted list of `dNdE`. 

        Notes
        -----
        This special function, together with `Spectra.__rsub__`, allows the use of the symbol - to subtract or subtract from `Spectra` objects. 

        See Also
        --------
        spectrum.Spectra.__rsub__
        """
        return self + -1*other 

    def __rsub__(self, other):
        """Subtracts one ``Spectra`` instance from another. 

        Parameters
        ----------
        other : Spectra

        Returns
        -------
        Spectra
            New ``Spectra`` instance which has the subtracted list of `dNdE`. 

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
            New ``Spectra`` instance with the `dNdE` negated.
        """
        return -1*self

    def __mul__(self, other):
        """Takes the product of two ``Spectra`` instances. 

        Parameters
        ----------
        other : Spectra, int or float

        Returns
        -------
        Spectra
            New ``Spectra`` instance which is an element-wise product of the ``Spectrum`` objects in each ``Spectra``. 

        Notes
        -----
        This special function, together with `Spectra.__rmul__`, allows the use of the symbol * to add ``Spectra`` objects together.

        See Also
        --------
        spectrum.Spectra.__rmul__
        """
        if np.issubdtype(type(other), float) or np.issubdtype(type(other), int):
            return Spectra([other*spec for spec in self])
        elif np.issubclass_(type(other), Spectra):
            if self.rs != other.rs or self.eng != other.eng:
                raise TypeError("the two spectra do not have the same redshift or abscissae.")
            return Spectra([spec1*spec2 for spec1,spec2 in zip(self, other)])
        else:
            raise TypeError("can only multiply Spectra or scalars.")

    def __rmul__(self, other):
        """Takes the product of two ``Spectra`` instances. 

        Parameters
        ----------
        other : Spectra, int or float

        Returns
        -------
        Spectra
            New ``Spectra`` instance which is an element-wise product of the ``Spectrum`` objects in each ``Spectra``. 

        Notes
        -----
        This special function, together with `Spectra.__mul__`, allows the use of the symbol * to add ``Spectra`` objects together.

        See Also
        --------
        spectrum.Spectra.__mul__
        """
        if np.issubdtype(type(other), float) or np.issubdtype(type(other), int):
            return Spectra([other*spec for spec in self])
        elif np.issubclass_(type(other), Spectra):
            if self.rs != other.rs or self.eng != other.eng:
                raise TypeError("the two spectra do not have the same redshift or abscissae.")
            return Spectra([spec2*spec1 for spec1,spec2 in zip(self, other)])
        else:
            raise TypeError("can only multiply Spectra or scalars.")

    def __truediv__(self,other):
        """Divides ``Spectra`` by a number or another ``Spectra``. 

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectra

        Notes
        -----
        This special function, together with `Spectra.__rtruediv__`, allows the use of the symbol / to divide ``Spectra`` objects by a number or another ``Spectra``. 

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
        """Divides ``Spectra`` by a number or another ``Spectra``. 

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectra

        Notes
        -----
        This special function, together with `Spectra.__truediv__`, allows the use of the symbol / to divide ``Spectra`` objects by a number or another ``Spectra``. 

        See Also
        --------
        spectrum.Spectra.__truediv__
        """
        invSpec = Spectra([1./spec for spec in self])

        return other*invSpec   

    def integrate_each_spec(self,weight=None):
        """Sums each ``Spectrum``, each `eng` bin weighted by `weight`. 

        Equivalent to contracting `weight` with each `dNdE` in ``Spectra``, `weight` should have length `self.length`. 

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
            weight = np.ones(self.eng.size)

        if isinstance(weight,np.ndarray):
            return np.array([spec.contract(weight) for spec in self])

        else:
            raise TypeError("mat must be an ndarray.")

    def sum_specs(self,weight=None):
        """Sums the spectrum in each energy bin, weighted by `mat`. 

        Equivalent to contracting `mat` with `[spec.dNdE[i] for spec in spec_arr]` for all `i`. `mat` should have length `self.rs.size`. 

        Parameters
        ----------
        weight : ndarray or ``Spectrum``, optional
            The weight in each redshift bin, with weight of 1 for every bin if not specified.

        Returns
        -------
        ndarray or ``Spectrum``
            An array or ``Spectrum`` of weight sums, one for each energy in `self.eng`, with length `self.length`. 

        """
        if weight is None:
            weight = np.ones(self.rs.size)

        if isinstance(weight, np.ndarray):
            return np.dot(weight, self.grid_values)
        elif isinstance(weight, Spectrum):
            new_dNdE = np.dot(weight.dNdE, self.grid_values)
            return Spectrum(self.eng, new_dNdE)
        else:
            raise TypeError("mat must be an ndarray.")

    def rebin(self, out_eng):
        """ Re-bins all ``Spectrum`` objects according to a new abscissa.

        Rebinning conserves total number and total energy.
        
        Parameters
        ----------
        out_eng : ndarray
            The new abscissa to bin into. If `self.eng` has values that are smaller than `out_eng[0]`, then the new underflow will be filled. If `self.eng` has values that exceed `out_eng[-1]`, then an error is returned.

        See Also
        --------
        spectrum.Spectrum.rebin
        """
        for spec in self:
            spec.rebin(out_eng)

        self.eng = out_eng

    def append(self, spec):
        """Appends a new ``Spectrum``. 

        Parameters
        ----------
        spec : Spectrum
            The new spectrum to append.
        """
        if not np.array_equal(self.eng, spec.eng):
            raise TypeError("new Spectrum does not have the same energy abscissa.")
        if self.rs.size > 1 and self.rs[-1] < spec.rs: 
            raise TypeError("new Spectrum has a larger redshift than the current last entry.")

        self.spec_arr.append(spec)
        self.rs = np.append(self.rs, spec.rs)

    def plot(self, ind, step=1):
        """Plots the contained ``Spectrum`` objects. 

        Parameters
        ----------
        ind : int or tuple of int
            Index of ``Spectrum`` to plot, or a tuple of indices providing a range of ``Spectrum`` to plot. 

        step : int, optional
            The number of steps to take before choosing one ``Spectrum`` to plot. 

        Returns
        -------
        matplotlib.figure
        """
        fig = plt.figure()
        if np.issubdtype(type(ind), int):
            plt.plot(self.eng, self.spec_arr[ind].dNdE)
            return fig
        elif np.issubdtype(type(ind), tuple):
            spec_to_plot = np.stack([self.spec_arr[i].dNdE for i in np.arange(ind[0], ind[1], step)], 
                axis=-1)
            plt.plot(self.eng, spec_to_plot)
            return fig
        else:
            raise TypeError("ind should be either an integer or a tuple of integers.")
