""" Non-physics convenience and mathematical functions."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def dict_from_inhom_list(l, key):
    """Get hdf5 compatible dictionary from an inhomogeneous list of arrays.

    Parameters
    ----------
    l : list
        List of arrays.
    key : str
        Key to use for dictionary.
    """
    d = {}
    for i, arr in enumerate(l):
        d[key + str(i)] = arr
    return d


def inhom_list_from_dict(d, key):
    """Get inhomogeneous list of arrays from dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to extract arrays from.
    key : str
        Key to use for dictionary.
    """
    l = []
    i = 0
    while key + str(i) in d:
        l.append(d[key + str(i)])
        i += 1
    return l


def arrays_equal(ndarray_list):
    """Checks if a list of arrays are all equal.

    Parameters
    ----------
    ndarray_list : sequence of ndarrays
        List of arrays to compare.

    Returns
    -------
        bool
            True if equal, False otherwise.
    """
    first_row = ndarray_list[0]
    for row in ndarray_list[1:]:
        if not np.array_equal(first_row, row):
            return False
    return True

def arrays_close(ndarray_list, rtol=1e-05, atol=1e-08):
    """Checks if a list of arrays are all np.allclose.

    Parameters
    ----------
    ndarray_list : sequence of ndarrays
        List of arrays to compare.

    Returns
    -------
        bool
            True if all np.allclose, False otherwise.
    """
    first_row = ndarray_list[0]
    for row in ndarray_list[1:]:
        if not np.allclose(first_row, row, rtol=rtol, atol=atol):
            return False
    return True

def is_log_spaced(arr):
    """Checks for a log-spaced array.

    Parameters
    ----------
    arr : ndarray
        Array for checking.

    Returns
    -------
        bool
            True if equal, False otherwise.

    """
    return not bool(np.ptp(np.diff(np.log(arr))))

def compare_arr(ndarray_list):
    """ Prints the arrays in a suitable format for comparison.

    Parameters
    ----------
    ndarray_list : list of ndarray
        The list of 1D arrays to compare.

    Returns
    --------
    None
    """

    print(np.stack(ndarray_list, axis=-1))

def log_1_plus_x(x):
    """ Computes log(1+x) with greater floating point accuracy.

    Unlike ``scipy.special.log1p``, this can take ``float128``. However the performance is certainly slower. See [1]_ for details. If that trick does not work, the code reverts to a Taylor expansion.

    Parameters
    ----------
    x : float or ndarray
        The input value.

    Returns
    -------
    ndarray
        log(1+x).
    """
    ind_not_zero = ((1+x) - 1 != 0)
    expr = np.zeros_like(x)

    if np.any(ind_not_zero):
        expr[ind_not_zero] = (
            x[ind_not_zero]*np.log(1+x[ind_not_zero])
            /((1+x[ind_not_zero]) - 1)
        )

    if np.any(~ind_not_zero):
        expr[~ind_not_zero] = (
            x[~ind_not_zero] - x[~ind_not_zero]**2/2
            + x[~ind_not_zero]**3/3
            - x[~ind_not_zero]**4/4 + x[~ind_not_zero]**5/5
            - x[~ind_not_zero]**6/6 + x[~ind_not_zero]**7/7
            - x[~ind_not_zero]**8/8 + x[~ind_not_zero]**9/9
            - x[~ind_not_zero]**10/10 + x[~ind_not_zero]**11/11
        )
    return expr

def bernoulli(k):
    """ The kth Bernoulli number.

    This function is written as a look-up table for the first few Bernoulli numbers for speed. The Bernoulli number definition we use is:

    .. math::
       \\frac{x}{e^x - 1} \\equiv \\sum_{n = 0}^\\infty \\frac{B_n x^n}{n!} \\,.

    Parameters
    ----------
    k : int
        The Bernoulli number to return.

    Returns
    -------
    float
        The kth Bernoulli number.
    """

    import scipy.special as sp

    B_n = np.array([1, -1/2, 1/6, 0, -1/30,
        0, 1/42, 0, -1/30, 0, 5/66,
        0, -691/2730, 0, 7/6, 0, -3617/510,
        0, 43867/798, 0, -174611/330, 0, 854513/138
    ])

    if k <= 22:
        return B_n[k]
    else:
        return sp.bernoulli(k)[-1]

def log_series_diff(b, a):
    """ The Taylor series for log(1+b) - log(1+a).

    Parameters
    ----------
    a : ndarray
        Input for log(1+a).
    b : ndarray
        Input for log(1+b).

    Returns
    -------
    ndarray
        The Taylor series log(1+b) - log(1+a), up to the 11th order term.

    """
    return(
        - (b-a) - (b**2 - a**2)/2 - (b**3 - a**3)/3
        - (b**4 - a**4)/4 - (b**5 - a**5)/5 - (b**6 - a**6)/6
        - (b**7 - a**7)/7 - (b**8 - a**8)/8 - (b**9 - a**9)/9
        - (b**10 - a**10)/10 - (b**11 - a**11)/11
    )

def spence_series_diff(b, a):
    r""" Returns the Taylor series for Li\ :sub:`2`\ (b) - Li\ :sub:`2`\ (a).

    Li2 is the polylogarithm function defined by
    
    .. math::
       \\text{Li}_2(z) \\equiv \\sum_{k=1}^\\infty \\frac{z^k}{k^2} \\,.

    Parameters
    ----------
    a : ndarray
        Input for Li\ :sub:`2`\ (a).
    b : ndarray
        Input for Li\ :sub:`2`\ (b).

    Returns
    -------
    ndarray
        The Taylor series Li\ :sub:`2`\ (b) - Li\ :sub:`2`\ (a), up to the 11th order term.

    """

    return(
        (b - a) + (b**2 - a**2)/2**2 + (b**3 - a**3)/3**2
        + (b**4 - a**4)/4**2 + (b**5 - a**5)/5**2
        + (b**6 - a**6)/6**2 + (b**7 - a**7)/7**2
        + (b**8 - a**8)/8**2 + (b**9 - a**9)/9**2
        + (b**10 - a**10)/10**2 + (b**11 - a**11)/11**1
    )

def exp_expn(n, x):
    """ Returns :math:`e^x E_n(x)`.

    The exponential integral :math:`E_n(x)` is defined as

    .. math::
       E_n(x) \\equiv \\int_1^\\infty dt\\, \\frac{e^{-xt}}{t^n}

    Circumvents overflow error in ``np.exp`` by expanding the exponential integral in a series to the 5th or 6th order.  

    Parameters
    ----------
    n : {1,2}
        The order of the exponential integral.
    x : ndarray
        The argument of the function.

    Returns
    -------
    ndarray
        The value of :math:`e^x E_n(x)`. 

    """
    import scipy.special as sp

    x_flt64 = np.array(x, dtype='float64')

    low = x < 700
    high = ~low
    expr = np.zeros_like(x)

    if np.any(low):
        expr[low] = np.exp(x[low])*sp.expn(n, x_flt64[low])
    if np.any(high):
        if n == 1:
            # The relative error is roughly 1e-15 for 700, smaller for larger arguments.
            expr[high] = (
                1/x[high] - 1/x[high]**2 + 2/x[high]**3 - 6/x[high]**4
                + 24/x[high]**5
            )
        elif n == 2:
            # The relative error is roughly 6e-17 for 700, smaller for larger arguments.
            expr[high] = (
                1/x[high] - 2/x[high]**2 + 6/x[high]**3 - 24/x[high]**4
                + 120/x[high]**5 - 720/x[high]**6
            )
        else:
            raise TypeError('only supports n = 1 or 2 for x > 700.')

    return expr

def hyp2f1_func_real(n, x):
    """ Returns the real part of :math:`_2F_1(1, n+1, n+2, x)`.

    Avoids the need for complex numbers in ``scipy.special.hyp2f1``, which is very slow. The function definition is identical.

    Parameters
    ----------
    n : integer
        The order of :math:`_2F_1(1, n+1, n+2, x)` to evaluate.
    x : ndarray
        The main argument of the function.

    Returns
    -------
    ndarray
        The result of :math:`_2F_1(1, n+1, n+2, x)`.

    """

    x_gt_1 = x > 1.
    x_lt_1_large_abs = (x <= 1.) & (np.abs(x) > 0.5)
    x_small_abs = np.abs(x) <= 0.5
    expr = np.zeros_like(x)

    if np.any(x_gt_1):
        x_1 = x[x_gt_1]
        for j in 1.+np.arange(n):
            expr[x_gt_1] -= (n+1)/j*(1/x_1)**(n+1-j)
        expr[x_gt_1] -= (
            (n+1)*(1/x_1)**(n+1)
            *(np.log(x_1) + np.log1p(-1/x_1))
            # just log(x-1) but works for x ~ 2.
        )

    if np.any(x_lt_1_large_abs):
        x_2 = x[x_lt_1_large_abs]
        for j in 1.+np.arange(n):
            expr[x_lt_1_large_abs] -= (n+1)/j*(1/x_2)**(n+1-j)
        expr[x_lt_1_large_abs] -= (
            (n+1)*(1/x_2)**(n+1)*np.log1p(-x_2)
        )

    if np.any(x_small_abs):
        # Power series expansion needed in this region.
        x_3 = x[x_small_abs]
        for j in 1.+np.arange(20):
            expr[x_small_abs] += (n+1)/(n+j)*x_3**(j-1)

    return expr

def get_grid(a, b):
    """ Returns a 2D grid of coordinates from 2 1D arrays.

    Parameters
    ----------
    a : ndarray
        First array.
    b : ndarray
        Second array.

    Returns
    -------
    ndarray
        2D array with grid values from `a` and `b`.

    Notes
    -----
    This function returns an array that when passed to ``scipy.interpolate.RegularGridInterpolator`` produces the same result as ``scipy.interpolate.interp2d(a, b)``.
    """

    grid_list = np.meshgrid(a,b)

    # order = 'F' required so that the points are sorted by values
    # in a (index 1) first, followed by values in b (index 2).
    return np.transpose(np.array([m.flatten(order='F') for m in grid_list]))

def check_err(val, err, epsrel):
    """ Checks the relative error given a tolerance.

    Parameters
    ----------
    val : float or ndarray
        The computed value.
    err : float or ndarray
        The computed error.
    epsrel : float
        The target tolerance.

    Returns
    -------
    None

    """
    if np.max(np.abs(err/val)) > epsrel:
        print('Series relative error is: ', err/val)
        print('Relative error required is: ', epsrel)
        raise RuntimeError('Relative error in series too large.')

    return None

class Interpolator2D:

    """Interpolation function over a list of objects.

    Parameters
    ----------
    val_arr : list of objects
        List of objects, ``ndim = (arr0.size, arr1.size, ...)``
    arr0 : ndarray
        list of values along 0th dimension
    arr1 : ndarray
        list of values along 1st dimension

    Attributes
    ----------
    interp_func : function
        A 2D interpolation function over ``arr0`` and ``arr1``.
    _grid_vals : ndarray
        a nD array of input data
    """

    def __init__(self, arr0, name0, arr1, name1, val_arr, logInterp=False):

        if str(type(val_arr)) != "<class 'numpy.ndarray'>":
            raise TypeError('val_arr must be an ndarray')

        if len(arr0) != np.size(val_arr, 0):
            raise TypeError('0th dimension of val_arr must be the arr0')

        if len(arr1) != np.size(val_arr, 1):
            raise TypeError('1st dimension of val_arr (val_arr[0,:,0,0,...]) must be the arr1 dimension')

        self.arr0 = arr0
        setattr(self, name0, self.arr0)
        self.arr1 = arr1
        setattr(self, name1, self.arr1)
        self._grid_vals = val_arr

        self.logInterp = logInterp

        if not logInterp:
            # self.interp_func = RegularGridInterpolator((np.log(arr0), np.log(arr1)), self._grid_vals)
            self.interp_func = RegularGridInterpolator((arr0, arr1), self._grid_vals)
        else:
            self._grid_vals[self._grid_vals <= 0] = 1e-200
            self.interp_func = RegularGridInterpolator((np.log(arr0), np.log(arr1)), np.log(self._grid_vals))


    def get_val(self, val0, val1):

        # xe must lie between these values.
        if val0 > self.arr0[-1]:
            val0 = self.arr0[-1]
        if val0 < self.arr0[0]:
            val0 = self.arr0[0]

        if val1 > self.arr1[-1]:
            val1 = self.arr1[-1]
        if val1 < self.arr1[0]:
            val1 = self.arr1[0]

        if not self.logInterp:
            return np.squeeze(self.interp_func([val0, val1]))
        else:
            return np.exp(np.squeeze(self.interp_func([np.log(val0), np.log(val1)])))

    def get_vals(self, val0, vals1):

        # xe must lie between these values.
        if val0 > self.arr0[-1]:
            val0 = self.arr0[-1]
        if val0 < self.arr0[0]:
            val0 = self.arr0[0]

        vals1 = np.array(vals1)
        vals1[vals1 > self.arr1[-1]] = self.arr1[-1]
        vals1[vals1 < self.arr1[0]] = self.arr1[0]

        # points = np.transpose([val0 * np.ones_like(vals1), vals1])

        if not self.logInterp:
            points = np.transpose(
                [val0 * np.ones_like(vals1), vals1]
            )
            return self.interp_func(points)
        else:
            points = np.transpose([val0 * np.ones_like(vals1), vals1])
            return np.exp(self.interp_func(np.log(points)))



# class InterpolatorND:

#     """Interpolation function over list of objects

#     Parameters
#     ----------
#     val_arr : list of objects
#         List of objects, ndim = (arr0.size, arr1.size, ...)
#     arr0 : ndarray
#         list of values along 0th dimension
#     arr1 : ndarray
#         list of values along 1st dimension

#     Attributes
#     ----------
#     interp_func : function
#         A 2D interpolation function over xe and rs.
#     _grid_vals : ndarray
#         a nD array of input data
#     """

#     def __init__(self, arr0, name0, arr1, name1, val_arr, logInterp=False):

#         if str(type(val_arr)) != "<class 'numpy.ndarray'>":
#             raise TypeError('val_arr must be an ndarray')

#         if len(arr0) != np.size(val_arr, 0):
#             raise TypeError('0th dimension of val_arr must be the arr0')

#         if len(arr1) != np.size(val_arr, 1):
#             raise TypeError('1st dimension of val_arr (val_arr[0,:,0,0,...]) must be the arr1 dimension')

#         self.arr0 = arr0
#         setattr(self, name0, self.arr0)
#         self.arr1 = arr1
#         setattr(self, name1, self.arr1)
#         self._grid_vals = val_arr

#         self.logInterp = logInterp

#         if not logInterp:
#             # self.interp_func = RegularGridInterpolator((np.log(arr0), np.log(arr1)), self._grid_vals)
#             self.interp_func = RegularGridInterpolator((arr0, arr1), self._grid_vals)
#         else:
#             self._grid_vals[self._grid_vals <= 0] = 1e-200
#             self.interp_func = RegularGridInterpolator((np.log(arr0), np.log(arr1)), np.log(self._grid_vals))


#     def get_val(self, val0, val1):

#         # xe must lie between these values.
#         if val0 > self.arr0[-1]:
#             val0 = self.arr0[-1]
#         if val0 < self.arr0[0]:
#             val0 = self.arr0[0]

#         if val1 > self.arr1[-1]:
#             val1 = self.arr1[-1]
#         if val1 < self.arr1[0]:
#             val1 = self.arr1[0]

#         if not self.logInterp:
#             return np.squeeze(self.interp_func([val0, val1]))
#         else:
#             return np.exp(np.squeeze(self.interp_func([np.log(val0), np.log(val1)])))

#     def get_vals(self, val0, vals1):

#         # xe must lie between these values.
#         if val0 > self.arr0[-1]:
#             val0 = self.arr0[-1]
#         if val0 < self.arr0[0]:
#             val0 = self.arr0[0]

#         vals1 = np.array(vals1)
#         vals1[vals1 > self.arr1[-1]] = self.arr1[-1]
#         vals1[vals1 < self.arr1[0]] = self.arr1[0]

#         # points = np.transpose([val0 * np.ones_like(vals1), vals1])

#         if not self.logInterp:
#             points = np.transpose(
#                 [val0 * np.ones_like(vals1), vals1]
#             )
#             return self.interp_func(points)
#         else:
#             points = np.transpose([val0 * np.ones_like(vals1), vals1])
#             return np.exp(self.interp_func(np.log(points)))
