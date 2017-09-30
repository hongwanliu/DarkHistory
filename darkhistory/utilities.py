""" ``utilities`` contains useful, non-physics functions used in DarkHistory.


"""

import numpy as np

def arrays_equal(ndarray_list):
    """Check if the arrays contained in `ndarray_list` are equal.
        
    Parameters
    ----------
    ndarray_list : sequence of ndarrays
        List of arrays to compare.
    
    Returns
    -------
        bool
            True if equal, False otherwise.

    """

    same = True
    ind = 0
    while same and ind < len(ndarray_list) - 1:
        same = same & np.array_equal(ndarray_list[ind], 
            ndarray_list[ind+1])
        ind += 1
    return same

def is_log_spaced(arr):
    """Checks if `arr` is a log-spaced array.
        
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

def div_ignore_by_zero(a, b, val=0):
    """ Divides `a` by `b`, returning `val` if a divide-by-zero error occurs.

    Parameters
    ----------
    a : ndarray
        Numerator of the division.
    b : ndarray
        Denominator of the division.
    val : float
        Value given as the result of the division if a divide-by-zero error occurs.

    Returns
    -------
    ndarray
        The result of the division of the two arrays.
    
    """
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        c = np.true_divide(a,b)
        c[~ np.isfinite(c)] = val
    return c

def compare_arr(ndarray_list):
    """ Prints the arrays in a suitable format for comparison.

    Parameters
    ----------
    ndarray_list : list of ndarray
        The list of 1D arrays to compare.
    """

    print(np.stack(ndarray_list, axis=-1))

    return 0