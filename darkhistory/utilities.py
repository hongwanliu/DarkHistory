""" Non-physics functions used in darkhistory.

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

def log_1_plus_x(x):
    """ Computes log(1+x) with greater floating point accuracy. 

    See "What every computer scientist should know about floating-point arithmetic" by David Goldberg for details. If that trick does not work, the code reverts to a Taylor expansion.

    Parameters
    ----------
    x : ndarray
        The input value. 

    Returns
    -------
    ndarray
        log(1+x). 
    """
    ind_zero = ((1+x) - 1 != 0)
    expr = np.zeros(x.size)

    if np.any(ind_zero):
        expr[ind_zero] = (
            x[ind_zero]*np.log(1+x[ind_zero])/((1+x[ind_zero]) - 1)
        )

    if np.any(~ind_zero):
        expr[~ind_zero] = (
            x[~ind_zero] - x[~ind_zero]**2/2 + x[~ind_zero]**3/3
            - x[~ind_zero]**4/4 + x[~ind_zero]**5/5
            - x[~ind_zero]**6/6 + x[~ind_zero]**7/7
            - x[~ind_zero]**8/8 + x[~ind_zero]**9/9
            - x[~ind_zero]**10/10 + x[~ind_zero]**11/11
        )
    return expr

def diff_pow(a, b, n):
    """ Computes a^n - b^n with greater floating point accuracy. 

    Factorizes out the difference between a and b first. 

    Parameters
    ----------
    a : ndarray
        a^n to be computed. 
    b : ndarray
        b^n to be computed. 
    n : int
        The exponent. 

    Returns
    -------
    float
        The computed value. 
    """

    return a**n - b**n

    # if n == 0:
    #     return 0
    # elif n == 1:
    #     return a - b
    # elif n == 2:
    #     return (a+b)*(a-b)
    # elif n == 3:
    #     return (a-b)*(a**2 + a*b + b**2)
    # elif n == 4:
    #     return (a-b)*(a+b)*(a**2 + b**2)
    # elif n == 5:
    #     return (a-b)*(a**4 + a**3*b + a**2*b**2 + a*b**3 + b**4)
    # elif n == 6:
    #     return (a-b)*(a+b)*(a**2 - a*b + b**2)*(a**2 + a*b + b**2)
    # elif n == 7:
    #     return (a-b)*(a**6 + a**5*b + a**4*b**2 
    #                 + a**3*b**3 + a**2*b**4 + a*b**5 + b**6
    #     )
    # elif n == 8:
    #     return (a-b)*(a+b)*(a**2 + b**2)*(a**4 + b**4)
    # elif n == 9:
    #     return (a-b)*(a**2 + a*b + b**2)*(a**6 + a**3*b**3 + b**6)
    # elif n == 10:
    #     return (a-b)*(a+b)*(
    #         (a**4 - a**3*b + a**2*b**2 - a*b**3 + b**4)
    #         *(a**4 + a**3*b + a**2*b**2 + a*b**3 + b**4)
    #     )
    # elif n == 11:
    #     return (a-b)*(
    #         a**10 + a**9*b + a**8*b**2 + a**7*b**3 + a**6*b**4
    #         + a**5*b**5 + a**4*b**6 + a**3*b**7 + a**2*b**8
    #         + a*b**9 + b**10
    #     )
    # elif n == 12:
    #     return (a-b)*(a+b)*(a**2 + b**2)*(
    #         (a**2 - a*b + b**2)
    #         *(a**2 + a*b + b**2)
    #         *(a**4 - a**2*b**2 + b**4)
    #     )
    # else: 
    #     raise TypeError('n > 12 not supported.')

def bernoulli(k):

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

    """
    if np.max(np.abs(err/val)) > epsrel:
        print('Series relative error is: ', err/val)
        print('Relative error required is: ', epsrel)
        raise RuntimeError('Relative error in series too large.')
        
    return None
