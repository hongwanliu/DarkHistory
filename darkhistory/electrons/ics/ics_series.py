"""Series for computation of ICS spectrum without quadrature."""

import numpy as np 
import scipy.special as sp
from darkhistory.utilities import log_1_plus_x
from darkhistory.utilities import diff_pow
from darkhistory.utilities import check_err

def F1(a,b,epsrel=0):
    """Definite integral of x/[(exp(x) - 1)]. 

    Parameters
    ----------
    a : ndarray
        Lower limit of integration. 
    b : ndarray
        Upper limit of integration.
    err : float
        Error associated with series expansion. If zero, then the error is not computed.

    Returns
    -------
    float
        The resulting integral.

    Note
    ----
    For a or b > 0.01, the exact analytic expression is used, whereas below that we use a series expansion. This avoids numerical errors due to computation of log(1 - exp(-x)) and likewise in the `spence` function. Note that `scipy.special.spence` can only take `float64` numbers, so downcasting is necessary for 0.01 < x < 3. 
    
    """
    lowlim = 0.1
    upplim = 3

    def indef_int(x):
        if x < lowlim:
            # Excludes pi^2/6 to avoid catastrophic cancellation.
            return (
                x - x**2/4 + x**3/36 - x**5/3600 
                + x**7/211680 - x**9/10886400
            )
        elif x > upplim:
            n = np.arange(11) + 1
            return (
                x*log_1_plus_x(-np.exp(-x))
                - np.sum(np.exp(-n*x)/n**2, axis=0)
            )
        else:
            return (
                x*log_1_plus_x(-np.exp(-x))
                - sp.spence(
                    np.array(1. - np.exp(-x), dtype='float64')
                )
            )

    if a < lowlim and b < lowlim: 
        integral = (
            (b-a) - (b-a)*(b+a)/4 + diff_pow(b,a,3)/36 
            - diff_pow(b,a,5)/3600 + diff_pow(b,a,7)/211680 
            - diff_pow(b,a,9)/10886400
        )
        if epsrel > 0:
            err = diff_pow(b,a,11)/526901760
            check_err(integral, err, epsrel)

    elif a > upplim and b > upplim:
        spence_term = np.sum(
            np.array(
                [diff_pow(np.exp(-b), np.exp(-a), i)/i**2 
                    for i in np.arange(1,11)
                ]
            ), axis=0
        )

        integral = ( 
            b*log_1_plus_x(-np.exp(-b)) 
            - a*log_1_plus_x(-np.exp(-a))
            - spence_term
        )
        if epsrel > 0:
            err = diff_pow(np.exp(-11*b), np.exp(-11*a), 11)/121
            check_err(integral, err, epsrel)

    else:
        # Correct for missing pi^2/6 if necessary.
        piSquareOver6 = 0
        if a < lowlim and b >= lowlim:
            piSquareOver6 = np.pi**2/6
        integral = indef_int(b) - indef_int(a) + piSquareOver6
    return integral

def F0(a,b,epsrel=0):
    """Definite integral of 1/[(exp(x) - 1)]. 

    Parameters
    ----------
    a : ndarray
        Lower limit of integration. 
    b : ndarray
        Upper limit of integration.
    err : float
        Error associated with series expansion. If zero, then the error is not computed.

    Returns
    -------
    float
        The resulting integral.   

    """
    lowlim = 0.1
    upplim = 3
    
    def indef_int(x):
        if x < lowlim:
            # Excludes pi^2/6 to avoid catastrophic cancellation.
            return (
                x - x**2/4 + x**3/36 - x**5/3600 
                + x**7/211680 - x**9/10886400
            )
        elif x > upplim:
            n = np.arange(11) + 1
            return (
                x*log_1_plus_x(-np.exp(-x))
                - np.sum(np.exp(-n*x)/n**2, axis=0)
            )
        else:
            return (
                x*log_1_plus_x(-np.exp(-x))
                - sp.spence(
                    np.array(1. - np.exp(-x), dtype='float64')
                )
            )
