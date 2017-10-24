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
        low = (x < lowlim)
        high = (x > upplim)
        gen = ~(low | high)
        expr = np.zeros(x.size)
        
        # Excludes pi^2/6 to avoid catastrophic cancellation.
        if np.any(low):
            expr[low] = (
                x[low] - x[low]**2/4 + x[low]**3/36 
                - x[low]**5/3600 + x[low]**7/211680 - x[low]**9/10886400
            )
        if np.any(high):
            n = np.arange(11) + 1
            expr[high] = (
                x[high]*log_1_plus_x(-np.exp(-x[high]))
                - np.sum(
                    np.array(
                        [np.exp(-n*x)/n**2 for x in x[high]]
                    ), axis=1
                )
            )
        
        if np.any(gen):
            expr[gen] = (x[gen]*log_1_plus_x(-np.exp(-x[gen]))
                - sp.spence(
                    np.array(1. - np.exp(-x[gen]), dtype='float64')
                )
            )

        return expr

    integral = np.zeros(a.size)

    both_low = (a < lowlim) & (b < lowlim)
    both_high = (a > upplim) & (b > upplim)
    
    if np.any(both_low):

        integral[both_low] = (
                b[both_low]-a[both_low]
                - (b[both_low]-a[both_low])*(b[both_low]+a[both_low])/4 
                + diff_pow(b[both_low],a[both_low],3)/36 
                - diff_pow(b[both_low],a[both_low],5)/3600 
                + diff_pow(b[both_low],a[both_low],7)/211680 
                - diff_pow(b[both_low],a[both_low],9)/10886400
        )

        if epsrel > 0:
            err = diff_pow(b[both_low],a[both_low],11)/526901760
            check_err(integral[both_low], err, epsrel)

    if np.any(both_high):

        spence_term = np.sum(
            np.array(
                [diff_pow(
                    np.exp(-b[both_high]), 
                    np.exp(-a[both_high]), i
                )/i**2 for i in np.arange(1,11)]
            ), axis=0
        )

        integral[both_high] = (
            b[both_high]*log_1_plus_x(-np.exp(-b[both_high]))
            - a[both_high]*log_1_plus_x(-np.exp(-a[both_high]))
            - spence_term
        )

        if epsrel > 0:
            err = (
                diff_pow(np.exp(-b[both_high]), np.exp(-a[both_high]), 11)/11**2
            )
            check_err(integral[both_high], err, epsrel)

    gen_case = ~(both_low | both_high)

    if np.any(gen_case):
        integral[gen_case] = indef_int(b[gen_case]) - indef_int(a[gen_case])

    # Correct for missing pi^2/6 where necessary.
    a_low_b_notlow = (a < lowlim) & (b >= lowlim)

    integral[a_low_b_notlow] += np.pi**2/6

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
        
        return log_1_plus_x(-np.exp(-x))

    integral = np.zeros(a.size)

    both_low = (a < lowlim) & (b < lowlim)
    both_high = (a > upplim) & (b > upplim)

    if np.any(both_low):
        integral[both_low] = (
            np.log(b[both_low]/a[both_low]) 
            - (b[both_low]-a[both_low])/2 
            + (b[both_low]-a[both_low])*(b[both_low]+a[both_low])/24
            - diff_pow(b[both_low],a[both_low],4)/2880 
            + diff_pow(b[both_low],a[both_low],6)/181440
            - diff_pow(b[both_low],a[both_low],8)/9676800
            + diff_pow(b[both_low],a[both_low],10)/479001600
        )
        if epsrel > 0:
            err = -diff_pow(b[both_low],a[both_low],12)*691/15692092416000
            check_err(integral[both_low], err, epsrel)

    if np.any(both_high):
        integral[both_high] = np.sum(
            np.array(
                [-diff_pow(np.exp(-b[both_high]), np.exp(-a[both_high]), i)/i
                    for i in np.arange(1,11)
                ]
            ), axis=0
        )
        if epsrel > 0:
            err = -diff_pow(
                np.exp(-b[both_high]), 
                np.exp(-a[both_high]), 12
            )/12
            check_err(integral[both_high], err, epsrel)

    gen_case = ~(both_low | both_high)

    if np.any(gen_case):
        integral[gen_case] = indef_int(b[gen_case]) - indef_int(a[gen_case])

    return integral
