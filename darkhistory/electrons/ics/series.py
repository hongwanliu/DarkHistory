"""Nonrelativistic series expansions for ICS spectrum."""

import numpy as np 
import scipy.special as sp

from darkhistory.utilities import log_1_plus_x
from darkhistory.utilities import check_err
from darkhistory.utilities import bernoulli as bern
from darkhistory.utilities import log_series_diff
from darkhistory.utilities import spence_series_diff
from darkhistory.utilities import exp_expn

# General series expressions for integrals over Planck distribution.

def F1(a,b,epsrel=0):
    """Definite integral of x/[(exp(x) - 1)]. 

    Parameters
    ----------
    a : ndarray
        Lower limit of integration. Can be either 1D or 2D.  
    b : ndarray
        Upper limit of integration. Can be either 1D or 2D.
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

        inf = (x == np.inf)
        low = (x < lowlim)
        high = (x > upplim) & (~inf)
        gen = ~(low | high) & (~inf)
        expr = np.zeros(x.size)
        
        # Two different series for small and large x limit.

        # Excludes pi^2/6 to avoid catastrophic cancellation.
        if np.any(inf):
            expr[inf] = 0

        if np.any(low):
            expr[low] = (
                x[low] - x[low]**2/4 + x[low]**3/36 
                - x[low]**5/3600 + x[low]**7/211680 - x[low]**9/10886400
            )
        if np.any(high):
            n = np.arange(11) + 1
            expr[high] = (
                x[high]*log_1_plus_x(-np.exp(-x[high]))
                - np.exp(-x[high]) - np.exp(-2*x[high])/4
                - np.exp(-3*x[high])/9 - np.exp(-4*x[high])/16
                - np.exp(-5*x[high])/25 - np.exp(-6*x[high])/36
                - np.exp(-7*x[high])/49 - np.exp(-8*x[high])/64
                - np.exp(-9*x[high])/81 
                - np.exp(-10*x[high])/100
                - np.exp(-11*x[high])/121
            )
        
        if np.any(gen):
            expr[gen] = (x[gen]*log_1_plus_x(-np.exp(-x[gen]))
                - sp.spence(
                    np.array(1. - np.exp(-x[gen]), dtype='float64')
                )
            )

        return expr

    if a.ndim == 1 and b.ndim == 2:
        if b.shape[1] != a.size:
            raise TypeError('The second dimension of b must have the same length as a.')
        # Extend a to a 2D array.
        a = np.outer(np.ones(b.shape[0]), a)
    elif a.ndim == 2 and b.ndim == 1:
        if a.shape[1] != b.size:
            raise TypeError('The second dimension of a must have the same length as b.')
        b = np.outer(np.ones(a.shape[0]), b)

    # if both are 1D, then the rest of the code still works.

    integral = np.zeros(a.shape, dtype='float128')

    both_low = (a < lowlim) & (b < lowlim)
    both_high = (a > upplim) & (b > upplim)

    if np.any(both_low):

        # Use diff_pow to compute differences in powers accurately.

        integral[both_low] = (
                b[both_low]-a[both_low]
                - (b[both_low]-a[both_low])*(b[both_low]+a[both_low])/4 
                + (b[both_low]**3 - a[both_low]**3)/36 
                - (b[both_low]**5 - a[both_low]**5)/3600 
                + (b[both_low]**7 - a[both_low]**7)/211680 
                - (b[both_low]**9 - a[both_low]**9)/10886400
        )

        if epsrel > 0:
            err = (b[both_low]**11 - a[both_low]**11)/526901760
            check_err(integral[both_low], err, epsrel)

    if np.any(both_high):

        # Use a series for the spence function.

        spence_term = np.zeros_like(integral)

        spence_term[both_high] = spence_series_diff(
            np.exp(-b[both_high]),
            np.exp(-a[both_high])
        )

        b_inf = both_high & (b == np.inf)
        b_not_inf = both_high & (b != np.inf)
        
        integral[b_inf] = (
            - a[b_inf]*log_1_plus_x(-np.exp(-a[b_inf]))
            - spence_term[b_inf]
        )

        integral[b_not_inf] = (
            b[b_not_inf]*log_1_plus_x(-np.exp(-b[b_not_inf]))
            - a[b_not_inf]*log_1_plus_x(-np.exp(-a[b_not_inf]))
            - spence_term[b_not_inf]
        )

        # Use diff_pow if necessary
        if epsrel > 0:
            err = (
                np.exp(-b[both_high])**11 
                - np.exp(-a[both_high])**11
            )/11**2
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
        Lower limit of integration. Can be either 1D or 2D.
    b : ndarray
        Upper limit of integration. Can be either 1D or 2D.
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
        
        inf = (x == np.inf)
        low = (x <= 1e-10)
        high = (x > 1e-10) & (~inf)
        expr = np.zeros_like(x)

        if np.any(inf):
            expr[inf] = 0

        if np.any(high):
            expr[high] = log_1_plus_x(-np.exp(-x[high]))

        if np.any(low):
            expr[low] = (
                np.log(x[low]) - x[low]/2 + x[low]**2/24 
                - x[low]**4/2880 + x[low]**6/181440 
                - x[low]**8/9676800 + x[low]**10/479001600
            )

        return expr

    if a.ndim == 1 and b.ndim == 2:
        if b.shape[1] != a.size:
            raise TypeError('The second dimension of b must have the same length as a.')
        # Extend a to a 2D array.
        a = np.outer(np.ones(b.shape[0]), a)
    elif a.ndim == 2 and b.ndim == 1:
        if a.shape[1] != b.size:
            raise TypeError('The second dimension of a must have the same length as b.')
        b = np.outer(np.ones(a.shape[0]), b)

    # if both are 1D, then the rest of the code still works.

    integral = np.zeros(a.shape, dtype='float128')

    both_low = (a < lowlim) & (b < lowlim)
    both_high = (a > upplim) & (b > upplim)

    # Use diff_pow if necessary
    if np.any(both_low):
        integral[both_low] = (
            np.log(b[both_low]/a[both_low]) 
            - (b[both_low]-a[both_low])/2 
            + (b[both_low]-a[both_low])*(b[both_low]+a[both_low])/24
            - (b[both_low]**4 - a[both_low]**4)/2880 
            + (b[both_low]**6 - a[both_low]**6)/181440
            - (b[both_low]**8 - a[both_low]**8)/9676800
            + (b[both_low]**10 - a[both_low]**10)/479001600
        )
        if epsrel > 0:
            err = -(b[both_low]**12 - a[both_low]**12)*691/15692092416000
            check_err(integral[both_low], err, epsrel)

    if np.any(both_high):
        integral[both_high] = log_series_diff(
            np.exp(-b[both_high]),
            np.exp(-a[both_high])
        )

        if epsrel > 0:
            err = -(
                np.exp(-b[both_high])**12 - 
                np.exp(-a[both_high])**12
            )/12
            check_err(integral[both_high], err, epsrel)

    gen_case = ~(both_low | both_high)

    if np.any(gen_case):
        integral[gen_case] = indef_int(b[gen_case]) - indef_int(a[gen_case])

    return integral

def F_inv(a,b,tol=1e-10):
    """Definite integral of 1/[x(exp(x) - 1)]. 

    Parameters
    ----------
    a : ndarray
        Lower limit of integration. 
    b : ndarray
        Upper limit of integration.
    tol : float
        The relative tolerance to be reached.

    Returns
    -------
    float
        The resulting integral.   

    """

    # bound is fixed. If changed to another number, the exact integral from bound to infinity later in the code needs to be changed to the appropriate value.
    bound = 2.

    # Two different series to approximate this: below and above bound.

    

    def low_summand(x, k):
        if k == 1:
            return -1/x - np.log(x)/2
        else:
            return (
                bern(k)*(x**(k-1))/
                (sp.factorial(k)*(k-1))
            )
                # B_n for n odd, n > 1 is zero.


    def high_summand(x, k):
        return sp.expn(1, k*np.array(x, dtype='float64'))

    if a.ndim == 1 and b.ndim == 2:
        if b.shape[1] != a.size:
            raise TypeError('The second dimension of b must have the same length as a.')
        # Extend a to a 2D array.
        a = np.outer(np.ones(b.shape[0],dtype='float128'), a)
    elif a.ndim == 2 and b.ndim == 1:
        if a.shape[1] != b.size:
            raise TypeError('The second dimension of a must have the same length as b.')
        b = np.outer(np.ones(a.shape[0],dtype='float128'), b)

    # if both are 1D, then the rest of the code still works.

    integral = np.zeros(a.shape, dtype='float128')
    err = np.zeros_like(integral)
    next_term = np.zeros_like(integral)

    both_low  = (a < bound) & (b <  bound)
    low_high  = (a < bound) & (b >= bound)
    both_high = (a > bound) & (b >  bound)

    # Both low

    if np.any(both_low):

        low_sum_a = low_summand(a[both_low], 1)
        low_sum_b = low_summand(b[both_low], 1)
        integral[both_low] = low_sum_b - low_sum_a

        k_low = 2
        err_max = 10*tol
    
        while err_max > tol:
            
            next_term[both_low] = (
                low_summand(b[both_low], k_low)
                - low_summand(a[both_low], k_low)
            )

            err[both_low] = np.abs(
                np.divide(
                    next_term[both_low],
                    integral[both_low],
                    out = np.zeros_like(next_term[both_low]),
                    where = integral[both_low] != 0
                )
            )

            integral[both_low] += next_term[both_low]

            k_low += 2
            err_max = np.max(err[both_low])
            both_low &= (err > tol)

    # a low b high

    if np.any(low_high):

        # Evaluate the definite integral from a to 2, and then 2 to b.

        low_sum_a = low_summand(a[low_high], 1)
        high_sum_b = high_summand(b[low_high], 1)
        low_sum_bound = low_summand(bound, 1)

        # Exact integral from 2 to infinity.
        int_bound_inf = np.float128(0.053082306482669888568)
        int_a_bound = low_sum_bound - low_sum_a
        int_bound_b = int_bound_inf - high_sum_b

        integral[low_high] = int_a_bound + int_bound_b

        k_low = 2
        k_high = 2
        err_max = 10*tol

        next_term_a_bound = np.zeros_like(integral)
        next_term_bound_b = np.zeros_like(integral)

        while err_max > tol:

            next_term_a_bound[low_high] = (
                low_summand(bound, k_low)
                - low_summand(a[low_high], k_low)
            )
            # Only need to compute the next term for the b to inf integral.
            next_term_bound_b[low_high] = (
                -high_summand(b[low_high], k_high)
            )


            next_term[low_high] = (
                next_term_a_bound[low_high]
                + next_term_bound_b[low_high]
            )

            err[low_high] = np.abs(
                np.divide(
                    next_term[low_high], 
                    integral[low_high], 
                    out = np.zeros_like(next_term[low_high]),
                    where = integral[low_high] != 0
                )
            )

            integral[low_high] += next_term[low_high]

            k_low += 2
            k_high += 1
            err_max = np.max(err[low_high])
            low_high &= (err > tol)

    # Both high

    if np.any(both_high):

        high_sum_a = high_summand(a[both_high], 1)
        high_sum_b = high_summand(b[both_high], 1)
        integral[both_high] = high_sum_a - high_sum_b

        k_high = 2
        err_max = 10*tol

        while err_max > tol:

            next_term[both_high] = (
                high_summand(a[both_high], k_high)
                - high_summand(b[both_high], k_high)
            )

            err[both_high] = np.abs(
                np.divide(
                    next_term[both_high],
                    integral[both_high], 
                    out = np.zeros_like(next_term[both_high]),
                    where = integral[both_high] != 0
                )
            )

            integral[both_high] += next_term[both_high]

            k_high += 1
            err_max = np.max(err[both_high])
            both_high &= (err > tol)

    return integral, err

def F_log(a,b,tol=1e-10):
    """Definite integral of log(x)/(exp(x) - 1). 

    Parameters
    ----------
    a : ndarray
        Lower limit of integration. 
    b : ndarray
        Upper limit of integration.
    tol : float
        The relative tolerance to be reached.

    Returns
    -------
    float
        The resulting integral.   

    """

    # bound is fixed. If changed to another number, the exact integral from bound to infinity later in the code needs to be changed to the appropriate value.
    bound = 2.
    

    # Two different series to approximate this: below and above bound.

    def low_summand(x, k):
        if k == 1:
            return 1/2*(np.log(x)**2) - (x/2)*(np.log(x) - 1)
        else:
            return (
                bern(k)*(x**k)/
                (sp.factorial(k)*k**2)*(k*np.log(x) - 1)
            )
                # B_n for n odd, n > 1 is zero.
            
    def high_summand(x, k):
        # sp.expn does not support float128.

        inf = (x == np.inf)

        expr = np.zeros_like(x)
        expr[inf] = 0
        expr[~inf] = (
            1/k*(np.exp(-k*x[~inf])*np.log(x[~inf]) 
                + sp.expn(
                    1, k*np.array(x[~inf], dtype='float64')
                )
            )
        )

        return expr

    if a.ndim == 1 and b.ndim == 2:
        if b.shape[1] != a.size:
            raise TypeError('The second dimension of b must have the same length as a.')
        # Extend a to a 2D array.
        a = np.outer(np.ones(b.shape[0]), a)
    elif a.ndim == 2 and b.ndim == 1:
        if a.shape[1] != b.size:
            raise TypeError('The second dimension of a must have the same length as b.')
        b = np.outer(np.ones(a.shape[0]), b)

    # if both are 1D, then the rest of the code still works.

    integral = np.zeros(a.shape, dtype='float128')
    err = np.zeros_like(integral)
    next_term = np.zeros_like(integral)

    both_low  = (a < bound) & (b <  bound)
    low_high  = (a < bound) & (b >= bound)
    both_high = (a > bound) & (b >  bound)

    # Both low

    if np.any(both_low):

        low_sum_a = low_summand(a[both_low], 1)
        low_sum_b = low_summand(b[both_low], 1)
        integral[both_low] = low_sum_b - low_sum_a

        k_low = 2
        err_max = 10*tol
    
        while err_max > tol:
            
            next_term[both_low] = (
                low_summand(b[both_low], k_low)
                - low_summand(a[both_low], k_low)
            )

            err[both_low] = np.abs(
                np.divide(
                    next_term[both_low],
                    integral[both_low],
                    out = np.zeros_like(next_term[both_low]),
                    where = integral[both_low] != 0
                )
            )

            integral[both_low] += next_term[both_low]

            k_low += 2
            err_max = np.max(err[both_low])
            both_low &= (err > tol)

    # a low b high

    if np.any(low_high):

        # Evaluate the definite integral from a to 2, and then 2 to b.

        low_sum_a = low_summand(a[low_high], 1)
        high_sum_b = high_summand(b[low_high], 1)
        low_sum_bound = low_summand(bound, 1)

        # Exact integral from 2 to infinity.
        int_bound_inf = np.float128(0.15171347859984083704)
        int_a_bound = low_sum_bound - low_sum_a
        int_bound_b = int_bound_inf - high_sum_b

        integral[low_high] = int_a_bound + int_bound_b

        k_low = 2
        k_high = 2
        err_max = 10*tol

        next_term_a_bound = np.zeros_like(integral)
        next_term_bound_b = np.zeros_like(integral)

        while err_max > tol:

            next_term_a_bound[low_high] = (
                low_summand(bound, k_low)
                - low_summand(a[low_high], k_low)
            )
            # Only need to compute the next term for the b to inf integral.
            next_term_bound_b[low_high] = (
                -high_summand(b[low_high], k_high)
            )

            next_term[low_high] = (
                next_term_a_bound[low_high]
                + next_term_bound_b[low_high]
            )

            err[low_high] = np.abs(
                np.divide(
                    next_term[low_high], 
                    integral[low_high], 
                    out = np.zeros_like(next_term[low_high]),
                    where = integral[low_high] != 0
                )
            )

            integral[low_high] += next_term[low_high]

            k_low += 2
            k_high += 1
            err_max = np.max(err[low_high])
            low_high &= (err > tol)

    # Both high

    if np.any(both_high):

        high_sum_a = high_summand(a[both_high], 1)
        high_sum_b = high_summand(b[both_high], 1)
        integral[both_high] = high_sum_a - high_sum_b

        k_high = 2
        err_max = 10*tol

        while err_max > tol:

            next_term[both_high] = (
                high_summand(a[both_high], k_high)
                - high_summand(b[both_high], k_high)
            )

            err[both_high] = np.abs(
                np.divide(
                    next_term[both_high],
                    integral[both_high], 
                    out = np.zeros_like(next_term[both_high]),
                    where = integral[both_high] != 0
                )
            )

            integral[both_high] += next_term[both_high]

            k_high += 1
            err_max = np.max(err[both_high])
            both_high &= (err > tol)

    return integral, err

def F_x_log(a,b,tol=1e-10):
    """Definite integral of x log(x)/(exp(x) - 1). 

    Parameters
    ----------
    a : ndarray
        Lower limit of integration. 
    b : ndarray
        Upper limit of integration. 
    tol : float
        The relative tolerance to be reached. 

    Returns
    -------
    float
        The resulting integral. 
    """

    # bound is fixed. If changed to another number, the exact integral from bound to infinity later in the code needs to be changed to the appropriate value.
    bound = 2.

    def low_summand(x,k):
        if k==1:
            return x*np.log(x) - x - (x**2/2)*(2*np.log(x) - 1)/4
        else:
            return (
                bern(k)*(x**(k+1))/
                (sp.factorial(k)*(k+1)**2)*((k+1)*np.log(x) - 1)
            )

    def high_summand(x, k):

        inf = (x == np.inf)

        expr = np.zeros_like(x)
        expr[inf] = 0
        expr[~inf] = (
            1/k**2*(
                (1+k*x[~inf])*np.exp(-k*x[~inf])*np.log(x[~inf])
                + (1+k*x[~inf])*sp.expn(
                    1, k*np.array(x[~inf], dtype='float64')
                )
                + sp.expn(2, k*np.array(x[~inf], dtype='float64'))
            )
        )

        return expr

    if a.ndim == 1 and b.ndim == 2:
        if b.shape[1] != a.size:
            raise TypeError('The second dimension of b must have the same length as a.')
        # Extend a to a 2D array.
        a = np.outer(np.ones(b.shape[0]), a)
    elif a.ndim == 2 and b.ndim == 1:
        if a.shape[1] != b.size:
            raise TypeError('The second dimension of a must have the same length as b.')
        b = np.outer(np.ones(a.shape[0]), b)

    # if both are 1D, then the rest of the code still works.

    integral = np.zeros(a.shape, dtype='float128')
    err = np.zeros_like(integral)
    next_term = np.zeros_like(integral)

    both_low  = (a < bound) & (b <  bound)
    low_high  = (a < bound) & (b >= bound)
    both_high = (a > bound) & (b >  bound)

    # Both low

    if np.any(both_low):

        low_sum_a = low_summand(a[both_low], 1)
        low_sum_b = low_summand(b[both_low], 1)
        integral[both_low] = low_sum_b - low_sum_a

        k_low = 2
        err_max = 10*tol

        while err_max > tol:

            next_term[both_low] = (
                low_summand(b[both_low], k_low)
                - low_summand(a[both_low], k_low)
            )

            err[both_low] = np.abs(
                np.divide(
                    next_term[both_low],
                    integral[both_low],
                    out = np.zeros_like(next_term[both_low]),
                    where = integral[both_low] != 0
                )
            )

            integral[both_low] += next_term[both_low]

            k_low += 2
            err_max = np.max(err[both_low])
            both_low &= (err > tol)

    # a low b high

    if np.any(low_high):

        # Evaluate the definite integral from a to 2, and then 2 to b.

        low_sum_a = low_summand(a[low_high], 1)
        high_sum_b = high_summand(b[low_high], 1)
        low_sum_bound = low_summand(bound, 1)

        # Exact integral from 2 to infinity. 
        int_bound_inf = np.float128(0.4888742871822041)
        int_a_bound = low_sum_bound - low_sum_a
        int_bound_b = int_bound_inf - high_sum_b

        integral[low_high] = int_a_bound + int_bound_b

        k_low = 2
        k_high = 2
        err_max = 10*tol

        next_term_a_bound = np.zeros_like(integral)
        next_term_bound_b = np.zeros_like(integral)

        while err_max > tol:

            next_term_a_bound[low_high] = (
                low_summand(bound, k_low)
                - low_summand(a[low_high], k_low)
            )

            next_term_bound_b[low_high] = (
                -high_summand(b[low_high], k_high)
            )

            next_term[low_high] = (
                next_term_a_bound[low_high]
                + next_term_bound_b[low_high]
            )

            err[low_high] = np.abs(
                np.divide(
                    next_term[low_high],
                    integral[low_high],
                    out = np.zeros_like(next_term[low_high]),
                    where = integral[low_high] != 0
                )
            )

            integral[low_high] += next_term[low_high]

            k_low += 2
            k_high += 1
            err_max = np.max(err[low_high])
            low_high &= (err > tol)

    # Both high

    if np.any(both_high):

        high_sum_a = high_summand(a[both_high], 1)
        high_sum_b = high_summand(b[both_high], 1)
        integral[both_high] = high_sum_a - high_sum_b

        k_high = 2
        err_max = 10*tol

        while err_max > tol:

            next_term[both_high] = (
                high_summand(a[both_high], k_high)
                - high_summand(b[both_high], k_high)
            )

            err[both_high] = np.abs(
                np.divide(
                    next_term[both_high],
                    integral[both_high],
                    out = np.zeros_like(next_term[both_high]),
                    where = integral[both_high] != 0
                )
            )

            integral[both_high] += next_term[both_high]

            k_high += 1
            err_max = np.max(err[both_high])
            both_high &= (err > tol)

    return integral, err

def F_log_a(lowlim, a, tol=1e-10):
    """Integral of log(x+a)/(exp(x) - 1) from lowlim to infinity. 

    Parameters
    ----------
    a : ndarray
        Parameter in log(x+a). 
    lowlim : ndarray
        Lower limit of integration. 
    tol : float
        The relative tolerance to be reached. 

    Returns
    -------
    ndarray
        The resulting integral. 
    """

    bound = np.ones_like(lowlim,dtype='float128')*2.

    # Two different series to approximate this: below and above bound. 

    def low_summand(x, a, k):
        x_flt64 = np.array(x, dtype='float64')
        a_flt64 = np.array(a, dtype='float64')
        if k == 1:
            expr = np.zeros_like(x)
            a_pos = a > 0
            a_neg = a < 0
            if np.any(a_pos):
                expr[a_pos] = (
                    np.log(x[a_pos])*np.log(a[a_pos]) 
                    - sp.spence(1+x_flt64[a_pos]/a_flt64[a_pos])
                    - (
                        (x[a_pos]+a[a_pos])
                        *np.log(x[a_pos]+a[a_pos]) 
                        - x[a_pos]
                    )/2
                )
            if np.any(a_neg):
                expr[a_neg] = (
                    np.log(-x[a_neg]/a[a_neg])*np.log(x[a_neg]+a[a_neg])
                    + sp.spence(-x_flt64[a_neg]/a_flt64[a_neg])
                    - (
                        (x[a_neg]+a[a_neg])*np.log(x[a_neg]+a[a_neg]) 
                        - x[a_neg]
                    )/2
                )
            return expr
        
        else:
            return (
                bern(k)*x**k/(sp.factorial(k)*k)*(
                    np.log(x + a) - x/(a*(k+1))*np.real(
                        sp.hyp2f1(
                            1, k+1, k+2, -x_flt64/a_flt64 + 0j
                        )
                    )
                )
            )

    def high_summand(x, a, k):

        x_flt64 = np.array(x, dtype='float64')
        a_flt64 = np.array(a, dtype='float64')
        inf = (x == np.inf)

        expr = np.zeros_like(x)
        expr[inf] = 0
        expr[~inf] = (
            np.exp(-k*x[~inf])/k*(
                np.log(x[~inf] + a[~inf]) 
                + exp_expn(1, k*(x[~inf] + a[~inf]))
            )
        )

        return expr

    if a.ndim == 1 and lowlim.ndim == 2:
        if lowlim.shape[1] != a.size:
            raise TypeError('The second dimension of lowlim must have the same length as a.')
        # Extend a to a 2D array.
        a = np.outer(np.ones(lowlim.shape[0]), a)
    elif a.ndim == 2 and lowlim.ndim == 1:
        if a.shape[1] != lowlim.size:
            raise TypeError('The second dimension of a must have the same length as lowlim.')
        lowlim = np.outer(np.ones(a.shape[0]), lowlim)

    # if both are 1D, then the rest of the code still works.

    integral = np.zeros(lowlim.shape, dtype='float128')
    err = np.zeros_like(integral)
    next_term = np.zeros_like(integral)

    a_is_zero = (a == 0)
    low = (lowlim < 2) & ~a_is_zero
    high = ~low & ~a_is_zero

    if np.any(a_is_zero):
        integral[a_is_zero] = F_log(lowlim[a_is_zero], 
            np.ones_like(lowlim[a_is_zero])*np.inf,
            tol=tol
        )

    if np.any(low):

        integral[low] = (
            low_summand(bound[low], a[low], 1) 
            - low_summand(lowlim[low], a[low], 1)
            + high_summand(bound[low], a[low], 1)
        )
        k_low = 2
        k_high = 2
        err_max = 10*tol

        while err_max > tol:

            next_term[low] = (
                low_summand(bound[low], a[low], k_low) 
                - low_summand(lowlim[low], a[low], k_low)
                + high_summand(bound[low], a[low], k_high)
            )
            err[low] = np.abs(
                np.divide(
                    next_term[low],
                    integral[low],
                    out = np.zeros_like(next_term[low]),
                    where = integral[low] != 0
                )
            )

            integral[low] += next_term[low]

            k_low += 2
            k_high += 1
            err_max = np.max(err[low])
            low &= (err > tol)

    if np.any(high):

        integral[high] = high_summand(lowlim[high], a[high], 1)

        k_high = 2
        err_max = 10*tol

        while err_max > tol:
            next_term[high] = high_summand(lowlim[high], a[high], k_high)
            err[high] = np.abs(
                np.divide(
                    next_term[high],
                    integral[high],
                    out = np.zeros_like(next_term[high]),
                    where = integral[high] != 0
                )
            )

            integral[high] += next_term[high]

            k_high += 1
            err_max = np.max(err[high])
            high &= (err > tol)

    return integral, err

def F_x_log_a(lowlim, a, tol=1e-10):
    """Integral of x log (x+a)/(exp(x) - 1) from lowlim to infinity.

    Parameters
    ----------
    a : ndarray
        Parameter in x log(x+a).
    lowlim : ndarray
        Lower limit of integration. 
    tol : float
        The relative tolerance to be reached. 

    Returns
    -------
    ndarray
        The resulting integral. 
    """

    bound = np.ones_like(lowlim, dtype='float128')*2.

    # Two different series to approximate this: below and above bound. 

    def low_summand(x, a, k):
        x_flt64 = np.array(x, dtype='float64')
        a_flt64 = np.array(a, dtype='float64')
        if k == 1:
            return (
                x*(
                    np.log(a+x) - 1 
                    + np.real(sp.hyp2f1(
                        1, 1, 2, -x_flt64/a_flt64 + 0j)
                    )
                )
                -x**2/8*(
                    2*np.log(a+x) - 1
                    + np.real(sp.hyp2f1(
                        1, 2, 3, -x_flt64/a_flt64 + 0j)
                    )
                )
            )
        else:
            return (
                bern(k)*x**(k+1)/(sp.factorial(k)*(k+1)**2)*(
                    (k+1)*np.log(x+a) - 1
                    + np.real(sp.hyp2f1(
                        1, k+1, k+2, -x_flt64/a_flt64 + 0j
                    ))
                )
            )

    def high_summand(x, a, k):

        x_flt64 = np.array(x, dtype='float64')
        a_flt64 = np.array(a, dtype='float64')
        inf = (x == np.inf)

        expr = np.zeros_like(x)
        expr[inf] = 0
        expr[~inf] = (
            np.exp(-k*x[~inf])/k**2*(
                (1+k*x[~inf])*np.log(x[~inf] + a[~inf])
                + (1+k*x[~inf])*exp_expn(1, k*(x[~inf] + a[~inf]))
                + exp_expn(2, k*(x[~inf] + a[~inf]))
            )
        )

        return expr

    if a.ndim == 1 and lowlim.ndim == 2:
        if lowlim.shape[1] != a.size:
            raise TypeError('The second dimension of lowlim must have the same length as a.')
        # Extend a to a 2D array.
        a = np.outer(np.ones(lowlim.shape[0]), a)
    elif a.ndim == 2 and lowlim.ndim == 1:
        if a.shape[1] != lowlim.size:
            raise TypeError('The second dimension of a must have the same length as lowlim.')
        lowlim = np.outer(np.ones(a.shape[0]), lowlim)

    # if both are 1D, then the rest of the code still works.

    integral = np.zeros(lowlim.shape, dtype='float128')
    err = np.zeros_like(integral)
    next_term = np.zeros_like(integral)

    a_is_zero = (a == 0)
    low = (lowlim < 2) & ~a_is_zero
    high = ~low & ~a_is_zero

    if np.any(a_is_zero):
        integral[a_is_zero] = F_x_log(
            lowlim[a_is_zero],
            np.ones_like(lowlim[a_is_zero])*np.inf,
            tol = tol
        )

    if np.any(low):

        integral[low] = (
            low_summand(bound[low], a[low], 1)
            - low_summand(lowlim[low], a[low], 1)
            + high_summand(bound[low], a[low], 1)
        )
        k_low = 2
        k_high = 2
        err_max = 10*tol

        while err_max > tol:

            next_term[low] = (
                low_summand(bound[low], a[low], k_low) 
                - low_summand(lowlim[low], a[low], k_low)
                + high_summand(bound[low], a[low], k_high)
            )
            err[low] = np.abs(
                np.divide(
                    next_term[low],
                    integral[low],
                    out = np.zeros_like(next_term[low]),
                    where = integral[low] != 0
                )
            )

            integral[low] += next_term[low]

            k_low += 2
            k_high += 1
            err_max = np.max(err[low])
            low &= (err > tol)

    if np.any(high):

        integral[high] = high_summand(lowlim[high], a[high], 1)

        k_high = 2
        err_max = 10*tol

        while err_max > tol:
            next_term[high] = high_summand(lowlim[high], a[high], k_high)
            err[high] = np.abs(
                np.divide(
                    next_term[high],
                    integral[high],
                    out = np.zeros_like(next_term[high]),
                    where = integral[high] != 0
                )
            )

            integral[high] += next_term[high]

            k_high += 1
            err_max = np.max(err[high])
            high &= (err > tol)

    return integral, err


# Low beta expansion functions

def Q(beta, photeng, T, as_pairs=False):
    """ Computes the Q term.

    This term is used in the beta expansion method for computing the nonrelativistic ICS spectrum. 

    Parameters
    ----------
    beta : ndarray
        Velocity of the electron. 
    photeng : ndarray
        Secondary photon energy. 
    T : float
        CMB temperature
    as_pairs : bool, optional
        If true, treats eleceng and photeng as a paired list: produces eleceng.size == photeng.size values. Otherwise, gets the spectrum at each photeng for each eleceng, returning an array of length eleceng.size*photeng.size. 

    Returns
    -------
    ndarray
        The Q term. 

    """

    eta = photeng/T

    large = (eta > 0.01)
    small = ~large

    q2_at_0 = np.zeros(eta.size)
    q4_at_0 = np.zeros(eta.size)
    q6_at_0 = np.zeros(eta.size)
    q8_at_0 = np.zeros(eta.size)

    if np.any(large):

        n = eta[large]

        q2_at_0[large] = (
            4*n**2*T**2/(1 - np.exp(-n))**2*(
                np.exp(-n)*(n - 1) + np.exp(-2*n)
            )
        )

        q4_at_0[large] = (
            8*n**2*T**2/(1 - np.exp(-n))**4*(
                np.exp(-n)*(2*n**3 - 14*n**2 + 25*n - 9)
                + np.exp(-2*n)*(8*n**3 - 50*n + 27)
                + np.exp(-3*n)*(2*n**3 + 14*n**2 + 25*n - 27)
                + 9*np.exp(-4*n)
            )
        )

        q6_at_0[large] = (
            4*n**2*T**2/(1 - np.exp(-n))**6*(
                np.exp(-n)*(
                    16*n**5 - 272*n**4 + 1548*n**3 
                    - 3540*n**2 + 3075*n - 675
                )
                + np.exp(-2*n)*(
                    416*n**5 - 2720*n**4 + 3096*n**3
                    + 7080*n**2 - 12300*n + 3375
                )
                + 6*np.exp(-3*n)*(
                    176*n**5 - 1548*n**3 + 3075*n - 1125
                )
                + 2*np.exp(-4*n)*(
                    208*n**5 + 1360*n**4 + 1548*n**3
                    - 3540*n**2 - 6150*n + 3375
                )
                + np.exp(-5*n)*(
                    16*n**5 + 272*n**4 + 1548*n**3
                    + 3540*n**2 + 3075*n - 3375
                )
                + np.exp(-6*n)*675
            )
        )

        # Computed for error
        q8_at_0[large] = (
            16*n**2*T**2/(1 - np.exp(-n))**8*(
                np.exp(-n)*(
                    16*n**7 - 496*n**6 + 5776*n**5
                    - 32144*n**4 + 90006*n**3 - 122010*n**2 
                    + 69825*n - 11025
                )
                + np.exp(-2*n)*(
                    1920*n**7 - 27776*n**6 + 138624*n**5
                    - 257152*n**4 + 488040*n**2 
                    - 418950*n + 77175
                )
                + np.exp(-3*n)*(
                    19056*n**7 - 121520*n**6 + 86640*n**5
                    + 610736*n**4 - 810054*n**3 - 610050*n**2
                    + 1047375*n - 231525
                )
                + np.exp(-4*n)*(
                    38656*n**7 - 462080*n**5 + 1440096*n**3
                    - 1396500*n + 385875
                )
                + np.exp(-5*n)*(
                    19056*n**7 + 121520*n**6 + 86640*n**5
                    - 610736*n**4 - 810054*n**3
                    + 610050*n**2 + 1047375*n - 385875
                )
                + np.exp(-6*n)*(
                    1920*n**7 + 27776*n**6 + 138624*n**5
                    + 257152*n**4 - 488040*n**2
                    - 418950*n + 231525
                )
                + np.exp(-7*n)*(
                    16*n**7 + 496*n**6 + 5776*n**5
                    + 32144*n**4 + 90006*n**3 + 122010*n**2
                    + 69825*n - 77175
                )
                + 11025*np.exp(-8*n)
            )
        )

    if np.any(small):

        p = eta[small]

        q2_at_0[small] = T**2*(
            2*p**2 + p**5/45 - p**7/1260 + p**9/37800
        )

        q4_at_0[small] = T**2*(
            36*p**2 - 68*p**3/3 + 2*p**5 
            - 89*p**7/630 + 149*p**9/18900
        )

        q6_at_0[small] = T**2*(
            1350*p**2 - 1250*p**3 + 1123*p**5/5 
            - 2381*p**7/84 + 6373*p**9/2520
        )
        # Computed for error
        q8_at_0[small] = T**2*(
            88200*p**2 - 107800*p**3 + 165844*p**5/5
            - 141679*p**7/21 + 27247*p**9/30
        )

    if as_pairs:
        term = 2*(
            q2_at_0*beta**2/2
            + q4_at_0*beta**4/24
            + q6_at_0*beta**6/720
        )
        err = 2*q8_at_0*beta**8/40320
    else:
        term = 2*(
            np.outer(beta**2, q2_at_0/2)
            + np.outer(beta**4, q4_at_0/24)
            + np.outer(beta**6, q6_at_0/720)
        )
        err = np.outer(beta**8, 2*q8_at_0/40320)

    testing = False
    if testing:
        print('***** Diagnostics for Q *****')
        print('1st Term: ', 2*q2_at_0*beta**2/2)
        print('2nd Term: ', 2*q4_at_0*beta**4/24)
        print('3rd Term: ', 2*q6_at_0*beta**6/720)
        print('Error: ', err)
        print('***** End Diagnostics for Q *****')

    return term, err

def Q_and_K(beta, photeng, T, as_pairs=False):
    """ Computes the Q and K term. 

    This term is used in the beta expansion method for computing the nonrelativistic ICS spectrum.

    Parameters
    ----------
    beta : ndarray
        Velocity of the electron. 
    photeng : ndarray
        Secondary photon energy. 
    T : float
        CMB temperature
    as_pairs : bool, optional
        If true, treats eleceng and photeng as a paired list: produces eleceng.size == photeng.size values. Otherwise, gets the spectrum at each photeng for each eleceng, returning an array of length eleceng.size*photeng.size. 

    Returns
    -------
    ndarray
        The Q and K term. 

    """
    eta = photeng/T

    large = (eta > 0.01)
    small = ~large

    q2_at_0 = np.zeros(eta.size)
    q4_at_0 = np.zeros(eta.size)
    q6_at_0 = np.zeros(eta.size)
    q8_at_0 = np.zeros(eta.size)

    k4_at_0 = np.zeros(eta.size)
    k6_at_0 = np.zeros(eta.size)
    k8_at_0 = np.zeros(eta.size)

    if np.any(large):

        n = eta[large]

        q2_at_0[large] = (
            4*n**2*T**2/(1 - np.exp(-n))**2*(
                np.exp(-n)*(n - 1) + np.exp(-2*n)
            )
        )

        q4_at_0[large] = (
            8*n**2*T**2/(1 - np.exp(-n))**4*(
                np.exp(-n)*(2*n**3 - 14*n**2 + 25*n - 9)
                + np.exp(-2*n)*(8*n**3 - 50*n + 27)
                + np.exp(-3*n)*(2*n**3 + 14*n**2 + 25*n - 27)
                + 9*np.exp(-4*n)
            )
        )

        q6_at_0[large] = (
            4*n**2*T**2/(1 - np.exp(-n))**6*(
                np.exp(-n)*(
                    16*n**5 - 272*n**4 + 1548*n**3 
                    - 3540*n**2 + 3075*n - 675
                )
                + np.exp(-2*n)*(
                    416*n**5 - 2720*n**4 + 3096*n**3
                    + 7080*n**2 - 12300*n + 3375
                )
                + 6*np.exp(-3*n)*(
                    176*n**5 - 1548*n**3 + 3075*n - 1125
                )
                + 2*np.exp(-4*n)*(
                    208*n**5 + 1360*n**4 + 1548*n**3
                    - 3540*n**2 - 6150*n + 3375
                )
                + np.exp(-5*n)*(
                    16*n**5 + 272*n**4 + 1548*n**3
                    + 3540*n**2 + 3075*n - 3375
                )
                + np.exp(-6*n)*675
            )
        )

        # Computed for error
        q8_at_0[large] = (
            16*n**2*T**2/(1 - np.exp(-n))**8*(
                np.exp(-n)*(
                    16*n**7 - 496*n**6 + 5776*n**5
                    - 32144*n**4 + 90006*n**3 - 122010*n**2 
                    + 69825*n - 11025
                )
                + np.exp(-2*n)*(
                    1920*n**7 - 27776*n**6 + 138624*n**5
                    - 257152*n**4 + 488040*n**2 
                    - 418950*n + 77175
                )
                + np.exp(-3*n)*(
                    19056*n**7 - 121520*n**6 + 86640*n**5
                    + 610736*n**4 - 810054*n**3 - 610050*n**2
                    + 1047375*n - 231525
                )
                + np.exp(-4*n)*(
                    38656*n**7 - 462080*n**5 + 1440096*n**3
                    - 1396500*n + 385875
                )
                + np.exp(-5*n)*(
                    19056*n**7 + 121520*n**6 + 86640*n**5
                    - 610736*n**4 - 810054*n**3
                    + 610050*n**2 + 1047375*n - 385875
                )
                + np.exp(-6*n)*(
                    1920*n**7 + 27776*n**6 + 138624*n**5
                    + 257152*n**4 - 488040*n**2
                    - 418950*n + 231525
                )
                + np.exp(-7*n)*(
                    16*n**7 + 496*n**6 + 5776*n**5
                    + 32144*n**4 + 90006*n**3 + 122010*n**2
                    + 69825*n - 77175
                )
                + 11025*np.exp(-8*n)
            )
        )

        k4_at_0[large] = (
            -8*n**2*T**2/(1 - np.exp(-n))**4*(
                np.exp(-n)*(2*n**3 - 10*n**2 + 7*n + 1)
                + np.exp(-2*n)*(8*n**3 - 14*n - 3)
                + np.exp(-3*n)*(2*n**3 + 10*n**2 + 7*n + 3)
                - np.exp(-4*n)
            )
        )

        k6_at_0[large] = (
            -4*n**2*T**2/(1 - np.exp(-n))**6*(
                np.exp(-n)*(
                    16*n**5 - 208*n**4 + 788*n**3 
                    -996*n**2 + 303*n + 33
                )
                + np.exp(-2*n)*(
                    416*n**5 - 2080*n**4 + 1576*n**3
                    + 1992*n**2 - 1212*n - 165
                )
                + 6*np.exp(-3*n)*(
                    176*n**5 - 788*n**3 + 303*n + 55
                )
                + 2*np.exp(-4*n)*(
                    208*n**5 + 1040*n**4 + 788*n**3
                    - 996*n**2 - 606*n - 165
                )
                + np.exp(-5*n)*(
                    16*n**5 + 208*n**4 + 788*n**3
                    + 996*n**2 + 303*n + 165
                )
                - 33*np.exp(-6*n)
            )
        )

        k8_at_0[large] = (
            -16*n**2*T**2/(1 - np.exp(-n))**8*(
                np.exp(-n)*(
                    16*n**7 - 400*n**6 + 3536*n**5
                    - 13904*n**4 + 24814*n**3 - 17958*n**2
                    + 3459*n + 309
                )
                + np.exp(-2*n)*(
                    1920*n**7 - 22400*n**6 + 84864*n**5
                    - 111232*n**4 + 71832*n**2 - 20754*n - 2163
                )
                + np.exp(-3*n)*(
                    19056*n**7 - 98000*n**6 + 53040*n**5
                    + 264176*n**4 - 223326*n**3
                    - 89790*n**2 + 51885*n + 6489
                )
                + np.exp(-4*n)*(
                    38656*n**7 - 282880*n**5 + 397024*n**3
                    - 69180*n - 10815
                )
                + np.exp(-5*n)*(
                    19056*n**7 + 98000*n**6 + 53040*n**5
                    - 264176*n**4 - 223326*n**3 + 89790*n**2 
                    + 51885*n + 10815
                )
                + np.exp(-6*n)*(
                    1920*n**7 + 22400*n**6 + 84864*n**5
                    + 111232*n**4 - 71832*n**2 - 20754*n - 6489
                )
                + np.exp(-7*n)*(
                    16*n**7 + 400*n**6 + 3536*n**5
                    + 13904*n**4 + 24814*n**3
                    + 17958*n**2 + 3459*n + 2163
                )
                - 309*np.exp(-8*n)
            )
        )

    if np.any(small):

        p = eta[small]

        q4_at_0[small] = T**2*(
            36*p**2 - 68*p**3/3 + 2*p**5 
            - 89*p**7/630 + 149*p**9/18900
        )
        q6_at_0[small] = T**2*(
            1350*p**2 - 1250*p**3 + 1123*p**5/5
            - 2381*p**7/84 + 6373*p**9/2520
        )
        q8_at_0[small] = T**2*(
            88200*p**2 - 107800*p**3 + 165844*p**5/5
            - 141679*p**7/21 + 27247*p**9/30
        )
        k4_at_0[small] = T**2*(
            4*p**2 + 4*p**3 - 46*p**5/45 
            + 59*p**7/630 - 37*p**9/6300
        )
        k6_at_0[small] = T**2*(
            66*p**2 + 90*p**3 - 193*p**5/3
            + 5309*p**7/420 - 393*p**9/280
        )
        k8_at_0[small] = T**2*(
            2472*p**2 + 4200*p**3 - 17780*p**5/3
            + 31411*p**7/15 - 15931*p**9/42
        )

    Q_term = Q(beta, photeng, T, as_pairs=as_pairs)

    if as_pairs:
        term = Q_term[0] + 2*(
            (q4_at_0 + k4_at_0)*beta**2/24
            + (q6_at_0 + k6_at_0)*beta**4/720
        )
        err = Q_term[1] + 2*(
            (q8_at_0 + k8_at_0)*beta**6/40320
        )
    else:
        term = Q_term[0] + 2*(
            np.outer(beta**2, (q4_at_0 + k4_at_0)/24)
            + np.outer(beta**4, (q6_at_0 + k6_at_0)/720)
        )
        err = Q_term[1] + 2*(
            np.outer(beta**6, (q8_at_0 + k8_at_0)/40320)
        )

    testing = False
    if testing:
        print('***** Diagnostics for Q_and_K *****')
        print('Q Term: ', Q_term[0])
        print('1st Term: ', 2*(q4_at_0 + k4_at_0)*beta**2/24)
        print('2nd Term: ', 2*(q6_at_0 + k6_at_0)*beta**4/720)
        print('Error: ', err)
        print('***** End Diagnostics for Q_and_K *****')

    return term, err

def H_and_G(beta, photeng, T, as_pairs=False):
    """ Computes the H and G term. 

    This term is used in the beta expansion method for computing the nonrelativistic ICS spectrum. 

    Parameters
    ----------
    beta : ndarray
        Velocity of the electron. 
    photeng : ndarray
        Secondary photon energy. 
    T : float
        CMB temperature
    as_pairs : bool, optional
        If true, treats eleceng and photeng as a paired list: produces eleceng.size == photeng.size values. Otherwise, gets the spectrum at each photeng for each eleceng, returning an array of length eleceng.size*photeng.size. 

    Returns
    -------
    ndarray
        The H and G term. 

    """
    eta = photeng/T

    large = (eta > 0.01)
    small = ~large

    h3_at_0 = np.zeros(eta.size)
    h5_at_0 = np.zeros(eta.size)
    h7_at_0 = np.zeros(eta.size)

    g2_at_0 = np.zeros(eta.size)
    g4_at_0 = np.zeros(eta.size)
    g6_at_0 = np.zeros(eta.size)
    g8_at_0 = np.zeros(eta.size)

    if np.any(large):

        n = eta[large]

        h3_at_0[large] = (
            2*n**2*T**2/(1 - np.exp(-n))**3*(
                np.exp(-n)*(4*n**2 - 18*n + 15)
                + 2*np.exp(-2*n)*(2*n**2 + 9*n - 15)
                + 15*np.exp(-3*n)
            )
        )

        h5_at_0[large] = (
            2*n**2*T**2/(1 - np.exp(-n))**5*(
                np.exp(-n)*(
                    16*n**4 - 200*n**3 
                    + 760*n**2 - 1020*n + 405
                )
                + 4*np.exp(-2*n)*(
                    44*n**4 - 150*n**3 
                    - 190*n**2 + 765*n - 405
                )
                + 2*np.exp(-3*n)*(
                    88*n**4 + 300*n**3 
                    - 380*n**2 - 1530*n + 1215
                )
                + 4*np.exp(-4*n)*(
                    4*n**4 + 50*n**3 
                    + 190*n**2 + 255*n - 405
                ) 
                + 405*np.exp(-5*n)
            )
        )

        h7_at_0[large] = (
            2*n**2*T**2/(1 - np.exp(-n))**7*(
                np.exp(-n)*(
                    64*n**6 - 1568*n**5 + 13776*n**4 
                    - 54600*n**3 + 100380*n**2 - 78750*n 
                    + 20475
                )
                + 2*np.exp(-2*n)*(
                    1824*n**6 - 19600*n**5 + 61992*n**4 
                    - 27300*n**3 - 150570*n**2 + 196875*n 
                    - 61425
                )
                + np.exp(-3*n)*(
                    19328*n**6 - 62720*n**5 - 137760*n**4
                    + 436800*n**3 + 200760*n**2
                    - 787500*n + 307125
                )
                + 4*np.exp(-4*n)*(
                    4832*n**6 + 15680*n**5 - 34440*n**4
                    - 109200*n**3 + 50190*n**2
                    + 196875*n - 102375
                )
                + np.exp(-5*n)*(
                    3648*n**6 + 39200*n**5 + 123984*n**4
                    + 54600*n**3 - 301140*n**2
                    - 393750*n + 307125
                )
                + 2*np.exp(-6*n)*(
                    32*n**6 + 784*n**5 + 6888*n**4
                    + 27300*n**3 + 50190*n**2
                    + 39375*n - 61425
                )
                + 20475*np.exp(-7*n)
            )
        )

        g2_at_0[large] = (-4*n**2*T**2*np.exp(-n)
            /(1 - np.exp(-n))
        )

        g4_at_0[large] = (
            -16*n**2*T**2/(1 - np.exp(-n))**3*(
                np.exp(-n) * (n**2 - 3*n + 3)
                + np.exp(-2*n) * (n**2 + 3*n - 6)
                + 3 * np.exp(-3*n)
            )
        )

        g6_at_0[large] = (
            -32*n**2*T**2/(1 - np.exp(-n))**5*(
                np.exp(-n)*(
                    2*n**4 - 20*n**3 + 70*n**2 - 90*n + 45
                )
                + 2*np.exp(-2*n)*(
                    11*n**4 - 30*n**3 - 35*n**2 + 135*n - 90
                )
                + np.exp(-3*n)*(
                    22*n**4 + 60*n**3 - 70*n**2 - 270*n + 270
                )
                + 2*np.exp(-4*n)*(
                    n**4 + 10*n**3 + 35*n**2 + 45*n - 90
                ) 
                + 45 * np.exp(-5*n)
            )
        )

        g8_at_0[large] = (
            -256*n**2*T**2/(1 - np.exp(-n))**7*(
                np.exp(-n)*(
                    n**6 - 21*n**5 + 168*n**4 - 630*n**3
                    + 1155*n**2 - 945*n + 315
                )
                + 3*np.exp(-2*n)*(
                    19*n**6 - 175*n**5 + 504*n**4 - 210*n**3
                    - 1155*n**2 + 1575*n - 630
                )
                + np.exp(-3*n)*(
                    302*n**6 - 840*n**5 - 1680*n**4 + 5040*n**3 
                    + 2310*n**2 - 9450*n + 4725
                )
                + 2*np.exp(-4*n)*(
                    151*n**6 + 420*n**5 - 840*n**4 -2520*n**3 
                    + 1155*n**2 + 4725*n - 3150
                )
                + 3*np.exp(-5*n)*(
                    19*n**6 + 175*n**5 + 504*n**4 + 210*n**3
                    - 1155*n**2 - 1575*n + 1575
                )
                + np.exp(-6*n)*(
                    n**6 + 21*n**5 + 168*n**4 + 630*n**3
                    + 1155*n**2 + 945*n - 1890
                )
                + 315*np.exp(-7*n)
            )
        )

    if np.any(small):

        p = eta[small]

        h3_at_0[small] = T**2*(
            10*p - 15*p**2 + 11*p**3/2 - 31*p**5/120 
            + 37*p**7/3024 - 103*p**9/201600
        )

        h5_at_0[small] = T**2*(
            178*p - 405*p**2 + 475*p**3/2 - 205*p**5/8
            + 6925*p**7/3024 - 703*p**9/4480
        )

        h7_at_0[small] = T**2*(
            6858*p - 20475*p**2 + 33075*p**3/2 - 26369*p**5/8 
            + 71801*p**7/144 - 101903*p**9/1920
        )

        g2_at_0[small] = T**2*(
            -4*p + 2*p**2 - p**3/3 + p**5/180 - p**7/7560 
            + p**9/302400
        )

        g4_at_0[small] = T**2*(
            -32*p + 24*p**2 - 8*p**3 + 2*p**5/5 - 19*p**7/945 
            + 11*p**9/12600
        )

        g6_at_0[small] = T**2*(
            -736*p + 720*p**2 - 360*p**3 + 38*p**5
            - 667*p**7/189 + 211*p**9/840
        )

        g8_at_0[small] = T**2*(
            -33792*p + 40320*p**2 - 26880*p**3 + 4928*p**5 
            - 6752*p**7/9 + 1228*p**9/15
        )

    if as_pairs:
        term1 = 4*(h3_at_0/6 + h5_at_0/120*beta**2)*beta**2
        term2 = (
            4*(g4_at_0/24 + g6_at_0/720*beta**2)
                *beta**2*np.sqrt(1-beta**2)
        )
        term3 = 2*g2_at_0*beta**2*(-1/2 - 1/8*beta**2)
        err = (
            4*(h7_at_0/40320 + g8_at_0/40320*np.sqrt(1-beta**2))
            + 2*g2_at_0*(-1/16)
        )*beta**6
    else: 
        term1 = (
            4*np.outer(beta**2, h3_at_0/6)
            + 4*np.outer(beta**4, h5_at_0/120)
        )

        term2 = (
            4*np.outer(beta**2*np.sqrt(1-beta**2), g4_at_0/24)
            + 4*np.outer(
                beta**4*np.sqrt(1-beta**2), g6_at_0/720
            )
        )

        term3 = (
            2*np.outer(beta**2*(-1/2 - 1/8*beta**2), g2_at_0)
        )
        err = (
            4*np.outer(beta**6, h7_at_0/40320)
            + 4*np.outer(
                beta**6*np.sqrt(1-beta**2), 
                g8_at_0/40320
            )
            + 2*np.outer(-beta**6/16, g2_at_0)
        )

    term = term1+term2+term3

    testing = False
    if testing:
        print('***** Diagnostics for H_and_G *****')
        print('1st Term: ', 4*beta**2*(
                h3_at_0/6 + np.sqrt(1-beta**2)*g4_at_0/24
            ) + 2*g2_at_0*beta**2*(-1/2)
        )
        print('2nd Term: ', 4*beta**2*(
                h5_at_0/120*beta**2
                + np.sqrt(1-beta**2)*g6_at_0/720*beta**2
            ) + 2*g2_at_0*beta**2*(-1/8*beta**2)
        )
        print('Error: ', err)
        print('***** End Diagnostics for H_and_G *****')

    return term, err
    
# Energy loss series

def F1_up_down(beta, delta, T, as_pairs=False):
    """ Computes the F1_upscatter - F1_downscatter term.

    This term is used in the small parameter expansion method for computing the ICS energy loss spectrum. 

    Parameters
    ----------
    beta : ndarray
        Velocity of the electron. 
    delta : ndarray
        Energy gained from upscattering by the secondary photon. 
    T : float
        CMB temperature
    as_pairs : bool, optional
        If true, treats eleceng and photeng as a paired list: produces eleceng.size == photeng.size values. Otherwise, gets the spectrum at each photeng for each eleceng, returning an array of length eleceng.size*photeng.size. 

    Returns
    -------
    ndarray
        The F1_upscatter - F1_downscatter term. 

    """

    eta = delta/T 

    if as_pairs:
        A_all = eta/(2*beta)
    else:
        A_all = np.outer(1/(2*beta), eta)

    large = A_all > 0.01
    small = ~large

    term_eta = np.zeros_like(A_all)
    term_eta_3 = np.zeros_like(A_all)
    term_eta_5 = np.zeros_like(A_all)

    if np.any(large):

        A = A_all[large]

        term_eta[large] = A*np.exp(-A)/(1 - np.exp(-A))

        term_eta_3[large] = (
            (A-2)*np.exp(-A) + (A+2)*np.exp(-2*A)
        )/(24 * (1 - np.exp(-A))**3)

        term_eta_5[large] = (
            A*(np.exp(-A) + 11*np.exp(-2*A) + 11*np.exp(-3*A) + np.exp(-4*A))
            - 4*(np.exp(-A) + 3*np.exp(-2*A) - 3*np.exp(-3*A) - np.exp(-4*A))
        )/(1920 * (1 - np.exp(-A))**5)

    if np.any(small):

        A = A_all[small]

        term_eta[small] = 1 - A/2 + A**2/12 - A**4/720

        term_eta_3[small] = (840 - 84*A**2 + 5*A**4)/120960

        term_eta_5[small] = (-168 + 60*A**2 - 7*A**4)/9676800

    prefac = 8*beta**3/3 + 28*beta**5/15

    if as_pairs:

        return prefac*(
            term_eta*eta + term_eta_3*eta**3 + term_eta_5*eta**5
        )

    else:

        return np.transpose(
            prefac*np.transpose(
                np.outer(term_eta, eta) 
                + np.outer(term_eta_3, eta**3)
                + np.outer(term_eta_5, eta**5)
            )
        )

def F0_up_down_diff(beta, delta, T, as_pairs=False):
    """ Computes the F0_upscatter - F0_downscatter term.

    This term is used in the small parameter expansion method for computing the ICS energy loss spectrum. 

    Parameters
    ----------
    beta : ndarray
        Velocity of the electron. 
    delta : ndarray
        Energy gained from upscattering by the secondary photon. 
    T : float
        CMB temperature
    as_pairs : bool, optional
        If true, treats eleceng and photeng as a paired list: produces eleceng.size == photeng.size values. Otherwise, gets the spectrum at each photeng for each eleceng, returning an array of length eleceng.size*photeng.size. 

    Returns
    -------
    ndarray
        The F0_upscatter - F0_downscatter term. 

    """

    eta = delta/T 

    if as_pairs:
        A_all = eta/(2*beta)
    else:
        A_all = np.outer(1/(2*beta), eta)

    large = A_all > 0.01
    small = ~large

    term_eta_2 = np.zeros_like(A_all)
    term_eta_4 = np.zeros_like(A_all)
    term_eta_6 = np.zeros_like(A_all)

    if np.any(large):

        A = A_all[large]

        term_eta_2[large] = np.exp(-A)/(1 - np.exp(-A))

        term_eta_4[large] = (
            np.exp(-A) + np.exp(-2*A)
        )/(24*(1 - np.exp(-A))**3)

        term_eta_6[large] = (
            np.exp(-A) + 11*np.exp(-2*A) + 11*np.exp(-3*A) + np.exp(-4*A)
        )/(1920*(1 - np.exp(-A))**5)

    if np.any(small):

        A = A_all[small]

        term_eta_2[small] = 1/A - 1/2 + A/12 - A**3/720 + A**5/30240

        term_eta_4[small] = 1/(12*A**3) - A/2880 + A**3/36288 - A**5/691200

        term_eta_6[small] = (
            2661120/A**5 + 440*A - 77*A**3 + 7*A**5
        )/212889600

    prefac = -2*beta**2 - beta**4

    if as_pairs:

        return prefac*(
            term_eta_2*eta**2 + term_eta_4*eta**4 + term_eta_6*eta**6
        )

    else:

        return np.transpose(
            prefac*np.transpose(
                np.outer(term_eta_2, eta**2) 
                + np.outer(term_eta_4, eta**4)
                + np.outer(term_eta_6, eta**6)
            )
        )

def F0_up_down_sum(beta, delta, T, as_pairs=False):
    """ Computes the F0_upscatter + F0_downscatter term.

    This term is used in the small parameter expansion method for computing the ICS energy loss spectrum. 

    Parameters
    ----------
    beta : ndarray
        Velocity of the electron. 
    delta : ndarray
        Energy gained from upscattering by the secondary photon. 
    T : float
        CMB temperature
    as_pairs : bool, optional
        If true, treats eleceng and photeng as a paired list: produces eleceng.size == photeng.size values. Otherwise, gets the spectrum at each photeng for each eleceng, returning an array of length eleceng.size*photeng.size. 

    Returns
    -------
    ndarray
        The F0_upscatter + F0_downscatter term. 

    """
    eta = delta/T 

    if as_pairs:
        A_all = eta/(2*beta)
    else:
        A_all = np.outer(1/(2*beta), eta)

    large = A_all > 0.01
    small = ~large

    term_eta = np.zeros_like(A_all)
    term_eta_3 = np.zeros_like(A_all)
    term_eta_5 = np.zeros_like(A_all)

    if np.any(large):

        A = A_all[large]
        term_eta[large] = -2*log_1_plus_x(-np.exp(-A))

        term_eta_3[large] = (
            np.exp(-A)/(4 * (1 - np.exp(-A))**2)
        )

        term_eta_5[large] = (
            np.exp(-A) + 4*np.exp(-2*A) + np.exp(-3*A)
        )/(192*(1 - np.exp(-A))**4)

    if np.any(small):

        A = A_all[small]

        term_eta[small] = -2*np.log(A) + A - A**2/12 + A**4/1440

        term_eta_3[small] = 1/(4*A**2) - 1/48 + A**2/960 - A**4/24192

        term_eta_5[small] = (241920/A**4 + 336 - 80*A**2 + 7*A**4)/7741440

    prefac = 2*beta**3/3 + 13*beta**5/15

    if as_pairs:

        return prefac*(
            term_eta*eta + term_eta_3*eta**3 + term_eta_5*eta**5
        )

    else:

        return np.transpose(
            prefac*np.transpose(
                np.outer(term_eta, eta) 
                + np.outer(term_eta_3, eta**3)
                + np.outer(term_eta_5, eta**5)
            )
        )

def F_inv_up_down(beta, delta, T, as_pairs=False):
    """ Computes the F_inv_upscatter - F_inv_downscatter term.

    This term is used in the small parameter expansion method for computing the ICS energy loss spectrum. 

    Parameters
    ----------
    beta : ndarray
        Velocity of the electron. 
    delta : ndarray
        Energy gained from upscattering by the secondary photon. 
    T : float
        CMB temperature
    as_pairs : bool, optional
        If true, treats eleceng and photeng as a paired list: produces eleceng.size == photeng.size values. Otherwise, gets the spectrum at each photeng for each eleceng, returning an array of length eleceng.size*photeng.size. 

    Returns
    -------
    ndarray
        The F_inv_upscatter - F_inv_downscatter term. 

    """
    eta = delta/T 

    if as_pairs:
        A_all = eta/(2*beta)
    else:
        A_all = np.outer(1/(2*beta), eta)

    large = A_all > 0.01
    small = ~large

    term_eta_3 = np.zeros_like(A_all)
    term_eta_5 = np.zeros_like(A_all)
    term_eta_7 = np.zeros_like(A_all)

    if np.any(large):

        A = A_all[large]

        term_eta_3[large] = (np.exp(-A)/A)/(1 - np.exp(-A))

        term_eta_5[large] = (
            (A**2 + 2*A + 2)*np.exp(-A) + (A**2 - 2*A - 4)*np.exp(-2*A)
            + 2*np.exp(-3*A)    
        )/(24*A**3*(1 - np.exp(-A))**3)

        term_eta_7[large] = (
            (A**4 + 4*A**3 + 12*A**2 + 24*A + 24)*np.exp(-A)
            + (11*A**4 + 12*A**3 - 12*A**2 - 72*A - 96)*np.exp(-2*A)
            + (11*A**4 - 12*A**3 - 12*A**2 + 72*A + 144)*np.exp(-3*A)
            + (A**4 - 4*A**3 + 12*A**2 - 24*A - 96)*np.exp(-4*A)
            + 24*np.exp(-5*A)
        )/(1920*A**5*(1 - np.exp(-A))**5)

    if np.any(small):

        A = A_all[small]

        term_eta_3[small] = 1/A**2 - 1/(2*A) + 1/12 - A**2/720 + A**4/30240

        term_eta_5[small] = -(
            -241920/A**4 + 40320/A**3 + 112 - 16*A**2 + A**4
        )/967680

        term_eta_7[small] = (
            1/(16*A**6) - 1/(160*A**5) + 1/2419200 
            - A**2/6451200 + A**4/54743040
        )

    prefac = beta - beta**3/2

    if as_pairs:

        return prefac*(
            term_eta_3*eta**3 + term_eta_5*eta**5 + term_eta_7*eta**7
        )

    else:

        return np.transpose(
            prefac*np.transpose( 
                + np.outer(term_eta_3, eta**3)
                + np.outer(term_eta_5, eta**5)
                + np.outer(term_eta_7, eta**7)
            )
        )

def F_inv2_up_down(beta, delta, T, as_pairs=False):
    """ Computes the F_inv2_upscatter + F_inv2_downscatter term.

    This term is used in the small parameter expansion method for computing the ICS energy loss spectrum. 

    Parameters
    ----------
    beta : ndarray
        Velocity of the electron. 
    delta : ndarray
        Energy gained from upscattering by the secondary photon. 
    T : float
        CMB temperature
    as_pairs : bool, optional
        If true, treats eleceng and photeng as a paired list: produces eleceng.size == photeng.size values. Otherwise, gets the spectrum at each photeng for each eleceng, returning an array of length eleceng.size*photeng.size. 

    Returns
    -------
    ndarray
        The F_inv2_upscatter + F_inv2_downscatter term. 

    """
    eta = delta/T 

    if as_pairs:
        A_all = eta/(2*beta)
    else:
        A_all = np.outer(1/(2*beta), eta)

    large = A_all > 0.01
    small = ~large

    term_eta_4 = np.zeros_like(A_all)
    term_eta_6 = np.zeros_like(A_all)
    term_eta_8 = np.zeros_like(A_all)

    if np.any(large):

        A = A_all[large]

        term_eta_4[large] = (np.exp(-A)/A**2)/(1 - np.exp(-A))

        term_eta_6[large] = (
            (A**2 + 4*A + 6)*np.exp(-A) + (A**2 - 4*A - 12)*np.exp(-2*A)
            + 6*np.exp(-3*A)    
        )/(24*A**4*(1 - np.exp(-A))**3)

        term_eta_8[large] = (
            (A**4 + 8*A**3 + 36*A**2 + 96*A + 120)*np.exp(-A)
            + (11*A**4 + 24*A**3 - 36*A**2 - 288*A - 480)*np.exp(-2*A)
            + (11*A**4 - 24*A**3 - 36*A**2 + 288*A + 720)*np.exp(-3*A)
            + (A**4 - 8*A**3 + 36*A**2 - 96*A - 480)*np.exp(-4*A)
            + 120*np.exp(-5*A)
        )/(1920*A**6*(1 - np.exp(-A))**5)

    if np.any(small):

        A = A_all[small]

        term_eta_4[small] = (
            1/A**3 - 1/(2*A**2) + 1/(12*A) - A/720 + A**3/30240
            - A**5/1209600
        )

        term_eta_6[small] = (
            1/(2*A**5) - 1/(8*A**4) + 1/(144*A**3) + A/120960
            - A**3/1451520 + A**5/27371520   
        )

        term_eta_8[small] = (
            3/(16*A**7) - 1/(32*A**6) + 1/(960*A**5) - A/19353600
            + A**3/109486080 - 691*A**5/830269440000
        )

    prefac = -(1/3 - beta**2/6)

    if as_pairs:

        return prefac*(
            term_eta_4*eta**4 + term_eta_6*eta**6 + term_eta_8*eta**8
        )

    else:

        return np.transpose(
            prefac*np.transpose( 
                + np.outer(term_eta_4, eta**4)
                + np.outer(term_eta_6, eta**6)
                + np.outer(term_eta_8, eta**8)
            )
        )