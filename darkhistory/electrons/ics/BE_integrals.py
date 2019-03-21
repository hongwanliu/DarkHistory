"""Integrals over the Bose-Einstein distribution."""

import numpy as np 
import scipy.special as sp

from darkhistory.utilities import log_1_plus_x
from darkhistory.utilities import check_err
from darkhistory.utilities import bernoulli as bern
from darkhistory.utilities import log_series_diff
from darkhistory.utilities import spence_series_diff
from darkhistory.utilities import exp_expn
from darkhistory.utilities import hyp2f1_func_real

from scipy.integrate import quad

def F2(a,b,tol=1e-10):
    """Definite integral of x^2/[(exp(x) - 1)]

    Parameters
    ----------
    a : ndarray
        Lower limit of integration. Can be either 1D or 2D. 
    b : ndarray
        Upper limit of integration. Can be either 1D or 2D.
    tol : float
        The relative tolerance to be reached. Default is 1e-10. 

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
            return x**2/2 - x**3/6
        else:
            return(
                bern(k)*x**(k+2)/(sp.factorial(k)*(k+2))
            )
            # B_n for n odd, n > 1 is zero.

    def high_summand(x, k):

        inf = (x == np.inf)
        expr = np.zeros_like(x) 
        # gammaincc(n,x) = 1/gamma(n) * int_x^\infty t^{n-1}exp(-t) dt
        expr[~inf] = 2*sp.gammaincc(
            3, k*np.array(x[~inf], dtype='float64')
        )/k**3

        return expr 

    if a.ndim == 1 and b.ndim == 2:
        if b.shape[1] != a.size:
            raise TypeError('The second dimension of b must have the same length as a')
        # Extend a to a 2D array. 
        a = np.outer(np.ones(b.shape[0], dtype='float128'), a)
    elif a.ndim == 2 and b.ndim == 1:
        if a.shape[1] != b.size:
            raise TypeError('The second dimension of a must have the same length as b')
        b = np.outer(np.ones(a.shape[0], dtype='float128'), b)

    # if both are 1D, the rest of the code still works. 

    integral  = np.zeros(a.shape, dtype='float128')
    err       = np.zeros_like(integral)
    next_term = np.zeros_like(integral)

    both_low  = (a < bound) & (b <  bound)
    low_high  = (a < bound) & (b >= bound)
    both_high = (a > bound) & (b >  bound)

    # Both low

    if np.any(both_low):

        # Initialize first term of each sum for either limit, and set integral to that value.
        low_sum_a = low_summand(a[both_low], 1)
        low_sum_b = low_summand(b[both_low], 1)
        integral[both_low] = low_sum_b - low_sum_a

        # Index of summand.
        k_low = 2
        # Initialize error. 
        err_max = 10*tol 

        while err_max > tol:
            # Get next term.
            next_term[both_low] = (
                low_summand(b[both_low], k_low)
                - low_summand(a[both_low], k_low)
            )
            # Estimate the error
            err[both_low] = np.abs(
                np.divide(
                    next_term[both_low],
                    integral[both_low],
                    out = np.zeros_like(next_term[both_low]),
                    where = integral[both_low] != 0
                )
            )

            # Add the next term in the series to the integral.
            integral [both_low] += next_term[both_low]

            # Increment k_low. Increment by 2 since B_n is zero for odd n > 1. 
            k_low += 2

            # Set the errors. Only propagate parts where the errors are large to the next step.
            err_max = np.max(err[both_low])
            both_low &= (err > tol)

    # a low b high
    if np.any(low_high):

        # Evaluate the definite integral from a to 2, and then 2 to b.

        low_sum_a     =  low_summand(a[low_high], 1)
        high_sum_b    = high_summand(b[low_high], 1) 
        low_sum_bound =  low_summand(bound, 1)

        # Exact integral from 2 to infinity.
        int_bound_inf = np.float128(1.417948518338124870521)
        # First term in integral from a to bound
        int_a_bound   = low_sum_bound - low_sum_a
        # First term in integral from bound to infinity
        int_bound_b   = int_bound_inf - high_sum_b

        # Initialize the integral
        integral[low_high] = int_a_bound + int_bound_b

        # Counters, error estimate
        k_low  = 2
        k_high = 2

        err_max = 10*tol

        # Arrays for next term
        next_term_a_bound = np.zeros_like(integral)
        next_term_bound_b = np.zeros_like(integral)

        while err_max > tol:
            next_term_a_bound[low_high] = (
                low_summand(bound, k_low)
                - low_summand(a[low_high], k_low)
            )
            # Only need to compute the next term to correct high_sum_b, since int_bound_inf is exact. 
            next_term_bound_b[low_high] = (
                -high_summand(b[low_high], k_high)
            )

            next_term[low_high] = (
                next_term_a_bound[low_high]
                + next_term_bound_b[low_high]
            )

            # Error estimate
            err[low_high] = np.abs(
                np.divide(
                    next_term[low_high], 
                    integral[low_high],
                    out = np.zeros_like(next_term[low_high]),
                    where = integral[low_high] != 0
                )
            )

            # Add the next terms to the current integral.
            integral[low_high] += next_term[low_high]

            k_low  += 2
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



def F1(a,b,epsrel=0):
    """Definite integral of x/[(exp(x) - 1)]. 

    This is computed from the indefinite integral

    .. math::

        \\int dx \\frac{x}{e^x - 1} = x \\log\\left(1 - e^{-x} \\right)
        - \\text{Li}_2\\left(e^{-x}\\right) = 
        x \\log\\left(1 - e^{-x} \\right) - 
        \\text{Sp}\\left( 1 - e^{-x} \\right) + \\frac{\\pi^2}{6} \\,,
    
    where Sp is Spence's function, as implemented in ``scipy.special.spence``. 

    Parameters
    ----------
    a : ndarray
        Lower limit of integration. Can be either 1D or 2D.  
    b : ndarray
        Upper limit of integration. Can be either 1D or 2D.
    epsrel : float
        Target relative error associated with series expansion. If zero, then the error is not computed. Default is 0. If the error is larger than ``epsrel``, then the Taylor expansions used here are insufficient. Higher order terms can be added very easily, however.

    Returns
    -------
    float
        The resulting integral.

    Notes
    -----
    For a or b > 0.01, the exact analytic expression is used, whereas below that we use a series expansion. This avoids numerical errors due to computation of log(1 - exp(-x)) and likewise in the `spence` function. Note that `scipy.special.spence` can only take `float64` numbers, so downcasting is necessary for 0.01 < x < 3. 

    See Also
    ---------
    :func:`.log_1_plus_x`, :func:`.spence_series_diff`
    
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
    """Definite integral of (1/x)/(exp(x) - 1). 

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

        inf = (x == np.inf)
        expr = np.zeros_like(x)
        expr[~inf] = sp.expn(1, k*np.array(x[~inf], dtype='float64'))

        return expr

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

def F_inv_a(lowlim, a, tol=1e-10):
    """Integral of 1/((x+a)(exp(x) - 1)) from lowlim to infinity. 

    Parameters
    ----------
    a : ndarray
        Parameter in (x+a).
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
            expr = np.log(x)/a - np.log(x+a)/a - 0.5*x*(
                1/a - x/(2*a**2)
                *hyp2f1_func_real(1, -x/a)
            )
            # expr = np.log(x)/a - np.log(x+a)/a - 0.5*x*(
            #     1/a - x/(2*a**2)
            #     *np.real(sp.hyp2f1(1, 2, 3, -x_flt64/a_flt64 + 0j))
            # )
            return expr
        else:
            return bern(k)*x**k/(sp.factorial(k)*k)*(
                1/a - k*x/((k+1)*a**2)*hyp2f1_func_real(k, -x/a)
            )
            # return bern(k)*x**k/(sp.factorial(k)*k)*(
            #     1/a - k*x/((k+1)*a**2)*np.real(
            #         sp.hyp2f1(1, k+1, k+2, -x_flt64/a_flt64 + 0j)
            #     )
            # )

    def high_summand(x, a, k):

        x_flt64 = np.array(x, dtype='float64')
        a_flt64 = np.array(a, dtype='float64')
        inf = (x == np.inf)

        expr = np.zeros_like(x)
        expr[inf] = 0
        expr[~inf] = np.exp(-k*x[~inf])*exp_expn(
            1, k*(x[~inf] + a[~inf])
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
    low       = (lowlim < 2) & ~a_is_zero
    high      = ~low & ~a_is_zero

    if np.any(a_is_zero):
        integral[a_is_zero] = F_inv(
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
        k_low  = 2
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

def F_inv_n(a,b,n,tol=1e-10):
    """Definite integral of (1/x**n)/(exp(x) - 1)

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

    bound = np.float128(2.) 

    # Two different series to approximate this: below and above bound.

    def low_summand(x, k):
        if k == 1:
            init_sum = 0
            for j in np.arange(n):
                init_sum += bern(j)/sp.factorial(j)*x**(j-n)/(j-n)
            init_sum += bern(n)/sp.factorial(n)*np.log(x)
            return init_sum
        else:
            # B_n for n odd, n > 1 is zero.
            if np.mod(k+n-1, 2) == 0:
                return(
                    bern(k+n-1)/sp.factorial(k+n-1)*x**(k-1)/(k-1)
                )
            else:
                return(
                    bern(k+n)/sp.factorial(k+n)*x**k/k
                )

    def high_summand(x, k):
        
        inf = (x == np.inf)
        expr = np.zeros_like(x)
        expr[~inf] = (
            sp.expn(n, k*np.array(x[~inf], dtype='float64'))/x[~inf]**(n-1)
        )

        return expr

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

    integral = np.zeros_like(a, dtype='float128')
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
        int_bound_inf = quad(
            lambda x: 1/(x**n*(np.exp(x) - 1)),
            bound, np.inf, epsabs = 1e-16, epsrel=1e-16
        )[0]

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

def F_inv_3(a,b,tol=1e-10):
    """Definite integral of (1/x**3)/(exp(x) - 1). 

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
            return -1/(3*x**3) + 1/(4*x**2) - 1/(12*x)
        else:
            return (
                bern(k+2)*(x**(k-1))/(sp.factorial(k+2)*(k-1))
            )
            # B_n for n odd, n > 1 is zero.


    def high_summand(x, k):
        inf = (x == np.inf)
        expr = np.zeros_like(x)
        expr[~inf] = (
            sp.expn(3, k*np.array(x[~inf], dtype='float64'))/x[~inf]**2
        )

        return expr

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

    integral = np.zeros_like(a, dtype='float128')
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
        int_bound_inf = np.float128(0.0083036361900336)
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

def F_inv_5(a,b,tol=1e-10):
    """Definite integral of (1/x**5)/(exp(x) - 1). 

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
            return -1/(5*x**5) + 1/(8*x**4) - 1/(36*x**3) + 1/(720*x)
        else:
            return (
                bern(k+4)*(x**(k-1))/(sp.factorial(k+4)*(k-1))
            )
            # B_n for n odd, n > 1 is zero.


    def high_summand(x, k):
        inf = (x == np.inf)
        expr = np.zeros_like(x)
        expr[~inf] = (
            sp.expn(5, k*np.array(x[~inf], dtype='float64'))/x[~inf]**4
        )
        
        return expr

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

    integral = np.zeros_like(a, dtype='float128')
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
        int_bound_inf = np.float128(0.001483878955697788)
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
                    np.log(x + a) 
                    - x/(a*(k+1))*hyp2f1_func_real(k, -x/a)
                )
            )
            # return (
            #     bern(k)*x**k/(sp.factorial(k)*k)*(
            #         np.log(x + a) - x/(a*(k+1))*np.real(
            #             sp.hyp2f1(
            #                 1, k+1, k+2, -x_flt64/a_flt64 + 0j
            #             )
            #         )
            #     )
            # )

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

    bound = np.ones_like(lowlim, dtype='float128')*2

    # Two different series to approximate this: below and above bound. 

    def low_summand(x, a, k):
        # x_flt64 = np.array(x, dtype='float64')
        # a_flt64 = np.array(a, dtype='float64')
        if k == 1:
            return (
                x*(
                    np.log(a+x) - 1 
                    + hyp2f1_func_real(0, -x/a)
                )
                -x**2/8*(
                    2*np.log(a+x) - 1
                    + hyp2f1_func_real(1, -x/a)
                )
                # x*(
                #     np.log(a+x) - 1 
                #     + np.real(sp.hyp2f1(
                #         1, 1, 2, -x/a + 0j)
                #     )
                # )
                # -x**2/8*(
                #     2*np.log(a+x) - 1
                #     + np.real(sp.hyp2f1(
                #         1, 2, 3, -x/a + 0j)
                #     )
                # )
            )
        else:
            return (
                bern(k)*x**(k+1)/(sp.factorial(k)*(k+1)**2)*(
                    (k+1)*np.log(x+a) - 1
                    + hyp2f1_func_real(k, -x/a)
                )
            )
            # return (
            #     bern(k)*x**(k+1)/(sp.factorial(k)*(k+1)**2)*(
            #         (k+1)*np.log(x+a) - 1
            #         + np.real(sp.hyp2f1(
            #             1, k+1, k+2, -x/a + 0j
            #         ))
            #     )
            # )

    def high_summand(x, a, k):

        # x_flt64 = np.array(x, dtype='float64')
        # a_flt64 = np.array(a, dtype='float64')
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

    bound_flt64 = np.array(bound, dtype='float64')
    a_flt64 = np.array(a, dtype='float64')
    lowlim_flt64 = np.array(lowlim, dtype='float64')


    if np.any(low):
        
        integral[low] = (
            low_summand(bound_flt64[low], a_flt64[low], 1)
            - low_summand(lowlim_flt64[low], a_flt64[low], 1)
            + high_summand(bound_flt64[low], a_flt64[low], 1)
        )
        k_low = 2
        k_high = 2
        err_max = 10*tol

        while err_max > tol:

            next_term[low] = (
                low_summand(
                    bound_flt64[low], a_flt64[low], k_low
                ) 
                - low_summand(
                    lowlim_flt64[low], a_flt64[low], k_low
                )
                + high_summand(
                    bound_flt64[low], a_flt64[low], k_high
                )
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

        integral[high] = high_summand(
            lowlim_flt64[high], a_flt64[high], 1
        )

        k_high = 2
        err_max = 10*tol

        while err_max > tol:
            next_term[high] = high_summand(
                lowlim_flt64[high], a_flt64[high], k_high
            )
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