"""Nonrelativistic series expansions for ICS spectrum."""

import numpy as np 
import scipy.special as sp

from darkhistory.utilities import log_1_plus_x
from darkhistory.utilities import diff_pow
from darkhistory.utilities import check_err
from darkhistory.utilities import bernoulli as bern
from darkhistory.utilities import div_ignore_by_zero

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
        low = (x < lowlim)
        high = (x > upplim)
        gen = ~(low | high)
        expr = np.zeros(x.size)
        
        # Two different series for small and large x limit.

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

    integral = np.zeros(a.shape)

    both_low = (a < lowlim) & (b < lowlim)
    both_high = (a > upplim) & (b > upplim)
    
    if np.any(both_low):

        # Use diff_pow to compute differences in powers accurately.

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

        # Use a series for the spence function.

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
        
        return log_1_plus_x(-np.exp(-x))

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

    integral = np.zeros(a.shape)

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

def F_inv(a,b,test,tol=1e-10):
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

    def low_summand(x):
        k = 1
        while True:
            if k == 1:
                next_term = -1/x - np.log(x)/2
                k += 1
            else:
                next_term = (
                    bern(k)*(x**(k-1))/
                    (sp.factorial(k)*(k-1))
                )
                # B_n for n odd, n > 1 is zero.
                k += 2
            yield next_term

    def high_summand(x):
        k = 1
        while True:
            # sp.expn does not support float128.
            next_term = sp.expn(1, k*np.array(x, dtype='float64'))
            k += 1
            yield next_term

    err = 10*tol

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

    integral = np.zeros(a.shape)

    both_low  = (a < bound) & (b <  bound)
    low_high  = (a < bound) & (b >= bound)
    both_high = (a > bound) & (b >  bound)

    # Both low

    if np.any(both_low):

        low_sum_a = low_summand(a[both_low])
        low_sum_b = low_summand(b[both_low])
        integral[both_low] = next(low_sum_b) - next(low_sum_a)

        while err > tol:
            next_term = next(low_sum_b) - next(low_sum_a)
            err = np.max(np.abs(
                div_ignore_by_zero(
                    next_term, 
                    integral[both_low], 
                    0
                )
            ))
            integral[both_low] += next_term

    err = 10*tol

    # a low b high

    if np.any(low_high):

        # Evaluate the definite integral from a to 2, and then 2 to b.

        low_sum_a = low_summand(a[low_high])
        high_sum_b = high_summand(b[low_high])
        low_sum_bound = low_summand(bound)

        # Exact integral from 2 to infinity.
        int_bound_inf = np.float128(0.053082306482669888568)
        int_a_bound = next(low_sum_bound) - next(low_sum_a)
        int_bound_b = int_bound_inf - next(high_sum_b)
        integral[low_high] = int_a_bound + int_bound_b

        while err > tol:
            next_term_a_bound = next(low_sum_bound) - next(low_sum_a)
            # Only need to compute the next term for the b to inf integral.
            next_term_bound_b = - next(high_sum_b)
            next_term = next_term_a_bound + next_term_bound_b
            err = np.max(np.abs(
                div_ignore_by_zero(
                    next_term, 
                    integral[low_high], 
                    0
                )
            ))
            integral[low_high] += next_term

    err = 10*tol

    # Both high

    if np.any(both_high):

        high_sum_a = high_summand(a[both_high])
        high_sum_b = high_summand(b[both_high])
        integral[both_high] = next(high_sum_a) - next(high_sum_b)

        while err > tol:
            next_term = next(high_sum_a) - next(high_sum_b)
            err = np.max(np.abs(
                div_ignore_by_zero(
                    next_term,
                    integral[both_high],
                    0
                )
            ))
            integral[both_high] += next_term

    return integral

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

    def low_summand(x):
        k = 1
        while True:
            if k == 1:
                next_term = (1/2)*(np.log(x)**2) - (x/2)*(np.log(x) - 1)
                k += 1
            else:
                next_term = (
                    bern(k)*(x**k)/
                    (sp.factorial(k)*k**2)*(k*np.log(x) - 1)
                )
                # B_n for n odd, n > 1 is zero.
                k += 2
            yield next_term

    def high_summand(x):
        k = 1
        while True:
            # sp.expn does not support float128.
            next_term = (1/k)*(
                np.exp(-k*x)*np.log(x) + 
                sp.expn(1, k*np.array(x, dtype='float64'))
            )
            k += 1
            yield next_term

    err = 10*tol

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

    integral = np.zeros(a.shape)

    both_low  = (a < bound) & (b <  bound)
    low_high  = (a < bound) & (b >= bound)
    both_high = (a > bound) & (b >  bound)

    # Both low

    if np.any(both_low):

        low_sum_a = low_summand(a[both_low])
        low_sum_b = low_summand(b[both_low])
        integral[both_low] = next(low_sum_b) - next(low_sum_a)

        while err > tol:
            next_term = next(low_sum_b) - next(low_sum_a)
            err = np.max(np.abs(
                div_ignore_by_zero(
                    next_term, 
                    integral[both_low],
                    0
                )
            ))
            integral[both_low] += next_term

    err = 10*tol

    # a low b high

    if np.any(low_high):

        # Evaluate the definite integral from a to 2, and then 2 to b.

        low_sum_a = low_summand(a[low_high])
        high_sum_b = high_summand(b[low_high])
        low_sum_bound = low_summand(bound)

        # Exact integral from 2 to infinity.
        int_bound_inf = np.float128(0.15171347859984083704)
        int_a_bound = next(low_sum_bound) - next(low_sum_a)
        int_bound_b = int_bound_inf - next(high_sum_b)
        integral[low_high] = int_a_bound + int_bound_b

        while err > tol:
            next_term_a_bound = next(low_sum_bound) - next(low_sum_a)
            # Only need to compute the next term for the b to inf integral.
            next_term_bound_b = - next(high_sum_b)
            next_term = next_term_a_bound + next_term_bound_b
            err = np.max(np.abs(
                div_ignore_by_zero(
                    next_term, 
                    integral[low_high], 
                    0
                )
            ))
            integral[low_high] += next_term

    err = 10*tol

    # Both high

    if np.any(both_high):

        high_sum_a = high_summand(a[both_high])
        high_sum_b = high_summand(b[both_high])
        integral[both_high] = next(high_sum_a) - next(high_sum_b)

        while err > tol:
            next_term = next(high_sum_a) - next(high_sum_b)
            err = np.max(np.abs(
                div_ignore_by_zero(
                    next_term, 
                    integral[both_high],
                    0
                )
            ))
            integral[both_high] += next_term

    return integral

# Low beta expansion functions

def Q(beta, photeng, T, as_pairs=False):

    eta = photeng/T

    large = (eta > 0.01)
    small = ~large

    q2_at_0 = np.zeros(eta.size)
    q4_at_0 = np.zeros(eta.size)
    q6_at_0 = np.zeros(eta.size)
    q8_at_0 = np.zeros(eta.size)

    if np.any(large):

        q2_at_0[large] = (
            4*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**2*(
                np.exp(eta[large])*(eta[large] - 1) + 1
            )
        )

        q4_at_0[large] = (
            8*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**4*(
                np.exp(2*eta[large])*(8*eta[large]**3 
                    + (4*eta[large]**3 + 50*eta[large] - 36)
                        *np.cosh(eta[large])
                    + 2*(9 - 14*eta[large]**2)
                        *np.sinh(eta[large]) 
                    -50*eta[large] + 27
                ) + 9
            )
        )

        q6_at_0[large] = (
            4*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**6*(
                np.exp(3*eta[large])*(
                    -16*eta[large]**2*np.sinh(eta[large])*(
                        340*eta[large]**2 
                        + (68*eta[large]**2 + 885)
                            *np.cosh(eta[large]) - 885
                    )
                    +3*(
                        352*eta[large]**5 - 3096*eta[large]**3 
                        + 6150*eta[large] 
                        - 1125*np.sinh(eta[large]) 
                        + 900*np.sinh(2*eta[large]) - 2250
                    )
                    +np.cosh(eta[large])*(
                        832*eta[large]**5 + 6192*eta[large]**3 
                        - 24600*eta[large] + 10125
                    )
                    +np.cosh(2*eta[large])*(
                        32*eta[large]**5 + 3096*eta[large]**3 
                        + 6150*eta[large] - 4050
                    )   
                ) +675
            )
        )

        # Computed for error
        q8_at_0[large] = (
            16*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**8*(
                np.exp(7*eta[large])*(
                    16*eta[large]**7 - 496*eta[large]**6 
                    + 5776*eta[large]**5
                    - 32144*eta[large]**4 + 90006*eta[large]**3
                    - 122010*eta[large]**2 + 69825*eta[large] 
                    - 11025
                )
                + np.exp(6*eta[large])*(
                    1920*eta[large]**7 - 27776*eta[large]**6 
                    + 138624*eta[large]**5
                    - 257152*eta[large]**4 
                    + 488040*eta[large]**2
                    - 418950*eta[large] + 77175
                )
                + np.exp(5*eta[large])*(
                    19056*eta[large]**7 - 121520*eta[large]**6 
                    + 86640*eta[large]**5
                    + 610736*eta[large]**4 
                    - 810054*eta[large]**3 
                    - 610050*eta[large]**2
                    + 1047375*eta[large] - 231525
                )
                + np.exp(4*eta[large])*(
                    38656*eta[large]**7 - 462080*eta[large]**5 
                    + 1440096*eta[large]**3
                    - 1396500*eta[large] + 385875
                )
                + np.exp(3*eta[large])*(
                    19056*eta[large]**7 + 121520*eta[large]**6 
                    + 86640*eta[large]**5
                    - 610736*eta[large]**4 
                    - 810054*eta[large]**3
                    + 610050*eta[large]**2 + 1047375*eta[large]
                    - 385875
                )
                + np.exp(2*eta[large])*(
                    1920*eta[large]**7 + 27776*eta[large]**6 
                    + 138624*eta[large]**5
                    + 257152*eta[large]**4 
                    - 488040*eta[large]**2
                    - 418950*eta[large] + 231525
                )
                + np.exp(eta[large])*(
                    16*eta[large]**7 + 496*eta[large]**6 
                    + 5776*eta[large]**5
                    + 32144*eta[large]**4 + 90006*eta[large]**3
                    + 122010*eta[large]**2
                    + 69825*eta[large] - 77175
                )
                + 11025
            )
        )

    if np.any(small):

        q2_at_0[small] = T**2*(
            2*eta[small]**2 + eta[small]**5/45 
            - eta[small]**7/1260 + eta[small]**9/37800
        )

        q4_at_0[small] = T**2*(
            36*eta[small]**2 - 68*eta[small]**3/3 
            + 2*eta[small]**5 
            - 89*eta[small]**7/630 + 149*eta[small]**9/18900
        )

        q6_at_0[small] = T**2*(
            1350*eta[small]**2 - 1250*eta[small]**3 
            + 1123*eta[small]**5/5 
            - 2381*eta[small]**7/84 + 6373*eta[small]**9/2520
        )
        # Computed for error
        q8_at_0[small] = T**2*(
            88200*eta[small]**2 - 107800*eta[small]**3 
            + 165844*eta[small]**5/5
            - 141679*eta[small]**7/21 + 27247*eta[small]**9/30
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

        q2_at_0[large] = (
            4*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**2*(
                np.exp(eta[large])*(eta[large] - 1) + 1
            )
        )

        q4_at_0[large] = (
            8*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**4*(
                np.exp(2*eta[large])*(
                    8*eta[large]**3 
                    + (4*eta[large]**3 + 50*eta[large] - 36)
                        *np.cosh(eta[large])
                    + 2*(9 - 14*eta[large]**2)
                        *np.sinh(eta[large]) 
                    -50*eta[large] + 27
                ) + 9
            )
        )

        q6_at_0[large] = (
            4*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**6*(
                np.exp(3*eta[large])*(
                    -16*eta[large]**2*np.sinh(eta[large])*(
                        340*eta[large]**2 
                        + (68*eta[large]**2 + 885)
                            *np.cosh(eta[large]) 
                        - 885
                    )
                    +3*(
                        352*eta[large]**5 - 3096*eta[large]**3 
                        + 6150*eta[large] 
                        - 1125*np.sinh(eta[large]) 
                        + 900*np.sinh(2*eta[large]) - 2250
                    )
                    +np.cosh(eta[large])*(
                        832*eta[large]**5 + 6192*eta[large]**3 
                        - 24600*eta[large] + 10125
                    )
                    +np.cosh(2*eta[large])*(
                        32*eta[large]**5 + 3096*eta[large]**3 
                        + 6150*eta[large] - 4050
                    )   
                ) +675
            )
        )

        # Computed for error
        q8_at_0[large] = (
            16*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**8*(
                np.exp(7*eta[large])*(
                    16*eta[large]**7 - 496*eta[large]**6 
                    + 5776*eta[large]**5 - 32144*eta[large]**4 
                    + 90006*eta[large]**3 - 122010*eta[large]**2
                    + 69825*eta[large] - 11025
                )
                + np.exp(6*eta[large])*(
                    1920*eta[large]**7 - 27776*eta[large]**6 
                    + 138624*eta[large]**5 
                    - 257152*eta[large]**4 
                    + 488040*eta[large]**2 - 418950*eta[large] 
                    + 77175
                )
                + np.exp(5*eta[large])*(
                    19056*eta[large]**7 - 121520*eta[large]**6 
                    + 86640*eta[large]**5 + 610736*eta[large]**4
                    - 810054*eta[large]**3 
                    - 610050*eta[large]**2 + 1047375*eta[large] 
                    - 231525
                )
                + np.exp(4*eta[large])*(
                    38656*eta[large]**7 - 462080*eta[large]**5 
                    + 1440096*eta[large]**3
                    - 1396500*eta[large] + 385875
                )
                + np.exp(3*eta[large])*(
                    19056*eta[large]**7 + 121520*eta[large]**6 
                    + 86640*eta[large]**5 - 610736*eta[large]**4
                    - 810054*eta[large]**3
                    + 610050*eta[large]**2 + 1047375*eta[large]
                    - 385875
                )
                + np.exp(2*eta[large])*(
                    1920*eta[large]**7 + 27776*eta[large]**6 
                    + 138624*eta[large]**5
                    + 257152*eta[large]**4 
                    - 488040*eta[large]**2
                    - 418950*eta[large] + 231525
                )
                + np.exp(eta[large])*(
                    16*eta[large]**7 + 496*eta[large]**6 
                    + 5776*eta[large]**5
                    + 32144*eta[large]**4 + 90006*eta[large]**3 
                    + 122010*eta[large]**2
                    + 69825*eta[large] - 77175
                )
                + 11025
            )
        )

        k4_at_0[large] = (
            8*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**4*(
                np.exp(2*eta[large])*(
                    -8*eta[large]**3 
                    - 2*(2*eta[large]**3 + 7*eta[large] + 2)
                        *np.cosh(eta[large])
                    +(20*eta[large]**2 + 2)
                        *np.sinh(eta[large]) 
                    +14*eta[large] + 3
                ) +1
            )
        )

        k6_at_0[large] = (
            4*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**6*(
                -np.exp(3*eta[large])*(
                    6*(176*eta[large]**5 - 788*eta[large]**3 
                        + 303*eta[large] + 55
                    )
                    + 2*(16*eta[large]**5 + 788*eta[large]**3 
                        + 303*eta[large] + 99
                    ) * np.cosh(2*eta[large])
                    + (-4160*eta[large]*4 
                        + 3984*eta[large]**2 + 165
                    ) * np.sinh(eta[large])
                    + (832*eta[large]**5 + 3152*eta[large]**3
                        -8*(108*eta[large]**4 
                            + 498*eta[large]**2 + 33
                        )*np.sinh(eta[large])
                        -2424*eta[large] - 495
                    ) * np.cosh(eta[large])
                ) + 33
            )
        )

        k8_at_0[large] = (
            -16*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**8*(
                np.exp(7*eta[large])*(
                    16*eta[large]**7 - 400*eta[large]**6 
                    + 3536*eta[large]**5
                    - 13904*eta[large]**4 + 24814*eta[large]**3
                    - 17958*eta[large]**2
                    + 3459*eta[large] + 309
                )
                + np.exp(6*eta[large])*(
                    1920*eta[large]**7 - 22400*eta[large]**6 
                    + 84864*eta[large]**5
                    - 111232*eta[large]**4 + 71832*eta[large]**2
                    - 20754*eta[large] - 2163
                )
                + np.exp(5*eta[large])*(
                    19056*eta[large]**7 - 98000*eta[large]**6 
                    + 53040*eta[large]**5
                    + 264176*eta[large]**4 
                    - 223326*eta[large]**3
                    - 89790*eta[large]**2 + 51885*eta[large] 
                    + 6489
                )
                + np.exp(4*eta[large])*(
                    38656*eta[large]**7 - 282880*eta[large]**5 
                    + 397024*eta[large]**3
                    - 69180*eta[large] - 10815
                )
                + np.exp(3*eta[large])*(
                    19056*eta[large]**7 + 98000*eta[large]**6 
                    + 53040*eta[large]**5
                    - 264176*eta[large]**4 
                    - 223326*eta[large]**3
                    + 89790*eta[large]**2 
                    + 51885*eta[large] + 10815
                )
                + np.exp(2*eta[large])*(
                    1920*eta[large]**7 + 22400*eta[large]**6 
                    + 84864*eta[large]**5
                    + 111232*eta[large]**4 - 71832*eta[large]**2
                    - 20754*eta[large] - 6489
                )
                + np.exp(eta[large])*(
                    16*eta[large]**7 + 400*eta[large]**6 
                    + 3536*eta[large]**5
                    + 13904*eta[large]**4 + 24814*eta[large]**3
                    + 17958*eta[large]**2
                    + 3459*eta[large] + 2163
                )
                + 309
            )
        )

    if np.any(small):
        q4_at_0[small] = T**2*(
            36*eta[small]**2 - 68*eta[small]**3/3 
            + 2*eta[small]**5 
            - 89*eta[small]**7/630 + 149*eta[small]**9/18900
        )
        q6_at_0[small] = T**2*(
            1350*eta[small]**2 - 1250*eta[small]**3 
            + 1123*eta[small]**5/5
            - 2381*eta[small]**7/84 + 6373*eta[small]**9/2520
        )
        q8_at_0[small] = T**2*(
            88200*eta[small]**2 - 107800*eta[small]**3 + 165844*eta[small]**5/5
            - 141679*eta[small]**7/21 + 27247*eta[small]**9/30
        )
        k4_at_0[small] = T**2*(
            4*eta[small]**2 + 4*eta[small]**3 
            - 46*eta[small]**5/45 
            + 59*eta[small]**7/630 - 37*eta[small]**9/6300
        )
        k6_at_0[small] = T**2*(
            66*eta[small]**2 + 90*eta[small]**3 
            - 193*eta[small]**5/3
            + 5309*eta[small]**7/420 - 393*eta[small]**9/280
        )
        k8_at_0[small] = T**2*(
            2472*eta[small]**2 + 4200*eta[small]**3 
            - 17780*eta[small]**5/3
            + 31411*eta[small]**7/15 - 15931*eta[small]**9/42
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

        h3_at_0[large] = (
            2*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**3*(
                2*np.exp(eta[large])
                    *(2*eta[large]**2 + 9*eta[large] - 15)
                + np.exp(2*eta[large])
                    *(4*eta[large]**2 - 18*eta[large] + 15)
                +15
            )
        )

        h5_at_0[large] = (
            2*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**5*(
                np.exp(4*eta[large])*(
                    16*eta[large]**4 - 200*eta[large]**3 
                    + 760*eta[large]**2 - 1020*eta[large] + 405
                )
                + 4*np.exp(3*eta[large])*(
                    44*eta[large]**4 - 150*eta[large]**3 
                    - 190*eta[large]**2 + 765*eta[large] - 405
                )
                + 2*np.exp(2*eta[large])*(
                    88*eta[large]**4 + 300*eta[large]**3 
                    - 380*eta[large]**2 - 1530*eta[large] + 1215
                )
                + 4*np.exp(eta[large])*(
                    4*eta[large]**4 + 50*eta[large]**3 
                    + 190*eta[large]**2 + 255*eta[large] - 405
                ) + 405
            )
        )

        h7_at_0[large] = (
            2*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**7*(
                np.exp(6*eta[large])*(
                    64*eta[large]**6 - 1568*eta[large]**5 
                    + 13776*eta[large]**4 - 54600*eta[large]**3 
                    + 100380*eta[large]**2 - 78750*eta[large] 
                    + 20475
                )
                + 2*np.exp(5*eta[large])*(
                    1824*eta[large]**6 - 19600*eta[large]**5 
                    + 61992*eta[large]**4 - 27300*eta[large]**3 
                    - 150570*eta[large]**2 + 196875*eta[large] 
                    - 61425
                )
                + np.exp(4*eta[large])*(
                    19328*eta[large]**6 - 62720*eta[large]**5 
                    - 137760*eta[large]**4
                    + 436800*eta[large]**3 + 200760*eta[large]**2
                    - 787500*eta[large] + 307125
                )
                + 4*np.exp(3*eta[large])*(
                    4832*eta[large]**6 + 15680*eta[large]**5 
                    - 34440*eta[large]**4
                    - 109200*eta[large]**3 + 50190*eta[large]**2
                    + 196875*eta[large] - 102375
                )
                + np.exp(2*eta[large])*(
                    3648*eta[large]**6 + 39200*eta[large]**5 
                    + 123984*eta[large]**4
                    + 54600*eta[large]**3 - 301140*eta[large]**2
                    - 393750*eta[large] + 307125
                )
                + 2*np.exp(eta[large])*(
                    32*eta[large]**6 + 784*eta[large]**5 
                    + 6888*eta[large]**4
                    + 27300*eta[large]**3 + 50190*eta[large]**2
                    + 39375*eta[large] - 61425
                )
                + 20475
            )
        )

        g2_at_0[large] = (-4*eta[large]**2*T**2
            /(np.exp(eta[large]) - 1)
        )

        g4_at_0[large] = (
            -16*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**3*(
                np.exp(2*eta[large])
                    *(eta[large]**2 - 3*eta[large] + 3)
                + np.exp(eta[large])
                    *(eta[large]**2 + 3*eta[large] - 6)
                +3
            )
        )

        g6_at_0[large] = (
            -32*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**5*(
                np.exp(4*eta[large])*(
                    2*eta[large]**4 - 20*eta[large]**3 
                    + 70*eta[large]**2 - 90*eta[large] + 45
                )
                + 2*np.exp(3*eta[large])*(
                    11*eta[large]**4 - 30*eta[large]**3 
                    - 35*eta[large]**2 + 135*eta[large] - 90
                )
                + np.exp(2*eta[large])*(
                    22*eta[large]**4 + 60*eta[large]**3 
                    - 70*eta[large]**2 - 270*eta[large] + 270
                )
                + 2*np.exp(eta[large])*(
                    eta[large]**4 + 10*eta[large]**3 
                    + 35*eta[large]**2 + 45*eta[large] - 90
                ) + 45
            )
        )

        g8_at_0[large] = (
            -256*eta[large]**2*T**2/(np.exp(eta[large]) - 1)**7*(
                np.exp(6*eta[large])*(
                    eta[large]**6 - 21*eta[large]**5 
                    + 168*eta[large]**4 - 630*eta[large]**3
                    + 1155*eta[large]**2 - 945*eta[large] + 315
                )
                + 3*np.exp(5*eta[large])*(
                    19*eta[large]**6 - 175*eta[large]**5 
                    + 504*eta[large]**4 - 210*eta[large]**3
                    - 1155*eta[large]**2 + 1575*eta[large] - 630
                )
                + np.exp(4*eta[large])*(
                    302*eta[large]**6 - 840*eta[large]**5 
                    - 1680*eta[large]**4 + 5040*eta[large]**3 
                    + 2310*eta[large]**2 - 9450*eta[large] + 4725
                )
                + 2*np.exp(3*eta[large])*(
                    151*eta[large]**6 + 420*eta[large]**5 
                    - 840*eta[large]**4 -2520*eta[large]**3 
                    + 1155*eta[large]**2 + 4725*eta[large] - 3150
                )
                + 3*np.exp(2*eta[large])*(
                    19*eta[large]**6 + 175*eta[large]**5 
                    + 504*eta[large]**4 + 210*eta[large]**3
                    - 1155*eta[large]**2 - 1575*eta[large] + 1575
                )
                + np.exp(eta[large])*(
                    eta[large]**6 + 21*eta[large]**5 
                    + 168*eta[large]**4 + 630*eta[large]**3
                    + 1155*eta[large]**2 + 945*eta[large] - 1890
                )
                + 315
            )
        )

    if np.any(small):

        h3_at_0[small] = T**2*(
            10*eta[small] - 15*eta[small]**2 + 11*eta[small]**3/2
            - 31*eta[small]**5/120+ 37*eta[small]**7/3024 
            - 103*eta[small]**9/201600
        )

        h5_at_0[small] = T**2*(
            178*eta[small] - 405*eta[small]**2 
            + 475*eta[small]**3/2 - 205*eta[small]**5/8
            + 6925*eta[small]**7/3024 - 703*eta[small]**9/4480
        )

        h7_at_0[small] = T**2*(
            6858*eta[small] - 20475*eta[small]**2 
            + 33075*eta[small]**3/2 - 26369*eta[small]**5/8 
            + 71801*eta[small]**7/144 - 101903*eta[small]**9/1920
        )

        g2_at_0[small] = T**2*(
            -4*eta[small] + 2*eta[small]**2 - eta[small]**3/3 
            + eta[small]**5/180 - eta[small]**7/7560 
            + eta[small]**9/302400
        )

        g4_at_0[small] = T**2*(
            -32*eta[small] + 24*eta[small]**2 - 8*eta[small]**3 
            + 2*eta[small]**5/5 - 19*eta[small]**7/945 
            + 11*eta[small]**9/12600
        )

        g6_at_0[small] = T**2*(
            -736*eta[small] + 720*eta[small]**2 
            - 360*eta[small]**3 + 38*eta[small]**5
            -667*eta[small]**7/189 + 211*eta[small]**9/840
        )
        g8_at_0[small] = T**2*(
            -33792*eta[small] + 40320*eta[small]**2 
            - 26880*eta[small]**3 + 4928*eta[small]**5 
            - 6752*eta[small]**7/9 + 1228*eta[small]**9/15
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
    













