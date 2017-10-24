"""Nonrelativistic ICS spectrum functions."""

import numpy as np 
import scipy.special as sp
from darkhistory.utilities import log_1_plus_x
from darkhistory.utilities import diff_pow
from darkhistory.utilities import check_err

# General series expressions for integrals over Planck distribution.

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

    integral = np.zeros(a.size)

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

def F_inv(a,b,tol=1e-30):
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
                    sp.bernoulli(k)[-1]*(x**(k-1))/
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

    both_low  = (a < bound) & (b <  bound)
    low_high  = (a < bound) & (b >= bound)
    both_high = (a > bound) & (b >  bound)

    integral = np.zeros(a.size)

    # Both low

    if np.any(both_low):

        low_sum_a = low_summand(a[both_low])
        low_sum_b = low_summand(b[both_low])
        integral[both_low] = next(low_sum_b) - next(low_sum_a)

        while err > tol:
            next_term = next(low_sum_b) - next(low_sum_a)
            err = np.max(np.abs(next_term/integral[both_low]))
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
            err = np.max(np.abs(next_term/integral[low_high]))
            integral[low_high] += next_term

    err = 10*tol

    # Both high

    if np.any(both_high):

        high_sum_a = high_summand(a[both_high])
        high_sum_b = high_summand(b[both_high])
        integral[both_high] = next(high_sum_a) - next(high_sum_b)

        while err > tol:
            next_term = next(high_sum_a) - next(high_sum_b)
            err = np.max(np.abs(next_term/integral[both_high]))
            integral[both_high] += next_term

    return integral

def F_log(a,b,tol=1e-30):
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
                    sp.bernoulli(k)[-1]*(x**k)/
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

    both_low  = (a < bound) & (b <  bound)
    low_high  = (a < bound) & (b >= bound)
    both_high = (a > bound) & (b >  bound)

    integral = np.zeros(a.size)

    # Both low

    if np.any(both_low):

        low_sum_a = low_summand(a[both_low])
        low_sum_b = low_summand(b[both_low])
        integral[both_low] = next(low_sum_b) - next(low_sum_a)

        while err > tol:
            next_term = next(low_sum_b) - next(low_sum_a)
            err = np.max(np.abs(next_term/integral[both_low]))
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
            err = np.max(np.abs(next_term/integral[low_high]))
            integral[low_high] += next_term

    err = 10*tol

    # Both high

    if np.any(both_high):

        high_sum_a = high_summand(a[both_high])
        high_sum_b = high_summand(b[both_high])
        integral[both_high] = next(high_sum_a) - next(high_sum_b)

        while err > tol:
            next_term = next(high_sum_a) - next(high_sum_b)
            err = np.max(np.abs(next_term/integral[both_high]))
            integral[both_high] += next_term

    return integral

# Low beta expansion functions

def Q(beta, photeng, T):

    eta = photeng/T

    if eta > 0.01:

        q2_at_0 = 4*eta**2*T**2/(np.exp(eta) - 1)**2*(
            np.exp(eta)*(eta - 1) + 1
        )

        q4_at_0 = 8*eta**2*T**2/(np.exp(eta) - 1)**4*(
            np.exp(2*eta)*(8*eta**3 
                + (4*eta**3 + 50*eta - 36)*np.cosh(eta)
                + 2*(9 - 14*eta**2)*np.sinh(eta) -50*eta + 27
            ) + 9
        )

        q6_at_0 = 4*eta**2*T**2/(np.exp(eta) - 1)**6*(
            np.exp(3*eta)*(
                -16*eta**2*np.sinh(eta)*(340*eta**2 
                + (68*eta**2 + 885)*np.cosh(eta) - 885)
                +3*(
                    352*eta**5 - 3096*eta**3 
                    + 6150*eta - 1125*np.sinh(eta) 
                    + 900*np.sinh(2*eta) - 2250
                )
                +np.cosh(eta)*(
                    832*eta**5 + 6192*eta**3 - 24600*eta + 10125
                )
                +np.cosh(2*eta)*(
                    32*eta**5 + 3096*eta**3 + 6150*eta - 4050
                )   
            ) +675
        )

    else:

        q2_at_0 = T**2*(
            2*eta**2 + eta**5/45 - eta**7/1260 + eta**9/37800
        )

        q4_at_0 = T**2*(
            36*eta**2 - 68*eta**3/3 + 2*eta**5 
            - 89*eta**7/630 + 149*eta**9/18900
        )

        q6_at_0 = T**2*(
            1350*eta**2 - 1250*eta**3 + 1123*eta**5/5 
            - 2381*eta**7/84 + 6373*eta**9/2520
        )

    return 2*(
        q2_at_0*beta**2/2 
        + q4_at_0*beta**4/24 
        + q6_at_0*beta**6/720
    )

def term_Q_and_K(beta, photeng, T):

    eta = photeng/T

    if eta > 0.01:

        q2_at_0 = 4*eta**2*T**2/(np.exp(eta) - 1)**2*(
            np.exp(eta)*(eta - 1) + 1
        )

        q4_at_0 = 8*eta**2*T**2/(np.exp(eta) - 1)**4*(
            np.exp(2*eta)*(8*eta**3 
                + (4*eta**3 + 50*eta - 36)*np.cosh(eta)
                + 2*(9 - 14*eta**2)*np.sinh(eta) -50*eta + 27
            ) + 9
        )

        q6_at_0 = 4*eta**2*T**2/(np.exp(eta) - 1)**6*(
            np.exp(3*eta)*(
                -16*eta**2*np.sinh(eta)*(340*eta**2 
                + (68*eta**2 + 885)*np.cosh(eta) - 885)
                +3*(
                    352*eta**5 - 3096*eta**3 
                    + 6150*eta - 1125*np.sinh(eta) 
                    + 900*np.sinh(2*eta) - 2250
                )
                +np.cosh(eta)*(
                    832*eta**5 + 6192*eta**3 - 24600*eta + 10125
                )
                +np.cosh(2*eta)*(
                    32*eta**5 + 3096*eta**3 + 6150*eta - 4050
                )   
            ) +675
        )

        k4_at_0 = 8*eta**2*T**2/(np.exp(eta) - 1)**4*(
            np.exp(2*eta)*(
                -8*eta**3 - 2*(2*eta**3 + 7*eta + 2)*np.cosh(eta)
                +(20*eta**2 + 2)*np.sinh(eta) +14*eta + 3
            ) +1
        )

        k6_at_0 = 4*eta**2*T**2/(np.exp(eta) - 1)**6*(
            -np.exp(3*eta)*(
                6*(176*eta**5 - 788*eta**3 + 303*eta + 55)
                + 2*(16*eta**5 + 788*eta**3 + 303*eta + 99)
                    *np.cosh(2*eta)
                + (-4160*eta*4 + 3984*eta**2 + 165)*np.sinh(eta)
                + (832*eta**5 + 3152*eta**3
                    -8*(108*eta**4 + 498*eta**2 + 33)
                        *np.sinh(eta)
                    -2424*eta - 495
                )*np.cosh(eta)
            ) + 33
        )

    else:
        q2_at_0 = T**2*(
            2*eta**2 + eta**5/45 - eta**7/1260 + eta**9/37800
        )
        q4_at_0 = T**2*(
            36*eta**2 - 68*eta**3/3 + 2*eta**5 
            - 89*eta**7/630 + 149*eta**9/18900
        )
        q6_at_0 = T**2*(
            1350*eta**2 - 1250*eta**3 + 1123*eta**5/5
            - 2381*eta**7/84 + 6373*eta**9/2520
        )
        k4_at_0 = T**2*(
            4*eta**2 + 4*eta**3 - 46*eta**5/45 
            + 59*eta**7/630 - 37*eta**9/6300
        )
        k6_at_0 = T**2*(
            66*eta**2 + 90*eta**3 - 193*eta**5/3
            + 5309*eta**7/420 - 393*eta**9/280
        )

    return Q(beta, photeng, T) + 2*(
        (q4_at_0 + k4_at_0)*beta**2/24
        (q6_at_0 + k6_at_0)*beta**4/720
    )

def term_H_and_G(beta, photeng, T):

    eta = photeng/T

    if eta > 0.01:

        h3_at_0 = 2*eta**2*T**2/(np.exp(eta) - 1)**3*(
            2*np.exp(eta)*(2*eta**2 + 9*eta - 15)
            + np.exp(2*eta)*(4*eta**2 - 18*eta + 15)
            +15
        )

        h5_at_0 = 2*eta**2*T**2/(np.exp(eta) - 1)**5*(
            np.exp(4*eta)*(
                16*eta**4 - 200*eta**3 + 760*eta**2 
                - 1020*eta + 405
            )
            + 4*np.exp(3*eta)*(
                44*eta**4 - 150*eta**3 - 190*eta**2
                + 765*eta - 405
            )
            + 2*np.exp(2*eta)*(
                88*eta**4 + 300*eta**3 - 380*eta**2
                - 1530*eta + 1215
            )
            + 4*np.exp(eta)*(
                4*eta**4 + 50*eta**3 + 190*eta**2
                + 255*eta - 405
            ) + 405
        )

        g2_at_0 = -4*eta**2*T**2/(np.exp(eta) - 1)

        g4_at_0 = -16*eta**2*T**2/(np.exp(eta) - 1)**3*(
            np.exp(2*eta)*(eta**2-3*eta+3)
            + np.exp(eta)*(eta**2+3*eta-6)
            +3
        )

        g6_at_0 = -32*eta**2*T**2/(np.exp(eta) - 1)**5*(
            np.exp(4*eta)*(
                2*eta**4 - 20*eta**3 + 70*eta**2 - 90*eta + 45
            )
            + 2*np.exp(3*eta)*(
                11*eta**4 - 30*eta**3 - 35*eta**2 + 135*eta - 90
            )
            + np.exp(2*eta)*(
                22*eta**4 + 60*eta**3 - 70*eta**2 - 270*eta + 270
            )
            + 2*np.exp(eta)*(
                eta**4 + 10*eta**3 + 35*eta**2 + 45*eta - 90
            ) + 45
        )

    else: 

        h3_at_0 = T**2*(
            10*eta - 15*eta**2 + 11*eta**3/2 - 31*eta**5/120
            + 37*eta**7/3024 - 103*eta**9/201600
        )

        h5_at_0 = T**2*(
            178*eta - 405*eta**2 + 475*eta**3/2 - 205*eta**5/8
            + 6925*eta**7/3024 - 703*eta**9/4480
        )

        g2_at_0 = T**2*(
            -4*eta + 2*eta**2 - eta**3/3 + eta**5/180 
            - eta**7/7560 + eta**9/302400
        )

        g4_at_0 = T**2*(
            -32*eta + 24*eta**2 - 8*eta**3 + 2*eta**5/5 
            - 19*eta**7/945 + 11*eta**9/12600
        )

        g6_at_0 = T**2*(
            -736*eta + 720*eta**2 - 360*eta**3 + 38*eta**5
            -667*eta**7/189 + 211*eta**9/840
        )
        
    term1 = 4*beta**2*(h3_at_0/6 + h5_at_0/120*beta**2)
    term2 = 4*beta**2*np.sqrt(1-beta**2)*(
        g4_at_0/24 + g6_at_0/720*beta**2
    )
    term3 = 2*g2_at_0*beta**2*(-1/2 - 1/8*beta**2)

    return term1+term2+term3
    













