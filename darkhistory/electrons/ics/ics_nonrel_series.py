"""Nonrelativistic series expansions for ICS spectrum."""

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

def Q(beta, photeng, T, epsrel=0):

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

        # Computed for error
        q8_at_0 = 16*eta**2*T**2/(np.exp(eta) - 1)**8*(
            np.exp(7*eta)*(
                16*eta**7 - 496*eta**6 + 5776*eta**5
                - 32144*eta**4 + 90006*eta**3
                - 122010*eta**2 + 69825*eta - 11025
            )
            + np.exp(6*eta)*(
                1920*eta**7 - 27776*eta**6 + 138624*eta**5
                - 257152*eta**4 + 488040*eta**2
                - 418950*eta + 77175
            )
            + np.exp(5*eta)*(
                19056*eta**7 - 121520*eta**6 + 86640*eta**5
                + 610736*eta**4 - 810054*eta**3 - 610050*eta**2
                + 1047375*eta - 231525
            )
            + np.exp(4*eta)*(
                38656*eta**7 - 462080*eta**5 + 1440096*eta**3
                - 1396500*eta + 385875
            )
            + np.exp(3*eta)*(
                19056*eta**7 + 121520*eta**6 + 86640*eta**5
                - 610736*eta**4 - 810054*eta**3
                + 610050*eta**2 + 1047375*eta - 385875
            )
            + np.exp(2*eta)*(
                1920*eta**7 + 27776*eta**6 + 138624*eta**5
                + 257152*eta**4 - 488040*eta**2
                - 418950*eta + 231525
            )
            + np.exp(eta)*(
                16*eta**7 + 496*eta**6 + 5776*eta**5
                + 32144*eta**4 + 90006*eta**3 + 122010*eta**2
                + 69825*eta - 77175
            )
            + 11025
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
        # Computed for error
        q8_at_0 = T**2*(
            88200*eta**2 - 107800*eta**3 + 165844*eta**5/5
            - 141679*eta**7/21 + 27247*eta**9/30
        )

    term = 2*(
        q2_at_0*beta**2/2 
        + q4_at_0*beta**4/24 
        + q6_at_0*beta**6/720
    )

    err = 2*q8_at_0*beta**8/40320

    if epsrel > 0:
        check_err(term, err, epsrel)
        print('***** Diagnostics for Q *****')
        print('1st Term: ', 2*q2_at_0*beta**2/2)
        print('2nd Term: ', 2*q4_at_0*beta**4/24)
        print('3rd Term: ', 2*q6_at_0*beta**6/720)
        print('Error: ', err)
        print('***** End Diagnostics for Q *****')

    return term

def Q_and_K(beta, photeng, T, epsrel=0):

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
                -16*eta**2*np.sinh(eta)*(
                    340*eta**2 
                    + (68*eta**2 + 885)*np.cosh(eta) - 885
                )
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

        # Computed for error
        q8_at_0 = 16*eta**2*T**2/(np.exp(eta) - 1)**8*(
            np.exp(7*eta)*(
                16*eta**7 - 496*eta**6 + 5776*eta**5
                - 32144*eta**4 + 90006*eta**3
                - 122010*eta**2 + 69825*eta - 11025
            )
            + np.exp(6*eta)*(
                1920*eta**7 - 27776*eta**6 + 138624*eta**5
                - 257152*eta**4 + 488040*eta**2
                - 418950*eta + 77175
            )
            + np.exp(5*eta)*(
                19056*eta**7 - 121520*eta**6 + 86640*eta**5
                + 610736*eta**4 - 810054*eta**3 - 610050*eta**2
                + 1047375*eta - 231525
            )
            + np.exp(4*eta)*(
                38656*eta**7 - 462080*eta**5 + 1440096*eta**3
                - 1396500*eta + 385875
            )
            + np.exp(3*eta)*(
                19056*eta**7 + 121520*eta**6 + 86640*eta**5
                - 610736*eta**4 - 810054*eta**3
                + 610050*eta**2 + 1047375*eta - 385875
            )
            + np.exp(2*eta)*(
                1920*eta**7 + 27776*eta**6 + 138624*eta**5
                + 257152*eta**4 - 488040*eta**2
                - 418950*eta + 231525
            )
            + np.exp(eta)*(
                16*eta**7 + 496*eta**6 + 5776*eta**5
                + 32144*eta**4 + 90006*eta**3 + 122010*eta**2
                + 69825*eta - 77175
            )
            + 11025
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

        k8_at_0 = -16*eta**2*T**2/(np.exp(eta) - 1)**8*(
            np.exp(7*eta)*(
                16*eta**7 - 400*eta**6 + 3536*eta**5
                - 13904*eta**4 + 24814*eta**3 - 17958*eta**2
                + 3459*eta + 309
            )
            + np.exp(6*eta)*(
                1920*eta**7 - 22400*eta**6 + 84864*eta**5
                - 111232*eta**4 + 71832*eta**2
                - 20754*eta - 2163
            )
            + np.exp(5*eta)*(
                19056*eta**7 - 98000*eta**6 + 53040*eta**5
                + 264176*eta**4 - 223326*eta**3
                - 89790*eta**2 + 51885*eta + 6489
            )
            + np.exp(4*eta)*(
                38656*eta**7 - 282880*eta**5 + 397024*eta**3
                - 69180*eta - 10815
            )
            + np.exp(3*eta)*(
                19056*eta**7 + 98000*eta**6 + 53040*eta**5
                - 264176*eta**4 - 223326*eta**3
                + 89790*eta**2 + 51885*eta + 10815
            )
            + np.exp(2*eta)*(
                1920*eta**7 + 22400*eta**6 + 84864*eta**5
                + 111232*eta**4 - 71832*eta**2
                - 20754*eta - 6489
            )
            + np.exp(eta)*(
                16*eta**7 + 400*eta**6 + 3536*eta**5
                + 13904*eta**4 + 24814*eta**3 + 17958*eta**2
                + 3459*eta + 2163
            )
            + 309
        )

    else:
        q4_at_0 = T**2*(
            36*eta**2 - 68*eta**3/3 + 2*eta**5 
            - 89*eta**7/630 + 149*eta**9/18900
        )
        q6_at_0 = T**2*(
            1350*eta**2 - 1250*eta**3 + 1123*eta**5/5
            - 2381*eta**7/84 + 6373*eta**9/2520
        )
        q8_at_0 = T**2*(
            88200*eta**2 - 107800*eta**3 + 165844*eta**5/5
            - 141679*eta**7/21 + 27247*eta**9/30
        )
        k4_at_0 = T**2*(
            4*eta**2 + 4*eta**3 - 46*eta**5/45 
            + 59*eta**7/630 - 37*eta**9/6300
        )
        k6_at_0 = T**2*(
            66*eta**2 + 90*eta**3 - 193*eta**5/3
            + 5309*eta**7/420 - 393*eta**9/280
        )
        k8_at_0 = T**2*(
            2472*eta**2 + 4200*eta**3 - 17780*eta**5/3
            + 31411*eta**7/15 - 15931*eta**9/42
        )

    term = Q(beta, photeng, T) + 2*(
        (q4_at_0 + k4_at_0)*beta**2/24
        + (q6_at_0 + k6_at_0)*beta**4/720
    )

    err = 2*((q8_at_0 + k8_at_0)*beta**6/40320)

    if epsrel > 0:
        check_err(term, err, epsrel)
        print('***** Diagnostics for Q_and_K *****')
        Q(beta, photeng, T, epsrel=epsrel)
        print('1st Term: ', 2*(q4_at_0 + k4_at_0)*beta**2/24)
        print('2nd Term: ', 2*(q6_at_0 + k6_at_0)*beta**4/720)
        print('Error: ', err)
        print('***** End Diagnostics for Q_and_K *****')

    return term

def H_and_G(beta, photeng, T, epsrel=0):

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

        h7_at_0 = 2*eta**2*T**2/(np.exp(eta) - 1)**7*(
            np.exp(6*eta)*(
                64*eta**6 - 1568*eta**5 + 13776*eta**4
                - 54600*eta**3 + 100380*eta**2
                - 78750*eta + 20475
            )
            + 2*np.exp(5*eta)*(
                1824*eta**6 - 19600*eta**5 + 61992*eta**4
                - 27300*eta**3 - 150570*eta**2
                + 196875*eta - 61425
            )
            + np.exp(4*eta)*(
                19328*eta**6 - 62720*eta**5 - 137760*eta**4
                + 436800*eta**3 + 200760*eta**2
                - 787500*eta + 307125
            )
            + 4*np.exp(3*eta)*(
                4832*eta**6 + 15680*eta**5 - 34440*eta**4
                - 109200*eta**3 + 50190*eta**2
                + 196875*eta - 102375
            )
            + np.exp(2*eta)*(
                3648*eta**6 + 39200*eta**5 + 123984*eta**4
                + 54600*eta**3 - 301140*eta**2
                - 393750*eta + 307125
            )
            + 2*np.exp(eta)*(
                32*eta**6 + 784*eta**5 + 6888*eta**4
                + 27300*eta**3 + 50190*eta**2
                + 39375*eta - 61425
            )
            + 20475
        )

        g2_at_0 = -4*eta**2*T**2/(np.exp(eta) - 1)

        g4_at_0 = -16*eta**2*T**2/(np.exp(eta) - 1)**3*(
            np.exp(2*eta)*(eta**2 - 3*eta + 3)
            + np.exp(eta)*(eta**2 + 3*eta - 6)
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

        g8_at_0 = -256*eta**2*T**2/(np.exp(eta) - 1)**7*(
            np.exp(6*eta)*(
                eta**6 - 21*eta**5 + 168*eta**4 - 630*eta**3
                + 1155*eta**2 - 945*eta + 315
            )
            + 3*np.exp(5*eta)*(
                19*eta**6 - 175*eta**5 + 504*eta**4 - 210*eta**3
                - 1155*eta**2 + 1575*eta - 630
            )
            + np.exp(4*eta)*(
                302*eta**6 - 840*eta**5 - 1680*eta**4
                + 5040*eta**3 + 2310*eta**2 - 9450*eta + 4725
            )
            + 2*np.exp(3*eta)*(
                151*eta**6 + 420*eta**5 - 840*eta**4
                -2520*eta**3 + 1155*eta**2 + 4725*eta - 3150
            )
            + 3*np.exp(2*eta)*(
                19*eta**6 + 175*eta**5 + 504*eta**4 + 210*eta**3
                - 1155*eta**2 - 1575*eta + 1575
            )
            + np.exp(eta)*(
                eta**6 + 21*eta**5 + 168*eta**4 + 630*eta**3
                + 1155*eta**2 + 945*eta - 1890
            )
            + 315
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

        h7_at_0 = T**2*(
            6858*eta - 20475*eta**2 + 33075*eta**3/2
            - 26369*eta**5/8 + 71801*eta**7/144
            - 101903*eta**9/1920
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
        g8_at_0 = T**2*(
            -33792*eta + 40320*eta**2 - 26880*eta**3 
            + 4928*eta**5 - 6752*eta**7/9 + 1228*eta**9/15
        )

    term1 = 4*beta**2*(h3_at_0/6 + h5_at_0/120*beta**2)
    term2 = 4*beta**2*np.sqrt(1-beta**2)*(
        g4_at_0/24 + g6_at_0/720*beta**2
    )
    term3 = 2*g2_at_0*beta**2*(-1/2 - 1/8*beta**2)

    term = term1+term2+term3

    err = (4*beta**2*h7_at_0/40320*beta**4
        + 4*beta**2*np.sqrt(1-beta**2)*g8_at_0/40320*beta**4
        + 2*g2_at_0*beta**2*(-1/16*beta**4)
    )

    if epsrel > 0:
        check_err(term, err, epsrel)
        print('***** Diagnostics for Q_and_K *****')
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

    return term
    













