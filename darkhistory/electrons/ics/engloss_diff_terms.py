""" Functions for computing the ICS energy loss spectrum by small parameter expansion."""

import numpy as np 
import scipy.special as sp

from darkhistory.utilities import log_1_plus_x
from darkhistory.electrons.ics import BE_integrals as BE_int

def engloss_diff_expansion(beta, delta, T, as_pairs=False):
    """ Difference expansion term for the energy loss spectrum.

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
        The computed energy loss spectrum 

    """

    eta = delta/T 

    if as_pairs:
        A_all = eta/(2*beta)
    else:
        A_all = np.outer(1/(2*beta), eta)

    large = A_all > 0.01
    small = ~large

    print('    Computing integrals 1/6...')
    P_0        = BE_int.F0(A_all, np.ones_like(A_all)*np.inf)
    print('    Computing integrals 2/6...')
    P_minus_3  = BE_int.F_inv_n(A_all, np.ones_like(A_all)*np.inf, 3)[0]
    print('    Computing integrals 3/6...')
    P_minus_5  = BE_int.F_inv_n(A_all, np.ones_like(A_all)*np.inf, 5)[0]
    print('    Computing integrals 4/6...')
    P_minus_7  = BE_int.F_inv_n(A_all, np.ones_like(A_all)*np.inf, 7)[0]
    print('    Computing integrals 5/6...')
    P_minus_9  = BE_int.F_inv_n(A_all, np.ones_like(A_all)*np.inf, 9)[0]
    print('    Computing integrals 6/6...')
    P_minus_11 = BE_int.F_inv_n(A_all, np.ones_like(A_all)*np.inf, 11)[0]
    print('    Integrals computed!')

    term_beta_0  = (
        176/15*A_all*P_0 - 64/3*A_all**4*P_minus_3
        + 128/5*A_all**6*P_minus_5
    )
    term_beta_2  = (
        -1168/105*A_all*P_0 + 128/3*A_all**4*P_minus_3
        -2176/15*A_all**6*P_minus_5 + 1280/7*A_all**8*P_minus_7
    )
    term_beta_4  = (
        -64/3*A_all**4*P_minus_3 + 640/3*A_all**6*P_minus_5
        -768*A_all**8*P_minus_7 + 14336/15*A_all**10*P_minus_9
    )
    term_beta_6  = (
        -512/3465*A_all*P_0 - 1408/15*A_all**6*P_minus_5
        + 6912/7*A_all**8*P_minus_7 - 161792/45*A_all**10*P_minus_9
        + 49152/11*A_all**12*P_minus_11
    )

    if np.any(large):

        A = A_all[large]

        term_beta_2[large] += (
            - 32/3*A**2*np.exp(-A)/(1 - np.exp(-A))
            + 8/3*A**3*np.exp(-A)/(1 - np.exp(-A))**2
        )

        term_beta_4[large] += (
            - 512/15*A**2*np.exp(-A)/(1 - np.exp(-A))
            + 8/5*A**3*np.exp(-A)/(1 - np.exp(-A))**2 
            - 8/15*A**4*np.exp(-A)*(1 + np.exp(-A))/(1 - np.exp(-A))**3
            + 2/15*A**5*(np.exp(-A) + 4*np.exp(-2*A) + np.exp(-3*A))
                /(1 - np.exp(-A))**4
        )

        term_beta_6[large] += (
            - 416/3*A**2*np.exp(-A)/(1 - np.exp(-A))
            + 1184/105*A**3*np.exp(-A)/(1 - np.exp(-A))**2 
            - 256/315*A**4*(np.exp(-A) + np.exp(-2*A))/(1 - np.exp(-A))**3
            - 2/63*A**5*(np.exp(-A) + 4*np.exp(-2*A) + np.exp(-3*A))
                /(1 - np.exp(-A))**4
            - 4/315*A**6*(
                np.exp(-A) + 11*np.exp(-2*A) + 11*np.exp(-3*A) + np.exp(-4*A)
            )/(1 - np.exp(-A))**5
            + 1/315*A**7*(
                np.exp(-A) + 26*np.exp(-2*A) 
                + 66*np.exp(-3*A) + 26*np.exp(-4*A) + np.exp(-5*A)
            )/(1 - np.exp(-A))**6
        )

    if np.any(small):

        A = A_all[small] 

        term_beta_2[small] += (
            - 8*A + 16/3*A**2 - 10/9*A**3 + 7/270*A**5 - 1/1260*A**7
            + 11/453600*A**9 - 13/17962560*A**11
        )

        term_beta_4[small] += (
            - 164/5*A + 256/15*A**2 - 134/45*A**3 + 161/2700*A**5 
            - 19/9450*A**7 + 359/4536000*A**9 - 289/89812800*A**11
        )

        term_beta_6[small] += (
            - 40676/315*A + 208/3*A**2 - 1312/105*A**3 + 4651/18900*A**5
            - 416/59535*A**7 + 989/4536000*A**9 - 173/22453200*A**11
        )

    final_expr = (
        term_beta_0
        + np.transpose(beta**2*np.transpose(term_beta_2))
        + np.transpose(beta**4*np.transpose(term_beta_4))
        + np.transpose(beta**6*np.transpose(term_beta_6))
    )

    return final_expr