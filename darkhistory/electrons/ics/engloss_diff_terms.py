""" Functions for computing the ICS energy loss spectrum by small parameter expansion."""

import numpy as np 
import scipy.special as sp

from darkhistory.utilities import log_1_plus_x
from darkhistory.electrons.ics import bose_einstein_integrals as BE_int

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


    return np.transpose(
        prefac*np.transpose(
            eta*term_eta + eta**3*term_eta_3 + eta**5*term_eta_5
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

    return np.transpose(
        prefac*np.transpose(
            eta**2*term_eta_2 + eta**4*term_eta_4 + eta**6*term_eta_6
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

    return np.transpose(
        prefac*np.transpose(
            eta*term_eta + eta**3*term_eta_3 + eta**5*term_eta_5
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


    return np.transpose(
        prefac*np.transpose( 
            eta**3*term_eta_3 + eta**5*term_eta_5 + eta**7*term_eta_7
        )
    )

def F_rem(beta, delta, T, as_pairs=False):
    """ Computes the F_rem term.

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

    F_inv_3_int = BE_int.F_inv_3(A_all, np.ones_like(A_all)*np.inf)

    F_inv_5_int = BE_int.F_inv_5(A_all, np.ones_like(A_all)*np.inf)


    term_eta_4  = np.zeros_like(A_all) + F_inv_3_int[0]/6
    term_eta_6  = np.zeros_like(A_all) + F_inv_5_int[0]/15
    term_eta_8  = np.zeros_like(A_all)
    term_eta_10 = np.zeros_like(A_all)

    if np.any(large):

        A = A_all[large]

        term_eta_4[large] += -(1/6)*(np.exp(-A)/A**2)/(1 - np.exp(-A))

        term_eta_6[large] += (
            - (5*A**2 + 5*A + 21)*np.exp(-A)
            + (-5*A**2 + 5*A + 42)*np.exp(-2*A)
            - 21*np.exp(-3*A)
        )/(720 * A**4 * (1 - np.exp(-A))**3)

        term_eta_8[large] += (
            -(7*A**4 + 21*A**3 + 105*A**2 + 84*A + 660)*np.exp(-A)
            +(-77*A**4 - 63*A**3 + 105*A**2 + 252*A + 2640)*np.exp(-2*A)
            +(-77*A**4 + 63*A**3 + 105*A**2 - 252*A - 3960)*np.exp(-3*A)
            +(-7*A**4 + 21*A**3 - 105*A**2 + 84*A + 2640)*np.exp(-4*A)
            -660*np.exp(-5*A)
        )/(80640 * A**6 * (1 - np.exp(-A))**5)

        term_eta_10[large] += (
            - (5*A**6 + 25*A**5 + 177*A**4 
                + 552*A**3 + 2640*A**2 + 43560*A + 199080)*np.exp(-A)
            + (-285*A**6 - 625*A**5 - 1593*A**4 
                - 552*A**3 + 7920*A**2 + 217800*A + 1194480)*np.exp(-2*A)
            - 2*(755*A**6 + 500*A**5 - 885*A**4
                - 2208*A**3 +2640*A**2 + 217800*A + 1493100)*np.exp(-3*A)
            - 2*(755*A**6 - 500*A**5 - 885*A**4
                + 2208*A**3 + 2640*A**2 - 217800*A - 1990800)*np.exp(-4*A)
            + (-285*A**6 + 625*A**5 - 1593*A**4
                + 552*A**3 + 7920*A**2 - 217800*A - 2986200)*np.exp(-5*A)
            + (-5*A**6 + 25*A**5 - 177*A**4
                + 552*A**3 - 2640*A**2 + 43560*A + 1194480)*np.exp(-6*A)
            - 199080*np.exp(-7*A)
        )/(9676800 * A**8 * (1 - np.exp(-A))**7)

    if np.any(small):

        A = A_all[small]

        term_eta_4[small] += (
            -1209600/A**3 + 604800/A**2 - 100800/A + 1680*A - 40*A**3 + A**5  
        )/7257600

        term_eta_6[small] += (
            -1/(20*A**5) + 7/(480*A**4) - 1/(540*A**3) + 1/(14400*A)
            - A/226800 + 7*A**3/31104000 - A**5/102643200   
        )

        term_eta_8[small] += (
            -13/(840*A**7) + 11/(2688*A**6) - 1/(1680*A**5) 
            + 19/(1209600*A**3) - 1/(1270080*A)
            + 31*A/508032000 + 47*A**3/10059033600
            + 20039*A**5/66572513280000
        )

        term_eta_10[small] += (
            -13/(480*A**9) + 79/(7680*A**8) - 3/(2240*A**7)
            + 281/(24192000*A**5) - 1/(15240960*A**3)
            - 31*A/80472268800 + 1645271*A**3/26362715258880000
            - 23*A**5/3736212480000
        )

    prefac = 2*np.sqrt(1 - beta**2)

    return np.transpose(
        prefac*np.transpose(
            eta**4*term_eta_4 + eta**6*term_eta_6
            + eta**8*term_eta_8 + eta**10*term_eta_10
        )
    )

def engloss_diff_expansion(beta, delta, T, as_pairs=False):
    """ Computes the difference expansion term for the new spectrum.

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
        The computed energy loss spectrum 

    """

    eta = delta/T 

    if as_pairs:
        A_all = eta/(2*beta)
    else:
        A_all = np.outer(1/(2*beta), eta)

    large = A_all > 0.01
    small = ~large

    print('Computing integrals 1/6...')
    P_0        = BE_int.F0(A_all, np.ones_like(A_all)*np.inf)
    print('Computing integrals 2/6...')
    P_minus_3  = BE_int.F_inv_n(A_all, np.ones_like(A_all)*np.inf, 3)[0]
    print('Computing integrals 3/6...')
    P_minus_5  = BE_int.F_inv_n(A_all, np.ones_like(A_all)*np.inf, 5)[0]
    print('Computing integrals 4/6...')
    P_minus_7  = BE_int.F_inv_n(A_all, np.ones_like(A_all)*np.inf, 7)[0]
    print('Computing integrals 5/6...')
    P_minus_9  = BE_int.F_inv_n(A_all, np.ones_like(A_all)*np.inf, 9)[0]
    print('Computing integrals 6/6...')
    P_minus_11 = BE_int.F_inv_n(A_all, np.ones_like(A_all)*np.inf, 11)[0]
    print('Integrals computed!')

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