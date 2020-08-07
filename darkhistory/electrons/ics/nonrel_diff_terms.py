"""Functions for computing the nonrelativistic ICS spectrum by beta expansion."""

import numpy as np 
import scipy.special as sp

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

# New spectrum series expansions

def diff_expansion(beta, photeng, T, as_pairs=False):
    """ Computes the expansion for small beta and photeng/T. 

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
    tuple of ndarray
        The result of the expansion and the error. Note that the error is a significant overestimate, given by the beta**6 term, when the error should be O(beta**8). 

    """
    x1 = photeng/T

    large = (x1 > 0.01)
    small = ~large

    P_beta_0 = np.zeros(x1.size)
    P_beta_2 = np.zeros(x1.size)
    P_beta_4 = np.zeros(x1.size)
    P_beta_6 = np.zeros(x1.size)

    if np.any(large):

        y = x1[large]

        P_beta_0[large] = 32/3*y**2*np.exp(-y)/(1 - np.exp(-y))

        P_beta_2[large] = 32/9*y**3/(1 - np.exp(-y))**3*(
            np.exp(-2*y)*(y + 4) + np.exp(-y)*(y - 4)
        )

        P_beta_4[large] = 16/225*y**3/(1 - np.exp(-y))**5*(
            np.exp(-y)*(7*y**3 - 84*y**2 + 260*y - 200)
            + np.exp(-2*y)*(77*y**3 - 252*y**2 - 260*y + 600)
            + np.exp(-3*y)*(77*y**3 + 252*y**2 - 260*y - 600)
            + np.exp(-4*y)*(7*y**3 + 84*y**2 + 260*y + 200)
        )

        P_beta_6[large] = 16/4725*y**3/(1 - np.exp(-y))**7*(
            np.exp(-y)*(
                11*y**5 - 264*y**4 + 2142*y**3
                - 7224*y**2 + 9870*y - 4200
            )
            + 3*np.exp(-2*y)*(
                209*y**5 - 2200*y**4 + 6426*y**3
                -2408*y**2 - 9870*y + 7000
            )
            + 2*np.exp(-3*y)*(
                1661*y**5 - 5280*y**4 - 10710*y**3
                + 28896*y**2 + 9870*y - 21000
            )
            + 2*np.exp(-4*y)*(
                1661*y**5 + 5280*y**4 - 10710*y**3
                - 28896*y**2 + 9870*y + 21000
            )
            + 3*np.exp(-5*y)*(
                209*y**5 + 2200*y**4 + 6426*y**3
                + 2408*y**2 - 9870*y - 7000
            )
            + np.exp(-6*y)*(
                11*y**5 + 264*y**4 + 2142*y**3
                + 7224*y**2 + 9870*y + 4200
            )
        )

    if np.any(small):

        y = x1[small]

        P_beta_0[small] = (
            32/3*y -16/3*y**2 + 8/9*y**3 - 2/135*y**5 
            + 1/2835*y**7 - 1/113400*y**9 + 1/4490640*y**11
        )

        P_beta_2[small] = (
            -64/9*y + 32/27*y**3 - 4/45*y**5
            + 8/1701*y**7 - 1/4860*y**9 + 1/124740*y**11
        )

        P_beta_4[small] = (
            -256/225*y + 32/27*y**3 - 296/1125*y**5
            + 1208/42525*y**7 - 64/30375*y**9 + 389/3118500*y**11
        )

        P_beta_6[small] = (
            -832/1575*y + 32/27*y**3 - 1828/3375*y**5 + 31352/297675*y**7
            - 10669/850500*y**9 + 10267/9355500*y**11    
        )

    if as_pairs:
        ans = (
            P_beta_0 + P_beta_2*beta**2 
            + P_beta_4*beta**4 + P_beta_6*beta**6
        )
        err = P_beta_6*beta**6
        
    else:
        ans = (
            np.outer(np.ones_like(beta), P_beta_0)
            + np.outer(beta**2, P_beta_2)
            + np.outer(beta**4, P_beta_4)
            + np.outer(beta**6, P_beta_6)
        )
        err = np.outer(beta**6, P_beta_6)

    return ans,err
