"""Nonrelativistic ICS spectrum after integrating over CMB."""

import numpy as np 
from scipy.integrate import quad

from darkhistory.electrons.ics.ics_nonrel_series import *
from darkhistory.utilities import log_1_plus_x
from darkhistory import physics as phys

def spec_series(eleceng, photeng, T):
    """ Nonrelativistic ICS spectrum using the series method.

    Parameters
    ----------
    eleceng : ndarray
        Incoming electron energy. 
    photeng : ndarray
        Outgoing photon energy. 
    T : float
        CMB temperature. 

    Returns
    -------
    ndarray
        dN/(dt dE) of the outgoing photons, with abscissa photeng. 

    Note
    ----
    Insert note on the suitability of the method. 
    """

    gamma = eleceng/phys.me
    # Most accurate way of finding beta when beta is small, I think.
    beta = np.sqrt((eleceng**2/phys.me**2 - 1)/(gamma**2))

    lowlim = np.array([(1-b)/(1+b)*photeng/T for b in beta])
    upplim = np.array([(1+b)/(1-b)*photeng/T for b in beta])
    eta = photeng/T

    prefac = ( 
        phys.c*(3/8)*phys.thomson_xsec/(2*gamma**3*beta**2)
        * (8*np.pi/(phys.ele_compton*phys.me)**3) 
        * (1+beta**2)/beta**2*np.sqrt((1+beta)/(1-beta))
    )

    F1_low = np.array([F1(low, eta) for low in lowlim])
    F0_low = np.array([F0(low, eta) for low in lowlim])
    F_inv_low = np.array([F_inv(low, eta) for low in lowlim])
    F_log_low = np.array([F_log(low, eta) for low in lowlim])

    F1_upp = np.array([F1(eta, upp) for upp in upplim])
    F0_upp = np.array([F0(eta, upp) for upp in upplim])
    F_inv_upp = np.array([F_inv(eta, upp) for upp in upplim])
    F_log_upp = np.array([F_log(eta, upp) for upp in upplim])


    # CMB photon energy less than outgoing photon energy.

    # F1_low an array of size [eleceng, photeng]. 
    # We can take photeng*F1_vec_low for element-wise products. 
    # In the other dimension, we must take transpose(eleceng*transpose(x)).

    term_low_1 = F1_low * T**2

    term_low_2 = np.transpose(
        2*beta/(1+beta**2)*(1-beta)/(1+beta)
        * np.transpose(photeng*F0_low) * T
    )

    term_low_3 = np.transpose(
        -(1-beta)**2/(1+beta**2)
        * np.transpose((photeng**2)*F_inv_low)
    )

    term_low_4 = np.transpose(
        2*(1-beta)/(1+beta**2)*(log_1_plus_x(-beta) - log_1_plus_x(beta))
        * np.transpose(photeng*F0_low*T)
        + 2*(1-beta)/(1+beta**2) 
        * np.transpose(np.log(photeng)*photeng*F0_low*T)
    )

    term_low_5 = np.transpose(
        -2*(1-beta)/(1+beta**2) * np.transpose(
            photeng*(T*np.log(T)*F0_low + F_log_low*T)
        )
    )

    # CMB photon energy higher than outgoing photon energy

    term_high_1 = np.transpose(
        -(1-beta)/(1+beta) * np.transpose(F1_upp * T**2)
    )

    term_high_2 = np.transpose(
        2*beta/(1+beta**2) * np.transpose(photeng * F0_upp * T)
    )

    term_high_3 = np.transpose(
        (1-beta**2)/(1+beta**2) * np.transpose(photeng**2*F_inv_upp)
    )

    term_high_4 = np.transpose(
        - 2*(1-beta)/(1+beta**2)*(log_1_plus_x(beta) - log_1_plus_x(-beta))
        * np.transpose(photeng*F0_upp*T)
        - 2*(1-beta)/(1+beta**2) 
        * np.transpose(np.log(photeng)*photeng*F0_upp*T)
    )

    term_high_5 = np.transpose(
        2*(1-beta)/(1+beta**2) * np.transpose(
            photeng*(T*np.log(T)*F0_upp+ F_log_upp*T)
        )
    )

    testing = False
    if testing:
        print('***** Diagnostics *****')
        print('lowlim: ', lowlim)
        print('upplim: ', upplim)
        print('photeng/T: ', eta)

        print('***** epsilon < epsilon_1 *****')
        print('term_low_1: ', term_low_1)
        print('term_low_2: ', term_low_2)
        print('term_low_3: ', term_low_3)
        print('term_low_4: ', term_low_4)
        print('term_low_5: ', term_low_5)

        print('***** epsilon > epsilon_1 *****')
        print('term_high_1: ', term_high_1)
        print('term_high_2: ', term_high_2)
        print('term_high_3: ', term_high_3)
        print('term_high_4: ', term_high_4)
        print('term_high_5: ', term_high_5)

        print('***** Term Sums *****')
        print('term_low_1 + term_high_1: ', term_low_1 + term_high_1)
        print('term_low_2 + term_high_2: ', term_low_2 + term_high_2)
        print('term_low_3 + term_high_3: ', term_low_3 + term_high_3)
        print('term_low_4 + term_high_4: ', term_low_4 + term_high_4)
        print('term_low_5 + term_high_5: ', term_low_5 + term_high_5)
        
        print('***** Total Sum (Excluding Prefactor) *****')
        print(
            (1+beta**2)/beta**2*np.sqrt((1+beta)/(1-beta))*np.transpose(
                (term_low_1 + term_high_1)
                + (term_low_2 + term_high_2)
                + (term_low_3 + term_high_3)
                + (term_low_4 + term_high_4)
                + (term_low_5 + term_high_5)
            )
        )
        print('***** End Diagnostics *****')

    # Addition ordered to minimize catastrophic cancellation, but if this is important, you shouldn't be using this method.

    return np.transpose(
        prefac*np.transpose(
            (term_low_1 + term_high_1)
            + (term_low_2 + term_high_2)
            + (term_low_3 + term_high_3)
            + (term_low_4 + term_high_4)
            + (term_low_5 + term_high_5)
        )
    )

def spec_quad(eleceng_arr, photeng_arr, T):
    """ Nonrelativistic ICS spectrum using quadrature.

    Parameters
    ----------
    eleceng : ndarray
        Incoming electron energy. 
    photeng : ndarray
        Outgoing photon energy. 
    T : float
        CMB temperature. 

    Returns
    -------
    ndarray
        dN/(dt dE) of the outgoing photons, with abscissa photeng. 

    Note
    ----
    Insert note on the suitability of the method. 
    """

    gamma_arr = eleceng_arr/phys.me

    # Most accurate way of finding beta when beta is small, I think.
    beta_arr = np.sqrt((eleceng_arr**2/phys.me**2 - 1)/(gamma_arr**2))

    lowlim = np.array([(1-b)/(1+b)*photeng_arr for b in beta_arr])
    upplim = np.array([(1+b)/(1-b)*photeng_arr for b in beta_arr])

    def integrand(eps, eleceng, photeng):

        gamma = eleceng/phys.me
        beta = np.sqrt((eleceng**2/phys.me**2 - 1)/(gamma**2))


        prefac = ( 
            phys.c*(3/8)*phys.thomson_xsec/(2*gamma**3*beta**2)
            * (8*np.pi/(phys.ele_compton*phys.me)**3)
        )

        if eps/T < 100:
            prefac *= 1/(np.exp(eps/T) - 1)
        else:
            prefac = 0

        if eps < photeng:

            fac = (
                (1+beta**2)/beta**2*np.sqrt((1+beta)/(1-beta))*eps
                + 2/beta*np.sqrt((1-beta)/(1+beta))*photeng
                - (1-beta)**2/beta**2*np.sqrt((1+beta)/(1-beta))*(
                    photeng**2/eps
                )
                + 2/(gamma*beta**2)*photeng*np.log(
                    (1-beta)/(1+beta)*photeng/eps
                )
            )

        else:

            fac = (
                - (1+beta**2)/beta**2*np.sqrt((1-beta)/(1+beta))*eps
                + 2/beta*np.sqrt((1+beta)/(1-beta))*photeng 
                + (1+beta)/(gamma*beta**2)*photeng**2/eps 
                - 2/(gamma*beta**2)*photeng*np.log(
                    (1+beta)/(1-beta)*photeng/eps 
                )
            )

        return prefac*fac

    integral = np.array([
        [quad(integrand, low, upp, args=(eleceng, photeng), epsrel=1e-10, epsabs=0)[0] 
        for (low, upp, photeng) in zip(low_part, upp_part, photeng_arr)
        ] for (low_part, upp_part, eleceng) 
            in zip(lowlim, upplim, eleceng_arr)
    ]) 

    testing = False
    if testing:
        print('***** Diagnostics *****')
        print('***** Integral (Excluding Prefactor) *****')
        prefac = ( 
            phys.c*(3/8)*phys.thomson_xsec/(2*gamma_arr**3*beta_arr**2)
            * (8*np.pi/(phys.ele_compton*phys.me)**3) 
        )
        print(np.transpose(np.transpose(integral)/prefac))
        print('***** Integration with Error *****')
        print(np.array([
            [quad(integrand, low, upp, args=(eleceng, photeng), 
                epsabs = 0, epsrel=1e-10)
                for (low, upp, photeng) in zip(
                    low_part, upp_part, photeng_arr
                )
            ] for (low_part, upp_part, eleceng) 
                in zip(lowlim, upplim, eleceng_arr)
        ]))
        print('***** End Diagnostics *****')


    return integral

def spec_diff(eleceng, photeng, T):

    gamma = eleceng/phys.me
    # Most accurate way of finding beta when beta is small, I think.
    beta = np.sqrt((eleceng**2/phys.me**2 - 1)/(gamma**2))

    prefac = ( 
        phys.c*(3/8)*phys.thomson_xsec/(2*gamma**3*beta**2)
        * (8*np.pi/(phys.ele_compton*phys.me)**3)
    )

    Q_and_K_term = np.array([Q_and_K(b, photeng, T) 
        for b in beta])

    H_and_G_term = np.array([H_and_G(b, photeng, T)
        for b in beta])

    testing = True
    if testing:
        print('***** Diagnostics for spec_diff *****')
        print('beta: ', beta)
        np.array([Q_and_K(b, photeng, T, epsrel=1e-3) 
            for b in beta])

        np.array([H_and_G(b, photeng, T, epsrel=1e-3) 
            for b in beta])
        print('***** End Diagnostics for spec_diff *****')

    return np.transpose(
        prefac*np.transpose(Q_and_K_term + H_and_G_term)
    )


