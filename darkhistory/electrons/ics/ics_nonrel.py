"""Nonrelativistic ICS spectrum after integrating over CMB."""

import numpy as np 
from scipy.integrate import quad

from darkhistory.electrons.ics.ics_nonrel_series import *
from darkhistory.utilities import log_1_plus_x
from darkhistory.utilities import div_ignore_by_zero
from darkhistory import physics as phys


from tqdm import tqdm_notebook as tqdm

def spec_series(eleceng, photeng, T, as_pairs=False, test=True):
    """ Nonrelativistic ICS spectrum using the series method.

    Parameters
    ----------
    eleceng : ndarray
        Incoming electron energy. 
    photeng : ndarray
        Outgoing photon energy. 
    T : float
        CMB temperature. 
    as_pairs : bool
        If true, treats eleceng and photeng as a paired list: produces eleceng.size == photeng.size values. Otherwise, gets the spectrum at each photeng for each eleceng, returning an array of length eleceng.size*photeng.size. 

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

    if as_pairs:
        lowlim = (1-beta)/(1+beta)*photeng/T 
        upplim = (1+beta)/(1-beta)*photeng/T
    else: 
        lowlim = np.outer((1-beta)/(1+beta), photeng/T)
        upplim = np.outer((1+beta)/(1-beta), photeng/T)
    
    eta = photeng/T

    prefac = ( 
        phys.c*(3/8)*phys.thomson_xsec/(2*gamma**3*beta**2)
        * (8*np.pi/(phys.ele_compton*phys.me)**3) 
        * (1+beta**2)/beta**2*np.sqrt((1+beta)/(1-beta))
    )

    # print('term by term')
    # if test:
    #     print('yes')
    #     print(lowlim)
    #     print(eta)
    # else:
    #     print('no')
    #     print(lowlim[0:-1,0:-1])
    #     print(eta)


    F1_low = F1(lowlim, eta)
    F0_low = F0(lowlim, eta)
    F_inv_low = F_inv(lowlim, eta, test)
    F_log_low = F_log(lowlim, eta)

    # print('term-by-term')
    # if test:
    #     print(F_inv_low)
    # else:
    #     print(F_inv_low[0:-1,0:-1])

    F1_upp = F1(eta, upplim)
    F0_upp = F0(eta, upplim)
    F_inv_upp = F_inv(eta, upplim, test)
    F_log_upp = F_log(eta, upplim)

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
        [quad(integrand, low, upp, args=(eleceng, photeng), epsabs=0)[0] 
        for (low, upp, photeng) in zip(low_part, upp_part, photeng_arr)
        ] for (low_part, upp_part, eleceng) 
            in zip(tqdm(lowlim), upplim, eleceng_arr)
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

def spec_diff(eleceng, photeng, T, as_pairs=False):
    """ Nonrelativistic ICS spectrum by beta expansion.

    Parameters
    ----------
    eleceng : ndarray
        Incoming electron energy. 
    photeng : ndarray
        Outgoing photon energy. 
    T : float
        CMB temperature.
    as_pairs : bool
        If true, treats eleceng and photeng as a paired list: produces eleceng.size == photeng.size values. Otherwise, gets the spectrum at each photeng for each eleceng, returning an array of length eleceng.size*photeng.size. 

    Returns
    -------
    tuple of ndarrays
        dN/(dt dE) of the outgoing photons and the error, with abscissa given by (eleceng, photeng). 

    Note
    ----
    Insert note on the suitability of the method. 
    """
    gamma = eleceng/phys.me
    # Most accurate way of finding beta when beta is small, I think.
    beta = np.sqrt((eleceng**2/phys.me**2 - 1)/(gamma**2))

    testing = False
    if testing: 
        print('beta: ', beta)

    prefac = ( 
        phys.c*(3/8)*phys.thomson_xsec/(2*gamma**3*beta**2)
        * (8*np.pi/(phys.ele_compton*phys.me)**3)
    )

    Q_and_K_term = Q_and_K(beta, photeng, T, as_pairs=as_pairs)
    H_and_G_term = H_and_G(beta, photeng, T, as_pairs=as_pairs)

    term = np.transpose(
        prefac*np.transpose(
            Q_and_K_term[0] + H_and_G_term[0]
        )
    )

    err = np.transpose(
        prefac*np.transpose(
            Q_and_K_term[1] + H_and_G_term[1]
        )
    )

    return term, err

def nonrel_spec(eleceng, photeng, T):
    """ Nonrelativistic ICS spectrum.

    Switches between `spec_diff` and `spec_series`. 

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
    tuple of ndarrays
        dN/(dt dE) of the outgoing photons and the error, with abscissa given by (eleceng, photeng). 

    Note
    ----
    Insert note on the suitability of the method. 
    """

    gamma = eleceng/phys.me
    # Most accurate way of finding beta when beta is small, I think.
    beta = np.sqrt((eleceng**2/phys.me**2 - 1)/(gamma**2))
    eta = photeng/T 

    # 2D masks, dimensions (eleceng, photeng)
    beta_2D_mask = np.outer(beta, np.ones(eta.size))
    eta_2D_mask = np.outer(np.ones(beta.size), eta)
    eleceng_2D_mask = np.outer(eleceng, np.ones(photeng.size))
    photeng_2D_mask = np.outer(np.ones(eleceng.size), photeng)

    # 2D boolean arrays. 
    beta_2D_small = (beta_2D_mask < 0.01)
    eta_2D_small  = (eta_2D_mask < 0.1/beta_2D_mask) & (eta_2D_mask < 100)

    where_diff = (beta_2D_small & eta_2D_small)
    
    print('where_diff on (eleceng, photeng) grid: ')
    print(where_diff)

    spec = np.zeros((eleceng.size, photeng.size))
    epsrel = np.zeros((eleceng.size, photeng.size))

    spec_with_diff, err_with_diff = spec_diff(
        eleceng_2D_mask[where_diff].flatten(), 
        photeng_2D_mask[where_diff].flatten(), 
        T, as_pairs=True
    )

    spec[where_diff] = spec_with_diff.flatten()
    epsrel[where_diff] = np.abs(
        err_with_diff.flatten()/spec[where_diff]
    )
    
    print('spec from spec_diff: ')
    print(spec)
    print('epsrel from spec_diff: ')
    print(epsrel)

    where_series = (~where_diff) | (epsrel > 1e-6)
    print('where_series on (eleceng, photeng) grid: ')
    print(where_series)

    spec_with_series = spec_series(
        eleceng_2D_mask[where_series].flatten(),
        photeng_2D_mask[where_series].flatten(),
        T, as_pairs=True
    )

    spec[where_series] = spec_with_series.flatten()
    spec_with_series = np.array(spec)
    spec_with_series[~where_series] = 0
    print('spec from spec_series: ')
    print(spec_with_series)
    print('*********************')
    print('Final Result: ')
    print(spec)

    return spec





