"""Nonrelativistic ICS spectrum after integrating over CMB."""

import numpy as np 
from scipy.integrate import quad

from darkhistory.electrons.ics.series import *
from darkhistory.utilities import log_1_plus_x
from darkhistory.utilities import div_ignore_by_zero
from darkhistory import physics as phys


from tqdm import tqdm_notebook as tqdm

def nonrel_spec_series(eleceng, photeng, T, as_pairs=False):
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

    print('Computing spectra by analytic series...')

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

    prefac = np.float128( 
        phys.c*(3/8)*phys.thomson_xsec/(2*gamma**3*beta**2)
        * (8*np.pi/(phys.ele_compton*phys.me)**3) 
        * (1+beta**2)/beta**2*np.sqrt((1+beta)/(1-beta))
    )

    print('Computing series 1/8...')
    F1_low = F1(lowlim, eta)
    print('Computing series 2/8...')
    F0_low = F0(lowlim, eta)
    print('Computing series 3/8...')
    F_inv_low = F_inv(lowlim, eta)[0]
    print('Computing series 4/8...')
    F_log_low = F_log(lowlim, eta)[0]

    F1_upp = F1(eta, upplim)
    print('Computing series 5/8...')
    F0_upp = F0(eta, upplim)
    print('Computing series 6/8...')
    F_inv_upp = F_inv(eta, upplim)[0]
    print('Computing series 7/8...')
    F_log_upp = F_log(eta, upplim)[0]
    print('Computing series 8/8...')

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

    print('Computation by analytic series complete!')

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

def nonrel_spec_diff(eleceng, photeng, T, as_pairs=False):
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

    print('Computing spectra by an expansion in beta...')

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

    print('Computing Q and K terms...')
    Q_and_K_term = Q_and_K(beta, photeng, T, as_pairs=as_pairs)
    print('Computing H and G terms...')
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

    print('Computation by expansion in beta complete!')

    return term, err

def nonrel_spec(eleceng, photeng, T):
    """ Nonrelativistic ICS spectrum.

    Switches between `nonrel_spec_diff` and `nonrel_spec_series`. 

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

    print('Initializing...')

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
    eta_2D_small  = (eta_2D_mask < 0.1/beta_2D_mask)

    where_diff = (beta_2D_small & eta_2D_small)
    
    testing = False

    if testing:
        print('where_diff on (eleceng, photeng) grid: ')
        print(where_diff)

    spec = np.zeros((eleceng.size, photeng.size), dtype='float128')
    epsrel = np.zeros((eleceng.size, photeng.size), dtype='float128')

    spec_with_diff, err_with_diff = nonrel_spec_diff(
        eleceng_2D_mask[where_diff].flatten(), 
        photeng_2D_mask[where_diff].flatten(), 
        T, as_pairs=True
    )


    print('Computing errors for beta expansion method...')

    spec[where_diff] = spec_with_diff.flatten()
    epsrel[where_diff] = np.abs(
        np.divide(
            err_with_diff.flatten(),
            spec[where_diff],
            out = np.zeros_like(err_with_diff.flatten()),
            where = (spec[where_diff] != 0)
        )
    )
    
    if testing:
        print('spec from nonrel_spec_diff: ')
        print(spec)
        print('epsrel from nonrel_spec_diff: ')
        print(epsrel)

    where_series = (~where_diff) | (epsrel > 1e-3)

    if testing:
    
        print('where_series on (eleceng, photeng) grid: ')
        print(where_series)

    spec_with_series = nonrel_spec_series(
        eleceng_2D_mask[where_series].flatten(),
        photeng_2D_mask[where_series].flatten(),
        T, as_pairs=True
    )

    spec[where_series] = spec_with_series.flatten()

    if testing:
        spec_with_series = np.array(spec)
        spec_with_series[~where_series] = 0
        print('spec from nonrel_spec_series: ')
        print(spec_with_series)
        print('*********************')
        print('Final Result: ')
        print(spec)

    print('Spectrum computed!')

    return spec

def rel_spec(eleceng, photeng, T, inf_upp_bound=False, as_pairs=False):
    """ Relativistic ICS spectrum.

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
    print('Initializing...')

    gamma = eleceng/phys.me

    # Most accurate way of finding beta when beta is small, I think.

    if as_pairs:
        Gamma_eps_q = (
            np.divide(
                photeng/eleceng,
                1 - photeng/eleceng,
                out = np.zeros_like(photeng),
                where = 1 - photeng/eleceng != 0
            )
        )
        B = phys.me/(4*gamma)*Gamma_eps_q
        lowlim = B/T
        if inf_upp_bound:
            upplim = np.inf*np.ones_like(gamma)
        else:
            upplim = 4*(gamma**2)*B/T
        
    else: 
        photeng_to_eleceng = np.outer(1/eleceng, photeng)
        Gamma_eps_q = (
            np.divide(
                photeng_to_eleceng,
                1 - photeng_to_eleceng,
                out = np.zeros_like(photeng_to_eleceng),
                where = 1 - photeng_to_eleceng != 0
            )
        )
        B = np.transpose(
            phys.me/(4*gamma)*np.transpose(Gamma_eps_q)
        )
        lowlim = B/T
        if inf_upp_bound:
            upplim = np.inf*np.ones_like(photeng_to_eleceng)
        else:
            upplim = np.transpose(
                4*gamma**2*np.transpose(B)/T
            )
        
    spec = np.zeros_like(Gamma_eps_q)
    F1_int = np.zeros_like(Gamma_eps_q)
    F0_int = np.zeros_like(Gamma_eps_q)
    F_inv_int = np.zeros_like(Gamma_eps_q)
    F_log_int = np.zeros_like(Gamma_eps_q)

    term_1 = np.zeros_like(Gamma_eps_q)
    term_2 = np.zeros_like(Gamma_eps_q)
    term_3 = np.zeros_like(Gamma_eps_q)
    term_4 = np.zeros_like(Gamma_eps_q)

    good = (lowlim > 0)

    Q = np.zeros_like(Gamma_eps_q)

    Q[good] = (1/2)*Gamma_eps_q[good]**2/(1 + Gamma_eps_q[good])

    prefac = np.float128( 
        6*np.pi*phys.thomson_xsec*phys.c*T/(gamma**2)
        /(phys.ele_compton*phys.me)**3
    )

    print('Computing series 1/4...')
    F1_int[good] = F1(lowlim[good], upplim[good])
    print('Computing series 2/4...')
    F0_int[good] = F0(lowlim[good], upplim[good])
    print('Computing series 3/4...')
    F_inv_int[good] = F_inv(lowlim[good], upplim[good])[0]
    print('Computing series 4/4...')
    F_log_int[good] = F_log(lowlim[good], upplim[good])[0]

    term_1[good] = (1 + Q[good])*T*F1_int[good]
    term_2[good] = (
        (1 + 2*np.log(B[good]/T) - Q[good])*B[good]*F0_int[good]
    )
    term_3[good] = -2*B[good]*F_log_int[good]
    term_4[good] = -2*B[good]**2/T*F_inv_int[good]
    


    testing = True
    if testing:
        print('***** Diagnostics *****')
        print('gamma: ', gamma)
        print('lowlim: ', lowlim)
        print('lowlim*T: ', lowlim*T)
        print('upplim: ', upplim)
        print('upplim*T: ', upplim*T)
        print('Gamma_eps_q: ', Gamma_eps_q)
        print('Q: ', Q)
        print('B: ', B)

        print('***** Integrals *****')
        print('term_1: ', term_1)
        # term_1_quad = quad(
        #     lambda x: x/(np.exp(x) - 1), lowlim[0,0], 
        #     upplim[0,0], epsabs = 0, epsrel = 1e-10
        # )[0]*(1 + Q)*T
        # print('term_1 by quadrature: ', term_1_quad)
        print('term_2: ', term_2)
        print('term_3: ', term_3)
        print('term_4: ', term_4)
        print('Sum of terms: ', term_1+term_2+term_3+term_4)

        print('Final answer: ', 
            np.transpose(
                prefac*np.transpose(
                    term_1 + term_2 + term_3 + term_4
                )
            )
        )
        
        print('***** End Diagnostics *****')

    print('Relativistic Computation Complete!')

    spec[good] = (
        term_1[good] + term_2[good] + term_3[good] + term_4[good]
    )

    return np.transpose(
        prefac*np.transpose(spec)
    )

def engloss_spec_series(eleceng, delta, T, as_pairs=False):
    """Nonrelativistic ICS energy loss spectrum using the series method. 

    Parameters
    ----------
    eleceng : ndarray
        Incoming electron energy. 
    delta : ndarray
        Upscattered photon energy (only positive values). 
    T : float
        CMB temperature. 
    as_pairs : bool
        If true, treats eleceng and delta as a paired list: produces eleceng.size == photeng.size values. Otherwise, gets the spectrum at each delta for each eleceng, return an array of length eleceng.size*delta.size. 

    Returns
    -------
    ndarray
        dN/(dt d(delta)) of the outgoing photons, with abscissa delta.

    Note
    ----
    The final result dN/(dt d(delta)) is the *net* spectrum, i.e. the total number of photons upscattered by delta - number of photons downscattered by delta. 

    """

    print('Computing energy loss spectrum by analytic series...')

    gamma = eleceng/phys.me
    # Most accurate way of finding beta when beta is small, I think.
    beta = np.sqrt((eleceng**2/phys.me**2 - 1)/(gamma**2))

    if as_pairs:
        # neg denotes delta < 0
        lowlim_down = (1+beta)/(2*beta)*delta/T 
        lowlim_up = (1-beta)/(2*beta)*delta/T
    else:
        lowlim_down = np.outer((1+beta)/(2*beta), delta/T)
        lowlim_up = np.outer((1-beta)/(2*beta), delta/T)

    prefac = np.float128( 
        phys.c*(3/8)*phys.thomson_xsec/(2*gamma**3*beta**2)
        * (8*np.pi/(phys.ele_compton*phys.me)**3)
    )

    inf_array = np.inf*np.ones_like(lowlim_down)

    print('Computing upscattering loss spectra...')

    print('Computing series 1/7...')
    F1_up = F1(lowlim_up, inf_array)
    print('Computing series 2/7...')
    F0_up = F0(lowlim_up, inf_array)
    print('Computing series 3/7...')
    F_inv_up = F_inv(lowlim_up, inf_array)[0]
    print('Computing series 4/7...')
    F_x_log_up = F_x_log(lowlim_up, inf_array)[0]
    print('Computing series 5/7...')
    F_log_up = F_log(lowlim_up, inf_array)[0]
    print('Computing series 6/7...')
    F_x_log_a_up = F_x_log_a(lowlim_up, delta/T)[0]
    print('Computing series 7/7...')
    F_log_a_up = F_log_a(lowlim_up, delta/T)[0]

    print('Computing downscattering loss spectra...')

    print('Computing series 1/7...')
    F1_down = F1(lowlim_down, inf_array)
    print('Computing series 2/7...')
    F0_down = F0(lowlim_down, inf_array)
    print('Computing series 3/7...')
    F_inv_down = F_inv(lowlim_down, inf_array)[0]
    print('Computing series 4/7...')
    F_x_log_down = F_x_log(lowlim_down, inf_array)[0]
    print('Computing series 5/7...')
    F_log_down = F_log(lowlim_down, inf_array)[0]
    print('Computing series 6/7...')
    F_x_log_a_down = F_x_log_a(lowlim_down, -delta/T)[0]
    print('Computing series 7/7...')
    F_log_a_down = F_log_a(lowlim_down, -delta/T)[0]

    term_1_up = np.transpose(
        (
            (2/beta)*np.sqrt((1+beta)/(1-beta))
            + (2/beta)*np.sqrt((1-beta)/(1+beta))
            + 2/(gamma*beta**2)*np.log((1-beta)/(1+beta))
        )*np.transpose(T**2*F1_up)
    )

    term_0_up = np.transpose(
        (
            (2/beta)*np.sqrt((1-beta)/(1+beta))
            - 2*(1-beta)**2/beta**2*np.sqrt((1+beta)/(1-beta))
            + 2/(gamma*beta**2)*np.log((1-beta)/(1+beta))
        )*np.transpose(T*delta*F0_up)
    )

    term_inv_up = np.transpose(
        -(1-beta)**2/beta**2*np.sqrt((1+beta)/(1-beta))*
        np.transpose(delta**2*F_inv_up)
    )

    term_log_up = np.transpose(
        2/(gamma*beta**2)*np.transpose(
            -T**2*F_x_log_up - delta*T*F_log_up
            + T**2*F_x_log_a_up + delta*T*F_log_a_up
        )
    )

    term_1_down = np.transpose(
        (
            (2/beta)*np.sqrt((1-beta)/(1+beta))
            + (2/beta)*np.sqrt((1+beta)/(1-beta))
            - 2/(gamma*beta**2)*np.log((1+beta)/(1-beta))
        )*np.transpose(T**2*F1_down)
    )

    term_0_down = -np.transpose(
        (
            (2/beta)*np.sqrt((1+beta)/(1-beta))
            + 2*(1+beta)**2/beta**2*np.sqrt((1-beta)/(1+beta))
            - 2/(gamma*beta**2)*np.log((1+beta)/(1-beta))
        )*np.transpose(T*delta*F0_down)
    )

    term_inv_down = np.transpose(
        (1+beta)**2/beta**2*np.sqrt((1-beta)/(1+beta))*
        np.transpose(delta**2*F_inv_down)
    )

    term_log_down = np.transpose(
        2/(gamma*beta**2)*np.transpose(
            T**2*F_x_log_down - delta*T*F_log_down
            - T**2*F_x_log_a_down + delta*T*F_log_a_down
        )
    )

    testing = True
    if testing:
        print('***** Diagnostics *****')
        print('lowlim_up: ', lowlim_up)
        print('lowlim_down: ', lowlim_down)
        print('beta: ', beta)
        print('delta/T: ', delta/T)

        print('***** Individual terms *****')
        print('term_1_up: ', term_1_up)
        print('term_0_up: ', term_0_up)
        print('term_inv_up: ', term_inv_up)
        print('term_log_up: ', term_log_up)
        print('term_1_down: ', term_1_down)
        print('term_0_down: ', term_0_down)
        print('term_inv_down: ', term_inv_down)
        print('term_log_down: ', term_log_down)

        print('***** Upscatter and Downscatter Differences*****')
        print('term_1: ', term_1_up - term_1_down)
        print('term_0: ', term_0_up - term_0_down)
        print('term_inv: ', term_inv_up - term_inv_down)
        print('term_log: ', term_log_up - term_log_down)
        print('Sum three terms: ', term_0_up - term_0_down
            + term_inv_up - term_inv_down
            + term_log_up - term_log_down
        )
        print('Sum: ', term_1_up - term_1_down
            + term_0_up - term_0_down
            + term_inv_up - term_inv_down
            + term_log_up - term_log_down)

        print('***** Total Sum (Excluding Prefactor) *****')
        print('Upscattering loss spectrum: ')
        print(
            np.transpose(prefac*np.transpose(
                term_1_up + term_0_up + term_inv_up + term_log_up
            )
        ))
        print('Downscattering loss spectrum: ')
        print(
            np.transpose(prefac*np.transpose(
                term_1_down + term_0_down + term_inv_down 
                + term_log_down
            )
        ))
        print('***** End Diagnostics *****')

    return np.transpose(
        prefac*np.transpose(
            term_1_up + term_0_up + term_inv_up + term_log_up
            - term_1_down - term_0_down - term_inv_down
            - term_log_down
        )
    )

def engloss_spec_diff(eleceng, delta, T, as_pairs=False):
    """Nonrelativistic ICS energy loss spectrum by beta expansion. 

    Parameters
    ----------
    eleceng : ndarray
        Incoming electron energy. 
    delta : ndarray
        Energy gained from upscattering by the secondary photon. 
    T : float
        CMB temperature. 
    as_pairs : bool
        If true, treats eleceng and delta as a paired list: produces eleceng.size == photeng.size values. Otherwise, gets the spectrum at each delta for each eleceng, return an array of length eleceng.size*delta.size. 

    Returns
    -------
    ndarray
        dN/(dt d(delta)) of the outgoing photons, with abscissa delta.

    Note
    ----
    The final result dN/(dt d(delta)) is the *net* spectrum, i.e. the total number of photons upscattered by delta - number of photons downscattered by delta. 

    """

    print('Computing energy loss spectrum by beta expansion...')

    gamma = eleceng/phys.me
    beta = np.sqrt((eleceng**2/phys.me**2 - 1)/(gamma**2))

    prefac = ( 
        phys.c*(3/8)*phys.thomson_xsec/(2*gamma**3*beta**2)
        * (8*np.pi/(phys.ele_compton*phys.me)**3)
        * (T**2/beta**2)
    )

    print('(1/5) Computing F1_up - F1_down term...')
    F1_up_down_term = F1_up_down(beta, delta, T, as_pairs=as_pairs)
    print('(2/5) Computing F0_up - F0_down term...')
    F0_up_down_diff_term = F0_up_down_diff(beta, delta, T, as_pairs=as_pairs)
    print('(3/5) Computing F0_up + F0_down term...')
    F0_up_down_sum_term = F0_up_down_sum(beta, delta, T, as_pairs=as_pairs)
    print('(4/5) Computing F_inv_up - F_inv_down term...')
    F_inv_up_down_term = F_inv_up_down(beta, delta, T, as_pairs=as_pairs)
    print('(5/5) Computing F_inv2_up - F_inv2_down term...')
    F_inv2_up_down_term = F_inv2_up_down(beta, delta, T, as_pairs=as_pairs)

    print(F1_up_down_term*T**2/beta**2)
    print((F0_up_down_diff_term + F0_up_down_sum_term
            + F_inv_up_down_term 
            + F_inv2_up_down_term)*T**2/beta**2)
    print((F1_up_down_term + F0_up_down_diff_term 
            + F0_up_down_sum_term
            + F_inv_up_down_term + F_inv2_up_down_term)*T**2/beta**2)

    term = np.transpose(
        prefac*np.transpose(
            F1_up_down_term + F0_up_down_diff_term 
            + F0_up_down_sum_term
            + F_inv_up_down_term + F_inv2_up_down_term
        )
    )

    print('Computation by expansion in beta complete!')

    return term