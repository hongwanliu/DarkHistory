"""Nonrelativistic ICS spectrum after integrating over CMB."""

import numpy as np

from darkhistory.electrons.ics.bose_einstein_integrals import *
from darkhistory.electrons.ics.engloss_diff_terms import *
from darkhistory import physics as phys


from tqdm import tqdm_notebook as tqdm

def engloss_spec_series(eleceng, delta, T, as_pairs=False):
    """Nonrelativistic ICS energy loss spectrum using the series method. 

    Parameters
    ----------
    eleceng : ndarray
        Incoming electron energy. 
    delta : ndarray
        Energy gained by photon after upscattering (only positive values). 
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

    testing = False
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
        Energy gained by photon after upscattering (only positive values).  
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
    print('(5/5) Computing F_rem term...')
    F_rem_term = F_rem(beta, delta, T, as_pairs=as_pairs)

    testing = False
    if testing:
        print('***** Diagnostics *****')
        print('beta: ', beta)
        print('delta/T: ', delta/T)
        print('delta/(2*beta*T): ', delta/(2*beta*T))

        print('***** Individual terms *****')
        print('F1_up_down_term: ', F1_up_down_term)
        print('F0_up_down_diff_term: ', F0_up_down_diff_term)
        print('F0_up_down_sum_term: ', F0_up_down_sum_term)
        print('F_inv_up_down_term: ', F_inv_up_down_term)
        print('F_rem_term: ', F_rem_term)

        print('***** Total Sum (Excluding Prefactor) *****')
        print(
            np.transpose(
                prefac*np.transpose(
                    F1_up_down_term + F0_up_down_diff_term 
                    + F0_up_down_sum_term
                    + F_inv_up_down_term + F_rem_term
                )
            )
        )
        print('***** End Diagnostics *****')

    term = np.transpose(
        prefac*np.transpose(
            F1_up_down_term + F0_up_down_diff_term 
            + F0_up_down_sum_term
            + F_inv_up_down_term + F_rem_term
        )
    )

    print('Computation by expansion in beta complete!')

    return term

def engloss_spec(eleceng, delta, T):
    """ Energy loss ICS spectrum. 

    Switches between `engloss_spec_series` and `engloss_spec_diff`. 

    Parameters
    ----------
    eleceng : ndarray
        Incoming electron energy. 
    delta : ndarray
        Energy gained by photon after upscattering (only positive values). 
    T : float
        CMB temperature. 

    Returns
    -------
    ndarray
        dN/(dt d delta) of the outgoing photons, with abscissa given by (eleceng, delta). 
    """

    print('Initializing...')

    gamma = eleceng/phys.me
    beta = np.sqrt((eleceng**2/phys.me**2 - 1)/(gamma**2))
    eta = delta/T

    # 2D masks, dimensions (eleceng, delta)

    beta_2D_mask = np.outer(beta, np.ones_like(eta))
    eleceng_2D_mask = np.outer(eleceng, np.ones_like(eta))
    delta_2D_mask = np.outer(np.ones_like(eleceng), delta)

    beta_2D_small = beta_2D_mask < 0.05

    spec = np.zeros((eleceng.size, delta.size), dtype='float128')

    spec_with_diff = engloss_spec_diff(
        eleceng_2D_mask[beta_2D_small].flatten(),
        delta_2D_mask[beta_2D_small].flatten(),
        T, as_pairs=True
    )

    spec[beta_2D_small] = spec_with_diff.flatten()

    spec_with_series = engloss_spec_series(
        eleceng_2D_mask[~beta_2D_small].flatten(),
        delta_2D_mask[~beta_2D_small].flatten(),
        T, as_pairs=True
    )

    spec[~beta_2D_small] = spec_with_series.flatten()

    print('Energy loss spectrum computed!')

    return spec

