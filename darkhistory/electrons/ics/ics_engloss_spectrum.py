"""Nonrelativistic ICS spectrum after integrating over CMB."""

import numpy as np

from darkhistory import physics as phys
from darkhistory import utilities as utils
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.transferfunction import TransFuncAtRedshift

from darkhistory.electrons.ics.BE_integrals import *
from darkhistory.electrons.ics.engloss_diff_terms import *
import darkhistory.electrons.ics.ics_spectrum as ics_spectrum


from tqdm import tqdm_notebook as tqdm

def engloss_spec_series(eleckineng, delta, T, as_pairs=False):
    """Thomson ICS scattered electron energy loss spectrum, series method. 

    Parameters
    ----------
    eleckineng : ndarray
        Incoming electron kinetic energy. 
    delta : ndarray
        Energy lost by electron after one scatter (only positive values). 
    T : float
        CMB temperature. 
    as_pairs : bool
        If true, treats eleckineng and delta as a paired list: produces eleckineng.size == photeng.size values. Otherwise, gets the spectrum at each delta for each eleckineng, return an array of length eleckineng.size*delta.size. 

    Returns
    -------
    ndarray
        dN/(dt d(delta)) of the scattered electrons, with abscissa delta.

    """

    print('****** Energy Loss Spectrum by Analytic Series ******')

    gamma = eleckineng/phys.me + 1
    # Most accurate way of finding beta when beta is small, I think.
    beta = np.sqrt(eleckineng/phys.me*(gamma+1)/gamma**2)

    if as_pairs:
        # neg denotes delta < 0
        lowlim_down = (1+beta)/(2*beta)*delta/T 
        lowlim_up = (1-beta)/(2*beta)*delta/T
    else:
        lowlim_down = np.outer((1+beta)/(2*beta), delta/T)
        lowlim_up = np.outer((1-beta)/(2*beta), delta/T)

    prefac = np.float128(
        phys.c*(3/8)*phys.thomson_xsec/(4*gamma**2*beta**6)
        * (8*np.pi*T**2/(phys.ele_compton*phys.me)**3)
    )

    inf_array = np.inf*np.ones_like(lowlim_down)

    print('Computing upscattering loss spectra...')

    print('    Computing series 1/8...')
    F1_up = F1(lowlim_up, inf_array)
    print('    Computing series 2/8...')
    F0_up = F0(lowlim_up, inf_array)
    print('    Computing series 3/8...')
    F_inv_up = F_inv(lowlim_up, inf_array)[0]
    print('    Computing series 4/8...')
    F_x_log_up = F_x_log(lowlim_up, inf_array)[0]
    print('    Computing series 5/8...')
    F_log_up = F_log(lowlim_up, inf_array)[0]
    print('    Computing series 6/8...')
    F_x_log_a_up = F_x_log_a(lowlim_up, delta/T)[0]
    print('    Computing series 7/8...')
    F_log_a_up = F_log_a(lowlim_up, delta/T)[0]
    print('    Computing series 8/8...')
    F_inv_a_up = F_inv_a(lowlim_up, delta/T)[0]
    

    print('Computing downscattering loss spectra...')

    print('    Computing series 1/8...')
    F1_down = F1(lowlim_down, inf_array)
    print('    Computing series 2/8...')
    F0_down = F0(lowlim_down, inf_array)
    print('    Computing series 3/8...')
    F_inv_down = F_inv(lowlim_down, inf_array)[0]
    print('    Computing series 4/8...')
    F_x_log_down = F_x_log(lowlim_down, inf_array)[0]
    print('    Computing series 5/8...')
    F_log_down = F_log(lowlim_down, inf_array)[0]
    print('    Computing series 6/8...')
    F_x_log_a_down = F_x_log_a(lowlim_down, -delta/T)[0]
    print('    Computing series 7/8...')
    F_log_a_down = F_log_a(lowlim_down, -delta/T)[0]
    print('    Computing series 8/8...')
    F_inv_a_down = F_inv_a(lowlim_down, -delta/T)[0]


    ### Upscattered terms, i.e. delta > 0 ###
    term_inv_a_up = np.transpose(
        1/gamma**4*np.transpose((delta/T)**2*F_inv_a_up)
    )
    term_inv_up = np.transpose(
        -1/gamma**4*np.transpose((delta/T)**2*F_inv_up)
    )
    
    term_1_up = np.transpose(
        (
            -2/gamma**2*(3-beta**2)*(np.log1p(beta) - np.log1p(-beta))
            -3/gamma**4 + (1-beta)*(
                beta*(beta**2 + 3) - 1/gamma**2*(9 - 4*beta**2)
            )
        )*np.transpose(delta/T*F0_up)
    )

    term_log_up = np.transpose(
        -2/gamma**2*(3 - beta**2)*np.transpose(delta/T*F_log_up)
    )
    term_log_a_up = np.transpose(
        2/gamma**2*(3 - beta**2)*np.transpose(delta/T*F_log_a_up)
    )

    term_x_up = np.transpose(
        (
            -4/gamma**2*(3-beta**2)*(np.log1p(beta) - np.log1p(-beta))
            + 2*beta*(beta**2 + 3 + 1/gamma**2*(9 - 4*beta**2))
        )*np.transpose(F1_up)
    )

    term_x_log_up = np.transpose(
        -4/gamma**2*(3-beta**2)*np.transpose(F_x_log_up)
    )
    term_x_log_a_up = np.transpose(
        4/gamma**2*(3-beta**2)*np.transpose(F_x_log_a_up)
    )
    ### Downscattered terms, i.e. delta < 0 ###
    # We take the input delta > 0. 
    term_inv_a_down = np.transpose(
        -1/gamma**4*np.transpose((-delta/T)**2*F_inv_a_down)
    )
    term_inv_down = np.transpose(
        1/gamma**4*np.transpose((-delta/T)**2*F_inv_down)
    )
    
    term_1_down = np.transpose(
        (
            2/gamma**2*(3-beta**2)*(np.log1p(-beta) - np.log1p(beta))
            +3/gamma**4 + (1+beta)*(
                beta*(beta**2 + 3) + 1/gamma**2*(9 - 4*beta**2)
            )
        )*np.transpose(-delta/T*F0_down)
    )

    term_log_down = np.transpose(
        2/gamma**2*(3 - beta**2)*np.transpose(-delta/T*F_log_down)
    )
    term_log_a_down = np.transpose(
        -2/gamma**2*(3 - beta**2)*np.transpose(-delta/T*F_log_a_down)
    )

    term_x_down = np.transpose(
        (
            4/gamma**2*(3-beta**2)*(np.log1p(-beta) - np.log1p(beta))
            + 2*beta*(beta**2 + 3 + 1/gamma**2*(9 - 4*beta**2))
        )*np.transpose(F1_down)
    )

    term_x_log_down = np.transpose(
        4/gamma**2*(3-beta**2)*np.transpose(F_x_log_down)
    )
    term_x_log_a_down = np.transpose(
        -4/gamma**2*(3-beta**2)*np.transpose(F_x_log_a_down)
    )

    sum_terms = (
        term_inv_a_up - term_inv_a_down 
        + term_inv_up - term_inv_down
        + term_1_up - term_1_down 
        + term_log_up - term_log_down 
        + term_log_a_up - term_log_a_down 
        + term_x_up - term_x_down 
        + term_x_log_up - term_x_log_down 
        + term_x_log_a_up - term_x_log_a_down
    )

    print('****** Complete! ******')

    return np.transpose(prefac*np.transpose(sum_terms))

def engloss_spec_diff(eleckineng, delta, T, as_pairs=False):
    """Thomson ICS scattered electron energy loss spectrum, beta expansion. 

    Parameters
    ----------
    eleckineng : ndarray
        Incoming electron energy. 
    delta : ndarray
        Energy lost by electron after one scatter (only positive values).  
    T : float
        CMB temperature. 
    as_pairs : bool
        If true, treats eleckineng and delta as a paired list: produces eleckineng.size == photeng.size values. Otherwise, gets the spectrum at each delta for each eleckineng, return an array of length eleckineng.size*delta.size.

    Returns
    -------
    ndarray
        dN/(dt d(delta)) of the scattered electrons, with abscissa delta.

    """

    print('****** Energy Loss Spectrum by beta Expansion ******')

    gamma = eleckineng/phys.me + 1
    beta = np.sqrt(eleckineng/phys.me*(gamma+1)/gamma**2)

    prefac = (
        phys.c*(3/8)*phys.thomson_xsec/4
        *(8*np.pi/(phys.ele_compton*phys.me)**3)
        *T**2
    )

    diff_term = engloss_diff_expansion(beta, delta, T, as_pairs=as_pairs)

    term = np.transpose(prefac*np.transpose(diff_term))

    print('****** Complete! ******')

    return term

def engloss_spec(
    eleckineng, delta, T, 
    as_pairs=False, thomson_only=False, thomson_tf=None, rel_tf=None,
):
    """ Thomson ICS scattered electron energy loss spectrum. 

    Switches between :func:`.engloss_spec_series` and :func:`.engloss_spec_diff` in the Thomson regime. Also switches between Thomson and relativistic regimes automatically.

    Parameters
    ----------
    eleckineng : ndarray
        Incoming electron kinetic energy. 
    delta : ndarray
        Energy gained by photon after upscattering (only positive values). 
    T : float
        CMB temperature.
    as_pairs : bool, optional
        If true, treats eleckineng and photeng as a paired list: produces eleckineng.size == photeng.size values. Otherwise, gets the spectrum at each photeng for each eleckineng, returning an array of length eleckineng.size*photeng.size. 
    thomson_only : bool, optional
        If true, only returns the Thomson energy loss spectrum, and never switches to the relativistic case. 
    thomson_tf : TransFuncAtRedshift, optional
        Reference Thomson energy loss ICS spectrum. If specified, calculation is done by interpolating over the transfer function. 
    rel_tf : TransFuncAtRedshift, optional
        Reference relativistic energy loss ICS spectrum. If specified, calculation is done by interpolating over the transfer function. 

    Returns
    -------
    TransFuncAtRedshift or ndarray
        dN/(dt d Delta) of the outgoing photons (dt = 1 s). If as_pairs == False, returns a TransFuncAtRedshift, with abscissa given by (eleckineng, delta). Otherwise, returns an ndarray, with abscissa given by each pair of (eleckineng, delta). 
    """

    gamma = eleckineng/phys.me + 1
    eleceng = eleckineng + phys.me
    beta = np.sqrt(eleckineng/phys.me*(gamma+1)/gamma**2)
    eta = delta/T

    # where to switch between Thomson and relativistic treatments.
    if thomson_only:
        rel_bound = np.inf 
    else:
        rel_bound = 20

    # 2D masks have dimensions (eleceng, delta).

    if as_pairs:
        if eleckineng.size != delta.size:
            raise TypeError('delta and electron energy arrays must have the same length for pairwise computation.')
        gamma_mask = gamma
        beta_mask = beta
        eleckineng_mask = eleckineng 
        eleceng_mask = eleceng
        delta_mask = delta
        spec = np.zeros_like(gamma)
    else:
        gamma_mask = np.outer(gamma, np.ones_like(eta))
        beta_mask = np.outer(beta, np.ones_like(eta))
        eleckineng_mask = np.outer(eleckineng, np.ones_like(eta))
        eleceng_mask = np.outer(eleceng, np.ones_like(eta))
        delta_mask = np.outer(np.ones_like(eleckineng), delta)
        spec = np.zeros(
            (eleckineng.size, delta.size), dtype='float128'
        )

    beta_small = beta_mask < 0.1
    
    rel = gamma_mask > rel_bound

    y = T/phys.TCMB(400)

    if not thomson_only:
        if rel_tf != None:
            if as_pairs:
                raise TypeError('When reading from file, the keyword as_pairs is not supported.')
            # If the electron energy at which interpolation is to be taken is outside rel_tf, then an error should be returned, since the file has not gone up to high enough energies.
            #rel_tf = rel_tf.at_in_eng(y*eleceng[gamma > rel_bound])
            # If the photon energy at which interpolation is to be taken is outside rel_tf, then for large photon energies, we set it to zero, since the spectrum should already be zero long before. If it is below, nan is returned, and the results should not be used.

            rel_tf_interp = np.transpose(
                rel_tf.interp_func(
                    np.log(y*eleceng[gamma > rel_bound]), np.log(y*delta)
                )
            )    

            spec[rel] = y**4*rel_tf_interp.flatten()

        else:

            print(
                '###### RELATIVISTIC ENERGY LOSS SPECTRUM ######'
            )

            spec[rel] = ics_spectrum.rel_spec(
                eleceng_mask[rel],
                delta_mask[rel],
                T, inf_upp_bound=True, as_pairs=True 
            )

            print('###### COMPLETE! ######')

    if thomson_tf != None:
        
        thomson_tf_interp = np.transpose(
            thomson_tf.interp_func(
                np.log(eleckineng[gamma <= rel_bound]), np.log(delta/y)
            )
        )

        spec[~rel] = y**2*thomson_tf_interp.flatten()

    else:
        print('###### THOMSON ENERGY LOSS SPECTRUM ######')
        # beta_small obviously doesn't intersect with rel. 
        spec[beta_small] = engloss_spec_diff(
            eleckineng_mask[beta_small], 
            delta_mask[beta_small], T, as_pairs=True
        )

        spec[~beta_small & ~rel] = engloss_spec_series(
            eleckineng_mask[~beta_small & ~rel],
            delta_mask[~beta_small & ~rel], T, as_pairs=True
        )
        print('###### COMPLETE! ######')

    # Zero out spec values that are too small (clearly no scatters within the age of the universe), and numerical errors. Non-zero to take log interpolation later. 
    spec[spec < 1e-100] = 1e-100
    
    if as_pairs:
        return spec 
    else:

        rs = T/phys.TCMB(1)
        dlnz = -1./(phys.dtdz(rs)*rs)

        return TransFuncAtRedshift(
            spec, in_eng = eleckineng, eng = delta, 
            rs = np.ones_like(eleckineng)*rs, dlnz=dlnz,
            spec_type = 'dNdE', with_interp_func=True
        )

