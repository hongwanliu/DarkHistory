"""Functions for electron cooling."""

import numpy as np

import darkhistory.physics as phys
import darkhistory.utilities as utils
import darkhistory.spec.transferfunction as tf
import darkhistory.spec.spectools as spectools

from darkhistory.electrons.ics.ics_spectrum import ics_spec
from darkhistory.electrons.ics.ics_engloss_spectrum import engloss_spec

from scipy.linalg import solve_triangular


def get_elec_cooling_tf(
    raw_nonrel_tf, raw_rel_tf, raw_engloss_tf,
    coll_ion_sec_elec_specs, coll_exc_sec_elec_specs,
    eleceng, photeng, rs, xH, xHe=0, ics_engloss_data=None, 
    check_conservation_eng = False, verbose=False
):

    """Returns transfer function for complete electron cooling through ICS and atomic processes using a linear algebra method.

    Parameters
    ----------
    nonrel_tf : TransFuncAtRedshift
        Raw nonrelativistic primary electron ICS transfer function.
    rel_tf : string
        Raw relativistic primary electron ICS transfer function.
    engloss_tf_filename : string
        Raw primary electron ICS energy loss transfer function.
    coll_ion_sec_elec_specs : tuple of ndarray
        Normalized collisional ionization secondary electron spectra, order HI, HeI, HeII, indexed by eleceng (injection) x eleceng (abscissa).
    coll_exc_sec_elec_specs : tuple of ndarray
        Normalized collisional excitation secondary electron spectra, order HI, HeI, HeII, indexed by eleceng (injection) x eleceng (abscissa).
    eleceng : ndarray
        The electron *kinetic* energy abscissa.
    photeng : ndarray
        The photon energy abscissa.
    rs : float
        The redshift.
    xH : float
        Ionized hydrogen fraction, nHII/nH. 
    xHe : float, optional
        Singly-ionized helium fraction, nHe+/nH. Set to nHe/nH*xe if None.
    ics_engloss_data : EnglossRebinData
        Stores rebinning information for speed. 
    check_conservation_eng : bool
        If true, checks for energy conservation.
    verbose : bool
        If true, prints energy conservation checks.
    
    Returns
    -------

    tuple of TransFuncAtRedshift
        Transfer functions for photons and low energy electrons.

    Notes
    -----
    The raw transfer functions should be generated when the code package is first installed. The transfer function corresponds to the fully resolved
    photon spectrum after scattering by one electron.

    This version of the code uses a linear algebra method to solve for the spectra directly.

    """

    # Set the electron fraction. 
    xe = xH + xHe
    
    # v/c of electrons
    beta_ele = np.sqrt(1 - 1/(1 + eleceng/phys.me)**2)

    #####################################
    # Inverse Compton
    #####################################

    T = phys.TCMB(rs)

    # Photon transfer function for single primary electron single scattering.
    # This is dN/(dE dt), dt = 1 s.
    phot_ICS_tf = ics_spec(
        eleceng, photeng, T, nonrel_tf = raw_nonrel_tf, rel_tf = raw_rel_tf
    )

    # Downcasting speeds up np.dot
    phot_ICS_tf._grid_vals = phot_ICS_tf.grid_vals.astype('float64')

    # Energy loss transfer function for single primary electron
    # single scattering. This is dN/(dE dt), dt = 1 s.
    engloss_ICS_tf = engloss_spec(
        eleceng, photeng, T, nonrel_tf = raw_engloss_tf, rel_tf = raw_rel_tf
    )

    # Downcasting speeds up np.dot
    engloss_ICS_tf._grid_vals = engloss_ICS_tf.grid_vals.astype('float64')

    # Switch the spectra type here to type 'N'.
    if phot_ICS_tf.spec_type == 'dNdE':
        phot_ICS_tf.switch_spec_type()
    if engloss_ICS_tf.spec_type == 'dNdE':
        engloss_ICS_tf.switch_spec_type()


    # Define some useful lengths.
    N_eleceng = eleceng.size
    N_photeng = photeng.size

    # Create the secondary electron transfer functions.

    # ICS transfer function.
    elec_ICS_tf = tf.TransFuncAtRedshift(
        np.zeros((N_eleceng, N_eleceng)), in_eng = eleceng,
        rs = rs*np.ones_like(eleceng), eng = eleceng,
        dlnz = -1, spec_type = 'N'
    )

    if ics_engloss_data is not None:
        elec_ICS_tf._grid_vals = ics_engloss_data.rebin(
            engloss_ICS_tf.grid_vals
        )
    else:
        elec_ICS_tf._grid_vals = spectools.engloss_rebin_fast(
            eleceng, photeng, engloss_ICS_tf.grid_vals, eleceng
        )
    
    # Total upscattered photon energy.
    cont_loss_ICS_vec = np.zeros_like(eleceng)
    # Deposited energy, enforces energy conservation.
    deposited_ICS_vec = np.zeros_like(eleceng)
    
    #########################
    # Collisional Excitation  
    #########################


    # Collisional excitation rates.
    rate_vec_exc_HI = (
        (1 - xH)*phys.nH*rs**3 * phys.coll_exc_xsec(eleceng, species='HI') * beta_ele * phys.c
    )
    
    rate_vec_exc_HeI = (
        (phys.nHe/phys.nH - xHe)*phys.nH*rs**3 * phys.coll_exc_xsec(eleceng, species='HeI') * beta_ele * phys.c
    )
    
    rate_vec_exc_HeII = (
        xHe*phys.nH*rs**3 * phys.coll_exc_xsec(eleceng, species='HeII') * beta_ele * phys.c
    )

    # Normalized electron spectrum after excitation.
    elec_exc_HI_tf = tf.TransFuncAtRedshift(
        rate_vec_exc_HI[:, np.newaxis]*coll_exc_sec_elec_specs[0],
        in_eng = eleceng, rs = rs*np.ones_like(eleceng),
        eng = eleceng, dlnz = -1, spec_type  = 'N'
    )

    elec_exc_HeI_tf = tf.TransFuncAtRedshift(
        rate_vec_exc_HeI[:, np.newaxis]*coll_exc_sec_elec_specs[1],
        in_eng = eleceng, rs = rs*np.ones_like(eleceng),
        eng = eleceng, dlnz = -1, spec_type  = 'N'
    )

    elec_exc_HeII_tf = tf.TransFuncAtRedshift(
        rate_vec_exc_HeII[:, np.newaxis]*coll_exc_sec_elec_specs[2],
        in_eng = eleceng, rs = rs*np.ones_like(eleceng),
        eng = eleceng, dlnz = -1, spec_type  = 'N'
    )
   
    # Deposited energy for excitation.
    deposited_exc_vec = np.zeros_like(eleceng)


    #########################
    # Collisional Ionization  
    #########################

    # Collisional ionization rates.
    rate_vec_ion_HI = (
        (1 - xH)*phys.nH*rs**3 
        * phys.coll_ion_xsec(eleceng, species='HI') * beta_ele * phys.c
    )
    
    rate_vec_ion_HeI = (
        (phys.nHe/phys.nH - xHe)*phys.nH*rs**3 
        * phys.coll_ion_xsec(eleceng, species='HeI') * beta_ele * phys.c
    )
    
    rate_vec_ion_HeII = (
        xHe*phys.nH*rs**3
        * phys.coll_ion_xsec(eleceng, species='HeII') * beta_ele * phys.c
    )

    # Normalized secondary electron spectra after ionization.
    elec_spec_ion_HI   = (
        rate_vec_ion_HI[:,np.newaxis]   * coll_ion_sec_elec_specs[0]
    )
    elec_spec_ion_HeI  = (
        rate_vec_ion_HeI[:,np.newaxis]  * coll_ion_sec_elec_specs[1]
    )
    elec_spec_ion_HeII = (
        rate_vec_ion_HeII[:,np.newaxis] * coll_ion_sec_elec_specs[2]
    )

    # Construct TransFuncAtRedshift objects.
    elec_ion_HI_tf = tf.TransFuncAtRedshift(
        elec_spec_ion_HI, in_eng = eleceng, rs = rs*np.ones_like(eleceng), 
        eng = eleceng, dlnz = -1, spec_type = 'N'
    )
    elec_ion_HeI_tf = tf.TransFuncAtRedshift(
        elec_spec_ion_HeI, in_eng = eleceng, rs = rs*np.ones_like(eleceng), 
        eng = eleceng, dlnz = -1, spec_type = 'N'
    )
    elec_ion_HeII_tf = tf.TransFuncAtRedshift(
        elec_spec_ion_HeII, in_eng = eleceng, rs = rs*np.ones_like(eleceng), 
        eng = eleceng, dlnz = -1, spec_type = 'N'
    )

    # Deposited energy for ionization.
    deposited_ion_vec = np.zeros_like(eleceng)

     #############################################
    # Heating
    #############################################
    
    dE_heat_dt = phys.elec_heating_engloss_rate(eleceng, xe, rs)
    
    deposited_heat_vec = np.zeros_like(eleceng)

    # new_eleceng = eleceng - dE_heat_dt

    # if not np.all(new_eleceng[1:] > eleceng[:-1]):
    #     utils.compare_arr([new_eleceng, eleceng])
    #     raise ValueError('heating loss is too large: smaller time step required.')

    # # After the check above, we can define the spectra by
    # # manually assigning slightly less than 1 particle along
    # # diagonal, and a small amount in the bin below. 

    # # N_n-1 E_n-1 + N_n E_n = E_n - dE_dt
    # # N_n-1 + N_n = 1
    # # therefore, (1 - N_n) E_n-1 - (1 - N_n) E_n = - dE_dt
    # # i.e. N_n = 1 + dE_dt/(E_n-1 - E_n)

    elec_heat_spec_grid = np.identity(eleceng.size)
    elec_heat_spec_grid[0,0] -= dE_heat_dt[0]/eleceng[0]
    elec_heat_spec_grid[1:, 1:] += np.diag(
        dE_heat_dt[1:]/(eleceng[:-1] - eleceng[1:])
    )
    elec_heat_spec_grid[1:, :-1] -= np.diag(
        dE_heat_dt[1:]/(eleceng[:-1] - eleceng[1:])
    )


    #############################################
    # Initialization of secondary spectra 
    #############################################

    # Low and high energy boundaries
    loweng = 3000
    eleceng_high = eleceng[eleceng > loweng]
    eleceng_high_ind = np.arange(eleceng.size)[eleceng > loweng]
    eleceng_low = eleceng[eleceng <= loweng]
    eleceng_low_ind  = np.arange(eleceng.size)[eleceng <= loweng]


    if eleceng_low.size == 0:
        raise TypeError('Energy abscissa must contain a low energy bin below 3 keV.')

    # Empty containers for quantities.
    # Final secondary photon spectrum.
    sec_phot_tf = tf.TransFuncAtRedshift(
        np.zeros((N_eleceng, N_photeng)), in_eng = eleceng,
        rs = rs*np.ones_like(eleceng), eng = photeng,
        dlnz = -1, spec_type = 'N'
    )
    # Final secondary low energy electron spectrum.
    sec_lowengelec_tf = tf.TransFuncAtRedshift(
        np.zeros((N_eleceng, N_eleceng)), in_eng = eleceng,
        rs = rs*np.ones_like(eleceng), eng = eleceng,
        dlnz = -1, spec_type = 'N'
    )

    # Continuum energy loss rate per electron, dU_CMB/dt.
    CMB_upscatter_eng_rate = phys.thomson_xsec*phys.c*phys.CMB_eng_density(T)
    
    # Secondary scattered electron spectrum.
    sec_elec_spec_N_arr = (
        elec_ICS_tf.grid_vals
        + elec_exc_HI_tf.grid_vals 
        + elec_exc_HeI_tf.grid_vals 
        + elec_exc_HeII_tf.grid_vals
        + elec_ion_HI_tf.grid_vals 
        + elec_ion_HeI_tf.grid_vals 
        + elec_ion_HeII_tf.grid_vals
        + elec_heat_spec_grid
    )
    
    # Secondary photon spectrum (from ICS). 
    sec_phot_spec_N_arr = phot_ICS_tf.grid_vals
    
    # Deposited ICS array.
    deposited_ICS_eng_arr = (
        np.sum(elec_ICS_tf.grid_vals, axis=1)*eleceng
        - np.dot(elec_ICS_tf.grid_vals, eleceng)
        - (np.dot(sec_phot_spec_N_arr, photeng) - CMB_upscatter_eng_rate)
    )

    # Energy loss is not taken into account for eleceng > 20*phys.me
    deposited_ICS_eng_arr[eleceng > 20*phys.me - phys.me] -= ( 
        CMB_upscatter_eng_rate
    )

    # Continuum energy loss array.
    continuum_engloss_arr = CMB_upscatter_eng_rate*np.ones_like(eleceng)
    # Energy loss is not taken into account for eleceng > 20*phys.me
    continuum_engloss_arr[eleceng > 20*phys.me - phys.me] = 0
    
    # Deposited excitation array.
    deposited_exc_eng_arr = (
        phys.lya_eng*np.sum(elec_exc_HI_tf.grid_vals, axis=1)
        + phys.He_exc_eng['23s']*np.sum(elec_exc_HeI_tf.grid_vals, axis=1)
        + 4*phys.lya_eng*np.sum(elec_exc_HeII_tf.grid_vals, axis=1)
    )
    
    # Deposited ionization array.
    deposited_ion_eng_arr = (
        phys.rydberg*np.sum(elec_ion_HI_tf.grid_vals, axis=1)/2
        + phys.He_ion_eng*np.sum(elec_ion_HeI_tf.grid_vals, axis=1)/2
        + 4*phys.rydberg*np.sum(elec_ion_HeII_tf.grid_vals, axis=1)/2
    )

    # Deposited heating array.
    deposited_heat_eng_arr = dE_heat_dt
    
    # Remove self-scattering, re-normalize. 
    np.fill_diagonal(sec_elec_spec_N_arr, 0)
    
    toteng_no_self_scatter_arr = (
        np.dot(sec_elec_spec_N_arr, eleceng)
        + np.dot(sec_phot_spec_N_arr, photeng)
        - continuum_engloss_arr
        + deposited_ICS_eng_arr
        + deposited_exc_eng_arr
        + deposited_ion_eng_arr
        + deposited_heat_eng_arr
    )
    
    fac_arr = eleceng/toteng_no_self_scatter_arr
    
    sec_elec_spec_N_arr *= fac_arr[:, np.newaxis]
    sec_phot_spec_N_arr *= fac_arr[:, np.newaxis]
    continuum_engloss_arr  *= fac_arr
    deposited_ICS_eng_arr  *= fac_arr
    deposited_exc_eng_arr  *= fac_arr
    deposited_ion_eng_arr  *= fac_arr
    deposited_heat_eng_arr *= fac_arr
    
    # Zero out deposition/ICS processes below loweng. 
    deposited_ICS_eng_arr[eleceng < loweng]  = 0
    deposited_exc_eng_arr[eleceng < loweng]  = 0
    deposited_ion_eng_arr[eleceng < loweng]  = 0
    deposited_heat_eng_arr[eleceng < loweng] = 0
    
    continuum_engloss_arr[eleceng < loweng]  = 0
    
    sec_phot_spec_N_arr[eleceng < loweng] = 0
    
    # Scattered low energy and high energy electrons. 
    # Needed for final low energy electron spectra.
    sec_lowengelec_N_arr = np.identity(eleceng.size)
    sec_lowengelec_N_arr[eleceng >= loweng] = 0
    sec_lowengelec_N_arr[eleceng_high_ind[0]:, :eleceng_high_ind[0]] += sec_elec_spec_N_arr[eleceng_high_ind[0]:, :eleceng_high_ind[0]]

    sec_highengelec_N_arr = np.zeros_like(sec_elec_spec_N_arr)
    sec_highengelec_N_arr[:, eleceng_high_ind[0]:] = sec_elec_spec_N_arr[:, eleceng_high_ind[0]:]
    
    # T = E.T + Prompt
    deposited_ICS_vec  = solve_triangular(
        np.identity(eleceng.size) - sec_elec_spec_N_arr,
        deposited_ICS_eng_arr, lower=True, check_finite=False
    )
    deposited_exc_vec  = solve_triangular(
        np.identity(eleceng.size) - sec_elec_spec_N_arr, 
        deposited_exc_eng_arr, lower=True, check_finite=False
    )
    deposited_ion_vec  = solve_triangular(
        np.identity(eleceng.size) - sec_elec_spec_N_arr, 
        deposited_ion_eng_arr, lower=True, check_finite=False
    )
    deposited_heat_vec = solve_triangular(
        np.identity(eleceng.size) - sec_elec_spec_N_arr, 
        deposited_heat_eng_arr, lower=True, check_finite=False
    )
    
    cont_loss_ICS_vec = solve_triangular(
        np.identity(eleceng.size) - sec_elec_spec_N_arr, 
        continuum_engloss_arr, lower=True, check_finite=False
    )
    
    sec_phot_specs = solve_triangular(
        np.identity(eleceng.size) - sec_elec_spec_N_arr, 
        sec_phot_spec_N_arr, lower=True, check_finite=False
    )
    
    # Prompt: low energy e produced in secondary spectrum upon scattering (sec_lowengelec_N_arr).
    # T : high energy e produced (sec_highengelec_N_arr). 
    sec_lowengelec_specs = solve_triangular(
        np.identity(eleceng.size) - sec_highengelec_N_arr,
        sec_lowengelec_N_arr, lower=True, check_finite=False
    )
    
    sec_phot_tf._grid_vals = sec_phot_specs
    sec_lowengelec_tf._grid_vals = sec_lowengelec_specs

    # Conservation checks.
    failed_conservation_check = False

    if check_conservation_eng:

        conservation_check = (
            eleceng
            - np.dot(sec_lowengelec_tf.grid_vals, eleceng)
            + cont_loss_ICS_vec
            - np.dot(sec_phot_tf.grid_vals, photeng)
            - deposited_exc_vec
            - deposited_ion_vec
            - deposited_heat_vec
        )

        if np.any(np.abs(conservation_check/eleceng) > 0.1):
            failed_conservation_check = True

        if verbose or failed_conservation_check:

            for i,eng in enumerate(eleceng):

                print('***************************************************')
                print('rs: ', rs)
                print('injected energy: ', eng)

                print(
                    'Energy in low energy electrons: ',
                    np.dot(sec_lowengelec_tf.grid_vals[i], eleceng)
                )

                print('Energy in photons: ', 
                    np.dot(sec_phot_tf.grid_vals[i], photeng)
                )
                
                print('Continuum_engloss: ', cont_loss_ICS_vec[i])
                
                print(
                    'Energy in photons - Continuum: ', (
                        np.dot(sec_phot_tf.grid_vals[i], photeng)
                        - cont_loss_ICS_vec[i]
                    )
                )

                print(
                    'Deposited in ionization: ', deposited_ion_vec[i]
                )

                print(
                    'Deposited in excitation: ', deposited_exc_vec[i]
                )

                print(
                    'Deposited in heating: ', deposited_heat_vec[i]
                )

                print(
                    'Energy is conserved up to (%): ',
                    conservation_check[i]/eng*100
                )
                print('Deposited in ICS: ', deposited_ICS_vec[i])
                
                print(
                    'Energy conservation with deposited (%): ',
                    (conservation_check[i] - deposited_ICS_vec[i])/eng*100
                )
                print('***************************************************')
                
            if failed_conservation_check:
                raise RuntimeError('Conservation of energy failed.')

    return (
        sec_phot_tf, sec_lowengelec_tf,
        deposited_ion_vec, deposited_exc_vec, deposited_heat_vec,
        cont_loss_ICS_vec, deposited_ICS_vec
    )



    
