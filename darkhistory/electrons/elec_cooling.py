"""Functions for electron cooling."""

import numpy as np

import darkhistory.physics as phys
import darkhistory.utilities as utils
import darkhistory.spec.transferfunction as tf
import darkhistory.spec.spectools as spectools

from darkhistory.electrons.ics.ics_spectrum import ics_spec
from darkhistory.electrons.ics.ics_engloss_spectrum import engloss_spec


def get_elec_cooling_tf_fast(
    raw_nonrel_tf, raw_rel_tf, raw_engloss_tf,
    eleceng, photeng, rs, xe, xHe=0, verbose=False
):

    """Returns transfer function for complete electron cooling through ICS and atomic processes.

    Parameters
    ----------
    nonrel_tf : TransFuncAtRedshift
        Raw nonrelativistic primary electron ICS transfer function.
    rel_tf : string
        Raw relativistic primary electron ICS transfer function.
    engloss_tf_filename : string
        Raw primary electron ICS energy loss transfer function.
    eleceng : ndarray
        The electron *kinetic* energy abscissa.
    photeng : ndarray
        The photon energy abscissa.
    rs : float
        The redshift.
    xe : float
        Free electron fraction. 
    xHe : float, optional
        Singly-ionized helium fraction, nHe+/nH. Set to nHe/nH*xe if None.
    verbose : bool
        If true, prints energy conservation checks.
    
    Returns
    -------

    tuple of TransFuncAtRedshift
        Transfer functions for photons and low energy electrons.

    Note
    ----
    The raw transfer functions should be generated when the code package is first installed. The transfer function corresponds to the fully resolved
    photon spectrum after scattering by one electron.

    This version of the code works faster, but dispenses with energy conservation checks and several other safeguards. Use only with default abscissa, or when get_ics_cooling_tf works.

    """
    
    if xHe is None:
        xHe = xe*phys.nHe/phys.nH
        
    # v/c of electrons, important subsequently.
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

    elec_ICS_tf._grid_vals = spectools.engloss_rebin_fast(
        eleceng, photeng, engloss_ICS_tf.grid_vals, eleceng
    )
    
    # Total upscattered photon energy.
    cont_loss_ICS_vec = np.zeros_like(eleceng)
    # Deposited energy, enforces energy conservation.
    deposited_ICS_vec = np.zeros_like(eleceng)
    
    
    #####################
    # Excitation  
    #####################
    
    # Construct the rate matrices first. Secondary electron spectrum is an electron at in_eng - excitation energy, 
    # with a per second rate given by n*sigma*c.
    

    rate_matrix_exc_HI = np.diag(
        (1 - xe)*phys.nH*rs**3 * phys.coll_exc_xsec(eleceng, species='HI') * beta_ele * phys.c
    )
    
    rate_matrix_exc_HeI = np.diag(
        (phys.nHe/phys.nH - xHe)*phys.nH*rs**3 * phys.coll_exc_xsec(eleceng, species='HeI') * beta_ele * phys.c
    )
    
    rate_matrix_exc_HeII = np.diag(
        xHe*phys.nH*rs**3 * phys.coll_exc_xsec(eleceng, species='HeII') * beta_ele * phys.c
    )

    # Construct the TransFuncAtRedshift objects.
    # Electrons scatter from in_eng to in_eng - excitation energy.
    # Remove all of the columns (eng) that have energies below the excitation energy, 
    elec_exc_HI_tf = tf.TransFuncAtRedshift(
        np.squeeze(rate_matrix_exc_HI[:, np.where(eleceng > phys.lya_eng)]), 
        in_eng = eleceng, rs = rs*np.ones_like(eleceng), 
        eng = eleceng[eleceng > phys.lya_eng] - phys.lya_eng,
        dlnz = -1, spec_type = 'N'
    )
    elec_exc_HeI_tf = tf.TransFuncAtRedshift(
        np.squeeze(rate_matrix_exc_HeI[:, np.where(eleceng > phys.He_exc_eng)]), 
        in_eng = eleceng, rs = rs*np.ones_like(eleceng), 
        eng = eleceng[eleceng > phys.He_exc_eng] - phys.He_exc_eng,
        dlnz = -1, spec_type = 'N'
    )
    elec_exc_HeII_tf = tf.TransFuncAtRedshift(
        np.squeeze(rate_matrix_exc_HeII[:, np.where(eleceng > 4*phys.lya_eng)]), 
        in_eng = eleceng, rs = rs*np.ones_like(eleceng), 
        eng = eleceng[eleceng > 4*phys.lya_eng] - 4*phys.lya_eng,
        dlnz = -1, spec_type = 'N'
    )
    
    # Rebin these transfer functions back to eleceng.
    elec_exc_HI_tf.rebin(eleceng)
    elec_exc_HeI_tf.rebin(eleceng)
    elec_exc_HeII_tf.rebin(eleceng)
   
    # Deposited energy for excitation.
    deposited_exc_vec = np.zeros_like(eleceng)
    
    #####################
    # Ionization  
    #####################
    
    # Construct the rate vector first. Secondary electron spectrum is an electron at in_eng - excitation energy, 
    # with a per second rate given by n*sigma*c.
    rate_vec_ion_HI = (
        (1 - xe)*phys.nH*rs**3 * phys.coll_ion_xsec(eleceng, species='HI') * beta_ele * phys.c
    )
    
    rate_vec_ion_HeI = (
        (phys.nHe/phys.nH - xHe)*phys.nH*rs**3 * phys.coll_ion_xsec(eleceng, species='HeI') * beta_ele * phys.c
    )
    
    rate_vec_ion_HeII = (
        xHe*phys.nH*rs**3 * phys.coll_ion_xsec(eleceng, species='HeII') * beta_ele * phys.c
    )
    
    # Construct the spectra. 
    elec_spec_ion_HI = np.array(
        [rate*phys.coll_ion_sec_elec_spec(in_eng, eleceng, species='HI') for rate,in_eng in zip(rate_vec_ion_HI,eleceng)]
    )
    elec_spec_ion_HeI = np.array(
        [rate*phys.coll_ion_sec_elec_spec(in_eng, eleceng, species='HeI') for rate,in_eng in zip(rate_vec_ion_HeI,eleceng)]
    )
    elec_spec_ion_HeII = np.array(
        [rate*phys.coll_ion_sec_elec_spec(in_eng, eleceng, species='HeII') for rate,in_eng in zip(rate_vec_ion_HeII,eleceng)]
    )   
    
    # Construct the TransFuncAtRedshift objects.
    # Electrons scatter from in_eng to in_eng - excitation energy.
    # Remove all of the columns (eng) that have energies below the excitation energy, 
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

    # Start building sec_phot_tf and sec_lowengelec_tf.
    # Low energy regime first.

    sec_lowengelec_tf._grid_vals[:eleceng_low.size, :eleceng_low.size] = (
        np.identity(eleceng_low.size)
    )

    # Continuum energy loss rate per electron, dU_CMB/dt.
    CMB_upscatter_eng_rate = phys.thomson_xsec*phys.c*phys.CMB_eng_density(T)

    # High energy electron loop to get fully resolved spectrum.
    for i, eng in zip(eleceng_high_ind, eleceng_high):
        
        phot_ICS_N = phot_ICS_tf.grid_vals[i]
        
        elec_ICS_N      = elec_ICS_tf.grid_vals[i]
        
        elec_exc_HI_N   = elec_exc_HI_tf.grid_vals[i]
        elec_exc_HeI_N  = elec_exc_HeI_tf.grid_vals[i]
        elec_exc_HeII_N = elec_exc_HeII_tf.grid_vals[i]
        
        elec_ion_HI_N   = elec_ion_HI_tf.grid_vals[i]
        elec_ion_HeI_N  = elec_ion_HeI_tf.grid_vals[i]
        elec_ion_HeII_N = elec_ion_HeII_tf.grid_vals[i]
                
        elec_heat_spec = spectools.rebin_N_arr(np.array([1]), np.array([eng]), eleceng)
        elec_heat_spec.eng -= dE_heat_dt[i]
        elec_heat_spec.rebin(eleceng)
        elec_heat_N = elec_heat_spec.N
        
        sec_elec_spec_N = (
            elec_ICS_N 
            + elec_exc_HI_N + elec_exc_HeI_N + elec_exc_HeII_N
            + elec_ion_HI_N + elec_ion_HeI_N + elec_ion_HeII_N
            + elec_heat_N
        )
        sec_phot_spec_N = phot_ICS_N
        

        sec_elec_totN = np.sum(sec_elec_spec_N)
        # The *net* total energy of secondary electrons produced
        # per unit time.
        sec_elec_toteng = np.dot(sec_elec_spec_N, eleceng)
        # The total energy of secondary photons produced per unit time.
        sec_phot_toteng = np.dot(sec_phot_spec_N, photeng)
        # Deposited ICS energy per unit time, dD/dt.
        # Numerical error (should be zero except for numerics)
        deposited_ICS_eng = (
            np.sum(elec_ICS_N)*eng - np.dot(elec_ICS_N, eleceng)
            - (np.dot(phot_ICS_N, photeng) - CMB_upscatter_eng_rate)
        )
        # Deposited excitation energy. 
        deposited_exc_eng = (
            phys.lya_eng*np.sum(elec_exc_HI_N)
            + phys.He_exc_eng*np.sum(elec_exc_HeI_N)
            + 4*phys.lya_eng*np.sum(elec_exc_HeII_N)
        )
        # Deposited ionization energy. Remember that the secondary spectrum
        # has two electrons for each ionization event.
        deposited_ion_eng = (
            phys.rydberg*np.sum(elec_ion_HI_N/2)
            + phys.He_ion_eng*np.sum(elec_ion_HeI_N/2)
            + 4*phys.rydberg*np.sum(elec_ion_HeII_N/2)
        )
        # Deposited heating energy. 
        deposited_heat_eng = dE_heat_dt[i]
        
        # In the original code, the energy of the electron has gamma > 20,
        # then the continuum energy loss is assigned 
        # to deposited_eng instead.
        # I'm not sure if this is necessary, but let's be consistent with the
        # original code for now.

        continuum_engloss = CMB_upscatter_eng_rate
        
        if eng + phys.me > 20*phys.me:
            deposited_ICS_eng -= CMB_upscatter_eng_rate
            continuum_engloss = 0

        # Remove self-scattering.
        sec_elec_spec_N[i] = 0
        
        # Rescale.
        toteng_no_self_scatter = (
            np.dot(sec_elec_spec_N, eleceng)
            + np.dot(sec_phot_spec_N, photeng)
            - continuum_engloss
            + deposited_ICS_eng
            + deposited_exc_eng
            + deposited_ion_eng
            + deposited_heat_eng
        )
        
        fac = eng/toteng_no_self_scatter
        # Normalize to one electron. 
        
        sec_elec_spec_N    *= fac
        sec_phot_spec_N    *= fac
        continuum_engloss  *= fac
        deposited_ICS_eng  *= fac
        deposited_exc_eng  *= fac
        deposited_ion_eng  *= fac
        deposited_heat_eng *= fac

        # Get the full secondary photon spectrum. Type 'N'
        resolved_phot_spec_vals = np.dot(
            sec_elec_spec_N, sec_phot_tf.grid_vals
        )
        
        # Get the full secondary low energy electron spectrum. Type 'N'.

        resolved_lowengelec_spec_vals = np.dot(
            sec_elec_spec_N, sec_lowengelec_tf.grid_vals
        )
        
        # Add the resolved spectrum to the first scatter.
        sec_phot_spec_N += resolved_phot_spec_vals

        # Resolve the secondary electron continuum loss and deposition.
        continuum_engloss += np.dot(sec_elec_spec_N, cont_loss_ICS_vec)

        deposited_ICS_eng  += np.dot(sec_elec_spec_N, deposited_ICS_vec)
        deposited_exc_eng  += np.dot(sec_elec_spec_N, deposited_exc_vec)
        deposited_ion_eng  += np.dot(sec_elec_spec_N, deposited_ion_vec)
        deposited_heat_eng += np.dot(sec_elec_spec_N, deposited_heat_vec)

        # Now, append the resulting spectrum to the transfer function.
        # Do this without calling append of course: just add to the zeros
        # that fill the current row in _grid_vals.
        sec_phot_tf._grid_vals[i] += sec_phot_spec_N
        sec_lowengelec_tf._grid_vals[i] += resolved_lowengelec_spec_vals
        # Set the correct values in cont_loss_vec and deposited_vec.
        
        cont_loss_ICS_vec[i]  = continuum_engloss
        deposited_ICS_vec[i]  = deposited_ICS_eng
        deposited_exc_vec[i]  = deposited_exc_eng
        deposited_ion_vec[i]  = deposited_ion_eng
        deposited_heat_vec[i] = deposited_heat_eng

        
        check = True
        verbose = False
        failed_conservation_check = False
        
        if check:

            conservation_check = (
                eng
                - np.dot(sec_lowengelec_tf.grid_vals[i], eleceng)
                + cont_loss_ICS_vec[i]
                - np.dot(sec_phot_tf.grid_vals[i], photeng)
                - deposited_exc_vec[i]
                - deposited_ion_vec[i]
                - deposited_heat_vec[i]
            )

            if conservation_check/eng > 0.1:
                failed_conservation_check = True
                
            if verbose or failed_conservation_check:
                
                print('***************************************************')
                print('rs: ', rs)
                print('injected energy: ', eng)
                print(
                    'Energy in low energy electrons: ',
                    np.dot(sec_lowengelec_tf.grid_vals[i], eleceng)
                )
                print('Energy in photons: ', np.dot(sec_phot_tf.grid_vals[i], photeng))
                print('Continuum_engloss: ', cont_loss_ICS_vec[i])
                print(
                    'Energy in photons - Continuum: ',
                    np.dot(sec_phot_tf.grid_vals[i], photeng) - cont_loss_ICS_vec[i]
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
                    conservation_check/eng*100
                )
                print('Deposited in ICS: ', deposited_ICS_vec[i])
                print(
                    'Energy conservation with deposited (%): ',
                    (conservation_check - deposited_ICS_vec[i])/eng*100
                )
                print('***************************************************')
                
            if failed_conservation_check:
                raise RuntimeError('Conservation of energy failed.')


    return (
        sec_phot_tf, sec_lowengelec_tf,
        deposited_ion_vec, deposited_exc_vec, deposited_heat_vec,
        cont_loss_ICS_vec, deposited_ICS_vec
    )
    
