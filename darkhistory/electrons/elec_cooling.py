"""Function for calculating the electron cooling transfer function.
"""

import numpy as np

import darkhistory.physics as phys
import darkhistory.utilities as utils
import darkhistory.spec.transferfunction as tf
import darkhistory.spec.spectools as spectools

from darkhistory.config import load_data

from darkhistory.spec.spectrum import Spectrum

from darkhistory.electrons.ics.ics_spectrum import ics_spec
from darkhistory.electrons.ics.ics_engloss_spectrum import engloss_spec

from scipy.linalg import solve_triangular


def get_elec_cooling_tf(
    eleceng, photeng, rs, xHII, xHeII=0, 
    raw_thomson_tf=None, raw_rel_tf=None, raw_engloss_tf=None,
    coll_ion_sec_elec_specs=None, coll_exc_sec_elec_specs=None,
    ics_engloss_data=None, 
    check_conservation_eng=False, verbose=False,
    nBscale=1.,
):

    """Transfer functions for complete electron cooling through inverse Compton scattering (ICS) and atomic processes.
  

    Parameters
    ----------
    eleceng : ndarray, shape (m, )
        The electron kinetic energy abscissa.
    photeng : ndarray
        The photon energy abscissa.
    rs : float
        The redshift (1+z). 
    xHII : float
        Ionized hydrogen fraction, nHII/nH. 
    xHeII : float, optional
        Singly-ionized helium fraction, nHe+/nH. Default is 0. 
    raw_thomson_tf : TransFuncAtRedshift, optional
        Thomson ICS scattered photon spectrum transfer function. If None, uses the default transfer function. Default is None.
    raw_rel_tf : TransFuncAtRedshift, optional
        Relativistic ICS scattered photon spectrum transfer function. If None, uses the default transfer function. Default is None.
    raw_engloss_tf : TransFuncAtRedshift, optional
        Thomson ICS scattered electron net energy loss transfer function. If None, uses the default transfer function. Default is None.
    coll_ion_sec_elec_specs : tuple of 3 ndarrays, shapes (m, m), optional 
        Normalized collisional ionization secondary electron spectra, order HI, HeI, HeII, indexed by injected electron energy by outgoing electron energy. If None, the function calculates this. Default is None.
    coll_exc_sec_elec_specs : tuple of 3 ndarray, shapes (m, m), optional 
        Normalized collisional excitation secondary electron spectra, order HI, HeI, HeII, indexed by injected electron energy by outgoing electron energy. If None, the function calculates this. Default is None.
    ics_engloss_data : EnglossRebinData
        An `EnglossRebinData` object which stores rebinning information (based on ``eleceng`` and ``photeng``) for speed. Default is None.
    check_conservation_eng : bool
        If True, lower=True, checks for energy conservation. Default is False.
    verbose : bool
        If True, prints energy conservation checks. Default is False.
    nBscale : float
        The baryon number scaling factor. Default is 1.
    
    Returns
    -------

    tuple
        Transfer functions for electron cooling deposition and spectra.

    See Also
    ---------
    :class:`.TransFuncAtRedshift`
    :class:`.EnglossRebinData`
    :mod:`.ics`

    Notes
    -----
    
    The entries of the output tuple are (see Sec IIIC of the paper):

    0. The secondary propagating photon transfer function :math:`\\overline{\\mathsf{T}}_\\gamma`; 
    1. The low-energy electron transfer function :math:`\\overline{\\mathsf{T}}_e`; 
    2. Energy deposited into ionization :math:`\\overline{\\mathbf{R}}_\\text{ion}`; 
    3. Energy deposited into excitation :math:`\\overline{\\mathbf{R}}_\\text{exc}`; 
    4. Energy deposited into heating :math:`\\overline{\\mathbf{R}}_\\text{heat}`;
    5. Upscattered CMB photon total energy :math:`\\overline{\\mathbf{R}}_\\text{CMB}`, and
    6. Numerical error away from energy conservation.

    Items 2--5 are vectors that when dotted into an electron spectrum with abscissa ``eleceng``, return the energy deposited/CMB energy upscattered for that spectrum. 

    Items 0--1 are :class:`.TransFuncAtRedshift` objects. For each of these objects ``tf`` and a given electron spectrum ``elec_spec``, ``tf.sum_specs(elec_spec)`` returns the propagating photon/low-energy electron spectrum after cooling.

    The default version of the three ICS transfer functions that are required by this function is provided in :mod:`.tf_data`.

    """

    # Use default ICS transfer functions if not specified.

    #ics_tf = load_data('ics_tf')

    #raw_thomson_tf = ics_tf['thomson']
    #raw_rel_tf     = ics_tf['rel']
    #raw_engloss_tf = ics_tf['engloss']

    if coll_ion_sec_elec_specs is None:

        # Compute the (normalized) collisional ionization spectra.
        coll_ion_sec_elec_specs = (
            phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HI'),
            phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HeI'),
            phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HeII')
        )

    if coll_exc_sec_elec_specs is None:

        # Compute the (normalized) collisional excitation spectra.
        id_mat = np.identity(eleceng.size)

        # Electron with energy eleceng produces a spectrum with one particle
        # of energy eleceng - phys.lya.eng. Similar for helium. 
        coll_exc_sec_elec_tf_HI = tf.TransFuncAtRedshift(
            np.squeeze(id_mat[:, np.where(eleceng > phys.lya_eng)]),
            in_eng = eleceng, rs = -1*np.ones_like(eleceng),
            eng = eleceng[eleceng > phys.lya_eng] - phys.lya_eng,
            dlnz = -1, spec_type = 'N'
        )

        coll_exc_sec_elec_tf_HeI = tf.TransFuncAtRedshift(
            np.squeeze(
                id_mat[:, np.where(eleceng > phys.He_exc_eng['23s'])]
            ),
            in_eng = eleceng, rs = -1*np.ones_like(eleceng),
            eng = (
                eleceng[eleceng > phys.He_exc_eng['23s']] 
                - phys.He_exc_eng['23s']
            ), 
            dlnz = -1, spec_type = 'N'
        )

        coll_exc_sec_elec_tf_HeII = tf.TransFuncAtRedshift(
            np.squeeze(id_mat[:, np.where(eleceng > 4*phys.lya_eng)]),
            in_eng = eleceng, rs = -1*np.ones_like(eleceng),
            eng = eleceng[eleceng > 4*phys.lya_eng] - 4*phys.lya_eng,
            dlnz = -1, spec_type = 'N'
        )

        # Rebin the data so that the spectra stored above now have an abscissa
        # of eleceng again (instead of eleceng - phys.lya_eng for HI etc.)
        coll_exc_sec_elec_tf_HI.rebin(eleceng)
        coll_exc_sec_elec_tf_HeI.rebin(eleceng)
        coll_exc_sec_elec_tf_HeII.rebin(eleceng)

        # Put them in a tuple.
        coll_exc_sec_elec_specs = (
            coll_exc_sec_elec_tf_HI.grid_vals,
            coll_exc_sec_elec_tf_HeI.grid_vals,
            coll_exc_sec_elec_tf_HeII.grid_vals
        )

    # Set the electron fraction. 
    xe = xHII + xHeII
    
    # v/c of electrons
    beta_ele = np.sqrt(1 - 1/(1 + eleceng/phys.me)**2)

    #####################################
    # Inverse Compton
    #####################################

    T = phys.TCMB(rs)

    # Photon transfer function for single primary electron single scattering.
    # This is dN/(dE dt), dt = 1 s.
    phot_ICS_tf = ics_spec(
        eleceng, photeng, T, thomson_tf = raw_thomson_tf, rel_tf = raw_rel_tf
    )

    # Downcasting speeds up np.dot
    phot_ICS_tf._grid_vals = phot_ICS_tf.grid_vals.astype('float64')

    # Energy loss transfer function for single primary electron
    # single scattering. This is dN/(dE dt), dt = 1 s.
    engloss_ICS_tf = engloss_spec(
        eleceng, photeng, T, thomson_tf = raw_engloss_tf, rel_tf = raw_rel_tf
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
        (1 - xHII) * (phys.nH * nBscale) * rs**3 * phys.coll_exc_xsec(eleceng, species='HI') * beta_ele * phys.c
    )
    
    rate_vec_exc_HeI = (
        (phys.nHe/phys.nH - xHeII) * (phys.nH * nBscale) * rs**3 * phys.coll_exc_xsec(eleceng, species='HeI') * beta_ele * phys.c
    )
    
    rate_vec_exc_HeII = (
        xHeII * (phys.nH * nBscale) * rs**3 * phys.coll_exc_xsec(eleceng, species='HeII') * beta_ele * phys.c
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
        (1 - xHII) * (phys.nH * nBscale) * rs**3 
        * phys.coll_ion_xsec(eleceng, species='HI') * beta_ele * phys.c
    )
    
    rate_vec_ion_HeI = (
        (phys.nHe/phys.nH - xHeII) * (phys.nH * nBscale) * rs**3 
        * phys.coll_ion_xsec(eleceng, species='HeI') * beta_ele * phys.c
    )
    
    rate_vec_ion_HeII = (
        xHeII * (phys.nH * nBscale) * rs**3
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
    
    dE_heat_dt = phys.elec_heating_engloss_rate(eleceng, xe, rs, nBscale=nBscale)
    
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
    sec_highengelec_N_arr[:, eleceng_high_ind[0]:] = (
        sec_elec_spec_N_arr[:, eleceng_high_ind[0]:]
    )
    
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

    # Subtract continuum from sec_phot_specs. After this point, 
    # sec_phot_specs will contain the *distortions* to the CMB. 

    # Normalized CMB spectrum. 
    norm_CMB_spec = Spectrum(
        photeng, phys.CMB_spec(photeng, phys.TCMB(rs)), spec_type='dNdE'
    )
    norm_CMB_spec /= norm_CMB_spec.toteng()

    # Get the CMB spectrum upscattered from cont_loss_ICS_vec. 
    upscattered_CMB_grid = np.outer(cont_loss_ICS_vec, norm_CMB_spec.N)

    # Subtract this spectrum from sec_phot_specs to get the final
    # transfer function.

    sec_phot_tf._grid_vals = sec_phot_specs - upscattered_CMB_grid
    sec_lowengelec_tf._grid_vals = sec_lowengelec_specs

    # Conservation checks.
    failed_conservation_check = False

    if check_conservation_eng:

        conservation_check = (
            eleceng
            - np.dot(sec_lowengelec_tf.grid_vals, eleceng)
            # + cont_loss_ICS_vec
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
                    'Fraction of Energy in low energy electrons: ',
                    np.dot(sec_lowengelec_tf.grid_vals[i], eleceng)/eng
                )

                # print('Energy in photons: ', 
                #     np.dot(sec_phot_tf.grid_vals[i], photeng)
                # )
                
                # print('Continuum_engloss: ', cont_loss_ICS_vec[i])
                
                print(
                    'Fraction of Energy in photons - Continuum: ', (
                        np.dot(sec_phot_tf.grid_vals[i], photeng)/eng
                        # - cont_loss_ICS_vec[i]
                    )
                )

                print(
                    'Fraction Deposited in ionization: ', 
                    deposited_ion_vec[i]/eng
                )

                print(
                    'Fraction Deposited in excitation: ', 
                    deposited_exc_vec[i]/eng
                )

                print(
                    'Fraction Deposited in heating: ', 
                    deposited_heat_vec[i]/eng
                )

                print(
                    'Energy is conserved up to (%): ',
                    conservation_check[i]/eng*100
                )
                print('Fraction Deposited in ICS (Numerical Error): ', 
                    deposited_ICS_vec[i]/eng
                )
                
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



    
