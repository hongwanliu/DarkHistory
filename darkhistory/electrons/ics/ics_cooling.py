"""Electron cooling through ICS."""

import numpy as np
import pickle

import darkhistory.physics as phys
import darkhistory.spec.transferfunction as tf
import darkhistory.spec.spectools as spectools

from darkhistory.spec.spectrum import Spectrum

from darkhistory.electrons.ics.ics_spectrum import ics_spec
from darkhistory.electrons.ics.ics_engloss_spectrum import engloss_spec

def get_ics_cooling_tf(
    raw_thomson_tf, raw_rel_tf, raw_engloss_tf,
    eleceng, photeng, rs, fast=True
):

    """Transfer function for complete electron cooling through ICS.

    Parameters
    ----------
    raw_thomson_tf : TransFuncAtRedshift
        Raw Thomson ICS scattered photon spectrum transfer function.
    raw_rel_tf : TransFuncAtRedshift
        Raw relativistic ICS scattered photon spectrum transfer function.
    raw_engloss_tf : TransFuncAtRedshift
        Raw Thomson ICS scattered electron net energy loss spectrum transfer function.
    eleceng : ndarray
        The electron *kinetic* energy abscissa.
    photeng : ndarray
        The photon energy abscissa.
    rs : float
        The redshift (1+z).
    fast : bool, optional
        If True, uses optimized code (with very little checks)

    Returns
    -------

    tuple of TransFuncAtRedshift
        Transfer functions for photons and low energy electrons.

    Notes
    -----
    The raw transfer functions should be generated when the code package is first installed. The transfer function corresponds to the fully resolved
    photon spectrum after scattering by one electron.

    """

    if fast:
        return get_ics_cooling_tf_fast(
            raw_thomson_tf, raw_rel_tf, raw_engloss_tf,
            eleceng, photeng, rs
        )


    T = phys.TCMB(rs)

    # Photon transfer function for single primary electron single scattering.
    # This is dN/(dE dt), dt = 1 s.
    ICS_tf = ics_spec(
        eleceng, photeng, T, thomson_tf = raw_thomson_tf, rel_tf = raw_rel_tf
    )

    # Downcasting speeds up np.dot
    ICS_tf._grid_vals = ICS_tf.grid_vals.astype('float64')

    # Energy loss transfer function for single primary electron
    # single scattering. This is dN/(dE dt), dt = 1 s.
    engloss_tf = engloss_spec(
        eleceng, photeng, T, thomson_tf = raw_engloss_tf, rel_tf = raw_rel_tf
    )

    # Downcasting speeds up np.dot
    engloss_tf._grid_vals = engloss_tf.grid_vals.astype('float64')

    # Define some useful lengths.
    N_eleceng = eleceng.size
    N_photeng = photeng.size

    # Create the secondary electron transfer function.

    sec_elec_tf = tf.TransFuncAtRedshift(
        np.zeros((N_eleceng, N_eleceng)), in_eng = eleceng,
        rs = rs*np.ones_like(eleceng), eng = eleceng,
        dlnz = -1, spec_type = 'dNdE'
    )

    # append_sec_elec_tf = sec_elec_tf.append

    # Change from energy loss spectrum to secondary electron spectrum.
    for i, in_eng in enumerate(eleceng):
        spec = engloss_tf[i]
        spec.engloss_rebin(in_eng, eleceng, fast=True)
        # Add to the appropriate row.
        sec_elec_tf._grid_vals[i] += spec.dNdE


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
    # Total upscattered photon energy.
    cont_loss_vec = np.zeros_like(eleceng)
    # Deposited energy, enforces energy conservation.
    deposited_vec = np.zeros_like(eleceng)

    # Test input electron to get the spectra.
    delta_spec = np.zeros_like(eleceng)

    # Start building sec_phot_tf and sec_lowengelec_tf.
    # Low energy regime first.

    ####################################
    # OLD: for loop to add identity.   #
    # Not very clever.                 #
    ####################################


    # for i, eng in zip(eleceng_low_ind, eleceng_low):
    #     # Zero out delta function test spectrum, set it correctly
    #     # for the loop ahead.
    #     delta_spec *= 0
    #     delta_spec[i] = 1
    #     # Add the trivial secondary electron spectrum to the
    #     # transfer function.
    #     sec_lowengelec_tf._grid_vals[i] += delta_spec

    ####################################
    # NEW: Just set the relevant       #
    # part to be the identity matrix   #
    ####################################

    sec_lowengelec_tf._grid_vals[:eleceng_low.size, :eleceng_low.size] = (
        np.identity(eleceng_low.size)
    )

    # Continuum energy loss rate, dU_CMB/dt.
    CMB_upscatter_eng_rate = phys.thomson_xsec*phys.c*phys.CMB_eng_density(T)


    # High energy electron loop to get fully resolved spectrum.
    for i, eng in zip(eleceng_high_ind, eleceng_high):

        # print('Check energies and indexing: ')
        # print(i, eleceng[i], eng)

        sec_phot_spec = ICS_tf[i]
        if sec_phot_spec.spec_type == 'dNdE':
            sec_phot_spec.switch_spec_type()

        sec_elec_spec = sec_elec_tf[i]
        if sec_elec_spec.spec_type == 'dNdE':
            sec_elec_spec.switch_spec_type()

        # sec_elec_spec_2 = sec_elec_tf_2[i]
        # if sec_elec_spec_2.spec_type == 'dNdE':
        #     sec_elec_spec_2.switch_spec_type()

        # The total number of primaries scattered is equal to the total number of scattered *photons*.
        # The scattered electrons is obtained from the *net* energy loss, and
        # so is not indicative of number of scatters.
        tot_N_scatter = sec_phot_spec.totN()
        # The total energy of primary electrons which is scattered per unit time.
        tot_eng_scatter = tot_N_scatter*eng
        # The *net* total number of secondary photons produced
        # per unit time.
        sec_elec_N = sec_elec_spec.totN()
        # The *net* total energy of secondary electrons produced
        # per unit time.
        sec_elec_toteng = sec_elec_spec.toteng()
        # The total energy of secondary photons produced per unit time.
        sec_phot_toteng = sec_phot_spec.toteng()
        # Deposited energy per unit time, dD/dt.
        deposited_eng = sec_elec_spec.totN()*eng - sec_elec_toteng - (sec_phot_toteng - CMB_upscatter_eng_rate)

        print('-------- Injection Energy: ', eng)
        print(
            '-------- No. of Scatters (Analytic): ',
            phys.thomson_xsec*phys.c*phys.CMB_N_density(T)
        )
        print(
            '-------- No. of Scatters (Computed): ',
            tot_N_scatter
        )
        gamma_elec = 1 + eng/phys.me
        beta_elec  = np.sqrt(eng/phys.me*(gamma_elec+1)/gamma_elec**2)
        print(
            '-------- Energy lost (Analytic): ',
            (4/3)*phys.thomson_xsec*phys.c*phys.CMB_eng_density(T)*(
                gamma_elec**2 * beta_elec**2
            )
        )
        print(
            '-------- Energy lost (Computed from photons): ',
            engloss_tf[i].toteng()
        )
        print(
            '-------- Energy lost (Computed from electrons): ',
            sec_elec_spec.totN()*eng - sec_elec_toteng
        )
        print(
            '-------- Energy of upscattered photons: ',
            CMB_upscatter_eng_rate
        )
        print(
            '-------- Energy in secondary photons (Computed): ',
            sec_phot_toteng
        )
        print(
            '-------- Energy in secondary photons (Analytic): ',
            phys.thomson_xsec*phys.c*phys.CMB_eng_density(T)*(
                1 + (4/3)* gamma_elec**2 * beta_elec**2
            )
        )
        print(
            '-------- Energy gain from photons: ',
            sec_phot_toteng - CMB_upscatter_eng_rate
        )
        print('-------- Deposited Energy: ', deposited_eng)

        # In the original code, the energy of the electron has gamma > 20,
        # then the continuum energy loss is assigned to deposited_eng instead.
        # I'm not sure if this is necessary, but let's be consistent with the
        # original code for now.

        continuum_engloss = CMB_upscatter_eng_rate

        if eng + phys.me > 20*phys.me:
            deposited_eng -= CMB_upscatter_eng_rate
            continuum_engloss = 0

        # Normalize to one secondary electron.

        sec_phot_spec /= sec_elec_N
        sec_elec_spec /= sec_elec_N
        continuum_engloss /= sec_elec_N
        deposited_eng /= sec_elec_N

        # Remove self-scattering.

        selfscatter_engfrac = (
            sec_elec_spec.N[i]
        )
        scattered_engfrac = 1 - selfscatter_engfrac

        sec_elec_spec.N[i] = 0

        sec_phot_spec /= scattered_engfrac
        sec_elec_spec /= scattered_engfrac
        continuum_engloss /= scattered_engfrac
        deposited_eng /= scattered_engfrac

        # Get the full secondary photon spectrum. Type 'N'
        resolved_phot_spec = sec_phot_tf.sum_specs(sec_elec_spec.N)

        # Get the full secondary low energy electron spectrum. Type 'N'.
        resolved_lowengelec_spec = (
            sec_lowengelec_tf.sum_specs(sec_elec_spec.N)
        )

        # Add the resolved spectrum to the first scatter.
        sec_phot_spec += resolved_phot_spec

        # Resolve the secondary electron continuum loss and deposition.
        continuum_engloss += np.dot(sec_elec_spec.N, cont_loss_vec)

        # utils.compare_arr([sec_elec_spec.N, deposited_vec])
        deposited_eng += np.dot(sec_elec_spec.N, deposited_vec)

        # Now, append the resulting spectrum to the transfer function.
        # Do this without calling append of course: just add to the zeros
        # that fill the current row in _grid_vals.
        sec_phot_tf._grid_vals[i] += sec_phot_spec.N
        sec_lowengelec_tf._grid_vals[i] += resolved_lowengelec_spec.N

        # Set the correct values in cont_loss_vec and deposited_vec.
        cont_loss_vec[i] = continuum_engloss
        deposited_vec[i] = deposited_eng

        # Conservation of energy check. Check that it is 1e-10 of eng.


        conservation_check = (
            eng
            - resolved_lowengelec_spec.toteng()
            + cont_loss_vec[i]
            - sec_phot_spec.toteng()
                         )

        # print('***************************************************')
        # print('injected energy: ', eng)
        # print('low energy e: ', resolved_lowengelec_spec.toteng())
        # print('scattered phot: ', sec_phot_spec.toteng())
        # print('continuum_engloss: ', cont_loss_vec[i])
        # print('diff: ', sec_phot_spec.toteng() - cont_loss_vec[i])
        # print('energy is conserved up to (%): ', conservation_check/eng*100)
        # print('deposited: ', deposited_vec[i])
        # print(
        #     'energy conservation with deposited (%): ',
        #     (conservation_check - deposited_vec[i])/eng*100
        # )
        # print('***************************************************')

        if (
            conservation_check/eng > 0.01
        ):
            print('***************************************************')
            print('rs: ', rs)
            print('injected energy: ', eng)
            print('low energy e: ', resolved_lowengelec_spec.toteng())
            print('scattered phot: ', sec_phot_spec.toteng())
            print('continuum_engloss: ', cont_loss_vec[i])
            print('diff: ', sec_phot_spec.toteng() - cont_loss_vec[i])
            print('energy is conserved up to (%): ', conservation_check/eng*100)
            print('deposited: ', deposited_vec[i])
            print(
                'energy conservation with deposited (%): ',
                (conservation_check - deposited_vec[i])/eng*100
            )
            print('***************************************************')

            raise RuntimeError('Conservation of energy failed.')

        # Force conservation of energy.
        # deposited_vec[i] += conservation_check

    return (sec_phot_tf, sec_lowengelec_tf, cont_loss_vec, deposited_vec)


def get_ics_cooling_tf_fast(
    raw_thomson_tf, raw_rel_tf, raw_engloss_tf,
    eleceng, photeng, rs
):

    """ Transfer function for complete electron cooling through ICS.

    Parameters
    ----------
    raw_thomson_tf : TransFuncAtRedshift
        Raw Thomson ICS scattered photon spectrum transfer function.
    raw_rel_tf : TransFuncAtRedshift
        Raw relativistic ICS scattered photon spectrum transfer function.
    raw_engloss_tf : TransFuncAtRedshift
        Raw Thomson ICS scattered electron net energy loss spectrum transfer function.
    eleceng : ndarray
        The electron *kinetic* energy abscissa.
    photeng : ndarray
        The photon energy abscissa.
    rs : float
        The redshift (1+z). 

    Returns
    -------

    tuple of TransFuncAtRedshift
        Transfer functions for photons and low energy electrons.

    Notes
    -----
    The raw transfer functions should be generated when the code package is first installed. The transfer function corresponds to the fully resolved
    photon spectrum after scattering by one electron.

    This version of the code works faster, but dispenses with energy conservation checks and several other safeguards. Use only with default abscissa, or when get_ics_cooling_tf works.

    """

    T = phys.TCMB(rs)

    # Photon transfer function for single primary electron single scattering.
    # This is dN/(dE dt), dt = 1 s.
    ICS_tf = ics_spec(
        eleceng, photeng, T, thomson_tf = raw_thomson_tf, rel_tf = raw_rel_tf
    )

    # Downcasting speeds up np.dot
    ICS_tf._grid_vals = ICS_tf.grid_vals.astype('float64')

    # Energy loss transfer function for single primary electron
    # single scattering. This is dN/(dE dt), dt = 1 s.
    engloss_tf = engloss_spec(
        eleceng, photeng, T, thomson_tf = raw_engloss_tf, rel_tf = raw_rel_tf
    )

    # Downcasting speeds up np.dot
    engloss_tf._grid_vals = engloss_tf.grid_vals.astype('float64')

    # Switch the spectra type here to type 'N'.
    if ICS_tf.spec_type == 'dNdE':
        ICS_tf.switch_spec_type()
    if engloss_tf.spec_type == 'dNdE':
        engloss_tf.switch_spec_type()


    # Define some useful lengths.
    N_eleceng = eleceng.size
    N_photeng = photeng.size

    # Create the secondary electron transfer function.

    sec_elec_tf = tf.TransFuncAtRedshift(
        np.zeros((N_eleceng, N_eleceng)), in_eng = eleceng,
        rs = rs*np.ones_like(eleceng), eng = eleceng,
        dlnz = -1, spec_type = 'N'
    )

    sec_elec_tf._grid_vals = spectools.engloss_rebin_fast(
        eleceng, photeng, engloss_tf.grid_vals, eleceng
    )

    # Change from energy loss spectrum to secondary electron spectrum.
    # for i, in_eng in enumerate(eleceng):
    #     spec = engloss_tf[i]
    #     spec.engloss_rebin(
    #         in_eng, eleceng, out_spec_type='N', fast=True
    #     )
    #     # Add to the appropriate row.
    #     sec_elec_tf._grid_vals[i] += spec.N

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
    # Total upscattered photon energy.
    cont_loss_vec = np.zeros_like(eleceng)
    # Deposited energy, enforces energy conservation.
    deposited_vec = np.zeros_like(eleceng)

    # Test input electron to get the spectra.
    delta_spec = np.zeros_like(eleceng)

    # Start building sec_phot_tf and sec_lowengelec_tf.
    # Low energy regime first.

    sec_lowengelec_tf._grid_vals[:eleceng_low.size, :eleceng_low.size] = (
        np.identity(eleceng_low.size)
    )

    # Continuum energy loss rate, dU_CMB/dt.
    CMB_upscatter_eng_rate = phys.thomson_xsec*phys.c*phys.CMB_eng_density(T)


    # High energy electron loop to get fully resolved spectrum.
    for i, eng in zip(eleceng_high_ind, eleceng_high):

        # print('Check energies and indexing: ')
        # print(i, eleceng[i], eng)

        sec_phot_spec_N = ICS_tf._grid_vals[i]

        sec_elec_spec_N = sec_elec_tf._grid_vals[i]

        # The total number of primaries scattered is equal to the total number of scattered *photons*.
        # The scattered electrons is obtained from the *net* energy loss, and
        # so is not indicative of number of scatters.
        tot_N_scatter = np.sum(sec_phot_spec_N)
        # The total energy of primary electrons which is scattered per unit time.
        tot_eng_scatter = tot_N_scatter*eng
        # The *net* total number of secondary photons produced
        # per unit time.
        sec_elec_totN = np.sum(sec_elec_spec_N)
        # The *net* total energy of secondary electrons produced
        # per unit time.
        sec_elec_toteng = np.dot(sec_elec_spec_N, eleceng)
        # The total energy of secondary photons produced per unit time.
        sec_phot_toteng = np.dot(sec_phot_spec_N, photeng)
        # Deposited energy per unit time, dD/dt.
        # Numerical error (should be zero except for numerics)
        deposited_eng = sec_elec_totN*eng - sec_elec_toteng - (sec_phot_toteng - CMB_upscatter_eng_rate)

        diagnostics = False

        if diagnostics:
            print('-------- Injection Energy: ', eng)
            print(
                '-------- No. of Scatters (Analytic): ',
                phys.thomson_xsec*phys.c*phys.CMB_N_density(T)
            )
            print(
                '-------- No. of Scatters (Computed): ',
                tot_N_scatter
            )
            gamma_elec = 1 + eng/phys.me
            beta_elec  = np.sqrt(eng/phys.me*(gamma_elec+1)/gamma_elec**2)
            print(
                '-------- Energy lost (Analytic): ',
                (4/3)*phys.thomson_xsec*phys.c*phys.CMB_eng_density(T)*(
                    gamma_elec**2 * beta_elec**2
                )
            )
            print(
                '-------- Energy lost (Computed from photons): ',
                engloss_tf[i].toteng()
            )
            print(
                '-------- Energy lost (Computed from electrons): ',
                sec_elec_totN*eng - sec_elec_toteng
            )
            print(
                '-------- Energy of upscattered photons: ',
                CMB_upscatter_eng_rate
            )
            print(
                '-------- Energy in secondary photons (Computed): ',
                sec_phot_toteng
            )
            print(
                '-------- Energy in secondary photons (Analytic): ',
                phys.thomson_xsec*phys.c*phys.CMB_eng_density(T)*(
                    1 + (4/3)* gamma_elec**2 * beta_elec**2
                )
            )
            print(
                '-------- Energy gain from photons: ',
                sec_phot_toteng - CMB_upscatter_eng_rate
            )
            print('-------- Deposited Energy: ', deposited_eng)


        # In the original code, the energy of the electron has gamma > 20,
        # then the continuum energy loss is assigned to deposited_eng instead.
        # I'm not sure if this is necessary, but let's be consistent with the
        # original code for now.

        continuum_engloss = CMB_upscatter_eng_rate

        if eng + phys.me > 20*phys.me:
            deposited_eng += CMB_upscatter_eng_rate
            continuum_engloss = 0

        # Normalize to one secondary electron.

        sec_phot_spec_N /= sec_elec_totN
        sec_elec_spec_N /= sec_elec_totN
        continuum_engloss /= sec_elec_totN
        deposited_eng /= sec_elec_totN

        # Remove self-scattering.

        selfscatter_engfrac = (
            sec_elec_spec_N[i]
        )
        scattered_engfrac = 1 - selfscatter_engfrac

        sec_elec_spec_N[i] = 0

        sec_phot_spec_N /= scattered_engfrac
        sec_elec_spec_N /= scattered_engfrac
        continuum_engloss /= scattered_engfrac
        deposited_eng /= scattered_engfrac

        # Get the full secondary photon spectrum. Type 'N'
        resolved_phot_spec_vals = np.dot(
            sec_elec_spec_N, sec_phot_tf._grid_vals
        )
        # Get the full secondary low energy electron spectrum. Type 'N'.

        # resolved_lowengelec_spec_vals = np.dot(
        #     sec_elec_spec_N, sec_lowengelec_tf._grid_vals
        # )

        # The resolved lowengelec spectrum is simply one electron
        # in the bin just below 3 keV.
        # Added directly to sec_lowengelec_tf. Removed the dot for speed.
        # resolved_lowengelec_spec_vals = np.zeros_like(eleceng)
        # resolved_lowengelec_spec_vals[eleceng_low_ind[-1]] += 1

        # Add the resolved spectrum to the first scatter.
        sec_phot_spec_N += resolved_phot_spec_vals

        # Resolve the secondary electron continuum loss and deposition.
        continuum_engloss += np.dot(sec_elec_spec_N, cont_loss_vec)

        deposited_eng += np.dot(sec_elec_spec_N, deposited_vec)

        # Now, append the resulting spectrum to the transfer function.
        # Do this without calling append of course: just add to the zeros
        # that fill the current row in _grid_vals.
        sec_phot_tf._grid_vals[i] += sec_phot_spec_N
        sec_lowengelec_tf._grid_vals[i, eleceng_low_ind[-1]] += 1
        # Set the correct values in cont_loss_vec and deposited_vec.
        cont_loss_vec[i] = continuum_engloss
        deposited_vec[i] = deposited_eng

        check = False

        if check:

            conservation_check = (
                eng
                - np.dot(resolved_lowengelec_spec_vals, eleceng)
                + cont_loss_vec[i]
                - np.dot(sec_phot_spec_N, photeng)
            )

            if (
                conservation_check/eng > 0.01
            ):
                print('***************************************************')
                print('rs: ', rs)
                print('injected energy: ', eng)
                print(
                    'low energy e: ',
                    np.dot(resolved_lowengelec_spec_vals, eleceng)
                )
                print('scattered phot: ', np.dot(sec_phot_spec_N, photeng))
                print('continuum_engloss: ', cont_loss_vec[i])
                print(
                    'diff: ',
                    np.dot(sec_phot_spec_N, photeng) - cont_loss_vec[i]
                )
                print(
                    'energy is conserved up to (%): ',
                    conservation_check/eng*100
                )
                print('deposited: ', deposited_vec[i])
                print(
                    'energy conservation with deposited (%): ',
                    (conservation_check - deposited_vec[i])/eng*100
                )
                print('***************************************************')

                raise RuntimeError('Conservation of energy failed.')


    return (sec_phot_tf, sec_lowengelec_tf, cont_loss_vec, deposited_vec)

