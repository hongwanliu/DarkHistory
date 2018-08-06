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
    raw_nonrel_tf, raw_rel_tf, raw_engloss_tf, eleceng, photeng, rs
):

    """Returns transfer function for complete electron cooling through ICS.

    Parameters
    ----------
    nonrel_tf : TransFuncAtRedshift
        Raw nonrelativistic primary electron ICS transfer function.
    rel_tf : string
        Raw relativistic primary electron ICS transfer function. 
    engloss_tf_filename : string
        Raw primary electron ICS energy loss transfer function. 
    eleceng : ndarray
        The electron kinetic energy abscissa. 
    photeng : ndarray
        The photon energy abscissa. 
    rs : float
        The redshift.

    Returns
    -------

    tuple of TransFuncAtRedshift
        Transfer functions for photons and low energy electrons.

    Note
    ----
    The raw transfer functions should be generated when the code package is first installed. The transfer function corresponds to the fully resolved
    photon spectrum after scattering by one electron. 

    """

    T = phys.TCMB(rs)

    # Photon transfer function for single primary electron single scattering.
    # This is dN/(dE dt), dt = 1 s.
    ICS_tf = ics_spec(
        eleceng, photeng, T, nonrel_tf = raw_nonrel_tf, rel_tf = raw_rel_tf
    )

    # Downcasting speeds up np.dot
    ICS_tf._grid_vals = ICS_tf.grid_vals.astype('float64')

    # Energy loss transfer function for single primary electron 
    # single scattering. This is dN/(dE dt), dt = 1 s. 
    engloss_tf = engloss_spec(
        eleceng, photeng, T, nonrel_tf = raw_engloss_tf, rel_tf = raw_rel_tf
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

    append_sec_elec_tf = sec_elec_tf.append

    # Change from energy loss spectrum to secondary electron spectrum.
    for i, in_eng in enumerate(eleceng):
        spec = engloss_tf[i]
        spec.engloss_rebin(in_eng, eleceng)
        # Add to the appropriate row.
        sec_elec_tf._grid_vals[i] += spec.dNdE

    # Low and high energy boundaries
    loweng = 250
    eleceng_high = eleceng[eleceng > loweng]
    eleceng_high_ind = np.arange(eleceng.size)[eleceng > loweng]
    eleceng_low = eleceng[eleceng <= loweng]
    eleceng_low_ind  = np.arange(eleceng.size)[eleceng <= loweng]

    if eleceng_low.size == 0:
        raise TypeError('Energy abscissa must contain a low energy bin below 250 eV.')

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
    for i, eng in zip(eleceng_low_ind, eleceng_low):
        # Zero out delta function test spectrum, set it correctly
        # for the loop ahead. 
        delta_spec *= 0
        delta_spec[i] = 1
        # Add the trivial secondary electron spectrum to the 
        # transfer function. 
        sec_lowengelec_tf._grid_vals[i] += delta_spec

    for i, eng in zip(eleceng_high_ind, eleceng_high):
        # Zero out delta function test spectrum, set it correctly
        # for the loop ahead. 
        delta_spec *= 0
        delta_spec[i] = 1

        # Put the delta function in a Spectrum.
        pri_elec_spec = Spectrum(eleceng, delta_spec, rs=rs, spec_type='N')
        
        # Get the scattered photons, dN/(dE dt). 
        # Using delta_spec returns type 'dNdE', which is right.
        sec_phot_spec = ICS_tf.sum_specs(delta_spec)
        # Switch to type 'N'.
        if sec_phot_spec.spec_type == 'dNdE':
            sec_phot_spec.switch_spec_type()
        # Get the scattered electrons, dNe/(dE dt).
        # Using delta_spec returns type 'dNdE', which is right.
        sec_elec_spec = sec_elec_tf.sum_specs(delta_spec)
        # Switch to type 'N'.
        if sec_elec_spec.spec_type == 'dNdE':
            sec_elec_spec.switch_spec_type()

        # Continuum energy loss rate, dU_CMB/dt. 
        continuum_engloss = phys.thomson_xsec*phys.c*phys.CMB_eng_density(T)

        # The total number of primaries scattered is equal to the total number of secondaries scattered. 
        pri_elec_totN = sec_elec_spec.totN()
        # The total energy of primary electrons which is scattered per unit time. 
        pri_elec_toteng = pri_elec_totN*eng
        # The total energy of secondary electrons produced per unit time. 
        sec_elec_toteng = sec_elec_spec.toteng()
        # The total energy of secondary photons produced per unit time. 
        sec_phot_toteng = sec_phot_spec.toteng()
        # Deposited energy per unit time, dD/dt. 
        deposited_eng = pri_elec_toteng - sec_elec_toteng - (sec_phot_toteng - continuum_engloss)

        # In the original code, the energy of the electron has gamma > 20, 
        # then the continuum energy loss is assigned to deposited_eng instead. 
        # I'm not sure if this is necessary, but let's be consistent with the 
        # original code for now. 

        if eng + phys.me > 20*phys.me:
            deposited_eng += continuum_engloss
            continuum_engloss = 0

        # Normalize to one primary electron.
        
        sec_phot_spec /= pri_elec_totN
        sec_elec_spec /= pri_elec_totN
        continuum_engloss /= pri_elec_totN
        deposited_eng /= pri_elec_totN
        
        # Remove self-scattering.
        
        ############################################
        # OLD: rescale by energy in the last bin.  #
        ############################################

        # selfscatter_engfrac = (
        #     sec_elec_spec.N[i]*eleceng[i]/(sec_elec_spec.totN()*eng)
        # )
        # scattered_engfrac = 1 - selfscatter_engfrac

        ############################################
        # NEW: rescale by N in the last bin.       #
        ############################################

        selfscatter_Nfrac = sec_elec_spec.N[i]/sec_elec_spec.totN()
        scattered_Nfrac = 1 - selfscatter_Nfrac



        sec_elec_spec.N[i] = 0

        # sec_phot_spec /= scattered_engfrac
        # sec_elec_spec /= scattered_engfrac
        # continuum_engloss /= scattered_engfrac
        # deposited_eng /= scattered_engfrac

        sec_phot_spec /= scattered_Nfrac
        sec_elec_spec /= scattered_Nfrac
        continuum_engloss /= scattered_Nfrac
        deposited_eng /= scattered_Nfrac

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
        
        conservation_check = (eng + continuum_engloss 
                          - resolved_lowengelec_spec.toteng()
                          - sec_phot_spec.toteng()
                          - deposited_eng
                         )

        print('***************************************************')
        print('injected energy: ', eng)
        print('low energy e: ', resolved_lowengelec_spec.toteng())
        print('scattered phot: ', sec_phot_spec.toteng())
        print('continuum_engloss: ', continuum_engloss)
        print('diff: ', sec_phot_spec.toteng() - continuum_engloss)
        print('deposited: ', deposited_eng)
        print('***************************************************')
        
        if (
            conservation_check/eng > 1e-8
            and conservation_check/continuum_engloss > 1e-8
        ):
            print(conservation_check/eng)
            print(conservation_check/continuum_engloss)
            raise RuntimeError('Conservation of energy failed.')
            
        # Force conservation of energy. 
        deposited_vec[i] += conservation_check

    return (sec_phot_tf, sec_lowengelec_tf)
       
########################################
# OLD VERSION OF get_ics_cooling_tf.   #
# Rewritten due to too many appends.   #
########################################


# def get_ics_cooling_tf(
#     raw_nonrel_tf, raw_rel_tf, raw_engloss_tf, eleceng, photeng, rs
# ):
#     """Returns transfer function for complete electron cooling through ICS.

#     Parameters
#     ----------
#     nonrel_tf : TransFuncAtRedshift
#         Raw nonrelativistic primary electron ICS transfer function.
#     rel_tf : string
#         Raw relativistic primary electron ICS transfer function. 
#     engloss_tf_filename : string
#         Raw primary electron ICS energy loss transfer function. 
#     eleceng : ndarray
#         The electron kinetic energy abscissa. 
#     photeng : ndarray
#         The photon energy abscissa. 
#     rs : float
#         The redshift.

#     Returns
#     -------

#     tuple of TransFuncAtRedshift
#         Transfer functions for photons and low energy electrons.

#     Note
#     ----
#     The raw transfer functions should be generated when the code package is first installed.

#     """

#     T = phys.TCMB(rs)

#     # Photon transfer function for primary electron single scattering.
#     ICS_tf = ics_spec(
#         eleceng, photeng, T, nonrel_tf = raw_nonrel_tf, rel_tf = raw_rel_tf
#     )
#     # Downcasting speeds up np.dot
#     ICS_tf._grid_vals = ICS_tf.grid_vals.astype('float64')

#     # Energy loss transfer function for primary electron single scattering.
#     engloss_tf = engloss_spec(
#         eleceng, photeng, T, nonrel_tf = raw_engloss_tf, rel_tf = raw_rel_tf
#     )
#     # Downcasting speeds up np.dot
#     engloss_tf._grid_vals = engloss_tf.grid_vals.astype('float64')

#     # Secondary electron transfer function from primary electron. 

#     sec_elec_tf = tf.TransFuncAtRedshift([])
#     append_sec_elec_tf = sec_elec_tf.append

#     for i, in_eng in enumerate(engloss_tf.in_eng):
#         spec = engloss_tf[i]
#         spec.engloss_rebin(in_eng, engloss_tf.in_eng)
#         append_sec_elec_tf(spec)


#     # Low and high energy boundaries
#     loweng = 250
#     eleceng_high = eleceng[eleceng > loweng]
#     eleceng_high_ind = np.arange(eleceng.size)[eleceng > loweng]
#     eleceng_low = eleceng[eleceng <= loweng]
#     eleceng_low_ind  = np.arange(eleceng.size)[eleceng <= loweng]

#     if eleceng_low.size == 0:
#         raise TypeError('Energy abscissa must contain a low energy bin below 250 eV.')

#     # Empty containers for quantities.
#     # Final full photon spectrum.
#     sec_phot_tf = tf.TransFuncAtRedshift([], dlnz=-1, spec_type='N')
#     # Final low energy electron spectrum. 
#     sec_lowengelec_tf = tf.TransFuncAtRedshift([], dlnz=-1, spec_type='N')
#     # Total upscattered photon energy. 
#     cont_loss_vec = np.zeros_like(eleceng)
#     # Deposited energy, enforces energy conservation. 
#     deposited_vec = np.zeros_like(eleceng)

#     # Low energy electrons. 
#     delta_spec = np.zeros_like(eleceng)

#     append_sec_phot_tf = sec_phot_tf.append
#     append_sec_lowengelec_tf = sec_lowengelec_tf.append

#     for i, eng in zip(eleceng_low_ind, eleceng_low):
#         # Construct the delta function spectrum. 
#         delta_spec *= 0
#         delta_spec[i] = 1
#         # Secondary electrons and photons. Trivial for low energy. 
#         sec_phot_spec = Spectrum(
#             photeng, np.zeros_like(photeng), spec_type='N'
#         )
#         sec_phot_spec.in_eng = eng
#         sec_phot_spec.rs = rs

#         sec_elec_spec = Spectrum(
#             eleceng, delta_spec, spec_type='N'
#         )
#         sec_elec_spec.in_eng = eng
#         sec_elec_spec.rs = rs
#         append_sec_phot_tf(sec_phot_spec)
#         append_sec_lowengelec_tf(sec_elec_spec)

#     # High energy electrons. 
#     for i,eng in zip(eleceng_high_ind, eleceng_high):

#         # Initialize the single electron. 
#         delta_spec = np.zeros_like(eleceng)
#         delta_spec[i] = 1

#         pri_elec_spec = Spectrum(eleceng, delta_spec, rs=rs, spec_type='N')

#         # Get the secondary photons, dN_gamma/(dE_gamma dt). 
#         # mode and out_mode specifies the input and output of the function. 
#         # 'N' means an array of numbers, while 'dNdE' will input or output a Spectrum.
#         # If 'N' is chosen, we need to specify the abscissa 'eng_arr' and list
#         # of numbers, 'N_arr'. new_eng determines the output abscissa.
#         sec_phot_spec = spectools.scatter(
#             ICS_tf, pri_elec_spec, new_eng = photeng
#         )

#         # Get the secondary electrons, dN_e/(dE_e dt).
#         sec_elec_spec = spectools.scatter(
#             sec_elec_tf, pri_elec_spec, new_eng = eleceng
#         )

#         # Continuum energy loss rate, dU_CMB/dt. 
#         continuum_engloss = phys.thomson_xsec*phys.c*phys.CMB_eng_density(T)
        
#         # The total number of primaries scattered is equal to the total number of secondaries scattered. 
#         pri_elec_totN = sec_elec_spec.totN()
#         # The total energy of primary electrons which is scattered per unit time. 
#         pri_elec_toteng = pri_elec_totN*eng
#         # The total energy of secondary electrons produced per unit time. 
#         sec_elec_toteng = sec_elec_spec.toteng()
#         # The total energy of secondary photons produced per unit time. 
#         sec_phot_toteng = sec_phot_spec.toteng()
#         # Deposited energy per unit time, dD/dt. 
#         deposited_eng = pri_elec_toteng - sec_elec_toteng - (sec_phot_toteng - continuum_engloss)

#         # In the original code, the energy of the electron has gamma > 20, 
#         # then the continuum energy loss is assigned to deposited_eng instead. 
#         # I'm not sure if this is necessary, but let's be consistent with the 
#         # original code for now. 

#         if eng + phys.me > 20*phys.me:
#             deposited_eng += continuum_engloss
#             continuum_engloss = 0
        
#         # Normalize to one primary electron.
        
#         sec_phot_spec /= pri_elec_totN
#         sec_elec_spec /= pri_elec_totN
#         continuum_engloss /= pri_elec_totN
#         deposited_eng /= pri_elec_totN
        
#         # Remove self-scattering.
        
#         selfscatter_engfrac = sec_elec_spec.N[i]*eleceng[i]/(sec_elec_spec.totN()*eng)
#         scattered_engfrac = 1 - selfscatter_engfrac

#         sec_elec_spec.N[i] = 0

#         sec_phot_spec /= scattered_engfrac
#         sec_elec_spec /= scattered_engfrac
#         continuum_engloss /= scattered_engfrac
#         deposited_eng /= scattered_engfrac
        
#         # First, we ensure that the secondary electron array has the same 
#         # length as the transfer function (which we have been appending `Spectrum`
#         # objects to from before). 

#         scattered_elec_spec = Spectrum(eleceng[0:i], sec_elec_spec.N[0:i], spec_type='N')

#         # Scatter into photons. For this first high energy case, this
#         # trivially gives no photons, but for higher energies, this is non-trivial.

#         resolved_phot_spec = spectools.scatter(
#             sec_phot_tf, scattered_elec_spec, new_eng = photeng
#         )
        
#         # Scatter into low energy electrons. For this first high energy case, 
#         # it's also trivial, since the transfer matrix is the identity. 

#         resolved_lowengelec_spec = spectools.scatter(
#             sec_lowengelec_tf, scattered_elec_spec, new_eng = eleceng
#         )
        
#         sec_phot_spec += resolved_phot_spec
#         sec_elec_spec = resolved_lowengelec_spec
        
#         # In this case, this is trivial because they are all zero, but becomes
#         # non-trivial for higher energies.
#         continuum_engloss += np.dot(scattered_elec_spec.N, cont_loss_vec[0:i])
#         deposited_eng += np.dot(scattered_elec_spec.N, deposited_vec[0:i])
       
#         # Set the properties of the spectra
#         sec_phot_spec.in_eng = eng
#         sec_elec_spec.in_eng = eng
#         sec_phot_spec.rs = rs
#         sec_elec_spec.rs = rs

#         # OLD: Append to the transfer function

#         append_sec_phot_tf(sec_phot_spec)
#         append_sec_lowengelec_tf(sec_elec_spec)

#         # Set the correct values in the cont_loss_vec and deposited_vec
#         cont_loss_vec[i] = continuum_engloss
#         deposited_vec[i] = deposited_eng
        
#         # Conservation of energy check. Check that it is 1e-10 of eng.
        
#         conservation_check = (eng + continuum_engloss 
#                           - sec_elec_spec.toteng()
#                           - sec_phot_spec.toteng()
#                           - deposited_eng
#                          )
#         print('***************************************************')
#         print('injected energy: ', eng)
#         print('low energy e: ', resolved_lowengelec_spec.toteng())
#         print('scattered phot: ', sec_phot_spec.toteng())
#         print('continuum_engloss: ', continuum_engloss)
#         print('deposited: ', deposited_eng)
#         print('***************************************************')
        
#         if (
#             conservation_check/eng > 1e-8
#             and conservation_check/continuum_engloss > 1e-8
#         ):
#             print(conservation_check/eng)
#             print(conservation_check/continuum_engloss)
#             raise RuntimeError('Conservation of energy failed.')
            
#         # Force conservation of energy. 
#         deposited_vec[i] += conservation_check

#     return (sec_phot_tf, sec_lowengelec_tf)

















