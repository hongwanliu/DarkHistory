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
        The electron energy abscissa. 
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
    The raw transfer functions should be generated when the code package is first installed.

    """

    T = phys.TCMB(rs)

    # Photon transfer function for primary electron single scattering.
    ICS_tf = ics_spec(
        eleceng, photeng, T, nonrel_tf = raw_nonrel_tf, rel_tf = raw_rel_tf
    )

    # Energy loss transfer function for primary electron single scattering.
    engloss_tf = engloss_spec(
        eleceng, photeng, T, nonrel_tf = raw_engloss_tf, rel_tf = raw_rel_tf
    )

    # Secondary electron transfer function from primary electron. 
    sec_elec_tf = tf.TransFuncAtRedshift([])

    for i, in_eng in enumerate(engloss_tf.in_eng):
        spec = engloss_tf[i]
        spec.engloss_rebin(in_eng, engloss_tf.in_eng)
        sec_elec_tf.append(spec)

    # Low and high energy boundaries
    loweng = 250
    eleceng_high = eleceng[eleceng > loweng]
    eleceng_high_ind = np.arange(eleceng.size)[eleceng > loweng]
    eleceng_low = eleceng[eleceng <= loweng]
    eleceng_low_ind  = np.arange(eleceng.size)[eleceng <= loweng]

    # Empty containers for quantities.
    # Final full photon spectrum.
    sec_phot_tf = tf.TransFuncAtRedshift([], dlnz=-1, spec_type='N')
    # Final low energy electron spectrum. 
    sec_lowengelec_tf = tf.TransFuncAtRedshift([], dlnz=-1, spec_type='N')
    # Total upscattered photon energy. 
    cont_loss_vec = np.zeros_like(eleceng)
    # Deposited energy, enforces energy conservation. 
    deposited_vec = np.zeros_like(eleceng)

    # Low energy electrons. 
    delta_spec = np.zeros_like(eleceng)
    for i, eng in zip(eleceng_low_ind, eleceng_low):
        # Construct the delta function spectrum. 
        delta_spec *= 0
        delta_spec[i] = 1
        # Secondary electrons and photons. Trivial for low energy. 
        sec_phot_spec = Spectrum(
            photeng, np.zeros_like(photeng), spec_type='N'
        )
        sec_phot_spec.in_eng = eng
        sec_phot_spec.rs = rs

        sec_elec_spec = Spectrum(
            eleceng, delta_spec, spec_type='N'
        )
        sec_elec_spec.in_eng = eng
        sec_elec_spec.rs = rs
        # Append the spectra to the transfer functions. 
        sec_phot_tf.append(sec_phot_spec)
        sec_lowengelec_tf.append(sec_elec_spec)


    # High energy electrons. 
    for i,eng in zip(eleceng_high_ind, eleceng_high):

        # Initialize the single electron. 
        delta_spec = np.zeros_like(eleceng)
        delta_spec[i] = 1

        pri_elec_spec = Spectrum(eleceng, delta_spec, rs=rs, spec_type='N')

        # Get the secondary photons, dN_gamma/(dE_gamma dt). 
        # mode and out_mode specifies the input and output of the function. 
        # 'N' means an array of numbers, while 'dNdE' will input or output a Spectrum.
        # If 'N' is chosen, we need to specify the abscissa 'eng_arr' and list
        # of numbers, 'N_arr'. new_eng determines the output abscissa.
        sec_phot_spec = spectools.scatter(
            ICS_tf, pri_elec_spec, new_eng = photeng
        )
        sec_phot_spec.switch_spec_type()

        # Get the secondary electrons, dN_e/(dE_e dt).
        sec_elec_spec = spectools.scatter(
            sec_elec_tf, pri_elec_spec, new_eng = eleceng
        )
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
        sec_phot_totN = sec_phot_spec.toteng()
        # Deposited energy per unit time, dD/dt. 
        deposited_eng = pri_elec_toteng - sec_elec_toteng - (sec_phot_totN - continuum_engloss)

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
        
        selfscatter_engfrac = sec_elec_spec.N[i]*eleceng[i]/(sec_elec_spec.totN()*eng)
        scattered_engfrac = 1 - selfscatter_engfrac

        sec_elec_spec.N[i] = 0

        sec_phot_spec /= scattered_engfrac
        sec_elec_spec /= scattered_engfrac
        continuum_engloss /= scattered_engfrac
        deposited_eng /= scattered_engfrac
        
        # First, we ensure that the secondary electron array has the same 
        # length as the transfer function (which we have been appending `Spectrum`
        # objects to from before). 

        scattered_elec_spec = Spectrum(eleceng[0:i], sec_elec_spec.N[0:i], spec_type='N')

        # Scatter into photons. For this first high energy case, this
        # trivially gives no photons, but for higher energies, this is non-trivial.
        resolved_phot_spec = spectools.scatter(
            sec_phot_tf, scattered_elec_spec, new_eng = photeng
        )

        # Scatter into low energy electrons. For this first high energy case, 
        # it's also trivial, since the transfer matrix is the identity. 
        resolved_lowengelec_spec = spectools.scatter(
            sec_lowengelec_tf, scattered_elec_spec, new_eng = eleceng
        )
        
        sec_phot_spec += resolved_phot_spec
        sec_elec_spec = resolved_lowengelec_spec
        
        # In this case, this is trivial because they are all zero, but becomes
        # non-trivial for higher energies.
        continuum_engloss += np.dot(scattered_elec_spec.N, cont_loss_vec[0:i])
        deposited_eng += np.dot(scattered_elec_spec.N, deposited_vec[0:i])
       
        # Set the properties of the spectra
        sec_phot_spec.in_eng = eng
        sec_elec_spec.in_eng = eng
        sec_phot_spec.rs = rs
        sec_elec_spec.rs = rs

        # Append to the transfer function
        sec_phot_tf.append(sec_phot_spec)
        sec_lowengelec_tf.append(sec_elec_spec)

        # Set the correct values in the cont_loss_vec and deposited_vec
        cont_loss_vec[i] = continuum_engloss
        deposited_vec[i] = deposited_eng
        
        # Conservation of energy check. Check that it is 1e-10 of eng.
        
        conservation_check = (eng + continuum_engloss 
                          - sec_elec_spec.toteng()
                          - sec_phot_spec.toteng()
                          - deposited_eng
                         )
        
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

















