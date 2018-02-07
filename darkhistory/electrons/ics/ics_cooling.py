"""Electron cooling through ICS."""

import numpy as np 
import pickle

import darkhistory.physics as phys
import darkhistory.spec.transferfunction as tf

from darkhistory.spec.spectrum import Spectrum

from darkhistory.electrons.ics.ics_spectrum import ics_spec 
from darkhistory.electrons.ics.ics_engloss_spectrum import engloss_spec 

def ICS_photon_tf(
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
        spec_elec_spec.rs = rs
        # Append the spectra to the transfer functions. 
        sec_phot_tf.append(sec_phot_spec)
        sec_lowengelec_tf.append(sec_elec_spec)















