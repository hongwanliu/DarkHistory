import sys
sys.path.append("../..")

import numpy as np

from darkhistory import physics as phys
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec import spectools
from darkhistory.low_energy import lowE_electrons
from darkhistory.low_energy import lowE_photons

def compute_fs(MEDEA_interp, elec_spec, phot_spec, x, dE_dVdt_inj, dt, highengdep, cmbloss, method="old", separate_higheng=False):
    """ Compute f(z) fractions for continuum photons, photoexcitation of HI, and photoionization of HI, HeI, HeII

    Given a spectrum of deposited electrons and photons, resolve their energy into
    H ionization, and ionization, H excitation, heating, and continuum photons in that order.

    Parameters
     ----------
    phot_spec : Spectrum object
        spectrum of photons. Assumed to be in dNdE mode. spec.totN() should return number _per baryon_.
    elec_spec : Spectrum object
        spectrum of electrons. Assumed to be in dNdE mode. spec.totN() should return number _per baryon_.
    x : list of floats
        number of (HI, HeI, HeII) divided by nH at redshift photon_spectrum.rs
    dE_dVdt_inj : float
        DM energy injection rate, dE/dVdt injected.  This is for unclustered DM (i.e. without structure formation).
    dt : float
        time in seconds over which these spectra were deposited.
    highengdep : list of floats
        total amount of energy deposited by high energy particles into {H_ionization, H_excitation, heating, continuum} per baryon per time, in that order.
    cmbloss : float
        Total amount of energy in upscattered photons that came from the CMB, per baryon per time, (1/n_B)dE/dVdt
    method : {'old','ion','new'}
        'old': All photons >= 13.6eV ionize hydrogen, within [10.2, 13.6)eV excite hydrogen, < 10.2eV are labelled continuum.
        'ion': Same as 'old', but now photons >= 13.6 can ionize HeI and HeII also.
        'new': Same as 'ion', but now [10.2, 13.6)eV photons treated more carefully.
    separate_higheng : bool, optional
        If True, returns separate high energy deposition. 

    Returns
    -------
    ndarray or tuple of ndarray
    f_c(z) for z within spec.rs +/- dt/2
    The order of the channels is {H Ionization, He Ionization, H Excitation, Heating and Continuum} 

    NOTE
    ----
    The CMB component hasn't been subtracted from the continuum photons yet
    Think about the exceptions that should be thrown (elec_spec.rs should equal phot_spec.rs)
    """

    # np.array syntax below needed so that a fresh copy of eng and N are passed to the
    # constructor, instead of simply a reference. 
    ion_bounds = spectools.get_bounds_between(phot_spec.eng, phys.rydberg)
    ion_engs = np.exp((np.log(ion_bounds[1:])+np.log(ion_bounds[:-1]))/2)

    ionized_elec = Spectrum(
        ion_engs,
        phot_spec.totN(bound_type="eng", bound_arr=ion_bounds),
        rs=phot_spec.rs,
        spec_type='N'
    )

    new_eng = ion_engs - phys.rydberg
    ionized_elec.shift_eng(new_eng)

    # rebin so that ionized_elec may be added to elec_spec
    ionized_elec.rebin(elec_spec.eng)

    tmp_elec_spec = Spectrum(np.array(elec_spec.eng), np.array(elec_spec.N), rs=elec_spec.rs, spec_type='N')
    tmp_elec_spec.N += ionized_elec.N

    f_phot = lowE_photons.compute_fs(
        phot_spec, x, dE_dVdt_inj, dt, method
    )
    #print(phot_spec.rs, f_phot[0], phot_spec.toteng(), cmbloss, dE_dVdt_inj)

    f_elec = lowE_electrons.compute_fs(
        MEDEA_interp, tmp_elec_spec, 1-x[0], dE_dVdt_inj, dt
    )

    #print('photons:', f_phot[2], f_phot[3]+f_phot[4], f_phot[1], 0, f_phot[0])
    #print('electrons:', f_elec[2], f_elec[3], f_elec[1], f_elec[4], f_elec[0])

    # f_low is {H ion, He ion, Lya Excitation, Heating, Continuum}
    f_low = np.array([
        f_phot[2]+f_elec[2],
        f_phot[3]+f_phot[4]+f_elec[3],
        f_phot[1]+f_elec[1],
        f_elec[4],
        f_phot[0]+f_elec[0] - cmbloss*phys.nB*phot_spec.rs**3 / dE_dVdt_inj
    ])

    f_high = np.array([
        highengdep[0], 0, highengdep[1],
        highengdep[2], highengdep[3]
    ]) * phys.nB * phot_spec.rs**3 / dE_dVdt_inj

    if separate_higheng:
        return (f_low, f_high)
    else:
        return f_low + f_high
