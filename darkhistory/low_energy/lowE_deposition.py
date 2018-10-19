import sys
sys.path.append("../..")

import numpy as np

from darkhistory import physics as phys
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec import spectools
from darkhistory.low_energy import lowE_electrons
from darkhistory.low_energy import lowE_photons

def compute_fs(spec_elec, spec_phot, x, dE_dVdt_inj, dt, cmbloss, method="old"):
    """ Compute f(z) fractions for continuum photons, photoexcitation of HI, and photoionization of HI, HeI, HeII

    Given a spectrum of deposited electrons and photons, resolve their energy into
    H ionization, and ionization, H excitation, heating, and continuum photons in that order.

    Parameters
     ----------
    spec_phot : Spectrum object
        spectrum of photons. Assumed to be in dNdE mode. spec.totN() should return number _per baryon_ per time.
    spec_elec : Spectrum object
        spectrum of electrons. Assumed to be in dNdE mode. spec.totN() should return number _per baryon_ per time.
    x : list of floats
        number of (HI, HeI, HeII) divided by nH at redshift photon_spectrum.rs
    dE_dVdt_inj : float
        DM energy injection rate, dE/dVdt injected.  This is for unclustered DM (i.e. without structure formation).
    dt : float
        time in seconds over which these spectra were deposited.
    cmbloss : float
        Total amount of energy in upscattered photons that came from the CMB, per baryon per time, (1/n_B)dE/dVdt
    method : {'old','ion','new'}
        'old': All photons >= 13.6eV ionize hydrogen, within [10.2, 13.6)eV excite hydrogen, < 10.2eV are labelled continuum.
        'ion': Same as 'old', but now photons >= 13.6 can ionize HeI and HeII also.
        'new': Same as 'ion', but now [10.2, 13.6)eV photons treated more carefully.

    Returns
    -------
    tuple of floats
    f_c(z) for z within spec.rs +/- dt/2
    The order of the channels is {continuum photons, HI excitation, HI ionization, HeI ion, HeII ion}

    NOTE
    ----
    The CMB component hasn't been subtracted from the continuum photons yet
    Think about the exceptions that should be thrown (spec_elec.rs should equal spec_phot.rs)
    """

    # np.array syntax below needed so that a fresh copy of eng and N are passed to the
    # constructor, instead of simply a reference. 
    ion_bounds = spectools.get_bounds_between(spec_phot.eng, phys.rydberg)
    ion_engs = np.exp((np.log(ion_bounds[1:])+np.log(ion_bounds[:-1]))/2)

    ionized_elec = Spectrum(
        ion_engs,
        spec_phot.totN(bound_type="eng", bound_arr=ion_bounds),
        rs=spec_phot.rs,
        spec_type='N'
    )

    new_eng = ion_engs - phys.rydberg
    ionized_elec.shift_eng(new_eng)

    # rebin so that ionized_elec may be added to spec_elec
    ionized_elec.rebin(spec_elec.eng)

    tmp_spec_elec = Spectrum(np.array(spec_elec.eng), np.array(spec_elec.N), rs=spec_elec.rs, spec_type='N')
    tmp_spec_elec.N += ionized_elec.N

    f_phot = lowE_photons.compute_fs(
        spec_phot, x, dE_dVdt_inj, dt, method
    )

    f_elec = lowE_electrons.compute_fs(
        tmp_spec_elec, 1-x[0], dE_dVdt_inj, dt
    )

    f_final = np.array([
        f_phot[2]+f_elec[2],
        f_phot[3]+f_phot[4]+f_elec[3],
        f_phot[1]+f_elec[1],
        f_elec[4],
        f_phot[0]+f_elec[0] - cmbloss * phys.nB * spec_phot.rs**3 / dE_dVdt_inj
    ])

    return f_final
