import sys
sys.path.append("../..")

import numpy as np

from darkhistory import physics as phys
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec import spectools
from darkhistory.low_energy import lowE_electrons
from darkhistory.low_energy import lowE_photons

def compute_fs(spec_elec, spec_phot, x, dE_dVdt, time_step, method="old"):
    """ Compute f(z) fractions for continuum photons, photoexcitation of HI, and photoionization of HI, HeI, HeII

    Given a spectrum of deposited photons, resolve its energy into continuum photons,
    HI excitation, and HI, HeI, HeII ionization in that order.

    Parameters
     ----------
    spec_phot : Spectrum object
        spectrum of photons. Assumed to be in dNdE mode. spec.toteng() should return energy _per baryon_ per time.
    spec_elec : Spectrum object
        spectrum of electrons. Assumed to be in dNdE mode. spec.toteng() should return energy _per baryon_ per time.
    x : list of floats
        number of (HI, HeI, HeII) divided by nH at redshift photon_spectrum.rs
    dE_dVdt : float
        DM energy injection rate, dE/dVdt injected.
    time_step : float
        The time-step associated with the deposited spectra.
    method : {'old','ion','new'}
        'old': All photons >= 13.6eV ionize hydrogen, within [10.2, 13.6)eV excite hydrogen, < 10.2eV are labelled continuum.
        'ion': Same as 'old', but now photons >= 13.6 can ionize HeI and HeII also.
        'new': Same as 'ion', but now [10.2, 13.6)eV photons treated more carefully.

    Returns
    -------
    tuple of floats
    f_c(z) for z within spec.rs +/- time_step/2
    The order of the channels is {continuum photons, HI excitation, HI ionization, HeI ion, HeII ion}

    NOTE
    ----
    The CMB component hasn't been subtracted from the continuum photons yet
    Think about the exceptions that should be thrown (spec_elec.rs should equal spec_phot.rs)
    """

    ion_indx = spectools.get_indx(
        spec_phot.eng, phys.rydberg
    ) #Check this, also check spec_phot.eng?

    # np.array syntax below needed so that a fresh copy of eng and N are passed to the
    # constructor, instead of simply a reference. 

    ionized_elec = Spectrum(
        np.array(spec_phot.eng[ion_indx:]), np.array(spec_phot.N[ion_indx:]), rs=spec_phot.rs, spec_type='N'
    ) #Change this so that it uses totN(bin_bounds)

    new_eng = ionized_elec.eng - phys.rydberg
    if new_eng[0] < 0: #Is this the best way to do this?
        new_eng = np.insert(new_eng[1:], 0, 1e-12)
    ionized_elec.shift_eng(new_eng)

    # rebin so that ionized_elec may be added to spec_elec
    indx = ionized_elec.eng.size
    ionized_elec.rebin(spec_elec.eng[:indx+1])

    # Changed this so that spec_elec is not modified. 
    # spec_elec.N[:indx+1] += ionized_elec.N
    tmp_spec_elec = Spectrum(np.array(spec_elec.eng), np.array(spec_elec.N), rs=spec_elec.rs, spec_type='N')
    tmp_spec_elec.N[:indx+1] += ionized_elec.N

    f_phot = lowE_photons.compute_fs(
        spec_phot, x, dE_dVdt, time_step, method
    )

    f_elec = lowE_electrons.compute_fs(
        tmp_spec_elec, 1-x[0], dE_dVdt, time_step
    )

    return f_phot + f_elec
