import sys
sys.path.append("../..")

import numpy as np
import physics as phys

def compute_dep_inj_ionization_ratio(photon_spectrum, n, tot_inj, time_step):
    """ Given a spectrum of photons, deposit its energy into HI excitation and HI, HeI, HeII ionization in that order.  The
        spectrum must provide the number density of photons within each bin.
        Q: can photons heat the IGM?  Should this method keep track of the fact that x_e, xHII, etc. are changing?

    Parameters
    ----------
    photon_spectrum : Spectrum object
        spectrum of photons
    n : list of floats
        density of (HI, HeI, HeII).
    tot_inj : float
        total energy injected by DM
    time_step : float
        amount of time during which energy deposition takes place

    Returns
    -------
    tuple of floats
        Ratio of deposited energy to a given channel over energy deposited by DM.
        The order of the channels is HI excitation and HI, HeI, HeII ionization
    """

    # Assume 100% of photons within window [10.2eV, 13.6eV] are absorbed
    excite_HI = photon_spectrum.totN(bound_type='eng', bound_arr=np.array([phys.lya_eng,phys.rydberg]))[0]/(tot_inj*time_step)
    # [# photons absorbed] / [unit time] = N(E) \sigma(E) n_species c
    ionHI = sum(photon_spectrum.N*phys.photo_ion_xsec(photon_spectrum.eng,'H0'))*n[0]*phys.c/tot_inj
    ionHeI = sum(photon_spectrum.N*phys.photo_ion_xsec(photon_spectrum.eng,'He0'))*n[1]*phys.c/tot_inj
    ionHeII = sum(photon_spectrum.N*phys.photo_ion_xsec(photon_spectrum.eng,'He1'))*n[2]*phys.c/tot_inj

    return excite_HI, ionHI, ionHeI, ionHeII
