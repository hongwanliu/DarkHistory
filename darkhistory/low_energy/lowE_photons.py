import sys
sys.path.append("../..")

import numpy as np
import physics as phys

def compute_dep_inj_ionization_ratio(photon_spectrum, n, tot_inj):
    """ Given a photon spectrum object, deposit its energy into HI, HeI, and HeII ionization in that order
        Q: can photons heat the IGM?  Should this method keep track of the fact that x_e, xHII, etc. are changing?

    Parameters
    ----------
    photon_spectrum : Spectrum object
        spectrum of photons
    n : (float,float,float)
        density of (HI, HeI, HeII).
    tot_inj : float
        total energy injected by DM

    Returns
    -------
    list of floats
        Ratio of deposited energy to a given channel over energy deposited by DM.
        The order of the channels is HI, HeI, and HeII ionization
    """

    # [# photons absorbed] / [unit time] = N(E) \sigma(E) n_species c
    ionHI = sum(photon_spectrum.N*phys.photo_ion_xsec(photon_spectrum.eng,'H0'))*n[0]*phys.c/tot_inj
    ionHeI = sum(photon_spectrum.N*phys.photo_ion_xsec(photon_spectrum.eng,'He0'))*n[1]*phys.c/tot_inj
    ionHeII = sum(photon_spectrum.N*phys.photo_ion_xsec(photon_spectrum.eng,'He1'))*n[2]*phys.c/tot_inj

    return ionHI, ionHeI, ionHeII
