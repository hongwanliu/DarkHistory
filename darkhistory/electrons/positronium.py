"""Positronium annihilation functions."""

import numpy as np 

from darkhistory.spectrum import Spectrum
from darkhistory.spectools import discretize
from darkhistory.spectools import rebin_N_arr

def ortho_photon_spec(eng):
    """ Returns the photon spectrum from orthopositronium annihilation. 

    This is normalized to a single annihilation, producing 3 photons. 

    Parameters
    ----------
    eng : ndarray
        The energy abscissa. 

    Returns
    -------
    Spectrum
        The resulting photon spectrum. 
    """

    fac = np.pi**2 - 9

    # dN/dE, unnormalized.
    # 3/fac normalizes dN/dE so that integral is 3, and 
    # integral of E dN/dE is 2*phys.me.
    def norm_spec(eng):
        norm_eng = eng/phys.me
        fac = np.pi**2 - 9
        if eng < phys.me:
            return 3/fac*(
                2*(2.-norm_eng)/norm_eng 
                + 2*(1. - norm_eng)*norm_eng/(2. - norm_eng)**2
                + 4*np.log(1. - norm_eng)*(
                    (1. - norm_eng)/norm_eng**2 
                    - (1. - norm_eng)**2/(2. - norm_eng)**3
                )
            )
        else:
            return 0

    return discretize(eng, norm_spec)

def para_photon_spec(eng):
    """ Returns the photon spectrum from parapositronium annihilation. 

    This is normalized to a single annihilation, producing 2 photons. 

    Parameters
    ----------
    eng : ndarray
        The energy abscissa. 

    Returns
    -------
    Spectrum
        The resulting photon spectrum. 
    """

    return rebin_N_arr(np.array([2]), np.array([phys.me]), eng)

def weighted_photon_spec(eng):
    """ Returns the weighted photon spectrum from positronium annihilation.

    This assumes 3/4 ortho- and 1/4 para-, normalized to a single 
    annihilation. 

    Parameters
    ----------
    eng : ndarray
        The energy abscissa. 

    Returns
    -------
    Spectrum
        The resulting photon spectrum. 
    """

    return 3/4*ortho_photon_spec(eng) + 1/4*para_photon_spec(eng)
