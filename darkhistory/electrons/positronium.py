"""Positronium annihilation functions."""

import numpy as np 

from darkhistory import physics as phys 
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectools import discretize
from darkhistory.spec.spectools import rebin_N_arr

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
        The resulting photon :class:`.Spectrum` object. 

    See Also
    --------
    :func:`.discretize` 
    """

    fac = np.pi**2 - 9

    # dN/d(E/phys.me), unnormalized.
    # 3/fac normalizes dN/d(E/phys.me) so that integral is 3, and 
    # integral of E dN/d(E/phys.me) is 2*phys.me.
    def norm_spec(eng):
        norm_eng = eng/phys.me
        fac = np.pi**2 - 9
        if eng < phys.me:
            if norm_eng > 0.001:
                return 3/fac/phys.me*(
                    2*(2.-norm_eng)/norm_eng 
                    + 2*(1. - norm_eng)*norm_eng/(2. - norm_eng)**2
                    + 4*np.log1p(-norm_eng)*(
                        (1. - norm_eng)/norm_eng**2 
                        - (1. - norm_eng)**2/(2. - norm_eng)**3
                    )
                )
            elif norm_eng <= 0.001:
                return 3/fac/phys.me*(
                    5/3*norm_eng + 1/3*norm_eng**2 - 2/15*norm_eng**3
                    - 1/5*norm_eng**4 - 29/210*norm_eng**5 
                    - 13/210*norm_eng**6 - 2/315*norm_eng**7
                    + 8/315*norm_eng**8 + 137/3465*norm_eng**9
                    + 149/3465*norm_eng**10
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
        The resulting photon :class:`.Spectrum` object. 

    See Also
    --------
    :func:`.rebin_N_arr`
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
        The resulting photon :class:`.Spectrum` object. 
    """

    return 3/4*ortho_photon_spec(eng) + 1/4*para_photon_spec(eng)
