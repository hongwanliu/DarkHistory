""" PPPC4DMID [1]_ [2]_ Information and Functions. 

"""

import sys
import numpy as np
import json

# from config import data_path
from darkhistory.config import load_data

import darkhistory.physics as phys
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectools import rebin_N_arr


# Mass threshold for mDM to annihilate into the primaries.
mass_threshold = {
    'elec_delta': phys.mass['e'], 'phot_delta': 0.,
    'e_L'   : phys.mass['e'],   'e_R': phys.mass['e'], 
    'e': phys.mass['e'],
    'mu_L'  : phys.mass['mu'], 'mu_R': phys.mass['mu'], 
    'mu': phys.mass['mu'],
    'tau_L' : phys.mass['tau'], 'tau_R' : phys.mass['tau'], 
    'tau': phys.mass['tau'],
    'q'     : 0.,             'c': phys.mass['c'],   
    'b'     : phys.mass['b'], 't': phys.mass['t'],
    'W_L': phys.mass['W'], 'W_T'   : phys.mass['W'], 'W': phys.mass['W'],
    'Z_L': phys.mass['Z'], 'Z_T'   : phys.mass['Z'], 'Z': phys.mass['Z'],
    'g': 0., 'gamma' : 0., 'h': phys.mass['h'],
    'nu_e': 0., 'nu_mu' : 0., 'nu_tau': 0.,
    'VV_to_4e'   : 2*phys.mass['e'], 'VV_to_4mu' : 2*phys.mass['mu'], 
    'VV_to_4tau' : 2*phys.mass['tau']
}

def get_pppc_spec(mDM, eng, pri, sec, decay=False):

    """ Returns the PPPC4DMID spectrum. 

    This is the secondary spectrum to e+e-/photons normalized to one annihilation or decay event to the species specified in ``pri``. These results include electroweak corrections. The full list of allowed channels is: 

    - :math:`\\delta`\ -function injections: ``elec_delta, phot_delta``
    - Leptons: ``e_L, e_R, e, mu_L, mu_R, mu, tau_L, tau_R, tau``
    - Quarks:  ``q, c, b, t``
    - Gauge bosons: ``gamma, g, W_L, W_T, W, Z_L, Z_T, Z``
    - Higgs: ``h``

    ``elec_delta`` and ``phot_delta`` assumes annihilation or decay to two electrons and photons respectively with no EW corrections or ISR/FSR. 

    Variables with subscripts, e.g. ``e_L``, correspond to particles with different polarizations. These polarizations are suitably averaged to obtain the spectra returned in their corresponding variables without subscripts, e.g. ``e``. 

    Parameters
    ----------
    mDM : float
        The mass of the annihilating/decaying dark matter particle (in eV). 
    eng : ndarray
        The energy abscissa for the output spectrum (in eV). 
    pri : string
        One of the available channels (see above). 
    sec : {'elec', 'phot'}
        The secondary spectrum to obtain. 
    decay : bool, optional
        If ``True``, returns the result for decays.

    Returns
    -------
    Spectrum
        Output :class:`.Spectrum` object, ``spec_type == 'dNdE'``.
    
    """

    if decay:
        # Primary energies is for 1 GeV decay = 0.5 GeV annihilation.
        _mDM = mDM/2.
    else:
        _mDM = mDM

    if _mDM < mass_threshold[pri]:
        # This avoids the absurd situation where mDM is less than the 
        # threshold but we get a nonzero spectrum due to interpolation.
        raise ValueError('mDM is below the threshold to produce pri particles.')

    if pri == 'elec_delta':
        # Exact kinetic energy of each electron.
        if not decay:
            eng_elec = mDM - phys.me
        else:
            eng_elec = (mDM - 2*phys.me)/2
        # Find the correct bin in eleceng.
        eng_to_inj = eng[eng < eng_elec][-1]
        # Place 2*eng_elec worth of electrons into that bin. Use
        # rebinning to accomplish this.

        if sec == 'elec':
            return rebin_N_arr(
                np.array([2 * eng_elec / eng_to_inj]), 
                np.array([eng_to_inj]), 
                eng
            )

        elif sec == 'phot':
            return Spectrum(
                eng, np.zeros_like(eng), spec_type='dNdE'
            )

        else:
            raise ValueError('invalid sec.')

    if pri == 'phot_delta':
        # Exact kinetic energy of each photon. 
        if not decay:
            eng_phot = mDM
        else:
            eng_phot = mDM/2
        
        if sec == 'elec':
            return Spectrum(
                eng, np.zeros_like(eng), spec_type='dNdE'
            )
        elif sec == 'phot':
            # Find the correct bin in photeng.
            eng_to_inj = eng[eng < eng_phot][-1]
            # Place 2*eng_phot worth of photons into that bin. Use
            # rebinning to accomplish this. 
            return rebin_N_arr(
                np.array([2 * eng_phot / eng_to_inj]), 
                np.array([eng_to_inj]), 
                eng
            )

        else:
            raise ValueError('invalid sec.')
    
    log10x = np.log10(eng/_mDM)

    # Refine the binning so that the spectrum is accurate. 
    # Do this by checking that in the relevant range, there are at
    # least 50,000 bins. If not, double (unless an absurd number
    # of bins already). 

    if (
        log10x[(log10x < 1) & (log10x > 1e-9)].size > 0
        and log10x.size < 500000
    ):
        while log10x[(log10x < 1) & (log10x > 1e-9)].size < 50000:
            log10x = np.interp(
                np.arange(0, log10x.size-0.5, 0.5), 
                np.arange(log10x.size), 
                log10x
            )
    # Get the interpolator. 
    dlNdlxIEW_interp = load_data('pppc')

    # Get the spectrum from the interpolator.
    dN_dlog10x = 10**dlNdlxIEW_interp[sec][pri].get_val(_mDM/1e9, log10x)

    # Recall that dN/dE = dN/dlog10x * dlog10x/dE
    x = 10**log10x
    spec = Spectrum(x*_mDM, dN_dlog10x/(x*_mDM*np.log(10)), spec_type='dNdE')
    
    # Rebin down to the original binning.

    # The highest bin of spec.eng should be the same as eng[-1], based on
    # the interpolation strategy above. However, sometimes a floating point
    # error is picked up. We'll get rid of this so that rebin doesn't
    # complain.
    spec.eng[-1] = eng[-1]
    spec.rebin(eng)
        
    return spec


