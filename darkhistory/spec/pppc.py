""" PPPC4DMID [1]_ [2]_ Information and Functions. 

"""

import sys
import numpy as np
import json

# from config import data_path
from config import load_data

import darkhistory.physics as phys
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectools import rebin_N_arr
from scipy.interpolate import interp1d
from scipy.integrate import quad

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
    'VV_to_4tau' : 2*phys.mass['tau'],
    'pi': phys.mass['pi'], 'pi0': phys.mass['pi0']
}

# Function for Lorentz boosting energy spectra
def boost_elec_spec(y, spec, Emin=None, Emax=None):
    """
    Returns isotropic spectrum of electrons boosted by Lorentz factor y.
    
    Parameters
    ----------
    y : float
        Lorentz boost factor
    spec : ndarray
        Energy spectrum of electrons in original frame
    Emax : float
        Maximum energy for spectrum

    Returns
    -------
    Spectrum
    
    """
    # Boost velocity
    b = np.sqrt(1-(1/y**2))

    # Total energy
    toteng = spec.eng + phys.me

    # Momentum of electron from energy
    def p(e):
        return np.sqrt(e**2 - phys.me**2)
    
    new_dNdE = np.zeros_like(spec.dNdE)
    integrand = interp1d(toteng, spec.dNdE/p(toteng), kind='linear', bounds_error=False, fill_value=0.)

    for i, E in enumerate(toteng):
        # Limits of integration
        if Emax is None:
            Eupper = y * (E + b*p(E))
        else:
            Eupper = min(Emax, y * (E + b*p(E)))

        if Emin is None:
            Elower = y * (E - b*p(E))
        else:
            Elower = max(Emin, y * (E - b*p(E)))
         
        # Only do integration where the limits make sense
        if Elower > Eupper:
            new_dNdE[i] = 0
        else:
            intspec, err = quad(integrand, Elower, Eupper)
            new_dNdE[i] = 1/(2*b*y) * intspec

    return Spectrum(spec.eng, new_dNdE, spec_type='dNdE')

def get_pppc_spec(mDM, eng, pri, sec, decay=False):

    """ Returns the PPPC4DMID spectrum. 

    This is the secondary spectrum to e+e-/photons normalized to one annihilation or decay event to the species specified in ``pri``. These results include electroweak corrections. The full list of allowed channels is: 

    - :math:`\\delta`\ -function injections: ``elec_delta, phot_delta``
    - Leptons: ``e_L, e_R, e, mu_L, mu_R, mu, tau_L, tau_R, tau``
    - Quarks:  ``q, c, b, t``
    - Gauge bosons: ``gamma, g, W_L, W_T, W, Z_L, Z_T, Z``
    - Higgs: ``h``
    - Mesons: ``pi``

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
        # Find the correct bin in photeng.
        eng_to_inj = eng[eng < eng_phot][-1]
        # Place 2*eng_phot worth of photons into that bin. Use
        # rebinning to accomplish this. 
        if sec == 'elec':
            return Spectrum(
                eng, np.zeros_like(eng), spec_type='dNdE'
            )

        elif sec == 'phot':
            return rebin_N_arr(
                np.array([2 * eng_phot / eng_to_inj]), 
                np.array([eng_to_inj]), 
                eng
            )

        else:
            raise ValueError('invalid sec.')

    # Spectrum of electrons from decay/annihilation to muons 
    if pri == 'mu':
        if sec == 'elec':
            # Useful masses
            me = phys.me
            mu = phys.mass['mu']

            # eng is kinetic energy, eleceng is total energy, KE + me
            eleceng = eng + phys.me
            # Electron momentum
            pmu = np.sqrt(eleceng**2 - phys.me**2)
            # Lorentz gamma from boosting from muon rest frame to DM rest frame
            if not decay:
                y = mDM/mu
            else:
                y = mDM/2/mu

            # Formula for dNdE of electrons in muon rest frame
            dNdE_rest = 8*pmu/mu**2 * ( 2*eleceng/mu * (3-4*eleceng/mu) + phys.me**2/mu**2 * (6*eleceng/mu - 4) )
            dNdE_rest[eng > (mu**2 + phys.me**2)/(2*mu)] = 0.
            dNdE_rest = Spectrum(eng, dNdE_rest, spec_type='dNdE')

            # dNdE of electrons boosted to the dark matter frame
            dNdE_DM = boost_elec_spec(y, dNdE_rest, Emin=me, Emax=(mu**2 + me**2)/(2*mu))

            return dNdE_DM

        elif sec == 'phot':
            return Spectrum(
                eng, np.zeros_like(eng), spec_type='dNdE'
            )

    # Spectrum of electrons from decay/annihilation to charged pions    
    if pri == 'pi':
        if sec == 'elec':
            # Useful masses
            me = phys.me
            mu = phys.mass['mu']
            mp = phys.mass['pi']
            # eng is kinetic energy, eleceng is total energy, KE + m
            eleceng = eng + phys.me
            # Electron momentum
            pmu = np.sqrt(eleceng**2 - phys.me**2)
            # Lorentz gamma from boosting from muon rest frame to pion rest frame
            y1 = (mp**2 + mu**2)/(2*mp*mu)
            # Lorentz gamma from boosting from pion rest frame to DM rest frame
            if not decay:
                y2 = mDM/mp
            else:
                y2 = mDM/2/mp

            # Formula for dNdE of electrons in muon rest frame
            dNdE_rest = 8*pmu/mu**2 * ( 
                            2*eleceng/mu * (3-4*eleceng/mu) + phys.me**2/mu**2 * (6*eleceng/mu - 4) )
            dNdE_rest[eng > (mu**2 + phys.me**2)/(2*mu)] = 0.
            dNdE_rest = Spectrum(eng, dNdE_rest, spec_type='dNdE')

            # dNdE of electrons boosted to the pion frame
            dNdE_pi = boost_elec_spec(y1, dNdE_rest, Emin=me, Emax=(mu**2 + me**2)/(2*mu))
            # dNdE of electrons boosted to the dark matter frame
            next_cut = eng[np.where(dNdE_pi.dNdE==0)[0][1]]
            dNdE_DM = boost_elec_spec(y2, dNdE_pi, Emax=next_cut)

            return dNdE_DM

        elif sec == 'phot':
            return Spectrum(
                eng, np.zeros_like(eng), spec_type='dNdE'
            )

    # Spectrum of electrons from decay/annihilation to neutral pions
    if pri == 'pi0':
        # Pion mass
        mp = phys.mass['pi0']
        # Lorentz gamma from boosting from muon rest frame to DM rest frame
        if not decay:
            y = mDM/mp
        else:
            y = mDM/2/mp
        b = np.sqrt(1-(1/y**2))

        # Photon line emission from pion decay becomes
        # box spectrum when boosted
        dNdE = np.zeros_like(eng)
        # Box width
        Emin = y*(1-b) * mp/2
        Emax = y*(1+b) * mp/2
        dNdE[(eng > Emin) & (eng < Emax)] = 1/b/y/mp

        if sec == 'elec':
            return Spectrum(
                eng, np.zeros_like(eng), spec_type='dNdE'
            )
        if sec == 'phot':
            return Spectrum(
                eng, dNdE, spec_type='dNdE'
            )


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


