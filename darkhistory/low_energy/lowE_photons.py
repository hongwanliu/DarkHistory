import sys
sys.path.append("../..")

import numpy as np
import darkhistory.physics as phys
import darkhistory.spec.spectools as spectools

def kappa_DM(photon_spectrum, xe):
    """ Compute kappa_DM of the modified tla.

    Parameters
    ----------
    photon_spectrum : Spectrum object
        spectrum of photons. Assumed to be in dNdE mode. spec.toteng() should return Energy per baryon.

    Returns
    -------
    kappa_DM : float
        The added photoionization rate due to products of DM.
    """
    eng = photon_spectrum.eng
    R_Lya = phys.rate_2p1s(xe,rs)
    Lambda = phys.width_2s1s

    # The bin number containing 10.2eV
    lya_index = spectools.get_indx(eng, phys.lya_eng)

    # The bins between 10.2eV and 13.6eV
    exc_bounds = spectools.get_bounds_between(
        eng, phys.lya_eng, phys.rydberg
    )

    # Convenient variables
    rs = photon_spectrum.rs
    Tcmb = phys.TCMB(rs)
    Lambda = phys.width_2s1s

    # Effect on 2p state due to DM products
    kappa_2p = (
        photon_spectrum.dNdE[lya_index] *
        (phys.hbar * np.pi / phys.lya_eng)**2 * phys.c**3
    )

    # Effect on 2s state
    kappa_2s = 0

    # Effect on 2s state due to DM products

    return (kappa_2p*3*R_lya/4 + kappa_2s*Lambda/4)/(3*R_lya/4 + Lambda/4)

def compute_dep_inj_ratio(photon_spectrum, x, tot_inj, time_step, method='old'):
    """ Compute f(z) fractions for continuum photons, photoexcitation of HI, and photoionization of HI, HeI, HeII

    Given a spectrum of deposited photons, resolve its energy into continuum photons,
    HI excitation, and HI, HeI, HeII ionization in that order.

    Parameters
    ----------
    photon_spectrum : Spectrum object
        spectrum of photons. Assumed to be in dNdE mode. spec.toteng() should return energy per baryon.
    n : list of floats
        number of (HI, HeI, HeII) divided by nH at redshift photon_spectrum.rs
    tot_inj : float
        total energy injected by DM, dE/dVdt |_inj
    method : {'old','ion','new'}
        'old': All photons >= 13.6eV ionize hydrogen, within [10.2, 13.6)eV excite hydrogen, < 10.2eV are labelled continuum.
        'ion': Same as 'old', but now photons >= 13.6 can ionize HeI and HeII also.
        'new': Same as 'ion', but now [10.2, 13.6)eV photons treated more carefully.

    Returns
    -------
    tuple of floats
        Ratio of deposited energy to a given channel over energy deposited by DM.
        The order of the channels is {continuum photons, HI excitation, HI ionization, HeI ion, HeII ion}
    """
    f_continuum, f_excite_HI, f_HI, f_HeI, f_HeII = 0,0,0,0,0

    eng = photon_spectrum.eng
    rs = photon_spectrum.rs
    n = x * phys.nH * rs**3

    # norm_factor converts from total deposited energy to f_c(z) = (dE/dVdt)dep / (dE/dVdt)inj
    norm_factor = phys.nB / time_step / tot_inj

    # All photons below 10.2eV get deposited into the continuum
    f_continuum = photon_spectrum.toteng(
        bound_type='eng',
        bound_arr=np.array([eng[0],phys.lya_eng])
    )[0] * norm_factor

    # Treatment of photoexcitation

    # The bin number containing 10.2eV
    lya_index = spectools.get_indx(eng, phys.lya_eng)
    # The bin number containing 13.6eV
    ryd_index = spectools.get_indx(eng, phys.rydberg)

    if(method != 'new'):
        # All photons between 10.2eV and 13.6eV are deposited into excitation
        tot_excite_eng = (
            photon_spectrum.toteng(
                bound_type='eng',
                bound_arr=np.array([phys.lya_eng,phys.rydberg])
            )[0]
        )
        f_excite_HI = tot_excite_eng * norm_factor
    else:
        # Only photons in the 10.2eV bin participate in 1s->2p excitation.
        # 1s->2s transition handled more carefully.

        # Convenient variables
        Tcmb = phys.TCMB(rs)

        chi = phys.nHe/phys.nH
        xHeIII = chi - x[1] - x[2]
        xHII = 1 - x[0]
        xe = xHII + x[2] + 2*xHeIII

        beta = phys.beta_ion(Tcmb)
        peebC = phys.peebles_C(xe,rs)
        peeb_numerator = 3*phys.rate_2p1s(xe,rs)/4 + phys.width_2s1s/4
        kappa = kappa_DM(photon_spectrum, xe)

        # When beta = 0, 1-C = 0, but their ratio is finite
        if(np.abs(beta/peeb_numerator) < 1e-8):
            const = peeb_numerator
        else:
            const = beta/(1-peebC)

        f_excite_HI = 4 * peebC * constant * kappa * phys.lya_eng * n[0] / tot_inj

    # Treatment of photoionization

    # Bin boundaries of photon spectrum capable of photoionization, and number of photons in those bounds.
    ion_bounds = spectools.get_bounds_between(eng, phys.rydberg, eng[-1])
    ion_Ns = photon_spectrum.totN(bound_type='eng', bound_arr=ion_bounds)

    if method == 'old':
        # All photons above 13.6 eV deposit their 13.6eV into HI ionization
        tot_ion_eng = phys.rydberg * ion_Ns
        f_HI = tot_ion_eng * norm_factor
    else:
        # Photons may also deposit their energy into HeI and HeII single ionization

        # Probability of being absorbed within time step dt in channel a is P_a = \sigma(E)_a n_a c*dt
        ionHI, ionHeI, ionHeII = [phys.photo_ion_xsec(photon_spectrum.eng[ryd_index:],channel) * n[i]
                                  for i,channel in enumerate(['H0','He0','He1'])]

        # Relative likelihood of photoionization of HI is then P_HI/sum(P_a)
        totList = ionHI + ionHeI + ionHeII
        ionHI, ionHeI, ionHeII = [ llist/totList for llist in [ionHI, ionHeI, ionHeII] ]

        f_HI, f_HeI, f_HeII = [
            sum(ion_Ns * llist * norm_factor)
            for llist in [phys.rydberg*ionHI, phys.He_ion_eng*ionHeI, 4*phys.rydberg*ionHeII]
        ]

    return f_continuum, f_excite_HI, f_HI, f_HeI, f_HeII
