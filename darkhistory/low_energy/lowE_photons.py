import sys
sys.path.append("../..")

import numpy as np
import darkhistory.physics as phys
import darkhistory.spec.spectools as spectools

def compute_dep_inj_ratio(photon_spectrum, n, tot_inj, method='old'):
    """ Given a spectrum of deposited photons, resolve its energy into continuum photons, HI excitation, and HI, HeI, HeII ionization in that order.  The
        spectrum must provide the energy density of photons per unit time within each bin, not just the total energy within each bin.
        Q: can photons heat the IGM?  Should this method keep track of the fact that x_e, xHII, etc. are changing?

    Parameters
    ----------
    photon_spectrum : Spectrum object
        spectrum of photons. Assumed to be in dNdE mode. spec.toteng() should return dE/dVdt.
    n : list of floats
        density of (HI, HeI, HeII) at redshift photon_spectrum.rs
    tot_inj : float
        total energy injected by DM
    method : {'old','ion','new'}
        'old': All photons >= 13.6eV ionize hydrogen, within [10.2, 13.6)eV excite hydrogen, < 10.2eV are labelled continuum.
        'ion': Same as 'old', but now photons >= 13.6 can ionize HeI and HeII also.
        'new': Same as 'ion', but now [10.2, 13.6)eV photons treated more carefully

    Returns
    -------
    tuple of floats
        Ratio of deposited energy to a given channel over energy deposited by DM.
        The order of the channels is {continuum photons, HI excitation, HI ionization, HeI ion, HeII ion}
    """
    f_continuum, excite_HI, f_HI, f_HeI, f_HeII = 0,0,0,0,0

    # All photons below 10.2eV get deposited into the continuum
    f_continuum = photon_spectrum.toteng(bound_type='eng', bound_arr=np.array([photon_spectrum.eng[0],phys.lya_eng]))[0]/tot_inj

    # eng[ion_index] (eng[lya_index]) is the first energy greater than phys.rydberg (phys.lya_eng)
    ion_index = np.searchsorted(photon_spectrum.eng,phys.rydberg)
    lya_index = np.searchsorted(photon_spectrum.eng,phys.lya_eng)

    # needs explanation
    ion_bin = spectools.get_bin_bound(photon_spectrum.eng)[ion_index]

    if(method != 'new'):
        # All photons between 10.2eV and 13.6eV are deposited into excitation
        f_excite_HI = photon_spectrum.toteng(bound_type='eng', bound_arr=np.array([phys.lya_eng,phys.rydberg]))[0]/tot_inj
    else:
        # Only photons in the 10.2eV bin participate in excitation

        # Convenient variables
        rs = photon_spectrum.rs
        Tcmb = phys.TCMB(rs)
        xe = sum(n)/phys.nH
        beta = phys.beta_ion(Tcmb)
        peebC = phys.peebles_C(xe,rs)
        factor = phys.rate_factor(xe, rs)

        # interpolated photon_spectrum.dNdE evaluated at E_lya
        dn_DM = (
            (photon_spectrum.dNdE[lya_index] - photon_spectrum.dNdE[lya_index-1])/
            (photon_spectrum.eng[lya_index] - photon_spectrum.eng[lya_index-1])
            *(phys.lya_eng - photon_spectrum.eng[lya_index-1]) + photon_spectrum.dNdE[lya_index-1]
        )

        # When beta = 0, 1-C = 0, but their ratio is finite
        if(np.abs(beta/factor) < 1e-8):
            const = factor
        else:
            const = beta/(1-peebC)

        f_excite_HI = (
            4 * dn_DM * n[0] * np.pi**2 * (phys.hbar * phys.c)**3 *
            peebC * factor / (phys.lya_eng * tot_inj)
        )

    if(method == 'old'):
        # All photons above 13.6 ionize HI
        f_HI = photon_spectrum.toteng(bound_type='eng', bound_arr=np.array([phys.rydberg,photon_spectrum.eng[-1]]))[0]/tot_inj
    else:
        # Probability of being absorbed within time step dt in channel a = \sigma(E)_a n_a c*dt
        # First convert from probability of being absorbed in channel 'a' to conditional probability given that these are deposited photons
        ionHI, ionHeI, ionHeII = [phys.photo_ion_xsec(photon_spectrum.eng[ion_index:],channel)*n[i] for i,channel in enumerate(['H0','He0','He1'])]
        totList = ionHI + ionHeI + ionHeII

        f_HI, f_HeI, f_HeII = [sum(photon_spectrum.eng[ion_index:]*photon_spectrum.N[ion_index:]*llist/totList)/tot_inj for llist in [ionHI, ionHeI, ionHeII]]

        #There's an extra piece of energy between 13.6 and the energy at ion_index. We expect 100% ionization within this region, so
        extra_HI, extra_HeI, extra_HeII = [phys.photo_ion_xsec(np.array([phys.rydberg+ion_bin])/2,channel)*n[i] for i, channel in enumerate(['H0','He0','He1'])]
        tot = extra_HI + extra_HeI + extra_HeII
        df_HI, df_HeI, df_HeII = [(photon_spectrum.toteng(bound_type='eng',bound_arr=np.array([phys.rydberg,ion_bin]))*extra/tot)[0]/tot_inj for extra in [extra_HI, extra_HeI, extra_HeII]]
        f_HI = f_HI + df_HI
        f_HeI = f_HeI + df_HeI
        f_HeII = f_HeII + df_HeII

    return f_continuum, f_excite_HI, f_HI, f_HeI, f_HeII
